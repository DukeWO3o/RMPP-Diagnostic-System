import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# 针对 AMD 新架构显卡 MIOpen 编译报错的绕过方案
# 在 ROCm 版本中，这会禁用 MIOpen 并回退到原生 ATen 算子
# ==========================================
torch.backends.cudnn.enabled = False

# ==========================================
# 1. 超参数与路径配置 (请核对你的路径)
# ==========================================
class Config:
    IMAGE_DIR = 'F:/2023original/DCM-RMPP/01_PNG'
    MASK_DIR = 'F:/2023original/DCM-RMPP/03_Binarized'
    SAVE_DIR = './models'

    IMG_SIZE = 512  # 训练分辨率，512 是速度和精度的极佳平衡
    BATCH_SIZE = 4  # 显存不足可调为 2；显存充裕可调为 8 或 16
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    VAL_SPLIT = 0.2  # 20% 数据用于验证
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


os.makedirs(Config.SAVE_DIR, exist_ok=True)


# ==========================================
# 2. 网络结构定义 (需与部署代码 100% 一致)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(SimpleUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)

        return self.outc(x)


# ==========================================
# 3. 数据集与数据加载器
# ==========================================
class CXRDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=512):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

        # 基础的数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取为单通道灰度图
        img = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = self.transform(img)
        mask = self.mask_transform(mask)

        # 确保 Mask 只有 0 和 1
        #mask = torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        mask = (mask > 0).float()
        return img, mask


# ==========================================
# 4. 损失函数 (BCE + Dice)
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # 先计算 BCE (注意 inputs 不需要过 sigmoid，因为 BCEWithLogitsLoss 内置了)
        BCE = self.bce(inputs, targets)

        # 计算 Dice
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return BCE + dice_loss


def calculate_dice_score(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# ==========================================
# 5. 主训练流程
# ==========================================
def main():
    print(f"🔥 开始训练准备，使用设备: {Config.DEVICE}")

    # 匹配文件名获取路径列表
    filenames = [f for f in os.listdir(Config.IMAGE_DIR) if f.endswith('.png')]
    image_paths = [os.path.join(Config.IMAGE_DIR, f) for f in filenames]
    # 假设你的 Mask 文件名与 Image 相同
    mask_paths = [os.path.join(Config.MASK_DIR, f) for f in filenames]

    # 划分训练集和验证集
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=Config.VAL_SPLIT, random_state=42
    )

    train_dataset = CXRDataset(train_imgs, train_masks, img_size=Config.IMG_SIZE)
    val_dataset = CXRDataset(val_imgs, val_masks, img_size=Config.IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型、优化器和损失函数
    model = SimpleUNet(n_channels=1, n_classes=1).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = DiceBCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda' if 'cuda' in Config.DEVICE else 'cpu')  # 自动混合精度

    best_dice = 0.0

    # 开始 Epoch 循环
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{Config.EPOCHS}] Train")

        for images, masks in train_pbar:
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

            optimizer.zero_grad()

            # 使用混合精度加速前向传播
            with torch.amp.autocast('cuda' if 'cuda' in Config.DEVICE else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            # 反向传播与优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 验证流程
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                with torch.amp.autocast('cuda' if 'cuda' in Config.DEVICE else 'cpu'):
                    outputs = model(images)
                val_dice += calculate_dice_score(outputs, masks).item()

        avg_val_dice = val_dice / len(val_loader)
        print(f"📈 验证集平均 Dice 系数: {avg_val_dice:.4f} | 训练集 Loss: {avg_train_loss:.4f}")

        # 根据验证集的 Dice 系数调整学习率
        scheduler.step(avg_val_dice)

        # 保存最佳模型权重
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            save_path = os.path.join(Config.SAVE_DIR, 'cxr_unet_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"🌟 发现更好的模型！已保存至 {save_path} (Dice: {best_dice:.4f})")

    print(f"✅ 训练完成！最佳验证集 Dice 系数: {best_dice:.4f}")


if __name__ == '__main__':
    main()