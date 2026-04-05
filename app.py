import streamlit as st
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
import joblib
import os
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import plotly.graph_objects as go

from unet_model import SimpleUNet

# ==========================================
# 0. 基础配置与多语言字典
# ==========================================
torch.backends.cudnn.enabled = False

st.set_page_config(page_title="MPP 辅助诊断系统", page_icon="🫁", layout="wide")

# 多语言配置字典
LANG = {
    "zh": {
        "title": "🫁 儿童肺炎支原体肺炎(MPP)智能辅助诊断系统",
        "step1": "📥 第 1 步：上传胸部 X 线片 (支持 .dcm 格式)",
        "step2": "🧑‍⚕️ 第 2 步：输入患儿年龄",
        "age_label": "患儿年龄 (岁)",
        "step3": "🚀 第 3 步：开始智能分析",
        "btn_run": "运行全流程预测",
        "init_sys": "正在初始化系统资产...",
        "processing": "正在处理图像并提取组学特征...",
        "col_std": "🖼️ 原始影像标准化",
        "col_seg": "🧪 全自动肺部分割",
        "cap_std": "标准化胸片 (1024x1024)",
        "cap_seg": "全自动分割结果 (绿色标注区域)",
        "res_title": "📊 预测分析结果",
        "tab_svm": "SVM 独立预测",
        "tab_integrated": "多模型综合分析",
        "prob_label": "患病概率",
        "x_axis": "风险概率",
        "avg_risk": "综合加权风险",
        "high_risk": "结论：高风险 (建议进一步临床检查)",
        "low_risk": "结论：低风险 (常规观察)",
        "info": "👈 请按步骤上传文件并输入信息进行分析。"
    },
    "en": {
        "title": "🫁 Pediatric MPP Intelligent Diagnostic System",
        "step1": "📥 Step 1: Upload Chest X-ray (.dcm)",
        "step2": "🧑‍⚕️ Step 2: Enter Patient Age",
        "age_label": "Patient Age (Years)",
        "step3": "🚀 Step 3: Start Intelligent Analysis",
        "btn_run": "Run Prediction Pipeline",
        "init_sys": "Initializing system assets...",
        "processing": "Processing image & extracting radiomics...",
        "col_std": "🖼️ Standardized Image",
        "col_seg": "🧪 Auto Lung Segmentation",
        "cap_std": "Standardized X-ray (1024x1024)",
        "cap_seg": "Auto-segmentation (Green area)",
        "res_title": "📊 Prediction Analysis",
        "tab_svm": "SVM Prediction",
        "tab_integrated": "Integrated Results",
        "prob_label": "Probability",
        "x_axis": "Probability of Risk",
        "avg_risk": "Weighted Avg Risk",
        "high_risk": "Conclusion: High Risk (Further clinical exam advised)",
        "low_risk": "Conclusion: Low Risk (Routine observation)",
        "info": "👈 Please follow the steps to upload file and analyze."
    }
}

if 'lang' not in st.session_state:
    st.session_state['lang'] = 'zh'

col_title, col_lang = st.columns([7, 3])
with col_lang:
    lang_choice = st.radio("Language", options=['中文', 'English'], horizontal=True, label_visibility="collapsed")
    st.session_state['lang'] = 'zh' if lang_choice == '中文' else 'en'

t = LANG[st.session_state['lang']]

with col_title:
    st.title(t["title"])
st.divider()

# ==========================================
# 1. 资源加载
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

UNET_MODEL_PATH = MODELS_DIR / "cxr_unet_best.pth"
SCALER_PATH = MODELS_DIR / "04_StandardScaler.pkl"
IMPUTATION_PATH = MODELS_DIR / "04_Imputation_Means.pkl"
RADIOMICS_CONFIG_PATH = CONFIG_DIR / "radiomics_config.yaml"


@st.cache_resource
def get_segmentation_model():
    model = SimpleUNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(str(UNET_MODEL_PATH), map_location='cpu'))
    model.eval()
    return model


@st.cache_resource
def get_ml_assets():
    scaler = joblib.load(str(SCALER_PATH))
    impute_values = joblib.load(str(IMPUTATION_PATH))
    model_names = ["LogisticRegression", "DecisionTree", "RandomForest", "MLP", "SVM", "XGBoost", "LightGBM",
                   "GaussianNB", "KNN"]
    models = {}
    for name in model_names:
        models[name] = joblib.load(str(MODELS_DIR / f"06_{name}_best_model.pkl"))
    return scaler, impute_values, models


# ==========================================
# 2. 图像处理函数
# ==========================================
def apply_standardization(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
        ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
        img = np.clip(img, float(wc) - float(ww) / 2, float(wc) + float(ww) / 2)
        img = (img - (float(wc) - float(ww) / 2)) / float(ww) * 255.0
    else:
        img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
    if np.mean([img[0:10, 0:10], img[0:10, -10:], img[-10:, 0:10], img[-10:, -10:]]) > 128:
        img = 255.0 - img
    target_size = (1024, 1024)
    old_h, old_w = img.shape[:2]
    scale = min(target_size[0] / old_w, target_size[1] / old_h)
    new_w, new_h = int(old_w * scale), int(old_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(target_size, dtype=np.uint8)
    top, left = (target_size[1] - new_h) // 2, (target_size[0] - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized.astype(np.uint8)
    return canvas


def run_segmentation(img_array, model):
    img_input = cv2.resize(img_array, (512, 512)) / 255.0
    img_tensor = torch.from_numpy(img_input).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)


def create_visual_overlay(img, mask):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask_color = np.zeros_like(img_color)
    mask_color[mask == 1] = [0, 255, 0]
    overlay = cv2.addWeighted(img_color, 0.7, mask_color, 0.3, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


# ==========================================
# 3. 主交互逻辑
# ==========================================
st.subheader(t["step1"])
uploaded_file = st.file_uploader("", type=['dcm'], label_visibility="collapsed")

if uploaded_file:
    st.subheader(t["step2"])
    age = st.number_input(t["age_label"], min_value=0.0, max_value=18.0, value=6.0, step=0.1)

    st.subheader(t["step3"])
    if st.button(t["btn_run"], type="primary"):
        with st.spinner(t["init_sys"]):
            seg_model = get_segmentation_model()
            scaler, impute_values, ml_models = get_ml_assets()

        # 图像处理
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{t['col_std']}**")
            std_img = apply_standardization(uploaded_file)
            st.image(std_img, caption=t["cap_std"], use_column_width=True)
        with col2:
            st.markdown(f"**{t['col_seg']}**")
            mask = run_segmentation(std_img, seg_model)
            overlay_view = create_visual_overlay(std_img, mask)
            st.image(overlay_view, caption=t["cap_seg"], use_column_width=True)

        # 特征提取与预测
        st.divider()
        st.subheader(t["res_title"])

        with st.spinner(t["processing"]):
            # 组学提取
            sitk_img = sitk.GetImageFromArray(std_img)
            sitk_mask = sitk.GetImageFromArray(mask)
            extractor = featureextractor.RadiomicsFeatureExtractor(str(RADIOMICS_CONFIG_PATH))
            features = extractor.execute(sitk_img, sitk_mask)

            # 数据对齐与标准化
            raw_feature_dict = {k.replace('original_', ''): float(v) for k, v in features.items() if
                                'diagnostics' not in k}
            all_radiomics_cols = list(scaler.feature_names_in_)
            full_df = pd.DataFrame([raw_feature_dict]).reindex(columns=all_radiomics_cols).fillna(impute_values)
            scaled_full_df = pd.DataFrame(scaler.transform(full_df), columns=all_radiomics_cols)

            # 最终 10 特征顺序 (需与训练时严格一致)
            FINAL_FEATURE_ORDER = ['age', 'firstorder_InterquartileRange', 'shape2D_MeshSurface',
                                   'firstorder_MeanAbsoluteDeviation', 'gldm_SmallDependenceHighGrayLevelEmphasis',
                                   'glcm_SumEntropy', 'firstorder_90Percentile', 'firstorder_Entropy',
                                   'glcm_ClusterTendency', 'glcm_JointEntropy']
            input_for_prediction = pd.DataFrame(columns=FINAL_FEATURE_ORDER)
            input_for_prediction.loc[0, 'age'] = age
            for col in FINAL_FEATURE_ORDER:
                if col != 'age': input_for_prediction.loc[0, col] = scaled_full_df.loc[0, col]

            # 安全重命名
            input_for_prediction.columns = [
                c.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_') for c in
                input_for_prediction.columns]
            X_final = input_for_prediction.astype(float)

            # 算法预测
            results = {name: model.predict_proba(X_final)[0, 1] for name, model in ml_models.items()}

            # AUC 加权计算
            auc_weights = {"LightGBM": 0.710298, "XGBoost": 0.707560, "SVM": 0.705357, "MLP": 0.696786,
                           "RandomForest": 0.692857, "LogisticRegression": 0.686190, "KNN": 0.681131,
                           "DecisionTree": 0.658571, "GaussianNB": 0.587857}
            total_auc = sum(auc_weights.values())
            weighted_avg_risk = sum(results[name] * (auc_weights[name] / total_auc) for name in results)

        # ==========================================
        # 结果展示：标签页切换
        # ==========================================
        tab1, tab2 = st.tabs([t["tab_svm"], t["tab_integrated"]])

        # ---------- 优化点：SVM 独立预测改为上下排列 ----------
        with tab1:
            svm_prob = results.get("SVM", 0.0)

            # 1. 顶部：大字体展示风险结论
            st.metric(f"SVM {t['prob_label']}", f"{svm_prob:.1%}")
            if svm_prob > 0.5:
                st.error(t["high_risk"])
            else:
                st.success(t["low_risk"])

            # 2. 下部：全宽展示仪表盘图表
            fig_svm = go.Figure(go.Indicator(
                mode="gauge+number",
                value=svm_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"SVM {t['prob_label']}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#ff4b4b" if svm_prob > 0.5 else "#28a745"},
                    'steps': [{'range': [0, 0.5], 'color': "lightgray"}, {'range': [0.5, 1], 'color': "gray"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}
                }
            ))
            # 稍微增加了高度 (350)，使其在全宽下更美观
            fig_svm.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_svm, use_column_width=True)

        # ---------- 综合模型页 ----------
        with tab2:
            st.metric(f"{t['avg_risk']} (AUC Weighted)", f"{weighted_avg_risk:.1%}")
            if weighted_avg_risk > 0.5:
                st.error(t["high_risk"])
            else:
                st.success(t["low_risk"])

            # 综合模型对比柱状图
            res_df = pd.DataFrame({"Algorithm": list(results.keys()), "Prob": list(results.values())}).sort_values(
                "Prob")
            fig_all = go.Figure(go.Bar(
                x=res_df["Prob"],
                y=res_df["Algorithm"],
                orientation='h',
                marker=dict(color=res_df["Prob"], colorscale='RdYlBu_r')
            ))
            fig_all.update_layout(xaxis_title=t["x_axis"], xaxis=dict(range=[0, 1]), height=400,
                                  margin=dict(l=0, r=0, t=30, b=0))
            fig_all.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_all, use_column_width=True)

else:
    st.info(t["info"])