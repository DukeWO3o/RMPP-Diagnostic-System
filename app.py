import streamlit as st
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
import joblib
import json
import os
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import plotly.graph_objects as go

from unet_model import SimpleUNet
import io

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
        "step2": "🧑‍⚕️ 第 2 步：输入患儿年龄及实验室指标",
        "age_years_label": "患儿年龄 (岁)",
        "age_months_label": "患儿年龄 (月)",
        "crp_label": "CRP (mg/L)",
        "wbc_label": "WBC (×10⁹/L)",
        "ldh_label": "LDH (U/L)",
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
        "step2": "🧑‍⚕️ Step 2: Enter age & lab values",
        "age_years_label": "Patient Age (Years)",
        "age_months_label": "Patient Age (Months)",
        "crp_label": "CRP (mg/L)",
        "wbc_label": "WBC (×10⁹/L)",
        "ldh_label": "LDH (U/L)",
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
CLINICAL_SCALER_PATH = MODELS_DIR / "04_ClinicalScaler.pkl"
IMPUTATION_PATH = MODELS_DIR / "04_Imputation_Means.pkl"
RADIOMICS_CONFIG_PATH = CONFIG_DIR / "radiomics_config.yaml"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# 演示模式配置：demo/ 目录下内置"脱敏"示例胸片，供审稿人/访客一键体验
DEMO_DIR = BASE_DIR / "demo"
DEMO_FILES = sorted(DEMO_DIR.glob("*.dcm")) if DEMO_DIR.is_dir() else []
# 每份示例图对应的建议临床输入（仅作演示默认值，可在页面上修改）
DEMO_CLIN_SUGGEST = {
    "rmpp_case_positive.dcm": ("示例①：肺炎支原体肺炎患儿胸片", 6, 0, 18.0, 11.0, 390.0),
    "rmpp_case_negative.dcm": ("示例②：非肺炎支原体对照患儿胸片", 4, 11, 1.5, 8.0, 300.0),
}


@st.cache_resource
def get_segmentation_model():
    model = SimpleUNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(str(UNET_MODEL_PATH), map_location='cpu'))
    model.eval()
    return model


def load_model_config():
    if MODEL_CONFIG_PATH.exists():
        with open(str(MODEL_CONFIG_PATH), "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def get_ml_assets():
    scaler = joblib.load(str(SCALER_PATH))
    impute_values = joblib.load(str(IMPUTATION_PATH))
    clinical_scaler = None
    if CLINICAL_SCALER_PATH.exists():
        clinical_scaler = joblib.load(str(CLINICAL_SCALER_PATH))
    cfg = load_model_config()
    if cfg and cfg.get("model_names"):
        model_names = cfg["model_names"]
    else:
        model_names = ["LogisticRegression", "DecisionTree", "RandomForest", "MLP", "SVM", "XGBoost",
                       "LightGBM", "GaussianNB", "KNN"]
    models = {}
    for name in model_names:
        models[name] = joblib.load(str(MODELS_DIR / f"06_{name}_best_model.pkl"))
    return scaler, impute_values, clinical_scaler, models, cfg


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

# ---- 演示模式：一键加载内置脱敏示例（无需下载/上传） ----
class _DemoFile(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name

if uploaded_file is not None:
    st.session_state["demo_active"] = False

if "demo_active" not in st.session_state:
    st.session_state["demo_active"] = False

if uploaded_file is None and DEMO_FILES:
    st.markdown("**💡 快速体验（内置脱敏示例胸片，无需上传）：**")
    btns = st.columns(len(DEMO_FILES))
    for col, f in zip(btns, DEMO_FILES):
        meta = DEMO_CLIN_SUGGEST.get(f.name)
        btn_label = meta[0] if meta else f.stem
        with col:
            if st.button(btn_label):
                st.session_state["demo_active"] = True
                st.session_state["demo_bytes"] = f.read_bytes()
                st.session_state["demo_name"] = f.name
                st.session_state["demo_auto_run"] = True
                if meta:
                    st.session_state["demo_age_y"] = meta[1]
                    st.session_state["demo_age_m"] = meta[2]
                    st.session_state["demo_crp"] = meta[3]
                    st.session_state["demo_wbc"] = meta[4]
                    st.session_state["demo_ldh"] = meta[5]

if uploaded_file is None and st.session_state.get("demo_active"):
    if st.button("✖ 关闭示例，改为自行上传"):
        st.session_state["demo_active"] = False
        st.session_state.pop("demo_bytes", None)
    else:
        b = st.session_state.get("demo_bytes")
        if b:
            uploaded_file = _DemoFile(b, st.session_state.get("demo_name", "demo.dcm"))

if uploaded_file:
    st.subheader(t["step2"])
    col_age_y, col_age_m = st.columns(2)
    with col_age_y:
        age_years = st.number_input(t["age_years_label"], min_value=0, max_value=18,
                                    value=int(st.session_state.get("demo_age_y", 6)), step=1)
    with col_age_m:
        age_months = st.number_input(t["age_months_label"], min_value=0, max_value=11,
                                     value=int(st.session_state.get("demo_age_m", 0)), step=1)
    # 自动换算为岁（小数），仅模型内部使用，不呈现给用户
    age = float(age_years) + float(age_months) / 12.0
    crp = st.number_input(t["crp_label"], min_value=0.0, max_value=500.0,
                          value=float(st.session_state.get("demo_crp", 10.0)), step=0.1)
    wbc = st.number_input(t["wbc_label"], min_value=0.0, max_value=100.0,
                          value=float(st.session_state.get("demo_wbc", 8.0)), step=0.1)
    ldh = st.number_input(t["ldh_label"], min_value=0.0, max_value=10000.0,
                          value=float(st.session_state.get("demo_ldh", 300.0)), step=1.0)

    st.subheader(t["step3"])
    if st.button(t["btn_run"], type="primary") or st.session_state.pop("demo_auto_run", False):
        with st.spinner(t["init_sys"]):
            seg_model = get_segmentation_model()
            scaler, impute_values, clinical_scaler, ml_models, model_cfg = get_ml_assets()

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
            clinical_inputs = {"age": age, "CRP": crp, "WBC": wbc, "LDH": ldh}
            cfg_pipe = bool(model_cfg) and model_cfg.get("pipeline_type") == "cv_internal_pipeline"

            if cfg_pipe:
                # ---------- 原始脚本的 CV 内部特征选择 pipeline 模型 ----------
                feature_order = model_cfg["feature_order"]
                full_row = {}
                for col in feature_order:
                    full_row[col] = clinical_inputs.get(col, raw_feature_dict.get(col, np.nan))
                full_df = pd.DataFrame([full_row]).reindex(columns=feature_order)
                if impute_values is not None:
                    full_df = full_df.fillna(pd.Series(impute_values).reindex(feature_order))
                scaled_full_df = pd.DataFrame(scaler.transform(full_df), columns=feature_order)

                sel_feats_map = model_cfg.get("model_selected_features", {})
                results = {}
                for name, model in ml_models.items():
                    feats = sel_feats_map.get(name)
                    X_model = scaled_full_df[feats] if feats else scaled_full_df
                    results[name] = float(model.predict_proba(X_model)[0, 1])
                auc_weights = model_cfg.get("internal_valid_auc", {}) or model_cfg.get("auc_weights", {})
            else:
                # ---------- 原有 app-ready 固定签名模型 ----------
                all_radiomics_cols = list(scaler.feature_names_in_)
                full_df = pd.DataFrame([raw_feature_dict]).reindex(columns=all_radiomics_cols).fillna(impute_values)
                scaled_full_df = pd.DataFrame(scaler.transform(full_df), columns=all_radiomics_cols)

                if model_cfg and model_cfg.get("final_feature_order"):
                    FINAL_FEATURE_ORDER = model_cfg["final_feature_order"]
                else:
                    FINAL_FEATURE_ORDER = ['age', 'firstorder_InterquartileRange', 'shape2D_MeshSurface',
                                           'firstorder_MeanAbsoluteDeviation', 'gldm_SmallDependenceHighGrayLevelEmphasis',
                                           'glcm_SumEntropy', 'firstorder_90Percentile', 'firstorder_Entropy',
                                           'glcm_ClusterTendency', 'glcm_JointEntropy']
                if clinical_scaler is not None:
                    clinical_scaler_cols = list(clinical_scaler.feature_names_in_)
                    clinical_df = pd.DataFrame([clinical_inputs]).reindex(columns=clinical_scaler_cols)
                    clinical_scaled = pd.DataFrame(clinical_scaler.transform(clinical_df),
                                                   columns=clinical_scaler_cols)
                else:
                    clinical_scaled = pd.DataFrame([clinical_inputs])
                input_for_prediction = pd.DataFrame(columns=FINAL_FEATURE_ORDER)
                for col in FINAL_FEATURE_ORDER:
                    if col in clinical_inputs:
                        input_for_prediction.loc[0, col] = clinical_scaled.loc[0, col]
                    else:
                        input_for_prediction.loc[0, col] = scaled_full_df.loc[0, col]
                input_for_prediction.columns = [
                    c.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(':', '_') for c in
                    input_for_prediction.columns]
                X_final = input_for_prediction.astype(float)
                results = {name: model.predict_proba(X_final)[0, 1] for name, model in ml_models.items()}
                auc_weights = model_cfg.get("auc_weights", {}) if model_cfg else {}

            # AUC 加权计算
            if not auc_weights:
                auc_weights = {"LightGBM": 0.710298, "XGBoost": 0.707560, "SVM": 0.705357, "MLP": 0.696786,
                               "RandomForest": 0.692857, "LogisticRegression": 0.686190, "KNN": 0.681131,
                               "DecisionTree": 0.658571, "GaussianNB": 0.587857}
            total_auc = sum(auc_weights.values())
            weighted_avg_risk = sum(results[name] * (auc_weights[name] / total_auc) for name in results)

            # 确定最佳模型：优先读配置，其次按内部验证 AUC 最大值，最后回退 LightGBM
            best_model_name = model_cfg.get("best_model") if model_cfg else None
            if not best_model_name and auc_weights:
                best_model_name = max(auc_weights, key=auc_weights.get)
            best_model_name = best_model_name or "LightGBM"

        # ==========================================
        # 结果展示：标签页切换
        # ==========================================
        tab1_label = t["tab_svm"].replace("SVM", best_model_name)
        tab1, tab2 = st.tabs([tab1_label, t["tab_integrated"]])

        # ---------- 最佳模型独立预测（当前最佳模型优先展示） ----------
        with tab1:
            best_prob = results.get(best_model_name, results.get("SVM", 0.0))

            # 1. 顶部：大字体展示风险结论
            st.metric(f"{best_model_name} {t['prob_label']}", f"{best_prob:.1%}")
            if best_prob > 0.5:
                st.error(t["high_risk"])
            else:
                st.success(t["low_risk"])

            # 2. 下部：全宽展示仪表盘图表
            fig_best = go.Figure(go.Indicator(
                mode="gauge+number",
                value=best_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{best_model_name} {t['prob_label']}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#ff4b4b" if best_prob > 0.5 else "#28a745"},
                    'steps': [{'range': [0, 0.5], 'color': "lightgray"}, {'range': [0.5, 1], 'color': "gray"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}
                }
            ))
            # 稍微增加了高度 (350)，使其在全宽下更美观
            fig_best.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_best, use_column_width=True)

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