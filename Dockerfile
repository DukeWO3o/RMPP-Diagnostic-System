# =============================================================================
# Hugging Face Spaces (Docker SDK) - 构建脚本
# 保持与本地一致的 Python 3.7 conda 环境, 端口 7860 (Spaces 默认端口)
# 部署方式: 把本 Dockerfile 放到 Space 仓库根目录, 与 app.py / unet_model.py /
#           config/ / models/ / environment.yml 放在同一层, push 后自动构建
# =============================================================================
FROM continuumio/miniconda3:4.12.0

LABEL maintainer="Radiomics Team"
LABEL description="RMPP Prediction Streamlit App (HF Spaces, Python 3.7)"

WORKDIR /app

# 1) 用 environment.yml 原样创建环境(radiomics 频道 + PyTorch CPU wheel)
COPY environment.yml /app/environment.yml
RUN conda config --add channels radiomics && \
    conda env create -f /app/environment.yml && \
    conda clean -afy

# 2) 拷贝应用代码与资产
COPY app.py unet_model.py /app/
COPY config /app/config
COPY models /app/models
COPY demo /app/demo

# 3) Spaces 默认监听 7860 端口
EXPOSE 7860
CMD ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "rmpp-app", \
     "streamlit", "run", "/app/app.py", \
     "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
