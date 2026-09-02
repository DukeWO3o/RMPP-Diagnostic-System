# RMPP-APP · 儿童肺炎支原体肺炎(MPP)智能辅助诊断系统

A bilingual (中文/English) Streamlit web app: upload a pediatric chest X-ray
(DICOM) → automatic lung segmentation (U-Net) → PyRadiomics feature extraction
→ ensemble ML prediction of refractory Mycoplasma pneumoniae pneumonia (RMPP).

## Run locally (Python 3.7 conda env)

```bash
conda env create -f environment.yml   # creates env "rmpp-app"
conda activate rmpp-app
streamlit run app.py
```

Open http://localhost:8501

## Quick demo (no upload needed)

Two de-identified demo chest X-rays are bundled under `demo/` — click
「示例①/示例②」on the page to run the full pipeline with one click.
Clinical inputs are pre-filled and can be edited.

> The demo DICOMs contain **no patient identifiers** (all PHI tags removed,
> pixel data preserved). Do not upload real patient images to public servers.

## Repository layout

```
app.py                  Streamlit main app
unet_model.py           U-Net segmentation network
config/                 PyRadiomics extraction config
models/                 Trained models + scaler + model_config.json
                        (cxr_unet_best.pth is Git-LFS, ~118 MB)
demo/                   De-identified demo DICOMs (one-click demo)
environment.yml         Conda env (python=3.7.1)
```

## Notes

- Requires Python 3.7.x; see `environment.yml` for pinned dependencies
  (pyradiomics via conda `radiomics` channel, torch 1.13.1+cpu).
- `models/model_config.json` declares the active feature signature; retrain
  artifacts replace the same files to update the app.
- Medical-use caution: intended for research/demonstration; validate locally
  before any clinical deployment.
