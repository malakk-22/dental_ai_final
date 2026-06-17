# Dental AI System — Graduation Project
## Complete Setup & Training Guide

---

## What This Project Does
A full AI pipeline that takes any dental X-ray and produces:
- **Tooth segmentation** (U-Net) — outlines every tooth and root
- **Pathology detection** (YOLOv8) — finds caries, lesions, bone loss
- **Diagnosis summary** — clinical text report
- **Treatment plan** — prioritized procedures

---

## Project Structure
```
dental_ai/
├── data/               ← put your downloaded datasets here
├── models/             ← saved model weights go here
├── training/           ← training scripts
│   ├── train_yolo.py
│   ├── train_unet.py
│   └── dataset_prep.py
├── api/                ← FastAPI backend
│   ├── main.py
│   └── inference.py
├── notebooks/          ← Jupyter exploration notebooks
└── utils/              ← helper functions
```

---

## Step-by-Step Instructions

### STEP 1 — Install requirements
```bash
pip install -r requirements.txt
```

### STEP 2 — Download datasets (free)
Run:
```bash
python training/dataset_prep.py
```
This downloads and prepares the UFBA-UESC dataset automatically.

### STEP 3 — Train detection model (YOLOv8)
```bash
python training/train_yolo.py --epochs 100 --batch 16
```

### STEP 4 — Train segmentation model (U-Net)
```bash
python training/train_unet.py --epochs 50 --batch 8
```

### STEP 5 — Start the API server
```bash
uvicorn api.main:app --reload --port 8000
```

### STEP 6 — Test it
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_xray.jpg"
```

---

## Hardware Requirements
- **Minimum:** 8GB RAM, any GPU (GTX 1060+)
- **Recommended:** 16GB RAM, RTX 3060+ or Google Colab Pro
- **Training time:** ~3 hours on RTX 3060, ~6 hours on Colab free
