"""
train_yolo.py
=============
Train a YOLOv8 model to detect dental pathologies in X-rays.

What it detects:
  tooth, caries, periapical lesion, bone loss,
  restoration, impacted tooth, calculus, crown

Usage:
    # Quick test (10 epochs)
    python training/train_yolo.py --epochs 10 --batch 8

    # Full training
    python training/train_yolo.py --epochs 100 --batch 16

    # Resume interrupted training
    python training/train_yolo.py --resume

Output:
    models/yolo_dental/weights/best.pt   ← use this for inference
    models/yolo_dental/weights/last.pt
    models/yolo_dental/results.csv       ← training metrics
"""

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"
MODEL_DIR = ROOT / "models" / "yolo_dental"

# ─── Augmentation config (dental-specific) ───────────────────────────────────
# X-rays are grayscale and low-contrast — these settings help a lot
AUG_CONFIG = dict(
    hsv_h=0.0,          # no hue shift (grayscale)
    hsv_s=0.0,          # no saturation shift
    hsv_v=0.4,          # brightness variation ±40% (simulates exposure differences)
    degrees=5,          # small rotation (patient positioning variation)
    translate=0.1,      # small translation
    scale=0.2,          # zoom variation
    fliplr=0.5,         # horizontal flip (left/right jaw symmetry)
    flipud=0.0,         # no vertical flip (teeth always point same way)
    mosaic=0.5,         # mosaic augmentation
    mixup=0.1,          # mixup for harder examples
    copy_paste=0.1,     # copy-paste augmentation
)


def train(epochs: int = 100, batch: int = 16, imgsz: int = 640,
          resume: bool = False, device: str = "auto"):
    """Train YOLOv8 on the dental dataset."""

    # Validate dataset exists
    if not DATA_YAML.exists():
        print(f"[!] Dataset not found at {DATA_YAML}")
        print("    Run: python training/dataset_prep.py first")
        return

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch:  {batch}")
    print(f"  ImgSz:  {imgsz}×{imgsz}\n")

    # Load base model
    # yolov8n = nano (fastest, least accurate) — good for Colab free tier
    # yolov8s = small (good balance)
    # yolov8m = medium (best for graduation project)
    if resume and (MODEL_DIR / "weights" / "last.pt").exists():
        print("  Resuming from last checkpoint...")
        model = YOLO(str(MODEL_DIR / "weights" / "last.pt"))
    else:
        model = YOLO("yolov8m.pt")  # downloads pretrained weights automatically
        print("  Loaded YOLOv8-medium pretrained on COCO")

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "models"),
        name="yolo_dental",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=30,            # early stopping if no improvement for 30 epochs
        save=True,
        save_period=10,         # checkpoint every 10 epochs
        val=True,
        plots=True,             # saves training plots as PNG
        verbose=True,
        device=device,
        workers=4,
        # dental-specific augmentations
        **AUG_CONFIG,
    )

    # Validate final model
    print("\n" + "="*55)
    print("  TRAINING COMPLETE — Running validation...")
    print("="*55)
    best_weights = MODEL_DIR / "weights" / "best.pt"
    model_best = YOLO(str(best_weights))
    metrics = model_best.val(data=str(DATA_YAML), imgsz=imgsz, device=device)

    print(f"\n  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    print(f"\n  Best weights saved → {best_weights}")
    print("\n  Next step:")
    print("    python training/train_unet.py")

    return results


def export_model(format: str = "onnx"):
    """Export trained model for deployment."""
    best = MODEL_DIR / "weights" / "best.pt"
    if not best.exists():
        print("[!] No trained model found. Train first.")
        return
    model = YOLO(str(best))
    model.export(format=format)
    print(f"  Exported to {format.upper()} format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 dental detector")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--device",  default="auto",
                        help="cpu / cuda / cuda:0 / mps")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--export",  default=None,
                        help="Export format after training: onnx / torchscript")
    args = parser.parse_args()

    results = train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
    )

    if args.export:
        export_model(format=args.export)
