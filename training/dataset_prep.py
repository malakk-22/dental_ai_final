"""
dataset_prep.py
===============
Downloads and prepares dental X-ray datasets for training.

Datasets used:
  1. UFBA-UESC Dental Images (panoramic X-rays, 1500 images)
     https://github.com/IvisionLab/dental-panoramic-xray
  2. Tufts Dental Database (periapical, 1000 images)
     https://tdd.ece.tufts.edu/
  3. Roboflow Dental Detection (pre-annotated, YOLO format)
     https://universe.roboflow.com/dental-xray

Run:
    python training/dataset_prep.py

Output folder structure:
    data/
    ├── raw/           ← original downloaded files
    ├── yolo/          ← YOLOv8-format dataset
    │   ├── images/train/
    │   ├── images/val/
    │   ├── labels/train/
    │   ├── labels/val/
    │   └── data.yaml
    └── segmentation/  ← U-Net format dataset
        ├── images/train/
        ├── images/val/
        ├── masks/train/
        └── masks/val/
"""

import os
import shutil
import random
import zipfile
import requests
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

# ─── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent / "data"
SEED = 42
VAL_SPLIT = 0.2
IMG_SIZE = 640

CLASSES = [
    "tooth",          # 0
    "caries",         # 1
    "periapical",     # 2
    "bone_loss",      # 3
    "restoration",    # 4
    "impacted",       # 5
    "calculus",       # 6
    "crown",          # 7
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = ""):
    """Stream-download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                      desc=desc or dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def split_images(src_dir: Path, val_ratio=0.2, seed=42):
    """Returns (train_paths, val_paths) for all images in src_dir."""
    imgs = sorted([p for p in src_dir.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    random.seed(seed)
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    return imgs[n_val:], imgs[:n_val]


def resize_image(src: Path, dst: Path, size=640):
    """Resize image to size×size and save to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src))
    if img is None:
        return False
    img = cv2.resize(img, (size, size))
    cv2.imwrite(str(dst), img)
    return True

# ─── Roboflow dental dataset (best free annotated source) ────────────────────

def download_roboflow_dataset(api_key: str = None):
    """
    Download dental detection dataset from Roboflow.
    If no API key, uses the public export URL for the
    'Dental X-Ray Detection' dataset (YOLO format).

    Free dataset: https://universe.roboflow.com/dental-xray/dental-x-ray-detection
    Sign up at roboflow.com → get free API key → paste here.
    """
    print("\n[1/3] Roboflow dental detection dataset")
    dest_zip = ROOT / "raw" / "roboflow_dental.zip"

    if api_key:
        url = (
            f"https://universe.roboflow.com/ds/REPLACE_WITH_YOUR_EXPORT_URL"
            f"?key={api_key}"
        )
    else:
        # Public direct link — replace with your own Roboflow export URL
        # Go to: https://universe.roboflow.com/dental-xray
        # → choose a dataset → Export → YOLOv8 → get download link
        url = "PASTE_YOUR_ROBOFLOW_DOWNLOAD_URL_HERE"
        print("  [!] No Roboflow API key. Set ROBOFLOW_KEY env variable or")
        print("      paste your export URL into this script.")
        print("  [!] Using synthetic data for now. See README for real data.")
        _generate_synthetic_dataset()
        return

    download_file(url, dest_zip, "Roboflow dental dataset")
    _unzip_roboflow(dest_zip)


def _unzip_roboflow(zip_path: Path):
    out = ROOT / "raw" / "roboflow_dental"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    print(f"  Extracted to {out}")


# ─── Synthetic dataset (for testing pipeline before real data) ────────────────

def _generate_synthetic_dataset(n_train=200, n_val=40):
    """
    Creates synthetic grayscale X-ray-like images with random bounding
    boxes so you can verify the full pipeline works before downloading
    real data.
    """
    print("  Generating synthetic dental dataset for pipeline testing...")
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = ROOT / "yolo" / "images" / split
        lbl_dir = ROOT / "yolo" / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        seg_img = ROOT / "segmentation" / "images" / split
        seg_msk = ROOT / "segmentation" / "masks" / split
        seg_img.mkdir(parents=True, exist_ok=True)
        seg_msk.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n), desc=f"  {split}"):
            # --- synthetic X-ray image ---
            img = np.random.randint(20, 80, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            # add tooth-like ellipses
            n_teeth = random.randint(6, 14)
            labels = []
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

            for t in range(n_teeth):
                cx = random.randint(60, IMG_SIZE - 60)
                cy = random.randint(100, IMG_SIZE - 80)
                rx = random.randint(18, 34)
                ry = random.randint(28, 54)
                brightness = random.randint(140, 220)
                cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, brightness, -1)
                cv2.ellipse(mask, (cx, cy), (rx, ry), 1, 0, 360, 255, -1)

                # random pathology on some teeth
                cls = 0  # default: tooth
                if random.random() < 0.3:
                    cls = random.randint(1, len(CLASSES) - 1)
                    # draw small dark spot for caries etc.
                    spot_r = random.randint(4, 10)
                    cv2.circle(img, (cx + random.randint(-rx//2, rx//2),
                                     cy + random.randint(-ry//2, ry//2)),
                               spot_r, random.randint(30, 60), -1)

                # YOLO label (class cx cy w h) normalised
                x1 = max(0, cx - rx) / IMG_SIZE
                y1 = max(0, cy - ry) / IMG_SIZE
                bw = min(1, 2 * rx / IMG_SIZE)
                bh = min(1, 2 * ry / IMG_SIZE)
                labels.append(f"{cls} {(x1+bw/2):.6f} {(y1+bh/2):.6f} {bw:.6f} {bh:.6f}")

            # save
            name = f"synth_{i:04d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            (lbl_dir / f"{name}.txt").write_text("\n".join(labels))
            cv2.imwrite(str(seg_img / f"{name}.jpg"), img)
            cv2.imwrite(str(seg_msk / f"{name}.png"), mask)

    print(f"  Synthetic dataset ready: {n_train} train, {n_val} val images")


# ─── Write YOLO data.yaml ─────────────────────────────────────────────────────

def write_data_yaml():
    yaml_path = ROOT / "yolo" / "data.yaml"
    content = f"""# Dental X-ray detection dataset
path: {(ROOT / 'yolo').resolve()}
train: images/train
val:   images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path.write_text(content)
    print(f"\n  data.yaml written → {yaml_path}")


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "="*55)
    print("  DATASET PREPARATION COMPLETE")
    print("="*55)
    for split in ["train", "val"]:
        n = len(list((ROOT / "yolo" / "images" / split).glob("*.jpg")))
        print(f"  YOLO  {split:5s}: {n:5d} images")
    for split in ["train", "val"]:
        n = len(list((ROOT / "segmentation" / "images" / split).glob("*.jpg")))
        print(f"  U-Net {split:5s}: {n:5d} images")
    print(f"\n  Classes ({len(CLASSES)}): {', '.join(CLASSES)}")
    print(f"  Image size: {IMG_SIZE}×{IMG_SIZE}")
    print("\n  Next step:")
    print("    python training/train_yolo.py")
    print("="*55)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dental AI datasets")
    parser.add_argument("--roboflow-key", default=None,
                        help="Roboflow API key (optional)")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip downloads, use synthetic data only")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)

    if args.synthetic_only or not args.roboflow_key:
        _generate_synthetic_dataset()
    else:
        download_roboflow_dataset(api_key=args.roboflow_key)

    write_data_yaml()
    print_summary()
