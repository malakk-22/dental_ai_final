"""
train_unet.py
=============
Train a U-Net with ResNet50 encoder for dental image segmentation.

Segments each pixel into:
  0 = background
  1 = tooth (crown)
  2 = root
  3 = bone
  4 = pathological region (caries, lesion, etc.)

Usage:
    python training/train_unet.py --epochs 50 --batch 8

Output:
    models/unet_dental_best.pth    ← best model weights
    models/unet_training_log.csv   ← epoch-by-epoch metrics
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "segmentation"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 5   # background + 4 dental classes
IMG_SIZE  = 512


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL: U-Net with ResNet50 encoder
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm + ReLU."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + skip connection + ConvBlock."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # handle size mismatch from odd input dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DentalUNet(nn.Module):
    """
    U-Net using a pretrained ResNet50 as the encoder.
    Encoder layers:
      layer0: conv1 + bn1 + relu (64 ch, /2)
      layer1: maxpool + res_layer1  (256 ch, /4)
      layer2: res_layer2            (512 ch, /8)
      layer3: res_layer3            (1024 ch, /16)
      layer4: res_layer4            (2048 ch, /32) ← bottleneck
    """
    def __init__(self, n_classes=N_CLASSES, pretrained=True):
        super().__init__()

        # ── Encoder (ResNet50) ──
        enc = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.enc0  = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 64, /2
        self.pool  = enc.maxpool
        self.enc1  = enc.layer1   # 256, /4
        self.enc2  = enc.layer2   # 512, /8
        self.enc3  = enc.layer3   # 1024, /16
        self.enc4  = enc.layer4   # 2048, /32

        # ── Bottleneck ──
        self.bottleneck = ConvBlock(2048, 1024)

        # ── Decoder ──
        self.up4 = UpBlock(1024, 1024, 512)
        self.up3 = UpBlock(512,  512,  256)
        self.up2 = UpBlock(256,  256,  128)
        self.up1 = UpBlock(128,  64,   64)

        # Final upsample to match input size
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)               # 64,  H/2
        e1 = self.enc1(self.pool(e0))   # 256, H/4
        e2 = self.enc2(e1)              # 512, H/8
        e3 = self.enc3(e2)              # 1024,H/16
        e4 = self.enc4(e3)              # 2048,H/32

        # Bottleneck
        b = self.bottleneck(e4)         # 1024,H/32

        # Decoder
        d4 = self.up4(b,  e3)           # 512, H/16
        d3 = self.up3(d4, e2)           # 256, H/8
        d2 = self.up2(d3, e1)           # 128, H/4
        d1 = self.up1(d2, e0)           # 64,  H/2

        out = self.up0(d1)              # 32,  H
        return self.head(out)           # n_classes, H


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class DentalSegDataset(Dataset):
    """
    Loads dental X-ray images + segmentation masks.

    Expected folder structure:
        data/segmentation/images/train/*.jpg
        data/segmentation/masks/train/*.png   ← grayscale, values 0-4
    """
    def __init__(self, split="train", img_size=IMG_SIZE, augment=True):
        self.img_dir  = DATA_DIR / "images" / split
        self.mask_dir = DATA_DIR / "masks"  / split
        self.img_size = img_size
        self.augment  = augment and (split == "train")

        self.img_paths = sorted(self.img_dir.glob("*.jpg")) + \
                         sorted(self.img_dir.glob("*.png"))
        if not self.img_paths:
            raise FileNotFoundError(
                f"No images found in {self.img_dir}\n"
                "Run: python training/dataset_prep.py"
            )

        # ImageNet normalisation (even for grayscale — convert to RGB)
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.img_paths)

    def _augment(self, img, mask):
        """Dental-appropriate augmentations."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img  = cv2.flip(img,  1)
            mask = cv2.flip(mask, 1)
        # Small rotation (±8°)
        angle = np.random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2),
                                    angle, 1.0)
        img  = cv2.warpAffine(img,  M, (img.shape[1], img.shape[0]))
        mask = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_NEAREST)
        # Brightness / contrast jitter (simulates X-ray exposure variation)
        alpha = np.random.uniform(0.8, 1.2)
        beta  = np.random.randint(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        # Load image (convert to RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Resize
        img  = cv2.resize(img,  (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # Augment
        if self.augment:
            img, mask = self._augment(img, mask)

        # Clamp mask values to valid class range
        mask = np.clip(mask, 0, N_CLASSES - 1)

        # To tensor
        img_t  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t  = self.norm(img_t)
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS
# ══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Dice loss — better than cross-entropy for imbalanced segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n, c, h, w = probs.shape
        # one-hot encode targets
        targets_oh = F.one_hot(targets, c).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Cross-entropy + Dice (0.5 each) — standard for medical segmentation."""
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(preds, targets, n_classes=N_CLASSES):
    """Mean IoU across all classes."""
    ious = []
    preds   = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(n_classes):
        pred_c   = (preds   == cls)
        target_c = (targets == cls)
        inter    = (pred_c & target_c).sum().float()
        union    = (pred_c | target_c).sum().float()
        if union == 0:
            continue
        ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(epochs=50, batch=8, lr=1e-4, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Epochs: {epochs}  |  Batch: {batch}  |  LR: {lr}\n")

    # Datasets
    train_ds = DentalSegDataset("train", augment=True)
    val_ds   = DentalSegDataset("val",   augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=2, pin_memory=True)
    print(f"  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # Model, loss, optimiser
    model = DentalUNet(n_classes=N_CLASSES, pretrained=True).to(device)
    criterion = CombinedLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-6
    )

    best_iou = 0.0
    log_path = MODEL_DIR / "unet_training_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "lr"])

        for epoch in range(1, epochs + 1):
            # ── Train ────────────────────────────────────────────────────────
            model.train()
            t_loss = 0.0
            for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch:03d}/{epochs} [train]",
                                    leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                optimiser.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                t_loss += loss.item()
            t_loss /= len(train_dl)

            # ── Validate ─────────────────────────────────────────────────────
            model.eval()
            v_loss, v_iou = 0.0, 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(val_dl, desc=f"Epoch {epoch:03d}/{epochs} [val  ]",
                                        leave=False):
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    v_loss += criterion(logits, masks).item()
                    preds   = logits.argmax(dim=1)
                    v_iou  += compute_iou(preds.cpu(), masks.cpu())
            v_loss /= len(val_dl)
            v_iou  /= len(val_dl)

            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]

            print(f"  [{epoch:03d}/{epochs}]  "
                  f"loss: {t_loss:.4f}  val_loss: {v_loss:.4f}  "
                  f"val_IoU: {v_iou:.4f}  lr: {cur_lr:.2e}")

            writer.writerow([epoch, f"{t_loss:.4f}", f"{v_loss:.4f}",
                             f"{v_iou:.4f}", f"{cur_lr:.2e}"])

            # Save best
            if v_iou > best_iou:
                best_iou = v_iou
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_iou": v_iou,
                    "n_classes": N_CLASSES,
                }, MODEL_DIR / "unet_dental_best.pth")
                print(f"  ✓ New best IoU: {best_iou:.4f} — checkpoint saved")

    print("\n" + "="*55)
    print(f"  TRAINING COMPLETE  |  Best val IoU: {best_iou:.4f}")
    print(f"  Weights → {MODEL_DIR / 'unet_dental_best.pth'}")
    print("\n  Next step:")
    print("    uvicorn api.main:app --reload --port 8000")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net dental segmentation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch,
          lr=args.lr, device=args.device)
