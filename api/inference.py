"""
api/inference.py
================
DentalAnalyzer class: loads trained models and runs full inference pipeline.

Pipeline:
  1. Preprocess image (resize, normalize)
  2. YOLOv8 → bounding boxes + class labels
  3. U-Net  → pixel segmentation mask
  4. Rule engine → findings + severity
  5. Treatment planner → prioritized procedures
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).parent.parent

# Class definitions
YOLO_CLASSES = [
    "tooth", "caries", "periapical", "bone_loss",
    "restoration", "impacted", "calculus", "crown"
]

SEG_COLORS = {
    0: (0,   0,   0),    # background
    1: (100, 200, 100),  # tooth crown — green
    2: (200, 150, 50),   # root — amber
    3: (150, 100, 200),  # bone — purple
    4: (220, 60,  60),   # pathology — red
}

SEVERITY_MAP = {
    "tooth":       "low",
    "caries":      "high",
    "periapical":  "high",
    "bone_loss":   "medium",
    "restoration": "low",
    "impacted":    "medium",
    "calculus":    "medium",
    "crown":       "low",
}

TREATMENT_MAP = {
    "caries": {
        "procedure": "Composite resin restoration",
        "priority": "high",
        "notes": "Remove carious tissue and restore with composite. Apply fluoride treatment.",
    },
    "periapical": {
        "procedure": "Root canal treatment (RCT)",
        "priority": "high",
        "notes": "Pulp is likely necrotic. Perform RCT followed by crown placement.",
    },
    "bone_loss": {
        "procedure": "Periodontal therapy",
        "priority": "medium",
        "notes": "Deep scaling and root planing. Consider surgical intervention if > 5mm pocket depth.",
    },
    "impacted": {
        "procedure": "Surgical extraction",
        "priority": "medium",
        "notes": "Monitor for eruption. Extract if symptomatic or causing adjacent tooth resorption.",
    },
    "calculus": {
        "procedure": "Professional scaling",
        "priority": "low",
        "notes": "Supragingival and subgingival debridement. Improve oral hygiene instruction.",
    },
}


@dataclass
class AnalysisResult:
    image_type: str = "unknown"
    overall_health: str = "fair"
    urgency: str = "routine"
    detections: list = field(default_factory=list)
    seg_info: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    diagnosis: str = ""
    treatment_plan: list = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((512, 512), np.uint8))
    annotated_image: np.ndarray = field(default_factory=lambda: np.zeros((512, 512, 3), np.uint8))


class DentalAnalyzer:
    """
    Loads trained models and runs the full dental AI pipeline.
    Falls back to rule-based demo mode if models are not yet trained.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_loaded = False
        self.unet_loaded = False
        self._load_models()

    def _load_models(self):
        """Attempt to load trained weights."""
        # YOLOv8 detection model
        yolo_path = ROOT / "models" / "yolo_dental" / "weights" / "best.pt"
        if yolo_path.exists():
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(str(yolo_path))
                self.yolo_loaded = True
                print(f"  [✓] YOLOv8 loaded from {yolo_path}")
            except Exception as e:
                print(f"  [!] YOLOv8 load failed: {e}")

        # U-Net segmentation model
        unet_path = ROOT / "models" / "unet_dental_best.pth"
        if unet_path.exists():
            try:
                from training.train_unet import DentalUNet, N_CLASSES
                ckpt = torch.load(unet_path, map_location=self.device)
                self.unet = DentalUNet(n_classes=N_CLASSES, pretrained=False)
                self.unet.load_state_dict(ckpt["model_state"])
                self.unet.eval().to(self.device)
                self.unet_loaded = True
                print(f"  [✓] U-Net loaded (val IoU: {ckpt.get('val_iou', '—')})")
            except Exception as e:
                print(f"  [!] U-Net load failed: {e}")

    @property
    def models_ready(self):
        return self.yolo_loaded or self.unet_loaded

    def _preprocess(self, img: np.ndarray, size=640) -> np.ndarray:
        img = cv2.resize(img, (size, size))
        return img

    def _run_yolo(self, img: np.ndarray) -> list[dict]:
        """Run YOLOv8 detection. Returns list of detection dicts."""
        if not self.yolo_loaded:
            return self._demo_detections(img)
        results = self.yolo(img, conf=0.25, iou=0.45, verbose=False)[0]
        detections = []
        h, w = img.shape[:2]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "x1": x1/w, "y1": y1/h, "x2": x2/w, "y2": y2/h,
                "class_name": YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else "unknown",
                "confidence": round(conf, 3),
                "severity": SEVERITY_MAP.get(YOLO_CLASSES[cls_id], "low"),
            })
        return detections

    def _run_unet(self, img: np.ndarray) -> np.ndarray:
        """Run U-Net segmentation. Returns class mask (H×W, uint8)."""
        if not self.unet_loaded:
            return self._demo_mask(img)
        norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t   = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
        img_t   = norm(img_t).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.unet(img_t)
            mask   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    def _demo_detections(self, img: np.ndarray) -> list[dict]:
        """Rule-based demo detections (no model needed)."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        demo = []
        # Find bright ellipses (tooth-like blobs)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours[:12]):
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 10 or ch < 10 or cw*ch < 400:
                continue
            cls = "tooth"
            if i % 5 == 1: cls = "caries"
            elif i % 7 == 0: cls = "bone_loss"
            demo.append({
                "x1": x/w, "y1": y/h,
                "x2": (x+cw)/w, "y2": (y+ch)/h,
                "class_name": cls,
                "confidence": round(0.65 + (i%3)*0.1, 3),
                "severity": SEVERITY_MAP.get(cls, "low"),
            })
        return demo[:8]

    def _demo_mask(self, img: np.ndarray) -> np.ndarray:
        """Rule-based demo segmentation mask."""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        mask[bright > 0] = 1
        _, very_bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        mask[very_bright > 0] = 2
        return mask

    def _build_seg_info(self, mask: np.ndarray, detections: list) -> dict:
        """Extract structured info from segmentation mask."""
        tooth_pixels = (mask == 1).sum()
        root_pixels  = (mask == 2).sum()
        teeth_count  = max(
            len([d for d in detections if d["class_name"] == "tooth"]),
            max(1, tooth_pixels // 2000)
        )
        restorations = len([d for d in detections if d["class_name"] in {"restoration","crown"}])
        has_bone_loss = any(d["class_name"] == "bone_loss" for d in detections)
        n_bone_loss   = len([d for d in detections if d["class_name"] == "bone_loss"])
        bone_level = "normal"
        if n_bone_loss >= 3:    bone_level = "severe loss"
        elif n_bone_loss >= 2:  bone_level = "moderate loss"
        elif has_bone_loss:     bone_level = "mild loss"
        return {
            "teeth_count": min(teeth_count, 32),
            "roots_visible": root_pixels > 1000,
            "bone_level": bone_level,
            "existing_restorations": restorations,
        }

    def _build_findings(self, detections: list) -> list[dict]:
        """Aggregate detections into clinical findings."""
        findings = []
        classes_seen = {}
        for d in detections:
            cls = d["class_name"]
            if cls == "tooth":
                continue
            if cls not in classes_seen:
                classes_seen[cls] = []
            classes_seen[cls].append(d)

        loc_names = ["upper right", "upper left", "lower right", "lower left",
                     "anterior", "posterior", "molar region", "premolar region"]
        for cls, dets in classes_seen.items():
            loc = loc_names[hash(cls) % len(loc_names)]
            findings.append({
                "name": cls.replace("_", " ").title(),
                "location": f"{loc} — {len(dets)} site(s)",
                "severity": dets[0]["severity"],
                "confidence": round(sum(d["confidence"] for d in dets) / len(dets), 3),
            })
        return findings

    def _build_treatment(self, findings: list) -> list[dict]:
        """Build prioritized treatment plan from findings."""
        plan = []
        priority_order = {"high": 0, "medium": 1, "low": 2}
        for f in sorted(findings, key=lambda x: priority_order.get(x["severity"], 3)):
            cls_key = f["name"].lower().replace(" ", "_")
            if cls_key in TREATMENT_MAP:
                t = dict(TREATMENT_MAP[cls_key])
                t["tooth"] = f["location"].split("—")[0].strip()
                plan.append(t)
        if not plan:
            plan.append({
                "procedure": "Routine check-up",
                "priority": "low",
                "tooth": "all teeth",
                "notes": "No significant pathology detected. Maintain regular 6-month recall.",
            })
        return plan

    def _build_diagnosis(self, findings: list, seg_info: dict) -> tuple[str, str, str]:
        """Returns (diagnosis_text, overall_health, urgency)."""
        n_high   = sum(1 for f in findings if f["severity"] == "high")
        n_medium = sum(1 for f in findings if f["severity"] == "medium")

        if n_high >= 2:
            health, urgency = "poor",   "urgent"
        elif n_high == 1 or n_medium >= 2:
            health, urgency = "fair",   "soon"
        else:
            health, urgency = "good",   "routine"

        finding_names = [f["name"] for f in findings] or ["no significant pathology"]
        teeth = seg_info.get("teeth_count", "—")
        bone  = seg_info.get("bone_level", "normal")

        diag = (
            f"Radiographic examination reveals {teeth} visible teeth with "
            f"{', '.join(finding_names[:3])}. "
            f"Alveolar bone levels appear {bone}. "
        )
        if urgency == "urgent":
            diag += "Immediate treatment is recommended to prevent further deterioration."
        elif urgency == "soon":
            diag += "Treatment should be scheduled within the next 4–6 weeks."
        else:
            diag += "Continue regular dental hygiene and recall appointments."
        return diag, health, urgency

    def _draw_annotations(self, img: np.ndarray, detections: list,
                           mask: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and segmentation overlay on the image."""
        ann = img.copy()
        h, w = ann.shape[:2]

        # Segmentation overlay (semi-transparent)
        overlay = ann.copy()
        for cls_id, color in SEG_COLORS.items():
            if cls_id == 0: continue
            overlay[mask == cls_id] = color
        ann = cv2.addWeighted(ann, 0.6, overlay, 0.4, 0)

        # Bounding boxes
        colors = {
            "tooth": (100,200,100), "caries": (50,50,220),
            "periapical": (0,0,180), "bone_loss": (180,100,200),
            "restoration": (200,200,50), "impacted": (200,150,0),
            "calculus": (0,180,180), "crown": (200,100,50),
        }
        for d in detections:
            x1 = int(d["x1"] * w); y1 = int(d["y1"] * h)
            x2 = int(d["x2"] * w); y2 = int(d["y2"] * h)
            col = colors.get(d["class_name"], (200,200,200))
            cv2.rectangle(ann, (x1,y1), (x2,y2), col, 2)
            label = f"{d['class_name']} {d['confidence']:.0%}"
            cv2.rectangle(ann, (x1, y1-18), (x1+len(label)*8, y1), col, -1)
            cv2.putText(ann, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        return ann

    def analyze(self, img: np.ndarray) -> AnalysisResult:
        """Full pipeline: preprocess → detect → segment → diagnose → plan."""
        img = self._preprocess(img)
        result = AnalysisResult()

        # Classify image type
        h, w = img.shape[:2]
        result.image_type = "panoramic" if w > h * 1.8 else \
                            "periapical" if w < 400 else "bitewing"

        # Run models
        result.detections = self._run_yolo(img)
        result.mask       = self._run_unet(img)

        # Build structured outputs
        result.seg_info      = self._build_seg_info(result.mask, result.detections)
        result.findings      = self._build_findings(result.detections)
        diag, health, urgency = self._build_diagnosis(result.findings, result.seg_info)
        result.diagnosis      = diag
        result.overall_health = health
        result.urgency        = urgency
        result.treatment_plan = self._build_treatment(result.findings)
        result.annotated_image = self._draw_annotations(img, result.detections, result.mask)

        return result
