"""
api/main.py
===========
FastAPI server that accepts a dental X-ray image and returns
segmentation, detections, diagnosis, and treatment plan.

Run:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /analyze       ← main endpoint, accepts multipart image
    GET  /health        ← check if models are loaded
    GET  /docs          ← auto-generated Swagger UI
"""

import io
import base64
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.inference import DentalAnalyzer, AnalysisResult

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dental AI System",
    description="AI-powered dental X-ray analysis: segmentation, detection, diagnosis & treatment planning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load models at startup ───────────────────────────────────────────────────
analyzer: Optional[DentalAnalyzer] = None

@app.on_event("startup")
async def startup():
    global analyzer
    print("\n[startup] Loading dental AI models...")
    try:
        analyzer = DentalAnalyzer()
        print("[startup] Models loaded successfully\n")
    except Exception as e:
        print(f"[startup] Warning: Could not load models — {e}")
        print("[startup] Server running in demo mode (no real inference)\n")


# ─── Response schemas ─────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float
    class_name: str
    confidence: float
    severity: str


class SegmentationInfo(BaseModel):
    teeth_count: int
    roots_visible: bool
    bone_level: str
    existing_restorations: int
    mask_base64: str   # PNG mask as base64 for frontend overlay


class Finding(BaseModel):
    name: str
    location: str
    severity: str
    confidence: float


class TreatmentStep(BaseModel):
    procedure: str
    priority: str
    tooth: str
    notes: str


class AnalysisResponse(BaseModel):
    success: bool
    processing_time_ms: int
    image_type: str
    overall_health: str
    urgency: str
    detections: list[BoundingBox]
    segmentation: SegmentationInfo
    findings: list[Finding]
    diagnosis: str
    treatment_plan: list[TreatmentStep]
    annotated_image_base64: str   # X-ray with overlays drawn on it


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": analyzer is not None and analyzer.models_ready,
        "yolo_loaded": analyzer.yolo_loaded if analyzer else False,
        "unet_loaded": analyzer.unet_loaded if analyzer else False,
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Upload a dental X-ray (JPG or PNG) and receive full AI analysis.
    """
    # Validate file type
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported"
        )

    # Read image bytes
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Run analysis
    t0 = time.monotonic()
    try:
        result: AnalysisResult = analyzer.analyze(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Encode mask and annotated image as base64 for frontend
    _, mask_buf = cv2.imencode(".png", result.mask)
    mask_b64    = base64.b64encode(mask_buf).decode()

    _, ann_buf  = cv2.imencode(".jpg", result.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ann_b64     = base64.b64encode(ann_buf).decode()

    return AnalysisResponse(
        success=True,
        processing_time_ms=elapsed_ms,
        image_type=result.image_type,
        overall_health=result.overall_health,
        urgency=result.urgency,
        detections=[BoundingBox(**d) for d in result.detections],
        segmentation=SegmentationInfo(
            teeth_count=result.seg_info["teeth_count"],
            roots_visible=result.seg_info["roots_visible"],
            bone_level=result.seg_info["bone_level"],
            existing_restorations=result.seg_info["existing_restorations"],
            mask_base64=mask_b64,
        ),
        findings=[Finding(**f) for f in result.findings],
        diagnosis=result.diagnosis,
        treatment_plan=[TreatmentStep(**t) for t in result.treatment_plan],
        annotated_image_base64=ann_b64,
    )


@app.get("/classes")
async def get_classes():
    """Return all detectable dental classes."""
    return {
        "detection_classes": [
            "tooth", "caries", "periapical", "bone_loss",
            "restoration", "impacted", "calculus", "crown"
        ],
        "segmentation_classes": [
            "background", "tooth_crown", "root", "bone", "pathology"
        ]
    }
