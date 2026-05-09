"""
dentiligence/api/main.py
=========================
FastAPI REST API — Dentiligence Neural Anatomy Generation Service.

Endpoints
---------
    POST /generate/anatomy        — Generate multi-modal anatomy output from query coords.
    POST /generate/radiograph     — Render a synthetic DRR from the neural field.
    POST /generate/haptic         — Compute haptic resistance map for robotic planning.
    GET  /pathologies             — List all injectable pathology types.
    POST /generate/full-dataset   — Generate a complete labelled training sample.
    GET  /health                  — Service health check.

All endpoints are stateless.  The model weights are loaded once at startup
and reused across requests (thread-safe inference via torch.no_grad()).

CORS is open for development; tighten `allow_origins` for production.

Authors : Dentiligence Core Team
Version : 0.1.0  (Series-A audit ready)
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Internal modules — adjust import path if running from repo root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.models import MultiHeadSIREN, HapticResistanceMapper, NUM_LABELS
from core.physics_sim import BeerLambertRayMarcher, ClinicalNoiseModule
from core.pathology_logic import PathologyEngine, PATHOLOGY_REGISTRY

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dentiligence.api")

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Dentiligence Neural Anatomy API",
    description = "High-fidelity synthetic dental data generation via Implicit Neural Representations.",
    version     = "0.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Tighten for production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL: Optional[MultiHeadSIREN]      = None
_HAPTIC: Optional[HapticResistanceMapper] = None


@app.on_event("startup")
async def _load_model() -> None:
    global _MODEL, _HAPTIC
    logger.info(f"Loading MultiHeadSIREN on device: {DEVICE}")
    weights_path = os.environ.get("MODEL_WEIGHTS_PATH", "")
    _MODEL = MultiHeadSIREN(
        hidden_features=256,
        hidden_layers=5,
        num_labels=NUM_LABELS,
    ).to(DEVICE)
    if weights_path and os.path.isfile(weights_path):
        _MODEL.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning("No pre-trained weights found — running with random initialisation.")
    _MODEL.eval()
    _HAPTIC = HapticResistanceMapper().to(DEVICE)
    logger.info("Model ready.")


def _get_model() -> MultiHeadSIREN:
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")
    return _MODEL


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CoordinateRequest(BaseModel):
    """3D coordinate batch for anatomy field queries."""

    coordinates: List[Tuple[float, float, float]] = Field(
        ...,
        description="List of (x, y, z) normalised coordinates ∈ [-1, 1].",
        min_items=1,
        max_items=100_000,
        example=[[0.0, 0.0, 0.0], [0.1, -0.2, 0.5]],
    )
    pathologies: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Optional list of pathology injectors. "
            "Each item: {\"name\": \"<pathology_type>\", \"params\": {...}}. "
            "See GET /pathologies for available types."
        ),
        example=[{"name": "periapical_granuloma", "params": {"severity": 0.7}}],
    )

    @validator("coordinates", each_item=True)
    def _check_coords(cls, v):
        if not all(-1.0 <= c <= 1.0 for c in v):
            raise ValueError("All coordinates must be in [-1, 1].")
        return v


class AnatomyResponse(BaseModel):
    """Multi-modal anatomy field output at queried points."""
    n_points          : int
    hu                : List[float]     = Field(..., description="Hounsfield Units per point.")
    label             : List[int]       = Field(..., description="Tissue label ID per point.")
    thermal_wpmk      : List[float]     = Field(..., description="Thermal conductivity W·m⁻¹·K⁻¹.")
    elasticity_gpa    : List[float]     = Field(..., description="Young's modulus GPa.")
    label_names       : List[str]       = Field(..., description="Human-readable tissue name per point.")


class RadiographRequest(BaseModel):
    """Parameters for DRR rendering."""
    pixel_h         : int   = Field(256, ge=32, le=1024, description="Detector height (pixels).")
    pixel_w         : int   = Field(256, ge=32, le=1024, description="Detector width (pixels).")
    n_samples       : int   = Field(128, ge=32, le=512,  description="Ray march samples per ray.")
    apply_noise     : bool  = Field(True, description="Apply ClinicalNoiseModule.")
    pathologies     : Optional[List[Dict[str, Any]]] = None


class HapticRequest(BaseModel):
    """Coordinate batch for haptic resistance map computation."""
    coordinates: List[Tuple[float, float, float]] = Field(..., min_items=1, max_items=100_000)
    pathologies: Optional[List[Dict[str, Any]]] = None
    weight_elasticity: float = Field(0.60, ge=0.0, le=1.0)
    weight_density   : float = Field(0.30, ge=0.0, le=1.0)
    weight_thermal   : float = Field(0.10, ge=0.0, le=1.0)


class FullDatasetRequest(BaseModel):
    """Parameters for a complete labelled training sample generation."""
    grid_resolution : int   = Field(64, ge=16, le=256, description="Voxel grid side length.")
    pathologies     : Optional[List[Dict[str, Any]]] = None
    apply_noise     : bool  = Field(True)
    render_drr      : bool  = Field(True, description="Include DRR in output.")


# ---------------------------------------------------------------------------
# Utility: build PathologyEngine from request payload
# ---------------------------------------------------------------------------

def _build_pathology_engine(
    pathology_list: Optional[List[Dict[str, Any]]]
) -> Optional[PathologyEngine]:
    if not pathology_list:
        return None
    configs = []
    for p in pathology_list:
        name   = p.get("name")
        params = p.get("params", {})
        if name not in PATHOLOGY_REGISTRY:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown pathology '{name}'. Available: {list(PATHOLOGY_REGISTRY)}",
            )
        configs.append((name, params))
    return PathologyEngine(configs).to(DEVICE)


# Reverse label lookup
_LABEL_ID_TO_NAME: Dict[int, str] = {v: k for k, v in {
    "background": 0, "enamel": 1, "dentin": 2, "pulp": 3, "cementum": 4,
    "pdl": 5, "alveolar_bone_dense": 6, "alveolar_bone_cancel": 7,
    "cortical_bone": 8, "gingiva": 9, "sinus_membrane": 10,
    "air_sinus": 11, "pathology": 12,
}.items()}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> Dict[str, str]:
    """Service liveness and device check."""
    return {"status": "ok", "device": str(DEVICE), "version": app.version}


@app.get("/pathologies", tags=["Catalogue"])
async def list_pathologies() -> Dict[str, Any]:
    """Return the full catalogue of injectable pathologies."""
    return {
        "total"     : len(PATHOLOGY_REGISTRY),
        "pathologies": list(PATHOLOGY_REGISTRY.keys()),
        "categories": {
            "micro_endo" : ["periapical_granuloma", "calcified_canal", "mb2_canal", "c_shaped_canal"],
            "surgical"   : ["impacted_molar", "ameloblastoma", "sinus_proximity"],
            "genetic"    : ["amelogenesis_imperfecta", "dens_in_dente", "regional_odontodysplasia"],
            "iatrogenic" : ["root_perforation", "overfilled_canal", "sinus_membrane_puncture"],
        },
    }


@app.post("/generate/anatomy", response_model=AnatomyResponse, tags=["Generation"])
async def generate_anatomy(req: CoordinateRequest) -> AnatomyResponse:
    """Query the Neural Anatomy Field at arbitrary 3D coordinates.

    Returns HU density, tissue label, thermal conductivity, and elasticity
    for each queried point — with optional pathology injection.
    """
    model = _get_model()
    pts   = torch.tensor(req.coordinates, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        physical = model.physical_output(pts)

    # Apply pathology if requested
    engine = _build_pathology_engine(req.pathologies)
    if engine is not None:
        with torch.no_grad():
            physical = engine.apply_to_physical(physical, pts)

    hu          = physical["hu"].squeeze(-1).cpu().tolist()
    labels      = physical["label"].cpu().tolist()
    thermal     = physical["thermal_wpmk"].squeeze(-1).cpu().tolist()
    elasticity  = physical["elasticity_gpa"].squeeze(-1).cpu().tolist()
    label_names = [_LABEL_ID_TO_NAME.get(int(l), f"unknown_{l}") for l in labels]

    return AnatomyResponse(
        n_points       = len(hu),
        hu             = hu,
        label          = labels,
        thermal_wpmk   = thermal,
        elasticity_gpa = elasticity,
        label_names    = label_names,
    )


@app.post("/generate/radiograph", tags=["Generation"])
async def generate_radiograph(req: RadiographRequest) -> Response:
    """Render a synthetic X-ray DRR using Beer-Lambert ray marching.

    Returns a raw 32-bit float NumPy array (binary) of shape (H, W).
    Use the `Content-Type: application/octet-stream` response directly,
    or request JSON if you prefer base64.
    """
    model = _get_model()

    noise_module = ClinicalNoiseModule() if req.apply_noise else None

    renderer = BeerLambertRayMarcher(
        model     = model,
        n_samples = req.n_samples,
        pixel_h   = req.pixel_h,
        pixel_w   = req.pixel_w,
    ).to(DEVICE)

    with torch.no_grad():
        image = renderer(noise_module=noise_module)  # (H, W)

    img_np = image.cpu().numpy().astype(np.float32)
    buf    = io.BytesIO()
    np.save(buf, img_np)
    buf.seek(0)

    return Response(
        content      = buf.read(),
        media_type   = "application/octet-stream",
        headers      = {
            "X-Image-Height" : str(req.pixel_h),
            "X-Image-Width"  : str(req.pixel_w),
            "X-Format"       : "numpy-float32",
        },
    )


@app.post("/generate/haptic", tags=["Generation"])
async def generate_haptic(req: HapticRequest) -> Dict[str, Any]:
    """Compute haptic / torque-resistance map for surgical robot planning.

    Returns a resistance score ∈ [0, 1] per query point:
      0 = air / negligible resistance
      1 = maximum (dense enamel / cortical bone)
    """
    model = _get_model()
    pts   = torch.tensor(req.coordinates, dtype=torch.float32, device=DEVICE)

    haptic_mapper = HapticResistanceMapper(
        weight_elasticity = req.weight_elasticity,
        weight_density    = req.weight_density,
        weight_thermal    = req.weight_thermal,
    ).to(DEVICE)

    with torch.no_grad():
        physical = model.physical_output(pts)

    engine = _build_pathology_engine(req.pathologies)
    if engine is not None:
        with torch.no_grad():
            physical = engine.apply_to_physical(physical, pts)

    with torch.no_grad():
        resistance = haptic_mapper(physical)  # (N, 1)

    return {
        "n_points"  : len(req.coordinates),
        "resistance": resistance.squeeze(-1).cpu().tolist(),
        "stats"     : {
            "min"  : float(resistance.min()),
            "max"  : float(resistance.max()),
            "mean" : float(resistance.mean()),
        },
    }


@app.post("/generate/full-dataset", tags=["Generation"])
async def generate_full_dataset(req: FullDatasetRequest) -> Response:
    """Generate a complete labelled training sample as an NPZ archive.

    NPZ contents:
        hu_volume         : (R, R, R)  float32 — Hounsfield Units
        label_volume      : (R, R, R)  int32   — Tissue labels
        thermal_volume    : (R, R, R)  float32 — W·m⁻¹·K⁻¹
        elasticity_volume : (R, R, R)  float32 — GPa
        haptic_volume     : (R, R, R)  float32 — Resistance [0,1]
        drr               : (R, R)     float32 — DRR (if requested)

    R = grid_resolution.
    """
    model = _get_model()
    R     = req.grid_resolution

    # Build coordinate grid
    coords_1d = torch.linspace(-1.0, 1.0, R, device=DEVICE)
    zz, yy, xx = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    pts = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)  # (R³, 3)

    CHUNK = 4096
    hu_list, label_list, thermal_list, elastic_list = [], [], [], []

    with torch.no_grad():
        for i in range(0, pts.shape[0], CHUNK):
            chunk = pts[i : i + CHUNK]
            phys  = model.physical_output(chunk)
            hu_list.append(phys["hu"].cpu())
            label_list.append(phys["label"].cpu())
            thermal_list.append(phys["thermal_wpmk"].cpu())
            elastic_list.append(phys["elasticity_gpa"].cpu())

    hu_vol       = torch.cat(hu_list).reshape(R, R, R).numpy().astype(np.float32)
    label_vol    = torch.cat(label_list).reshape(R, R, R).numpy().astype(np.int32)
    thermal_vol  = torch.cat(thermal_list).reshape(R, R, R).numpy().astype(np.float32)
    elastic_vol  = torch.cat(elastic_list).reshape(R, R, R).numpy().astype(np.float32)

    # Haptic volume
    haptic_mapper = HapticResistanceMapper().to(DEVICE)
    haptic_list   = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], CHUNK):
            chunk = pts[i : i + CHUNK]
            phys  = model.physical_output(chunk)
            haptic_list.append(haptic_mapper(phys).cpu())
    haptic_vol = torch.cat(haptic_list).reshape(R, R, R).numpy().astype(np.float32)

    arrays: Dict[str, np.ndarray] = {
        "hu_volume"        : hu_vol,
        "label_volume"     : label_vol,
        "thermal_volume"   : thermal_vol,
        "elasticity_volume": elastic_vol,
        "haptic_volume"    : haptic_vol,
    }

    # DRR
    if req.render_drr:
        renderer = BeerLambertRayMarcher(model, n_samples=128, pixel_h=R, pixel_w=R).to(DEVICE)
        noise    = ClinicalNoiseModule() if req.apply_noise else None
        with torch.no_grad():
            drr = renderer(noise_module=noise).cpu().numpy().astype(np.float32)
        arrays["drr"] = drr

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)

    return Response(
        content    = buf.read(),
        media_type = "application/octet-stream",
        headers    = {
            "X-Grid-Resolution" : str(R),
            "X-Format"          : "numpy-npz-compressed",
            "Content-Disposition": f'attachment; filename="dentiligence_sample_r{R}.npz"',
        },
    )
