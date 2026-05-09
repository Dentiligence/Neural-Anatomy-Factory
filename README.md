# Dentiligence — Neural Anatomy Factory
## High-Fidelity Synthetic Dental Data Infrastructure
### `v0.1.0` · Series-A Technical Audit Ready

---

## Overview

Dentiligence is a **Neural Anatomy Factory** built on Implicit Neural Representations (INR) using **SIREN** (Sinusoidal Representation Networks). It generates continuous, resolution-free, multi-modal synthetic dental anatomy data for training autonomous surgical robots and diagnostic AI systems.

Unlike volumetric (voxel) or mesh (STL) pipelines, a single learned neural field encodes the complete anatomical structure and can be queried at arbitrary resolution to produce any of 5 output tiers.

---

## Architecture Diagram

```
                        ┌─────────────────────────────────┐
                        │        MultiHeadSIREN           │
                        │   (x, y, z) → ℝ⁴ Neural Field  │
                        └────────────┬────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
     SIRENBackbone              PathologyEngine         ClinicalNoiseModule
   (shared sinusoidal trunk)  (delta injection)        (blood/saliva/metal)
              │
    ┌─────────┼──────────┬──────────┐
    │         │          │          │
 Head 1    Head 2     Head 3     Head 4
 Density   Labels    Thermal   Elasticity
    │         │          │          │
    ▼         ▼          ▼          ▼
┌───────┬─────────┬──────────┬──────────┐
│ Tier 2│ Tier 3  │ Tier 4   │ Tier 5   │
│  HU/  │ Semantic│ X-ray    │ Haptic   │
│ DICOM │ Labels  │ DRR Sim  │ Torque   │
└───────┴─────────┴──────────┴──────────┘
                        │
                    Tier 1
               Marching Cubes
               Surface Mesh (STL)
```

---

## 5-Tier Multi-Modal Output

| Tier | Name | Format | Consumer |
|------|------|--------|----------|
| 1 | Surface Geometry | STL (marching cubes from HU field) | Students, haptic simulators |
| 2 | Volumetric Density | RAW / DICOM (HU grid) | Radiology AI training |
| 3 | Semantic Segmentation | Label volume + PNG masks | Diagnostic model supervision |
| 4 | Radiographic Physics | DRR via Beer-Lambert ray marching | X-ray/CBCT AI training |
| 5 | Robotic Haptic Map | Resistance [0,1] per point | Surgical robot force feedback |

---

## Repository Structure

```
dentiligence/
├── core/
│   ├── __init__.py
│   ├── models.py           # MultiHeadSIREN, HapticResistanceMapper
│   ├── physics_sim.py      # BeerLambertRayMarcher, ClinicalNoiseModule, DualEnergyRenderer
│   └── pathology_logic.py  # PathologyEngine + all 13 pathology classes
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI REST endpoints
├── deployment/
│   ├── Dockerfile
│   └── vercel.json
├── requirements.txt
└── README.md
```

---

## Pathology Catalogue (13 classes, 4 categories)

### Micro-Endodontic
| ID | Pathology | Clinical Basis |
|----|-----------|---------------|
| `periapical_granuloma` | Periapical granuloma | Chronic apical periodontitis; HU -200 to -400 |
| `calcified_canal` | Calcified/obliterated canal | Secondary dentine +300-600 HU |
| `mb2_canal` | MB2 canal (Vertucci Class IV) | Stropko 1999: 70-95% prevalence |
| `c_shaped_canal` | C-shaped canal system | Mandibular 2nd molars, ribbon-like lumen |

### Surgical
| ID | Pathology | Clinical Basis |
|----|-----------|---------------|
| `impacted_molar` | Impacted 3rd molar | Winter's + Pell-Gregory classification |
| `ameloblastoma` | Ameloblastoma | Soap-bubble multilocular pattern |
| `sinus_proximity` | Sinus floor proximity | Schneiderian membrane proximity / dehiscence |

### Genetic / Rare
| ID | Pathology | Clinical Basis |
|----|-----------|---------------|
| `amelogenesis_imperfecta` | Amelogenesis Imperfecta | Hypoplastic / hypomaturation / hypocalcified |
| `dens_in_dente` | Dens in Dente | Oehlers Type I/II/III invagination |
| `regional_odontodysplasia` | Regional Odontodysplasia | Ghost tooth, HU -500 to -700 |

### Iatrogenic (Error Training Data)
| ID | Pathology | Clinical Basis |
|----|-----------|---------------|
| `root_perforation` | Root perforation | Strip perforation / crestal complication |
| `overfilled_canal` | Overfilled canal | Gutta-percha extrusion beyond apex |
| `sinus_membrane_puncture` | Sinus membrane puncture | Schneiderian membrane tear |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run API locally

```bash
uvicorn api.main:app --reload --port 8000
```

OpenAPI docs: http://localhost:8000/docs

### 3. Query anatomy field (Python client)

```python
import httpx, numpy as np

resp = httpx.post("http://localhost:8000/generate/anatomy", json={
    "coordinates": [[0.0, 0.0, 0.0], [0.1, -0.2, 0.5]],
    "pathologies": [
        {"name": "periapical_granuloma", "params": {"severity": 0.8}},
        {"name": "mb2_canal", "params": {}}
    ]
})
data = resp.json()
print(data["hu"])           # Hounsfield Units per point
print(data["label_names"])  # ['pulp', 'dentin', ...]
```

### 4. Generate full training NPZ

```python
resp = httpx.post("http://localhost:8000/generate/full-dataset", json={
    "grid_resolution": 64,
    "render_drr": True,
    "pathologies": [{"name": "impacted_molar", "params": {"angulation": "mesioangular"}}]
}, timeout=120)

arrays = np.load(io.BytesIO(resp.content))
print(arrays["hu_volume"].shape)    # (64, 64, 64)
print(arrays["label_volume"].shape) # (64, 64, 64)
print(arrays["drr"].shape)          # (64, 64)
```

---

## Physics Model Reference

### Beer-Lambert X-ray Attenuation

```
I = I₀ · exp(−∫ μ(x) ds)
μ(x) = μ_water · (1 + HU(x) / 1000)
μ_water = 0.1928 cm⁻¹  @ 70 keV (NIST XCOM)
```

### Tissue Property Ranges

| Tissue | HU | E (GPa) | κ (W/mK) |
|--------|----|---------|----------|
| Enamel | 2500–3000 | 80–90 | 0.9–1.2 |
| Dentin | 600–1100 | 18–25 | 0.5–0.6 |
| Pulp | -100–+50 | 0.002–0.003 | 0.1–0.2 |
| Cortical bone | 500–1500 | 14–20 | 0.4–0.6 |
| Air / Sinus | -1000 | ~0 | ~0.025 |

---

## Docker Deployment

```bash
cd deployment
docker build -f Dockerfile -t dentiligence:latest ..
docker run -p 8000:8000 \
  -v /path/to/weights:/app/weights:ro \
  -e MODEL_WEIGHTS_PATH=/app/weights/model.pt \
  dentiligence:latest
```

---

## Medical References

1. Sitzmann et al. (2020). *Implicit Neural Representations with Periodic Activation Functions*. NeurIPS.
2. Vertucci (1984). *Root canal anatomy of the human permanent teeth*. Oral Surg. 58(5).
3. Pell & Gregory (1933). *Impacted mandibular third molars*. Dental Cosmos.
4. Magne et al. (2002). *Enamel and dentin moduli*. J. Prosthet. Dent. 87(5).
5. Braden (1964). *Heat conduction in normal and carious human teeth*. J. Dent. Res.
6. Neville et al. (2015). *Oral & Maxillofacial Pathology*, 4th ed. Elsevier.
7. Oehlers (1957). *Dens invaginatus*. Oral Surg. Oral Med. Oral Pathol.
8. NIST XCOM Photon Cross Sections Database. https://physics.nist.gov/PhysRefData/Xcom/

---

## Licence

Proprietary — Dentiligence Inc. All rights reserved.
Not for distribution without written consent.
