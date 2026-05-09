"""
dentiligence/core/pathology_logic.py
======================================
PathologyEngine — Procedural Pathology Injection into Neural Anatomy Fields.

Design philosophy
-----------------
The engine modifies a :class:`MultiHeadSIREN` neural field *post-hoc* by
injecting pathology-specific delta functions (additive HU perturbations,
label overrides, and property changes) without retraining the backbone.

Each pathology is modelled as a differentiable spatial function that maps
3D coordinates to a ``PathologyDelta`` — a thin wrapper holding:
    - delta_hu      : Additive HU perturbation.
    - label_override: Optional semantic label replacement.
    - delta_elastic : Additive elasticity modifier (GPa).
    - delta_thermal : Additive thermal conductivity modifier.

Pathology categories
---------------------
    MICRO-ENDO  — C-shaped canals, MB2, calcified canals, periapical granuloma
    SURGERY     — Impacted 3rd molars, ameloblastoma, sinus proximity
    GENETIC     — Amelogenesis imperfecta, dens in dente, regional odontodysplasia
    IATROGENIC  — Root perforation, overfilled canals, sinus membrane puncture

Medical references
------------------
  - Vertucci (1984). Root canal anatomy of the human permanent teeth.
    Oral Surg. Oral Med. Oral Pathol. 58(5), 589-599.
  - Pell & Gregory (1933). Impacted mandibular third molars. Dental Cosmos.
  - Winter (1926). Impacted mandibular third molars. American Medical Book Co.
  - Neville et al. (2015). Oral & Maxillofacial Pathology, 4th ed.
  - Oehlers (1957). Dens invaginatus. Oral Surg. Oral Med. Oral Pathol.

Authors : Dentiligence Core Team
Version : 0.1.0  (Series-A audit ready)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .models import TISSUE_LABELS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PathologyLabel(IntEnum):
    """Extended label IDs for pathological structures."""
    PERIAPICAL_GRANULOMA  = 20
    CALCIFIED_CANAL       = 21
    MB2_CANAL             = 22
    C_SHAPED_CANAL        = 23
    IMPACTED_MOLAR        = 24
    AMELOBLASTOMA         = 25
    SINUS_PROXIMITY       = 26
    AMELOGENESIS_IMP      = 27
    DENS_IN_DENTE         = 28
    REGIONAL_ODONTO       = 29
    ROOT_PERFORATION      = 30
    OVERFILLED_CANAL      = 31
    SINUS_PUNCTURE        = 32


@dataclass
class PathologyDelta:
    """Spatial perturbation applied to the neural anatomy field at query points.

    All fields are ``(N, 1)`` tensors aligned with query coordinates.
    None means "no change" for that property.
    """
    delta_hu      : torch.Tensor           # Required: HU perturbation
    label_override: Optional[int] = None   # Integer label to assign
    delta_elastic : Optional[torch.Tensor] = None
    delta_thermal : Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Helper: Smooth signed-distance sphere mask
# ---------------------------------------------------------------------------

def _sphere_mask(
    pts     : torch.Tensor,   # (N, 3)
    centre  : torch.Tensor,   # (3,)
    radius  : float,
    sharpness: float = 20.0,
) -> torch.Tensor:            # (N, 1)
    """Differentiable soft sphere occupancy [0, 1]."""
    dist = (pts - centre).norm(dim=-1, keepdim=True)
    return torch.sigmoid(sharpness * (radius - dist))


def _ellipsoid_mask(
    pts    : torch.Tensor,   # (N, 3)
    centre : torch.Tensor,   # (3,)
    radii  : Tuple[float, float, float],
    sharpness: float = 15.0,
) -> torch.Tensor:
    """Soft ellipsoid occupancy."""
    r = torch.tensor(radii, device=pts.device, dtype=pts.dtype)
    sdf = ((pts - centre) / r).pow(2).sum(dim=-1, keepdim=True).sqrt()
    return torch.sigmoid(sharpness * (1.0 - sdf))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BasePathology(nn.Module):
    """Abstract pathology injector."""

    label: int = PathologyLabel.PERIAPICAL_GRANULOMA  # override in subclass

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        raise NotImplementedError


# ===========================================================================
# MICRO-ENDO PATHOLOGIES
# ===========================================================================

class PeriapicalGranuloma(BasePathology):
    """Periapical granuloma — chronic inflammatory lesion at root apex.

    Radiographic appearance: well-defined radiolucent lesion, round/oval.
    HU perturbation: ≈ -200 to -400 HU (soft tissue replacing bone).

    Args:
        apex_position : Normalised 3D coordinate of root apex.
        radius        : Lesion radius in normalised units (≈ 0.05–0.15).
        severity      : 0=mild, 1=severe (scales HU depression and size).
    """

    label = PathologyLabel.PERIAPICAL_GRANULOMA

    def __init__(
        self,
        apex_position : Tuple[float, float, float] = (0.0, 0.0, -0.7),
        radius        : float = 0.08,
        severity      : float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("centre", torch.tensor(apex_position))
        self.radius   = radius * (1.0 + 0.5 * severity)
        self.delta_hu_peak = -200.0 - 200.0 * severity

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        mask = _sphere_mask(pts, self.centre, self.radius)   # (N, 1)
        delta_hu = mask * self.delta_hu_peak
        return PathologyDelta(
            delta_hu       = delta_hu,
            label_override = int(self.label),
            delta_elastic  = mask * (-2.0),   # inflammatory tissue: softer
        )


class CalcifiedCanal(BasePathology):
    """Calcified/obliterated root canal.

    Pulp lumen replaced by secondary dentine; HU increases markedly.
    Clinically: +300 to +600 HU above normal pulp tissue.

    Args:
        canal_axis : Tuple of (start_xyz, end_xyz) in normalised coords.
        canal_radius : Canal radius (normalised).
    """

    label = PathologyLabel.CALCIFIED_CANAL

    def __init__(
        self,
        canal_start  : Tuple[float, float, float] = (0.0, 0.0, 0.0),
        canal_end    : Tuple[float, float, float] = (0.0, 0.0, -0.6),
        canal_radius : float = 0.015,
    ) -> None:
        super().__init__()
        self.register_buffer("start", torch.tensor(canal_start))
        self.register_buffer("end",   torch.tensor(canal_end))
        self.canal_radius = canal_radius

    def _capsule_sdf(self, pts: torch.Tensor) -> torch.Tensor:
        """Signed distance to a capsule (cylinder with hemispherical caps)."""
        ab  = self.end - self.start                      # (3,)
        ap  = pts - self.start                            # (N, 3)
        t   = (ap * ab).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        closest = self.start + t * ab
        return (pts - closest).norm(dim=-1, keepdim=True)

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        sdf  = self._capsule_sdf(pts)                    # (N, 1)
        mask = torch.sigmoid(20.0 * (self.canal_radius - sdf))
        delta_hu = mask * 450.0                          # mineralised +450 HU
        return PathologyDelta(
            delta_hu       = delta_hu,
            label_override = int(self.label),
            delta_elastic  = mask * 8.0,   # harder tissue
        )


class MB2Canal(BasePathology):
    """Mesiobuccal-2 (MB2) canal — Vertucci Class IV configuration.

    The MB2 is a second canal in the mesiobuccal root of maxillary molars,
    often missed clinically.  Models a separate parallel canal path with
    slightly smaller diameter than MB1.

    Prevalence: ~70–95% of maxillary first molars (Stropko 1999).
    """

    label = PathologyLabel.MB2_CANAL

    def __init__(
        self,
        mb1_start: Tuple[float, float, float] = (-0.05, 0.0, 0.2),
        mb1_end  : Tuple[float, float, float] = (-0.05, 0.0, -0.6),
        offset   : float = 0.04,   # buccal offset between MB1 and MB2
        radius   : float = 0.010,
    ) -> None:
        super().__init__()
        # MB2 is offset buccally from MB1
        s2 = list(mb1_start); s2[1] += offset
        e2 = list(mb1_end);   e2[1] += offset
        self.inner = CalcifiedCanal(tuple(s2), tuple(e2), canal_radius=radius)
        # MB2 has normal (non-calcified) pulp by default
        self.inner.label = int(self.label)

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        delta = self.inner(pts)
        # Override: MB2 is a normal patent canal (negative HU shift vs dentin)
        sdf  = self.inner._capsule_sdf(pts)
        mask = torch.sigmoid(20.0 * (self.inner.canal_radius - sdf))
        delta.delta_hu = mask * (-250.0)   # pulp-like HU, not calcified
        return delta


class CShapedCanal(BasePathology):
    """C-shaped root canal system (most common in mandibular second molars).

    The C-shaped canal presents as a ribbon-like lumen spanning the
    mesiobuccal-lingual fusion, with a characteristic C cross-section.

    Models the canal as a curved tube following a circular arc in the
    bucco-lingual plane.
    """

    label = PathologyLabel.C_SHAPED_CANAL

    def __init__(
        self,
        arc_centre : Tuple[float, float, float] = (0.0, 0.0, -0.3),
        arc_radius : float = 0.12,
        tube_radius: float = 0.02,
        arc_angle  : float = math.pi * 1.5,   # 270° arc
    ) -> None:
        super().__init__()
        self.register_buffer("arc_centre", torch.tensor(arc_centre))
        self.arc_radius  = arc_radius
        self.tube_radius = tube_radius
        self.arc_angle   = arc_angle

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        # Signed distance to a torus-sector (approximate by full torus)
        ac  = self.arc_centre
        p_rel = pts - ac
        q_xy = torch.stack([
            p_rel[:, 0:1].pow(2).add(p_rel[:, 1:2].pow(2)).sqrt() - self.arc_radius,
            p_rel[:, 2:3],
        ], dim=-1).norm(dim=-1, keepdim=True)
        mask = torch.sigmoid(20.0 * (self.tube_radius - q_xy))
        delta_hu = mask * (-220.0)
        return PathologyDelta(delta_hu=delta_hu, label_override=int(self.label))


# ===========================================================================
# SURGICAL PATHOLOGIES
# ===========================================================================

class ImpactedThirdMolar(BasePathology):
    """Impacted mandibular third molar (Winter's & Pell-Gregory classification).

    Impaction types encoded by (angulation, depth, relation_to_ramus):
        Angulation  : mesioangular(35°), distoangular(-35°), horizontal(90°), vertical(0°)
        Pell-Gregory depth: Class A (0.0), B (0.3), C (0.6)
        Ramus relation   : Class I (1.0), II (0.7), III (0.4)

    The tooth is modelled as a high-HU ellipsoid (enamel crown + dentin root)
    at an angulated position within the posterior mandible.
    """

    label = PathologyLabel.IMPACTED_MOLAR

    ANGULATION_MAP = {
        "mesioangular"  : 35.0,
        "distoangular"  : -35.0,
        "horizontal"    : 90.0,
        "vertical"      : 0.0,
    }

    def __init__(
        self,
        position    : Tuple[float, float, float] = (0.5, 0.0, -0.5),
        angulation  : str   = "mesioangular",
        pg_depth    : str   = "B",     # Pell-Gregory A/B/C
        pg_ramus    : str   = "II",    # Pell-Gregory I/II/III
    ) -> None:
        super().__init__()
        angle_deg = self.ANGULATION_MAP.get(angulation, 35.0)
        self.angle_rad = math.radians(angle_deg)
        self.register_buffer("position", torch.tensor(position))
        # Depth modifies z placement
        depth_offset = {"A": 0.0, "B": -0.1, "C": -0.2}.get(pg_depth, 0.0)
        self.crown_radii = (0.08, 0.06, 0.05 + depth_offset * 0.1)
        self.root_radii  = (0.03, 0.025, 0.12)

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        # Rotate query points about x-axis by angulation angle
        cos_a = math.cos(self.angle_rad)
        sin_a = math.sin(self.angle_rad)
        p_rel = pts - self.position
        y_rot = p_rel[:, 1:2] * cos_a - p_rel[:, 2:3] * sin_a
        z_rot = p_rel[:, 1:2] * sin_a + p_rel[:, 2:3] * cos_a
        pts_rot = torch.cat([p_rel[:, 0:1], y_rot, z_rot], dim=-1)

        # Crown (high HU enamel)
        crown_centre = torch.zeros(3, device=pts.device)
        crown_mask   = _ellipsoid_mask(pts_rot, crown_centre, self.crown_radii)
        # Root (slightly lower HU dentin)
        root_centre  = torch.tensor([0.0, 0.0, -0.15], device=pts.device)
        root_mask    = _ellipsoid_mask(pts_rot, root_centre, self.root_radii)

        delta_hu = crown_mask * 2800.0 + root_mask * 700.0   # enamel/dentin HU
        return PathologyDelta(
            delta_hu       = delta_hu,
            label_override = int(self.label),
            delta_elastic  = crown_mask * 85.0 + root_mask * 15.0,
        )


class Ameloblastoma(BasePathology):
    """Ameloblastoma — benign odontogenic epithelial tumour.

    Radiographic hallmark: multilocular 'soap-bubble' pattern.
    Modelled as a collection of overlapping radiolucent spheres.
    HU: -100 to -300 (cystic fluid replacing bone).

    Args:
        centre         : Lesion centroid in normalised coords.
        overall_radius : Outer boundary radius.
        n_locules      : Number of internal soap-bubble locules.
        severity       : 0 (minimal) → 1 (aggressive multilocular).
    """

    label = PathologyLabel.AMELOBLASTOMA

    def __init__(
        self,
        centre        : Tuple[float, float, float] = (0.3, 0.0, -0.2),
        overall_radius: float = 0.25,
        n_locules     : int   = 6,
        severity      : float = 0.6,
    ) -> None:
        super().__init__()
        self.register_buffer("centre", torch.tensor(centre))
        self.overall_radius = overall_radius
        self.severity = severity
        # Pre-compute locule centres as static buffers
        locule_centres = []
        for i in range(n_locules):
            angle = 2 * math.pi * i / n_locules
            r     = overall_radius * 0.5
            loc   = [
                centre[0] + r * math.cos(angle),
                centre[1] + r * math.sin(angle),
                centre[2] + (i % 2) * 0.05,
            ]
            locule_centres.append(loc)
        self.register_buffer("locule_centres", torch.tensor(locule_centres))   # (L, 3)
        self.locule_radius = overall_radius * 0.40

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        # Outer envelope
        outer_mask = _sphere_mask(pts, self.centre, self.overall_radius)
        # Soap-bubble: union of locules
        locule_masks = [
            _sphere_mask(pts, self.locule_centres[i], self.locule_radius)
            for i in range(self.locule_centres.shape[0])
        ]
        locule_union = torch.stack(locule_masks, dim=-1).max(dim=-1).values  # (N, 1)

        # Septa: regions inside outer but between locules → higher HU (bone septa)
        septa_mask = outer_mask * (1.0 - locule_union)
        cyst_mask  = outer_mask * locule_union

        hu_cyst  = cyst_mask  * (-200.0 * self.severity)
        hu_septa = septa_mask * (300.0)   # residual bone trabeculae

        return PathologyDelta(
            delta_hu       = hu_cyst + hu_septa,
            label_override = int(self.label),
            delta_elastic  = cyst_mask * (-5.0),
        )


class SinusFloorProximity(BasePathology):
    """Maxillary sinus floor proximity — thin bony partition or dehiscence.

    Models the Schneiderian membrane and cortical sinus floor thinning.
    Critical for implant planning and sinus lift procedures.

    Membrane HU: ~20–60 HU (soft tissue)
    Cortical floor: 400–800 HU (may reduce to near 0 in dehiscence).
    """

    label = PathologyLabel.SINUS_PROXIMITY

    def __init__(
        self,
        sinus_floor_z  : float = 0.3,     # normalised z of sinus floor
        floor_thickness: float = 0.015,
        dehiscence     : bool  = False,
    ) -> None:
        super().__init__()
        self.sinus_floor_z   = sinus_floor_z
        self.floor_thickness = floor_thickness
        self.dehiscence      = dehiscence

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        z   = pts[:, 2:3]
        # Soft sigmoid slab at sinus floor
        slab_mask = torch.sigmoid(30.0 * (self.floor_thickness / 2 - (z - self.sinus_floor_z).abs()))
        if self.dehiscence:
            # Create a gap (radiolucent) within the slab
            gap_mask = torch.sigmoid(40.0 * (0.03 - pts[:, 0:1].abs()))
            slab_mask = slab_mask * (1.0 - gap_mask)
            delta_hu  = slab_mask * (-800.0)   # bone loss
        else:
            delta_hu  = slab_mask * (-200.0)   # thin cortex
        return PathologyDelta(delta_hu=delta_hu, label_override=int(self.label))


# ===========================================================================
# GENETIC / RARE PATHOLOGIES
# ===========================================================================

class AmelogenesisImperfecta(BasePathology):
    """Amelogenesis Imperfecta — hereditary enamel formation defect.

    Three subtypes modelled:
        hypoplastic  : Thin enamel, normal mineralisation (HU −300 vs normal)
        hypomaturation: Normal thickness, reduced mineralisation
        hypocalcified: Soft, poorly mineralised enamel (lowest HU)

    Affects the entire enamel shell uniformly.
    """

    label = PathologyLabel.AMELOGENESIS_IMP

    HU_DELTAS = {
        "hypoplastic"  : -300.0,
        "hypomaturation": -500.0,
        "hypocalcified" : -700.0,
    }

    def __init__(
        self,
        subtype         : str   = "hypocalcified",
        crown_centre    : Tuple[float, float, float] = (0.0, 0.0, 0.5),
        enamel_thickness: float = 0.04,
        crown_radius    : float = 0.10,
    ) -> None:
        super().__init__()
        self.register_buffer("crown_centre", torch.tensor(crown_centre))
        self.crown_radius    = crown_radius
        self.enamel_thickness = enamel_thickness
        self.delta_hu_val    = self.HU_DELTAS.get(subtype, -500.0)

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        dist = (pts - self.crown_centre).norm(dim=-1, keepdim=True)
        # Enamel shell: between crown_radius-thickness and crown_radius
        outer = torch.sigmoid(20.0 * (self.crown_radius - dist))
        inner = torch.sigmoid(20.0 * (dist - (self.crown_radius - self.enamel_thickness)))
        enamel_mask = outer * inner
        return PathologyDelta(
            delta_hu       = enamel_mask * self.delta_hu_val,
            label_override = int(self.label),
            delta_elastic  = enamel_mask * (self.delta_hu_val / 90.0),   # proportional softening
        )


class DensInDente(BasePathology):
    """Dens in Dente — Oehlers Type III (tooth within a tooth).

    Characterised by an invagination of the enamel organ into the dental
    papilla, creating an enamel-lined cavity within the pulp.

    Oehlers classification:
        Type I  : Crown only (shallow)
        Type II : Extends beyond CEJ but remains within root
        Type III: Extends through root apex (most severe — modelled here)

    HU profile: enamel-lined channel (2500–2800 HU shell) surrounding a
    low-density core (-200 to +50 HU).
    """

    label = PathologyLabel.DENS_IN_DENTE

    def __init__(
        self,
        invagination_centre: Tuple[float, float, float] = (0.0, 0.0, 0.3),
        outer_radius       : float = 0.05,
        inner_radius       : float = 0.03,
        oehlers_type       : int   = 3,
    ) -> None:
        super().__init__()
        self.register_buffer("centre", torch.tensor(invagination_centre))
        self.outer_r  = outer_radius
        self.inner_r  = inner_radius
        self.depth    = {1: 0.1, 2: 0.3, 3: 0.6}.get(oehlers_type, 0.6)

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        dist = (pts - self.centre).norm(dim=-1, keepdim=True)
        enamel_shell = (
            torch.sigmoid(20.0 * (self.outer_r - dist)) *
            torch.sigmoid(20.0 * (dist - self.inner_r))
        )
        inner_cavity = torch.sigmoid(20.0 * (self.inner_r - dist))

        delta_hu = enamel_shell * 2700.0 + inner_cavity * (-150.0)
        return PathologyDelta(
            delta_hu       = delta_hu,
            label_override = int(self.label),
            delta_elastic  = enamel_shell * 85.0,
        )


class RegionalOdontodysplasia(BasePathology):
    """Regional odontodysplasia ('ghost teeth') — severe dysplastic condition.

    Affects a regional group of teeth; characterised by extremely thin, poorly
    mineralised enamel and dentin — the 'ghost tooth' radiographic appearance.
    HU may drop to -500 to -700 in affected regions.
    """

    label = PathologyLabel.REGIONAL_ODONTO

    def __init__(
        self,
        region_centre: Tuple[float, float, float] = (0.2, 0.0, 0.0),
        region_radius: float = 0.30,
        severity     : float = 0.8,
    ) -> None:
        super().__init__()
        self.register_buffer("centre", torch.tensor(region_centre))
        self.radius   = region_radius
        self.severity = severity

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        mask      = _sphere_mask(pts, self.centre, self.radius, sharpness=8.0)
        delta_hu  = mask * (-650.0 * self.severity)
        return PathologyDelta(
            delta_hu       = delta_hu,
            label_override = int(self.label),
            delta_elastic  = mask * (-40.0 * self.severity),
        )


# ===========================================================================
# IATROGENIC (ERROR) PATHOLOGIES
# ===========================================================================

class RootPerforation(BasePathology):
    """Iatrogenic root perforation — communication between canal & periodontium.

    Modelled as a small puncture through the root at a specified depth.
    Creates a localised radiolucent defect with surrounding inflammatory change.
    """

    label = PathologyLabel.ROOT_PERFORATION

    def __init__(
        self,
        perforation_site: Tuple[float, float, float] = (0.03, 0.0, -0.2),
        perforation_radius: float = 0.008,
    ) -> None:
        super().__init__()
        self.register_buffer("site", torch.tensor(perforation_site))
        self.radius = perforation_radius

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        mask = _sphere_mask(pts, self.site, self.radius, sharpness=40.0)
        # Perforation itself: radiolucent
        delta_hu = mask * (-400.0)
        # Surrounding inflammatory halo: slightly less radiolucent
        halo_mask = _sphere_mask(pts, self.site, self.radius * 3, sharpness=10.0)
        delta_hu += halo_mask * (-100.0) * (1.0 - mask)
        return PathologyDelta(delta_hu=delta_hu, label_override=int(self.label))


class OverfilledCanal(BasePathology):
    """Overfilled root canal — excess obturating material beyond apex.

    Gutta-percha / sealer extrusion creates a radiopaque periapical mass.
    HU: +800 to +2500 (dense obturating material).
    """

    label = PathologyLabel.OVERFILLED_CANAL

    def __init__(
        self,
        apex_position  : Tuple[float, float, float] = (0.0, 0.0, -0.68),
        overfill_length: float = 0.05,
        canal_radius   : float = 0.012,
    ) -> None:
        super().__init__()
        self.register_buffer("apex", torch.tensor(apex_position))
        self.overfill_length = overfill_length
        self.canal_radius    = canal_radius

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        # Overfill blob below apex
        overfill_centre = self.apex.clone()
        overfill_centre[2] -= self.overfill_length / 2
        mask = _ellipsoid_mask(
            pts, overfill_centre,
            (self.canal_radius * 2, self.canal_radius * 2, self.overfill_length),
        )
        delta_hu = mask * 1800.0   # dense gutta-percha
        return PathologyDelta(
            delta_hu       = delta_hu,
            label_override = int(self.label),
            delta_elastic  = mask * 5.0,
        )


class SinusMembranePuncture(BasePathology):
    """Iatrogenic sinus membrane perforation — Schneiderian membrane tear.

    Modelled as a localised radiolucent defect in the sinus floor and
    membrane, with possible air entry into the membrane.
    """

    label = PathologyLabel.SINUS_PUNCTURE

    def __init__(
        self,
        puncture_site : Tuple[float, float, float] = (0.0, 0.0, 0.32),
        puncture_radius: float = 0.02,
    ) -> None:
        super().__init__()
        self.register_buffer("site", torch.tensor(puncture_site))
        self.radius = puncture_radius

    def forward(self, pts: torch.Tensor) -> PathologyDelta:
        mask     = _sphere_mask(pts, self.site, self.radius, sharpness=30.0)
        delta_hu = mask * (-1000.0)   # air entry / disrupted membrane
        return PathologyDelta(delta_hu=delta_hu, label_override=int(self.label))


# ===========================================================================
# PathologyEngine — orchestrator
# ===========================================================================

PATHOLOGY_REGISTRY: Dict[str, type] = {
    # Micro-endo
    "periapical_granuloma" : PeriapicalGranuloma,
    "calcified_canal"      : CalcifiedCanal,
    "mb2_canal"            : MB2Canal,
    "c_shaped_canal"       : CShapedCanal,
    # Surgical
    "impacted_molar"       : ImpactedThirdMolar,
    "ameloblastoma"        : Ameloblastoma,
    "sinus_proximity"      : SinusFloorProximity,
    # Genetic/rare
    "amelogenesis_imperfecta"   : AmelogenesisImperfecta,
    "dens_in_dente"             : DensInDente,
    "regional_odontodysplasia"  : RegionalOdontodysplasia,
    # Iatrogenic
    "root_perforation"     : RootPerforation,
    "overfilled_canal"     : OverfilledCanal,
    "sinus_membrane_puncture": SinusMembranePuncture,
}


class PathologyEngine(nn.Module):
    """Orchestrates multi-pathology injection into a neural anatomy field.

    Pathologies are composed additively: the combined HU delta and label
    override are applied to the base model's physical output before
    downstream rendering.

    Args:
        pathology_configs : List of (pathology_name, kwargs_dict) pairs.
                            Pathologies are applied in order; later ones win
                            on label_override conflicts.

    Example::

        engine = PathologyEngine([
            ("periapical_granuloma", {"severity": 0.7}),
            ("mb2_canal", {}),
            ("calcified_canal", {"canal_start": (0, 0, 0.1)}),
        ])
        deltas = engine(pts)   # list of PathologyDelta
    """

    def __init__(self, pathology_configs: List[Tuple[str, dict]]) -> None:
        super().__init__()
        modules = {}
        for name, kwargs in pathology_configs:
            cls = PATHOLOGY_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown pathology '{name}'. Available: {list(PATHOLOGY_REGISTRY)}")
            modules[name] = cls(**kwargs)
        self.pathologies = nn.ModuleDict(modules)

    def forward(self, pts: torch.Tensor) -> List[PathologyDelta]:
        """Evaluate all pathology injectors at query points.

        Args:
            pts: ``(N, 3)`` normalised coordinates.
        Returns:
            List of :class:`PathologyDelta`, one per registered pathology.
        """
        return [p(pts) for p in self.pathologies.values()]

    def apply_to_physical(
        self,
        physical: Dict[str, torch.Tensor],
        pts     : torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inject all pathology deltas into a physical output dict in-place.

        Args:
            physical : Output of :meth:`MultiHeadSIREN.physical_output`.
            pts      : ``(N, 3)`` coordinates matching physical tensors.

        Returns:
            Modified physical dict with pathology perturbations applied.
        """
        deltas = self.forward(pts)
        hu = physical["hu"].clone()
        label = physical["label"].clone()
        elasticity = physical["elasticity_gpa"].clone()

        for delta in deltas:
            hu = hu + delta.delta_hu
            if delta.label_override is not None:
                # Apply label where mask is strong (delta_hu magnitude > 10 HU)
                strong_mask = delta.delta_hu.abs().squeeze(-1) > 10.0
                label[strong_mask] = delta.label_override
            if delta.delta_elastic is not None:
                elasticity = (elasticity + delta.delta_elastic).clamp(0.001, 90.0)

        physical = dict(physical)   # shallow copy
        physical["hu"]             = hu.clamp(-1000.0, 3000.0)
        physical["label"]          = label
        physical["elasticity_gpa"] = elasticity
        return physical
