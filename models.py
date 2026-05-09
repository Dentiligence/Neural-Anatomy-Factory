"""
dentiligence/core/models.py
============================
Multi-Head SIREN (Sinusoidal Representation Network) — Neural Anatomy Field.

Architecture
------------
  Input  : (x, y, z) ∈ ℝ³   — continuous, resolution-free spatial coordinates
  Output : (density, label_logits, thermal_conductivity, elasticity)

Coordinate convention (normalised to [-1, 1]):
    x → mesio-distal axis
    y → bucco-lingual axis
    z → apico-coronal axis

Physical property references
------------------------------
  HU range        : -1000 → +3000  (standard radiographic window)
  Elasticity (E)  : 0.001 → 90 GPa  [Magne et al., J. Prosthet. Dent. 2002]
  Thermal cond.   : 0.1   → 1.2  W·m⁻¹·K⁻¹  [Braden, J. Dent. Res. 1964]

Authors : Dentiligence Core Team
Version : 0.1.0  (Series-A audit ready)
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dental tissue label registry
# ---------------------------------------------------------------------------

TISSUE_LABELS: Dict[str, int] = {
    "background"          : 0,
    "enamel"              : 1,
    "dentin"              : 2,
    "pulp"                : 3,
    "cementum"            : 4,
    "pdl"                 : 5,   # Periodontal Ligament
    "alveolar_bone_dense" : 6,
    "alveolar_bone_cancel": 7,
    "cortical_bone"       : 8,
    "gingiva"             : 9,
    "sinus_membrane"      : 10,
    "air_sinus"           : 11,
    "pathology"           : 12,  # Generic; overridden by PathologyEngine
}
NUM_LABELS = len(TISSUE_LABELS)

# Physical unit ranges used for de-normalisation
HU_RANGE          = (-1000.0, 3000.0)
ELASTICITY_RANGE  = (0.001,   90.0)    # GPa
THERMAL_RANGE     = (0.1,     1.2)     # W·m⁻¹·K⁻¹


# ---------------------------------------------------------------------------
# SineLayer — core SIREN primitive
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    """Fully-connected layer with sinusoidal activation.

    Implements:
        h = sin(ω₀ · (W·x + b))

    Weight initialisation follows Sitzmann et al. (NeurIPS 2020):
      - First layer : W ~ U[-1/n_in,  1/n_in]
      - Other layers: W ~ U[-√(6/n_in)/ω₀, √(6/n_in)/ω₀]

    Args:
        in_features  : Input dimension.
        out_features : Output dimension.
        bias         : Include additive bias.
        is_first     : True for the network's first layer.
        omega_0      : Angular frequency ω₀ (default 30).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear  = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights(is_first)

    def _init_weights(self, is_first: bool) -> None:
        with torch.no_grad():
            n_in = self.linear.in_features
            bound = (1.0 / n_in) if is_first else (math.sqrt(6.0 / n_in) / self.omega_0)
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


# ---------------------------------------------------------------------------
# Shared SIREN backbone
# ---------------------------------------------------------------------------

class SIRENBackbone(nn.Module):
    """Shared sinusoidal trunk that maps (x,y,z) → latent feature vector.

    Args:
        in_features     : Spatial input dim (3).
        hidden_features : Layer width.
        hidden_layers   : Number of intermediate SineLayers.
        omega_0         : ω₀ for the first layer.
        omega_hidden    : ω₀ for subsequent layers.
    """

    def __init__(
        self,
        in_features: int     = 3,
        hidden_features: int = 256,
        hidden_layers: int   = 5,
        omega_0: float       = 30.0,
        omega_hidden: float  = 30.0,
    ) -> None:
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_hidden))
        self.net = nn.Sequential(*layers)
        self.out_features = hidden_features

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: ``(N, 3)`` normalised coordinates.
        Returns:
            latent: ``(N, hidden_features)``.
        """
        return self.net(xyz)


# ---------------------------------------------------------------------------
# Lightweight decoder head
# ---------------------------------------------------------------------------

class _DecoderHead(nn.Module):
    """Two-layer ReLU MLP attached to the shared backbone."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MultiHeadSIREN — main neural anatomy model
# ---------------------------------------------------------------------------

class MultiHeadSIREN(nn.Module):
    """Neural Anatomy Field with four specialised output heads.

    Each spatial query point returns:
    ┌──────────────────────┬──────────┬─────────────────────────────────┐
    │ Head                 │ Shape    │ Description                     │
    ├──────────────────────┼──────────┼─────────────────────────────────┤
    │ density              │ (N, 1)   │ Raw → HU via :meth:`to_hu`      │
    │ label_logits         │ (N, L)   │ Semantic tissue class scores    │
    │ thermal_conductivity │ (N, 1)   │ Sigmoid [0,1] (de-norm: W/mK)  │
    │ elasticity           │ (N, 1)   │ Sigmoid [0,1] (de-norm: GPa)   │
    └──────────────────────┴──────────┴─────────────────────────────────┘

    Example::

        model = MultiHeadSIREN()
        xyz   = torch.rand(2048, 3) * 2 - 1      # normalised coords
        out   = model.physical_output(xyz)
        print(out["hu"].shape)                    # (2048, 1)
        print(out["label"].unique())              # tissue class IDs
    """

    def __init__(
        self,
        in_features: int     = 3,
        hidden_features: int = 256,
        hidden_layers: int   = 5,
        num_labels: int      = NUM_LABELS,
        omega_0: float       = 30.0,
        omega_hidden: float  = 30.0,
    ) -> None:
        super().__init__()
        self.backbone       = SIRENBackbone(in_features, hidden_features, hidden_layers, omega_0, omega_hidden)
        lat                 = hidden_features
        self.head_density   = _DecoderHead(lat, 1)
        self.head_label     = _DecoderHead(lat, num_labels)
        self.head_thermal   = _DecoderHead(lat, 1)
        self.head_elasticity= _DecoderHead(lat, 1)

    # ------------------------------------------------------------------

    def forward(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Raw network outputs (unbounded density, logits, sigmoid props).

        Args:
            xyz: ``(N, 3)`` float tensor, coordinates ∈ [-1, 1].

        Returns:
            Dict with keys: ``density``, ``label_logits``,
            ``thermal_conductivity``, ``elasticity``.
        """
        lat = self.backbone(xyz)
        return {
            "density"             : self.head_density(lat),
            "label_logits"        : self.head_label(lat),
            "thermal_conductivity": torch.sigmoid(self.head_thermal(lat)),
            "elasticity"          : torch.sigmoid(self.head_elasticity(lat)),
        }

    # ------------------------------------------------------------------
    # Unit converters
    # ------------------------------------------------------------------

    @staticmethod
    def to_hu(density_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw density logit → Hounsfield Units ∈ [-1000, 3000]."""
        lo, hi = HU_RANGE
        return torch.sigmoid(density_raw).mul(hi - lo).add(lo).clamp(lo, hi)

    @staticmethod
    def to_thermal(t_norm: torch.Tensor) -> torch.Tensor:
        """De-normalise thermal conductivity → W·m⁻¹·K⁻¹."""
        lo, hi = THERMAL_RANGE
        return t_norm * (hi - lo) + lo

    @staticmethod
    def to_elasticity(e_norm: torch.Tensor) -> torch.Tensor:
        """De-normalise elasticity → GPa (Young's modulus)."""
        lo, hi = ELASTICITY_RANGE
        return e_norm * (hi - lo) + lo

    # ------------------------------------------------------------------

    def physical_output(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Query in physical units — primary interface for downstream modules.

        Returns:
            ``hu``            : (N,1) Hounsfield Units
            ``label``         : (N,)  integer tissue class
            ``label_prob``    : (N,L) softmax probability
            ``thermal_wpmk``  : (N,1) W·m⁻¹·K⁻¹
            ``elasticity_gpa``: (N,1) GPa
        """
        raw = self.forward(xyz)
        return {
            "hu"             : self.to_hu(raw["density"]),
            "label"          : raw["label_logits"].argmax(dim=-1),
            "label_prob"     : F.softmax(raw["label_logits"], dim=-1),
            "thermal_wpmk"   : self.to_thermal(raw["thermal_conductivity"]),
            "elasticity_gpa" : self.to_elasticity(raw["elasticity"]),
        }


# ---------------------------------------------------------------------------
# Haptic / Torque Resistance Mapper  (Tier 5 — Robotic output)
# ---------------------------------------------------------------------------

class HapticResistanceMapper(nn.Module):
    """Derives haptic torque-resistance maps from physical tissue properties.

    Composite model:
        R = w_e·Ê + w_d·ρ̂ + w_t·κ̂

    where Ê, ρ̂, κ̂ are normalised elasticity, HU density, and thermal proxy.

    Args:
        weight_elasticity : Contribution weight for Young's modulus.
        weight_density    : Contribution weight for tissue density.
        weight_thermal    : Contribution weight for thermal conductivity proxy.
    """

    def __init__(
        self,
        weight_elasticity: float = 0.60,
        weight_density   : float = 0.30,
        weight_thermal   : float = 0.10,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor([weight_elasticity, weight_density, weight_thermal]))

    def forward(self, physical: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            physical: Output of :meth:`MultiHeadSIREN.physical_output`.
        Returns:
            resistance: ``(N, 1)`` ∈ [0, 1].  0 = air, 1 = dense enamel.
        """
        e_norm = (physical["elasticity_gpa"] - ELASTICITY_RANGE[0]) / (ELASTICITY_RANGE[1] - ELASTICITY_RANGE[0])
        d_norm = (physical["hu"]              - HU_RANGE[0])          / (HU_RANGE[1]          - HU_RANGE[0])
        t_norm = (physical["thermal_wpmk"]    - THERMAL_RANGE[0])     / (THERMAL_RANGE[1]     - THERMAL_RANGE[0])
        feat   = torch.cat([e_norm, d_norm, t_norm], dim=-1)
        return (feat * self.weights).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
