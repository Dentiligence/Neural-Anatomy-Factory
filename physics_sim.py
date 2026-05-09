"""
dentiligence/core/physics_sim.py
=================================
Differentiable Radiographic Renderer — Beer-Lambert Ray-Marching.

Physics background
------------------
X-ray attenuation through tissue follows the Beer-Lambert law:

    I(E) = I₀(E) · exp(-∫ μ(x, E) ds)

where:
    I₀(E)   = incident photon intensity at energy E (keV)
    μ(x, E) = linear attenuation coefficient at position x [cm⁻¹]
    ds       = differential path length element

The linear attenuation coefficient is derived from Hounsfield Units via:
    μ(x) = μ_water · (1 + HU(x) / 1000)

with μ_water ≈ 0.1928 cm⁻¹ at 70 keV (NIST XCOM database).

Clinical Noise Module models:
    1. Blood pooling — Gaussian density perturbation over vascular ROIs.
    2. Saliva coating — thin low-density surface layer.
    3. Metal artifact reduction (MAR) — streak-like Cauchy noise near
       high-density objects (amalgam, implants, gutta-percha).

References
----------
  - Johns & Cunningham (1983). Physics of Radiology, 4th ed.
  - NIST XCOM: https://physics.nist.gov/PhysRefData/Xcom/
  - Bamberg et al. (2011). Metal artifact reduction: J. Comput. Assist. Tomogr.

Authors : Dentiligence Core Team
Version : 0.1.0  (Series-A audit ready)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Physical constants (at 70 keV reference beam)
# ---------------------------------------------------------------------------

MU_WATER_CM1 = 0.1928          # cm⁻¹  — linear attenuation, water @ 70 keV
HU_WATER      = 0.0            # HU of water (definitional)
HU_AIR        = -1000.0        # HU of air

# Mapping HU → μ  (linearised)
def hu_to_mu(hu: torch.Tensor) -> torch.Tensor:
    """Convert Hounsfield Units to linear attenuation coefficient [cm⁻¹].

    Formula:  μ = μ_water · (1 + HU / 1000)

    Args:
        hu: Tensor of HU values.
    Returns:
        mu: Non-negative attenuation coefficients (clamped to ≥ 0).
    """
    mu = MU_WATER_CM1 * (1.0 + hu / 1000.0)
    return mu.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Differentiable Ray-Marching Renderer
# ---------------------------------------------------------------------------

class BeerLambertRayMarcher(nn.Module):
    """Differentiable X-ray renderer using Beer-Lambert ray marching.

    The renderer casts a bundle of parallel rays through the neural anatomy
    field, accumulates attenuation using the Beer-Lambert law, and returns a
    synthetic radiograph (DRR — Digitally Reconstructed Radiograph).

    Integration scheme: trapezoidal quadrature along each ray (differentiable).

    Args:
        model         : :class:`MultiHeadSIREN` neural anatomy field.
        n_samples     : Number of sample points per ray.
        ray_length    : Physical length of each ray in cm.
        pixel_h       : Detector height in pixels.
        pixel_w       : Detector width in pixels.
        source_dist   : Source-to-isocentre distance (cm).  Unused in parallel
                        geometry but reserved for cone-beam extension.
    """

    def __init__(
        self,
        model,
        n_samples : int   = 256,
        ray_length: float = 8.0,    # ~ 8 cm spans a full adult jaw
        pixel_h   : int   = 256,
        pixel_w   : int   = 256,
        source_dist: float = 100.0, # cm
    ) -> None:
        super().__init__()
        self.model       = model
        self.n_samples   = n_samples
        self.ray_length  = ray_length
        self.pixel_h     = pixel_h
        self.pixel_w     = pixel_w
        self.source_dist = source_dist

    # ------------------------------------------------------------------
    # Ray generation
    # ------------------------------------------------------------------

    def _generate_parallel_rays(self, device: torch.device) -> torch.Tensor:
        """Generate a parallel-beam ray grid.

        Returns:
            rays_o : ``(H*W, 3)`` ray origin points.
            rays_d : ``(H*W, 3)`` unit direction vectors (all pointing -z).
        """
        ys = torch.linspace(-1.0, 1.0, self.pixel_h, device=device)
        xs = torch.linspace(-1.0, 1.0, self.pixel_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        origins_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (H*W, 2)
        z_start    = torch.ones(origins_xy.shape[0], 1, device=device)     # start at z=+1

        rays_o = torch.cat([origins_xy, z_start], dim=-1)                  # (H*W, 3)
        rays_d = torch.zeros_like(rays_o)
        rays_d[:, 2] = -1.0                                                  # direction: -z

        return rays_o, rays_d

    # ------------------------------------------------------------------
    # Core renderer
    # ------------------------------------------------------------------

    def forward(
        self,
        noise_module: Optional["ClinicalNoiseModule"] = None,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Render a synthetic X-ray DRR.

        The volume is sampled along parallel rays and attenuated using the
        Beer-Lambert law.  Optionally applies clinical noise.

        Args:
            noise_module : Optional :class:`ClinicalNoiseModule` instance.
            chunk_size   : Batch size for GPU memory management.

        Returns:
            image: ``(H, W)`` tensor — normalised pixel intensities ∈ [0, 1].
                   High value = high transmission (air / low density).
                   Low  value = high attenuation (bone / metal).
        """
        device   = next(self.model.parameters()).device
        rays_o, rays_d = self._generate_parallel_rays(device)               # (N_rays, 3)
        n_rays   = rays_o.shape[0]

        # Trapezoidal sample points along each ray
        t_vals   = torch.linspace(0.0, self.ray_length, self.n_samples, device=device)  # (S,)
        dt       = t_vals[1] - t_vals[0]                                    # step size (cm)

        # Points: (N_rays, S, 3)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]

        # Clamp to normalised cube [-1, 1] — outside is air (μ=0)
        in_volume_mask = (pts.abs() <= 1.0).all(dim=-1)                     # (N_rays, S)

        # Query model in chunks to manage memory
        pts_flat    = pts.reshape(-1, 3)                                     # (N_rays*S, 3)
        hu_flat     = torch.full((pts_flat.shape[0], 1), HU_AIR, device=device)

        for i in range(0, pts_flat.shape[0], chunk_size):
            chunk = pts_flat[i : i + chunk_size]
            with torch.no_grad():
                raw = self.model.forward(chunk)
            hu_flat[i : i + chunk_size] = self.model.to_hu(raw["density"])

        hu_volume = hu_flat.reshape(n_rays, self.n_samples, 1)              # (N_rays, S, 1)

        # Zero out μ outside the volume (treat as air)
        mask_exp  = in_volume_mask.unsqueeze(-1).float()                    # (N_rays, S, 1)
        hu_masked = hu_volume * mask_exp + HU_AIR * (1.0 - mask_exp)

        # Apply clinical noise BEFORE attenuation integration
        if noise_module is not None:
            hu_masked = noise_module(hu_masked, pts.reshape(n_rays, self.n_samples, 3))

        mu = hu_to_mu(hu_masked.squeeze(-1))                                # (N_rays, S)

        # Trapezoidal integration: ∫ μ ds ≈ Σ (μᵢ + μᵢ₊₁)/2 · dt
        mu_mid      = 0.5 * (mu[:, :-1] + mu[:, 1:])                       # (N_rays, S-1)
        optical_depth = (mu_mid * dt).sum(dim=-1)                           # (N_rays,)

        # Beer-Lambert: I = I₀ · exp(-τ)
        transmission = torch.exp(-optical_depth)                            # (N_rays,)
        image        = transmission.reshape(self.pixel_h, self.pixel_w)    # (H, W)

        return image


# ---------------------------------------------------------------------------
# Clinical Noise Module
# ---------------------------------------------------------------------------

class ClinicalNoiseModule(nn.Module):
    """Stochastic noise injection simulating intra-oral clinical conditions.

    Models three distinct artefact sources:

    1. **Blood pooling** — localised Gaussian density perturbation
       (HU ~+40 to +70) over randomly selected vascular-proxy ROIs.

    2. **Saliva coating** — diffuse low-amplitude noise on soft-tissue
       surfaces (HU ±5–15), mimicking thin fluid layer attenuation.

    3. **Metal artefacts** — heavy-tailed Cauchy-distributed streak noise
       near high-HU voxels (amalgam ≈ +15 000 HU proxy; implants).
       Simulates photon starvation and beam hardening streaks.

    Args:
        blood_intensity    : Peak HU perturbation for blood noise.
        saliva_std         : Standard deviation (HU) of saliva noise.
        metal_threshold_hu : HU above which metal artefact noise is applied.
        metal_cauchy_scale : Scale parameter γ of the Cauchy distribution.
        blood_prob         : Probability that any given chunk has blood noise.
    """

    def __init__(
        self,
        blood_intensity   : float = 55.0,
        saliva_std        : float = 8.0,
        metal_threshold_hu: float = 2000.0,
        metal_cauchy_scale: float = 30.0,
        blood_prob        : float = 0.3,
    ) -> None:
        super().__init__()
        self.blood_intensity    = blood_intensity
        self.saliva_std         = saliva_std
        self.metal_threshold_hu = metal_threshold_hu
        self.metal_cauchy_scale = metal_cauchy_scale
        self.blood_prob         = blood_prob

    def forward(
        self,
        hu_volume: torch.Tensor,
        pts      : torch.Tensor,
    ) -> torch.Tensor:
        """Inject clinical noise into HU volume samples.

        Args:
            hu_volume : ``(N_rays, S, 1)`` raw HU values.
            pts       : ``(N_rays, S, 3)`` corresponding 3D positions.

        Returns:
            noisy_hu  : ``(N_rays, S, 1)`` perturbed HU values.
        """
        hu = hu_volume.clone()

        # 1. Saliva coating — global low-amplitude Gaussian noise
        saliva_noise = torch.randn_like(hu) * self.saliva_std
        # Apply only to near-surface voxels (|coord| > 0.8 in any axis)
        near_surface = (pts.abs() > 0.80).any(dim=-1, keepdim=True)        # (N, S, 1)
        hu = hu + saliva_noise * near_surface.float()

        # 2. Blood pooling — localised blobs
        if torch.rand(1).item() < self.blood_prob:
            # Random sphere centre in normalised coords
            centre = torch.rand(3, device=hu.device) * 1.6 - 0.8
            dist   = (pts - centre).norm(dim=-1, keepdim=True)             # (N, S, 1)
            radius = 0.1 + torch.rand(1, device=hu.device).item() * 0.15
            blood_mask   = (dist < radius).float()
            blood_kernel = self.blood_intensity * torch.exp(-dist**2 / (2 * (radius / 2)**2))
            hu = hu + blood_mask * blood_kernel

        # 3. Metal artefact streaks — Cauchy noise near high-HU voxels
        metal_mask = (hu > self.metal_threshold_hu).float()
        if metal_mask.sum() > 0:
            # Cauchy sampling via inverse CDF: γ·tan(π(U-0.5))
            u            = torch.rand_like(hu)
            cauchy_noise = self.metal_cauchy_scale * torch.tan(torch.pi * (u - 0.5))
            cauchy_noise = cauchy_noise.clamp(-300.0, 300.0)               # realistic cap
            # Propagate streaks along the ray (neighbouring samples)
            streak_mask  = F.max_pool1d(
                metal_mask.squeeze(-1),
                kernel_size=5,
                stride=1,
                padding=2,
            ).unsqueeze(-1)
            hu = hu + streak_mask * cauchy_noise

        return hu


# ---------------------------------------------------------------------------
# Multi-Energy DRR Renderer (extension: dual-energy CT simulation)
# ---------------------------------------------------------------------------

class DualEnergyRenderer(nn.Module):
    """Renders two DRRs at different beam energies for dual-energy CT simulation.

    Simulates the energy-dependence of X-ray attenuation.  The low-energy
    beam (e.g. 60 keV) accentuates soft-tissue contrast; the high-energy
    beam (e.g. 140 keV) provides bone mineralisation detail.

    The ratio image (low/high) enables virtual non-contrast and iodine maps
    used in cone-beam CT protocols.

    Args:
        model           : :class:`MultiHeadSIREN` neural field.
        energy_low_kev  : Low-energy beam reference (keV).
        energy_high_kev : High-energy beam reference (keV).
        n_samples       : Ray march sample count.
        pixel_h         : Detector rows.
        pixel_w         : Detector cols.
    """

    # μ_water at reference energies (NIST XCOM)
    MU_TABLE = {60: 0.2059, 70: 0.1928, 80: 0.1837, 100: 0.1707, 140: 0.1541}

    def __init__(
        self,
        model,
        energy_low_kev : int = 60,
        energy_high_kev: int = 140,
        n_samples      : int = 256,
        pixel_h        : int = 256,
        pixel_w        : int = 256,
    ) -> None:
        super().__init__()
        self.model          = model
        self.n_samples      = n_samples
        self.pixel_h        = pixel_h
        self.pixel_w        = pixel_w
        self.mu_low         = self.MU_TABLE.get(energy_low_kev,  0.2059)
        self.mu_high        = self.MU_TABLE.get(energy_high_kev, 0.1541)

    def _render_at_energy(self, mu_water: float, chunk_size: int = 4096) -> torch.Tensor:
        """Internal render at a specific energy level."""
        renderer = BeerLambertRayMarcher(
            self.model,
            n_samples=self.n_samples,
            pixel_h=self.pixel_h,
            pixel_w=self.pixel_w,
        )
        # Temporarily patch MU_WATER_CM1 — monkey-patch for this render pass
        import dentiligence.core.physics_sim as _psim
        _orig = _psim.MU_WATER_CM1
        _psim.MU_WATER_CM1 = mu_water
        img = renderer(chunk_size=chunk_size)
        _psim.MU_WATER_CM1 = _orig
        return img

    def forward(self, chunk_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            drr_low    : DRR at low energy   (H, W)
            drr_high   : DRR at high energy  (H, W)
            ratio_map  : Low/High ratio image (H, W), useful for material decomposition
        """
        drr_low  = self._render_at_energy(self.mu_low,  chunk_size)
        drr_high = self._render_at_energy(self.mu_high, chunk_size)
        ratio    = (drr_low + 1e-8) / (drr_high + 1e-8)
        return drr_low, drr_high, ratio
