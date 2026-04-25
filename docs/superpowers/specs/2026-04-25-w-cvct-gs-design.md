# W-CVCT-GS: Wavelet-Disentangled Sub-Gaussians + Cross-View Color Triangulation

> Design spec for an innovation on top of the StereoGS pipeline (`train_stereo_gs.py`).
> Date: 2026-04-25
> Branch base: `test_idea/fast_foundation_stereo`

---

## 1. Motivation & Research Problem

### 1.1 What StereoGS does today

The current StereoGS pipeline (in `lib/stereo_gs/stereo_gs_model.py`, trained via `train_stereo_gs.py`) is a feed-forward stereo human-Gaussian-splatting framework:

```
Stereo pair → [frozen FFS] → backbone feats + disparity + cost volume + context
            → FeatureAdapter (1×1 conv, 3 scales)
            → CrossViewFusion (warp_attention | occlusion_aware)
            → ConfidenceExtractor
            → CAGS (FullResGaussianHead + AdaptiveSplitter, k_sub=3)
            → pts2render_cags
            → (optional) PostRefinement
```

Loss: `0.8·L1 + 0.2·(1-SSIM) + 0.5·Chamfer + 0.01·SparsityCAGS`.

### 1.2 Two unresolved weaknesses

**W1 — Sub-Gaussians have no semantic role.**
The CAGS module produces `k_sub=3` sub-Gaussians per pixel. The `LearnedSplitCriterion` is a black-box network that outputs three weights with no inductive bias. The sparsity loss is a blunt total-mass regularizer. Empirically, the three sub-Gaussians can collapse, scatter, or be redundant; there is no constraint encouraging them to specialize.

**W2 — Color is rigidly sourced from a single view.**
For each pixel-aligned Gaussian in the left view, color is taken directly from the left source image (analogously for the right view). There is no per-Gaussian appearance modulation, no cross-view evidence aggregation, and no explicit handling of occlusion boundaries. At edges, hair, semi-transparent regions, or sub-pixel offsets, this rigid sourcing leaves quality on the table.

### 1.3 Research statement

> **In feed-forward stereo Gaussian splatting, geometric structure should be supervised by a multi-scale frequency decomposition (wavelets), and appearance should be reconstructed by cross-view evidence triangulation. The two innovations close into one loop through a shared wavelet supervision in image space.**

---

## 2. Innovations Overview

| # | Innovation | Replaces / extends | Solves |
|---|---|---|---|
| **I-1** | **FDSG** — Frequency-Disentangled Sub-Gaussian supervision | `LearnedSplitCriterion` training signal + `cags_sparsity_weight` | W1: sub-Gaussian black-box |
| **I-2** | **CVCT** — Cross-View Color Triangulation | Pixel-direct color sourcing | W2: rigid single-view appearance |

**Coupling**: Both innovations are supervised by the same multi-band wavelet loss on the rendered image, which closes the loop:
- FDSG enforces *which* sub-Gaussian contributes to *which* frequency band.
- CVCT improves color, and its quality is verified by the same wavelet-band reconstruction.
- Together they form one supervisory signal — not two parallel losses.

### Working name: **W-CVCT-GS** (Wavelet-disentangled CVCT for Gaussian Splatting).

---

## 3. Innovation I-1: FDSG (Frequency-Disentangled Sub-Gaussians)

### 3.1 Hypothesis

The j-th sub-Gaussian should be **responsible only for reconstructing the j-th wavelet detail band** of the rendered image. The base (parent) Gaussian is responsible for the low-frequency residue.

```
Base Gaussian        →  reconstructs LL    (smooth body / low-frequency skin tone)
Sub-Gaussian k=1     →  reconstructs D1    (finest detail: hair, texture)
Sub-Gaussian k=2     →  reconstructs D2    (mid-scale: clothing folds)
Sub-Gaussian k=3     →  reconstructs D3    (coarse: silhouette, shading transitions)
```

This grants each sub-Gaussian a *physical role*, removing the ambiguity of W1.

### 3.2 Architecture changes

**No architecture changes** to the StereoGS pipeline. FFS, FeatureAdapter, CrossViewFusion, ConfidenceExtractor, FullResGaussianHead, AdaptiveSplitter, SubGaussianResidualHead all remain bit-identical.

The innovation is **entirely in the loss function** and a small augmentation of the rendering call to attribute outputs per sub-Gaussian level.

### 3.3 Three new loss terms

Notation: `DWT3(·)` is a 3-level Haar discrete wavelet transform (implemented as strided conv2d, no external dependency). It decomposes an image `I (B,3,H,W)` into:
- `LL` (low-frequency residue, `B,3,H/8,W/8`)
- `D_j = (LH_j, HL_j, HH_j)` for `j ∈ {1, 2, 3}` (detail bands at scale `2^j`)

#### L_band — multi-band rendering reconstruction
```
DWT3(I_rendered)  → {LL,    D1,    D2,    D3}
DWT3(I_GT)        → {LL*,   D1*,   D2*,   D3*}
L_band = β · ‖LL - LL*‖_1  +  Σ_j  α_j · ‖D_j - D_j*‖_1
```
Default band weights: `β=0.4, α₁=0.3, α₂=0.2, α₃=0.1`.

This is a global frequency-aware reconstruction loss; it is well-known in image restoration but novel as a supervision for feed-forward GS.

#### L_disentangle — sub-Gaussian frequency-band specialization (core novelty)
The j-th sub-Gaussian's rendering contribution must lie *only* in band `D_j`.

For each sub-Gaussian level `j ∈ {1,2,3}`:
```
I_full     = render(base + sub_1 + sub_2 + sub_3)
I_drop_j   = render(base + sub_{≠j})
ΔI_j       = I_full - I_drop_j                  # contribution of level-j sub-Gaussians
DWT3(ΔI_j) → {LL_Δ_j, D1_Δ_j, D2_Δ_j, D3_Δ_j}

L_disentangle_j = ‖LL_Δ_j‖_1  +  Σ_{i≠j} ‖D_i_Δ_j‖_1
L_disentangle   = Σ_j L_disentangle_j
```
This is the loss that gives each sub-Gaussian its physical meaning. It requires either four rendering passes per training step (naive), or an augmented rasterizer that returns per-level alpha-accumulated images in one pass (optimized; see §6.5).

#### L_active — GT-wavelet-guided sparsity (replaces existing CAGS sparsity)
```
E_j_GT = wavelet_energy(D_j*) ↑upsample to (B,1,H,W)
       normalized to [0,1] per-image

L_active = Σ_j  ‖ weights[:, j-1, :, :]  ·  (1 - E_j_GT) ‖_1
```
where `weights` is the output of `LearnedSplitCriterion`. This says: *sub-Gaussian level `j` may activate only where GT has band-`j` energy.*

This replaces (does not stack on) the existing `cags_sparsity_weight` regularizer, which is too coarse.

---

## 4. Innovation I-2: CVCT (Cross-View Color Triangulation)

### 4.1 Color generation formula

Each Gaussian's color is computed as a per-Gaussian, learnable mixture of cross-view evidence plus a bounded residual:

```
c_final = ω · c_left  +  (1 - ω) · c_right_warped  +  Δrgb
```

where:
- `c_left`: pixel color sampled from the source view this Gaussian belongs to.
- `c_right_warped`: pixel color from the *other* source view, warped via FFS disparity using `disparity_warp` already in `lib/stereo_gs/cross_view_fusion.py`.
- `ω ∈ [0,1]`: per-pixel learned visibility gate.
- `Δrgb ∈ [-ε, +ε]`: bounded residual correction (default ε = 0.05).

### 4.2 Module placement

Inserted between `_process_single_view_cags` (end of CAGS) and `pts2render_cags`. Color is the only attribute affected; xyz/rot/scale/opacity flow unchanged.

### 4.3 Three sub-components

| Component | Inputs | Architecture | Params |
|---|---|---|---|
| `ColorEvidence` | source_left, source_right, disparity | `disparity_warp` (parameter-free) | 0 |
| `VisibilityGate` | `fused_feat` (B,C,H,W) + `confidence` (B,1,H,W) | `Conv3×3(C+1, 32) → ReLU → Conv3×3(32,32) → ReLU → Conv1×1(32,1) → Sigmoid` | ~3K |
| `ResidualHead` | `shared_feat` from `FullResGaussianHead` | `Conv3×3(C', 16) → ReLU → Conv1×1(16, 3) → tanh × ε` | ~2K |

### 4.4 Sub-Gaussian inheritance

Sub-Gaussians do not get their own `ω` and `Δrgb` heads. Instead:
```
sub_ω      = parent_ω + 0.1 · tanh(δω)         # δω: 1 extra channel in SubGaussianResidualHead
sub_Δrgb   = parent_Δrgb · sub_opacity_weight  # natural attenuation by opacity
```
This keeps appearance consistent across parent and children, prevents discontinuities, and adds essentially no extra parameters.

### 4.5 Two new loss terms

#### L_cycle — cross-view photometric consistency
At pixels where both views see the same surface (`ω` near 0.5), the two color evidences must agree:
```
visibility_mask = 4 · ω · (1 - ω)        # peaks at ω=0.5, vanishes at endpoints
L_cycle = Σ_pixel  visibility_mask · ‖c_left - c_right_warped‖_1
```
The `4·ω·(1-ω)` mask automatically avoids occluded regions, so the loss does not penalize the model for legitimate disagreements.

#### L_omega_align — visibility soft alignment
A soft target for `ω` is constructed from the existing `OcclusionAwareFusion` confidence and the cycle-consistency mask:
```
ω_target = 0.5 in dual-visible regions,
           1.0 in left-exclusive regions,
           0.0 in right-exclusive regions
L_omega_align = ‖ω - ω_target.detach()‖_2^2
```
This grants `ω` an interpretable "visibility" semantics: visualizing `ω` directly produces an occlusion mask, which strengthens the paper's interpretability story.

---

## 5. End-to-end Loss

```
L_total =     0.8 · L_L1 + 0.2 · L_SSIM      ── RGB primary loss (preserved)
        +    λ_band      · L_band            ── multi-band reconstruction (FDSG)
        +    λ_dis       · L_disentangle     ── sub-Gaussian band specialization (FDSG core)
        +    λ_act       · L_active          ── GT-wavelet-guided sparsity (replaces existing sparsity)
        +    λ_cyc       · L_cycle           ── cross-view photometric consistency (CVCT core)
        +    λ_omega     · L_omega_align     ── visibility soft alignment (CVCT interpretability)
        +    λ_chamfer   · L_chamfer         ── left/right point-cloud alignment (preserved)
```

### 5.1 Default weights

| Term | Weight | Rationale |
|---|---|---|
| `L_L1 / L_SSIM` | 0.8 / 0.2 | Match StereoGS baseline |
| `λ_band` | 0.3 | Aligned to L1 magnitude |
| `λ_dis` | 0.5 | Core novelty; warm-up at 0.1 for first 5K post-Phase-2 steps |
| `λ_act` | 0.05 | Replaces existing 0.01 sparsity; can be larger because GT-guided |
| `λ_cyc` | 0.2 | Physical constraint; stable after warmup |
| `λ_omega` | 0.1 | Soft supervision; do not overpower primary loss |
| `λ_chamfer` | 0.5 | Preserve original |

### 5.2 Three-phase training schedule

The losses can fight each other if all enabled simultaneously. We use a staged schedule:

**Phase 1 — Baseline (steps 0 → 5K)**
- Active: `L_L1 + L_SSIM + L_chamfer`.
- CVCT clamps `ω = 0.5` and `Δrgb = 0`.
- Goal: reproduce StereoGS baseline; ensure geometry trains stably.

**Phase 2 — FDSG only (steps 5K → 30K)**
- Add: `L_band + L_active`.
- CVCT remains in identity mode.
- Goal: learn frequency-aware structure; sub-Gaussian activations align with GT wavelet energy.

**Phase 3 — Full system (steps 30K → 200K)**
- Add: `L_disentangle + L_cycle + L_omega_align`.
- Unlock CVCT (`ω` and `Δrgb` learnable).
- Goal: finetune to final performance.

### 5.3 Numerical stability

| Issue | Mitigation |
|---|---|
| High-freq band magnitudes are tiny | Apply `sign(x)·log(1 + |x|·k)` (k=10) compression in loss only |
| `L_disentangle` requires multiple renders | Naive 4-render in early iterations; switch to single-pass per-level alpha attribution after validation (§6.5) |
| `ω` collapses to 0 or 1 | `L_omega_align` provides soft target; add `0.01 · entropy(ω)` to encourage middle range |
| Loss-magnitude mismatch | Run 100 warmup steps, log per-term magnitudes, rebalance λ to same order |

---

## 6. Implementation Plan

### 6.1 New files

```
lib/stereo_gs/
├── wavelet_ops.py              ← Haar DWT3 (forward + inverse), per-band energy
├── cvct.py                     ← VisibilityGate, ResidualHead, ColorEvidence assembly
├── losses_freq.py              ← L_band, L_disentangle, L_active
└── losses_cvct.py              ← L_cycle, L_omega_align
```

### 6.2 Modified files

```
lib/stereo_gs/stereo_gs_model.py    ← Hook CVCT in CAGS forward; expose per-level rendering data
lib/GaussianRender.py               ← Augment pts2render_cags to support per-level rendering attribution
train_stereo_gs.py                  ← Three-phase scheduler; new loss logging; CVCT identity-mode toggle
config/stereo_gs_stage.yaml         ← New section `wcvct:` with all hyperparameters
config/stereo_human_config.py       ← Schema additions
```

### 6.3 Config additions (`config/stereo_gs_stage.yaml`)

```yaml
wcvct:
  enable: true

  # FDSG
  fdsg:
    wavelet_levels: 3
    wavelet_type: 'haar'
    band_weights: [0.3, 0.2, 0.1]   # α_1, α_2, α_3
    ll_weight: 0.4                  # β
    lambda_band: 0.3
    lambda_disentangle: 0.5
    lambda_disentangle_warmup: 0.1
    lambda_disentangle_warmup_steps: 5000
    lambda_active: 0.05
    log_compress_k: 10.0
    use_per_level_render: false     # Phase A: false (4-render). Phase B: true (single-pass)

  # CVCT
  cvct:
    residual_bound: 0.05            # ε
    visibility_hidden: 32
    residual_hidden: 16
    lambda_cycle: 0.2
    lambda_omega_align: 0.1
    lambda_omega_entropy: 0.01

  # Schedule
  schedule:
    phase1_end: 5000
    phase2_end: 30000
    total_steps: 200000

  # Disable/replace existing CAGS sparsity (becomes redundant with L_active)
  override_cags_sparsity: true
```

### 6.4 Wavelet implementation (Haar, no external dependency)

```python
# lib/stereo_gs/wavelet_ops.py
def haar_dwt2d_step(x):
    # x: (B, C, H, W); H, W even
    a = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) * 0.5
    h = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) * 0.5
    v = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) * 0.5
    d = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) * 0.5
    return a, h, v, d

def dwt3(x):
    LL_1, LH_1, HL_1, HH_1 = haar_dwt2d_step(x)
    LL_2, LH_2, HL_2, HH_2 = haar_dwt2d_step(LL_1)
    LL_3, LH_3, HL_3, HH_3 = haar_dwt2d_step(LL_2)
    return {
        'LL': LL_3,
        'D1': (LH_1, HL_1, HH_1),
        'D2': (LH_2, HL_2, HH_2),
        'D3': (LH_3, HL_3, HH_3),
    }
```

Pad input to multiple of 8 if needed.

### 6.5 L_disentangle rendering strategies

**Phase A (Naive, recommended for first runs)** — render four times per training step:
1. `I_full = render(base + sub_1 + sub_2 + sub_3)`
2. `I_drop_j = render(base + sub_{≠j})` for j ∈ {1,2,3}
At batch_size=1 this is feasible; for larger batches, gradient accumulation is preferred over batching the four renders.

**Phase B (Optimized, after Phase A validates idea)** — single rendering pass with per-Gaussian level tag:
- Tag each Gaussian with its level (0=base, 1/2/3=sub).
- Modify `pts2render_cags` to return four images, one per level, by accumulating each level into a separate output buffer.
- Net cost ≈ 1.2× of single render rather than 4×.
- Requires a small augmentation of the rasterizer wrapper in `gaussian_renderer/`.

### 6.6 CVCT identity-mode toggle

For Phase 1 and Phase 2, CVCT outputs must be forced to:
```python
if not cvct_unlocked:
    omega = torch.full_like(..., 0.5)
    delta_rgb = torch.zeros_like(...)
```
This is implemented via a flag on the CVCT module set by the trainer based on `total_steps`.

### 6.7 Estimated effort

| Stage | Work | Estimate |
|---|---|---|
| W1 | Haar DWT utilities + L_band + L_active | 1 day |
| W2 | L_disentangle (naive 4-render + per-level mask plumbing) | 2 days |
| W3 | CVCT modules + L_cycle + L_omega_align | 2 days |
| W4 | Modify trainer for three-phase schedule + per-loss logging | 1 day |
| W5 | Train 200K steps (Phase 1 → 2 → 3) | 4-5 days wall-clock |
| W6 | 7-row ablation table | 1-2 weeks |
| W7 | Paper writing + visualizations | 2-3 weeks |

Total dev time: ~6-7 days. Compute time: 1-2 weeks.

---

## 7. Evaluation

### 7.1 Metrics

| Category | Metric | Purpose |
|---|---|---|
| Overall quality | PSNR, SSIM, LPIPS | Compare to GPS-Gaussian, StereoGS-baseline |
| Frequency quality | Per-band PSNR (LL, D1, D2, D3) | Verify FDSG genuinely improves high-freq |
| Generalization | Cross-dataset PSNR drop (THumanMV ↔ HiAS) | Use existing `val_psnr_official` and `val_psnr_hias` logging |
| Interpretability | `ω` histogram, per-level activation rates | Visualization for supplementary |
| Efficiency | Inference FPS, parameter count, GPU memory | Paper Table 2 |

### 7.2 Ablation table

| ID | Config | Validates |
|---|---|---|
| M0 | StereoGS baseline (CAGS=learned, no wavelet, no CVCT) | Baseline |
| M1 | + `L_band` only | Frequency-domain supervision contribution |
| M2 | + `L_disentangle` | **FDSG core hypothesis** |
| M3 | + `L_active` (replaces sparsity) | GT-guided vs blind sparsity |
| M4 | + CVCT module (no `L_cycle`) | CVCT structure contribution |
| M5 | + CVCT + `L_cycle` | Cross-view consistency contribution |
| M6 | Full system (FDSG + CVCT) | Final |
| M7 | Full, no warmup schedule | Schedule necessity |

### 7.3 Visualizations for the paper

1. **Occlusion-boundary quality** (CVCT value): zoom-ins on hair / ear / clothing edges.
2. **High-frequency band reconstruction** (FDSG value): per-band rendered vs GT; per-sub-Gaussian contribution `ΔI_j`.
3. **Interpretability** (supplementary):
   - `ω` heatmap vs GT occlusion mask.
   - Sub-Gaussian activation masks vs GT wavelet energy (per level).

### 7.4 Risks

| Risk | Probability | Mitigation |
|---|---|---|
| `L_disentangle` collapses sub-Gaussians toward zero (degenerates to base only) | Medium | Warmup `λ_dis` from 0.1; entropy regularizer; checkpoint after Phase 2 to verify activation rates > 5% per level |
| 4-render exceeds GPU memory | Low at batch_size=1 | Switch to per-level alpha-accumulation single-pass render |
| `ω` cannot learn 0.5 mid-values | Medium | Increase `λ_omega`; add entropy term; visualize against occlusion masks |
| Per-band PSNR rises but overall PSNR does not | Low | Reduce `λ_band`; indicates over-emphasized frequency loss |
| **Generalization drops on cross-dataset val** | **Critical** | Track `val_psnr_official` vs `val_psnr_hias` every 500 steps; if gap widens, lower `λ_dis` or `λ_active` |

---

## 8. Out of Scope

- Architecture changes to FFS, FeatureAdapter, CrossViewFusion, ConfidenceExtractor, FullResGaussianHead, AdaptiveSplitter, SubGaussianResidualHead, or PostRefinement.
- Replacing Haar wavelet with learnable / db4 / non-orthogonal wavelets (deferred to follow-up).
- Adding per-Gaussian SH coefficients or view-dependent appearance beyond the bounded `Δrgb` residual.
- Modifying the Chamfer distance loss or scale-regularization terms.
- Changing the rendering backend (`diff_gaussian_rasterization`).

---

## 9. Open Questions for Implementation

These are deferred to the implementation plan (writing-plans skill):

1. Exact API for per-level rendering attribution in `pts2render_cags` (batched 4-render vs. extended rasterizer call).
2. Whether to compute DWT on `[0,1]` images or `[-1,1]` — likely `[0,1]` for stable magnitudes.
3. Whether `L_omega_align` should backprop through `OcclusionAwareFusion` confidence (currently `.detach()`) — start detached.
4. Image resolution: current pipeline outputs full-resolution maps; verify wavelet levels remain valid at the smallest scale (H/8 ≥ 32).
5. AMP / mixed precision compatibility with custom DWT — Haar is numerically stable, expect no issues.
