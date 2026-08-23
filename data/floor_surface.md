# Detection-floor ratio — sensitivity surface

`spatial_z` floor divided by the best non-compositional floor, at D_AUD=0.3, n_sim=500, n=15/group, alpha=0.025.

The single number 6.7x previously published is the value at `(D_AUD=0.24, rho=0)`. D_AUD=0.24 was an argmin over 25 single noisy draws and the last point of its own grid; rho=0 asserts spatially independent prediction noise, which is structurally impossible for a model whose head is `nn.Linear(hidden, low_rank_head=2048)` over 20,484 vertices.

| rho | spatial_z | raw_roi_mean | roi_minus_reference | glm_contrast_z | ratio |
|---|---|---|---|---|---|
| 0.0 | 0.1726 | 0.0296 | 0.021 | 0.029 | **8.22x** |
| 0.3 | 0.1858 | 0.0443 | 0.0482 | 0.0426 | **4.36x** |
| 0.6 | 0.1924 | 0.0555 | 0.0668 | 0.05 | **3.85x** |
| 0.9 | 0.2074 | 0.0636 | 0.0803 | 0.0616 | **3.37x** |

**Reported range: 3.4x – 8.2x** across rho in [0.0, 0.3, 0.6, 0.9].

This is a sensitivity range over a stipulated parameter, not a confidence interval and not min/max sampling noise: each row is a separate simulation at n_sim=500 with the same seed policy, so the spread between rows is the effect of rho, not of resampling.

Absolute floors remain in synthetic units and are computed at 15v15; S2 would run a different n. The ratio is the transferable quantity, and even it is conditional on the noise model.

## ⚠️ The ranking of the ALTERNATIVES reverses with rho — this changes the recommendation

| rho | raw_roi_mean | roi_minus_reference | glm_contrast_z | best |
|---|---|---|---|---|
| 0.0 | 0.0296 | **0.0210** | 0.0290 | `roi_minus_reference` |
| 0.3 | 0.0443 | 0.0482 | **0.0426** | `glm_contrast_z` |
| 0.6 | 0.0555 | 0.0668 | **0.0500** | `glm_contrast_z` |
| 0.9 | 0.0636 | 0.0803 | **0.0616** | `glm_contrast_z` |

`roi_minus_reference` is the most sensitive statistic at rho=0 and the **least** sensitive at every
rho >= 0.3. Its floor degrades 3.8x across the range; `glm_contrast_z` degrades 2.1x.

Mechanism: `roi_minus_reference` subtracts a fixed off-target region. When noise is spatially
independent that subtraction removes a per-clip gain and adds little variance. When noise is
correlated WITHIN parcels but not ACROSS them, the ROI and the reference fluctuate independently, so
the subtraction *adds* the reference parcel's variance instead of cancelling anything.

**Consequence for the paper.** Any statement of the form "the simplest non-compositional statistic
is also the most sensitive" is true only at rho=0. Since rho=0 is structurally impossible for TRIBE
(rank <= 2048 over 20,484 vertices), the recommendation must either be `glm_contrast_z`, or be
stated as conditional on a noise correlation nobody has measured. **rho for real TRIBE predictions
is UNMEASURED** — closing it needs the prediction cache, which does not exist yet.

What is robust across the whole range: `spatial_z` is the worst of the four at every rho.

## Choice of comparator — and a selection bias I built in

`floor_surface.py` originally reported the ratio against `min()` of the three alternatives, i.e. it
picked the best-performing alternative **separately at each rho**. That is a per-row selection on
the same data and it inflates the ratio at rho=0, where `roi_minus_reference` happens to win.

| comparator | range |
|---|---|
| best-of-three, selected per row | 3.4x - 8.2x  *(selection-inflated; do not lead with this)* |
| `glm_contrast_z`, fixed | **3.4x - 6.0x** |
| `roi_minus_reference`, fixed | 2.6x - 8.2x |

**Report the fixed-comparator range.** `glm_contrast_z` is the appropriate fixed comparator because
it is the best alternative at every realistic rho (>= 0.3); the resulting claim is
**"spatial_z's detection floor is 3.4x-6.0x worse than the best non-compositional statistic,
depending on a noise-correlation parameter we have not measured."**

The qualitative claim is unaffected by comparator choice: `spatial_z` is the worst of the four at
every rho tested, by at least 2.6x.
