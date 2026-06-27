# OT-CFM (gene-only) vs OT-CFM (naive-spatial) vs NicheFlow CFM vs NicheFlow VFM — metric comparison

Back-to-back metric comparison on the **same target slide (1.025)** with the **same neutral
classifier slide (1.026)**, now including the spatially-aware **`c2st_graph`** metric (a GCN over a
joint spatial-kNN graph with expression-only node features). Both NicheFlow models (CFM and VFM) are
evaluated at step 55000.

- **OT-CFM (gene-only)** — `reports/otcfm_1025/metrics.csv`. Unconditional, expression-only baseline;
  flat whole-slide output with **random placeholder coordinates** (aspatial); raw-gene PCA, 18 classes.
- **OT-CFM (naive-spatial)** — `reports/otcfm_spatial_1025/metrics.csv`. Same OT-CFM trained over the
  **concatenated** `[whitened-PCA expression | standardized coords]` vector, so it **jointly generates
  coordinates** (no conditioning, no OT weighting between blocks); raw-gene PCA target space, 18 classes.
- **NicheFlow CFM (unaligned)** — `reports/nicheflow_cfm_unaligned/metrics.csv`. Niche-shaped output;
  whitened shared PCA, 20 classes. Trained on the **unaligned** 1.024→1.025 pair.
- **NicheFlow VFM** — `reports/nicheflow_vfm/metrics.csv`. Same as CFM but the VFM variant, also
  trained on the unaligned pair; whitened shared PCA, 20 classes.

> **Comparability.**
> - **NicheFlow CFM vs VFM are directly comparable** — identical target, classifier, feature space
>   (whitened shared PCA), output shape, and both trained on the same unaligned pair. 
> - **OT-CFM gene-only vs naive-spatial are directly comparable** — same model/config, the only change
>   is whether coordinates are generated (`coord_mode=generate`) or random placeholders.
> - **OT-CFM family vs NicheFlow family is not strictly head-to-head**: the OT-CFM runs use a
>   **raw-gene PCA** (un-whitened) and 18 classes; NicheFlow uses a **whitened shared PCA** and 20
>   classes. So absolute expression distances (`ot_*/x`, `mmd2/x` — different PCA basis/scale) and
>   `ct/acc_real` (different classifier basis/classes) are confounded across families — marked ⚠.
> - **Coordinates for OT-CFM gene-only are random placeholders**
>
> Lower is better unless noted (↑). `1.000` = rounded; see the source CSVs for full precision.

## Distribution / two-sample (expression + position)

| Metric | OT-CFM gene-only | OT-CFM spatial | NicheFlow CFM (unaligned) | NicheFlow VFM | Expected |
|---|---|---|---|---|---|
| `c2st/acc` ↓ | 0.998 | 0.998 | 0.602 | 0.594 | →0.5 = indistinguishable from real |
| `c2st/auc` ↓ | 1.000 | 1.000 | 0.641 | 0.619 | →0.5 |
| `c2st/graph_acc` ↓ | 1.000 ⚠ | 0.999 | 0.629 | 0.609 | →0.5 (spatial-graph view) |
| `c2st/graph_auc` ↓ | 1.000 ⚠ | 1.000 | 0.680 | 0.652 | →0.5 |
| `c2st/pos_acc` ↓ | 0.634 ⚠ | 0.498 | 0.590 | 0.581 | →0.5 (position-only) |
| `mmd2/x` ↓ ⚠ | 0.369 | 0.339 | 0.003 | 0.003 | →0 = matched expression marginals |
| `mmd2/pos` ↓ | 0.073 ⚠ | −0.000 | 0.008 | 0.028 | →0 |
| `ot_w1/x` ↓ ⚠ | 29.07 | 28.80 | 5.47 | 5.21 | →0 (PCA-scale dependent) |
| `ot_w2/x` ↓ ⚠ | 35.94 | 35.57 | 5.60 | 5.38 | →0 |
| `ot_w1/pos` ↓ | 0.281 ⚠ | 0.049 | 0.192 | 0.214 | →0 |
| `ot_w2/pos` ↓ | 0.337 ⚠ | 0.058 | 0.228 | 0.277 | →0 |

> **What `c2st_graph` adds here.** For **OT-CFM naive-spatial**, the marginal coordinate distribution
> is essentially correct (`c2st/pos_acc 0.498`, `mmd2/pos ≈ 0`, tiny `ot_*/pos`) — yet the joint
> graph C2ST is still ~1.0. That meas the cells are placed with the right
> *marginal* geometry but the **expression↔position coupling is wrong**, which position-only metrics
> cannot see. For **NicheFlow CFM/VFM** the graph C2ST stays near chance (≈0.65–0.68 AUC), only
> marginally above the MLP `c2st` — their coupling holds up under the spatial view. (OT-CFM gene-only
> is ⚠: random coordinates make its graph topology meaningless; the ~1.0 only re-confirms its
> expression is trivially separable.)

## Geometry — point-set distances (⚠ OT-CFM gene-only coords are random placeholders)

| Metric | OT-CFM gene-only | OT-CFM spatial | NicheFlow CFM (unaligned) | NicheFlow VFM | Expected |
|---|---|---|---|---|---|
| `psd/mean` ↓ | 0.050 ⚠ | 0.014 | 0.020 | 0.020 | generated lands on real manifold |
| `psd/max` ↓ | 0.606 ⚠ | 0.245 | 0.156 | 0.164 | low worst-case |
| `spd/mean` ↓ | 0.013 ⚠ | 0.012 | 0.018 | 0.014 | real cloud is covered |
| `spd/max` ↓ | 0.044 ⚠ | 0.101 | 0.362 | 0.225 | low worst-case gap |

## Moran's I — spatial autocorrelation

| Metric | OT-CFM gene-only | OT-CFM spatial | NicheFlow CFM (unaligned) | NicheFlow VFM | Expected |
|---|---|---|---|---|---|
| `moran/real_mean` | 0.257 | 0.257 | 0.189 | 0.189 | reference (real structure) |
| `moran/gen_mean` → real_mean | −0.000 | 0.249 | 0.112 | 0.175 | match `real_mean` |
| `moran/corr` ↑ | 0.096  | 0.735 | 0.937 | 0.981 | →1 (per-gene match) |
| `moran/mae` ↓ | 0.257 | 0.078 | 0.077 | 0.023 | →0 |

## Cell-type classifier `ct/*` (classifier trained on slide 1.026)

| Metric | OT-CFM gene-only | OT-CFM spatial | NicheFlow CFM (unaligned) | NicheFlow VFM | Expected |
|---|---|---|---|---|---|
| `ct/acc_real` ⚠ | 0.524 | 0.625 | 0.324 | 0.292 | classifier sanity on real niches |
| `ct/acc_gen` ↑ | 0.340 | 0.594 | 0.326 | 0.308 | high = generated classifiable |
| `ct/acc_gap` ↓ | 0.184 | 0.031 | 0.002 | 0.016 | →0 = gen as classifiable as real |
| `ct/acc` ↑ | 0.531 | 0.777 | 0.544 | 0.606 | →1 = labels agree with paired-real |
| `ct/f1` ↑ | 0.368 | 0.733 | 0.494 | 0.571 | →1 |
| `ct/prop_kl` ↓ | 7.31 | 0.368 | 0.113 | 0.158 | →0 = composition matches |
| `ct/prop_tv` ↓ | 0.469 | 0.151 | 0.193 | 0.140 | →0 |
| `ct/prop_jsd` ↓ | 0.199 | 0.030 | 0.025 | 0.016 | →0 |

## Regression — matched ground truth (niche-shaped models only)

| Metric | OT-CFM gene-only | OT-CFM spatial | NicheFlow CFM (unaligned) | NicheFlow VFM | Expected |
|---|---|---|---|---|---|
| `x/mae` ↓ | — (skipped) | — (skipped) | 1.057 | 1.062 | →0 vs matched real niche |
| `x/mse` ↓ | — (skipped) | — (skipped) | 1.917 | 1.959 | →0 |
| `pos/mae` ↓ | — (skipped) | — (skipped) | 0.553 | 0.532 | →0 |
| `pos/mse` ↓ | — (skipped) | — (skipped) | 0.502 | 0.451 | →0 |

_Regression needs cell-for-cell matched ground truth, which only the niche-shaped NicheFlow outputs
carry; the flat OT-CFM slides skip it._

## Takeaways

- **NicheFlow (CFM & VFM) dominate the distribution match.** Both are near-inseparable from real on
  every C2ST view (`c2st/auc` ≈ 0.62–0.64, `c2st/graph_auc` ≈ 0.65–0.68), with tiny `mmd2/x`
  (≈0.003), while both OT-CFM variants sit at ~1.0 separability and ~100× larger `mmd2/x`.
- **VFM edges CFM on spatial fidelity**: VFM `moran/corr 0.981`/`mae 0.023` vs CFM `0.937`/`0.077`,
  lower graph C2ST (`0.652` vs `0.680` AUC), and higher label agreement (`ct/acc 0.606` vs `0.544`) —
  though CFM has the tighter classifier accuracy gap (`0.002` vs `0.016`).
- **Naive-spatial OT-CFM fixes coordinate *marginals* but not the *coupling*.** It nails position-only
  metrics (`pos_acc 0.498`, `ot_*/pos` ≈ 0.05–0.06, `moran/mae 0.078` vs gene-only `0.257`) and is the
  best `ct/*` agreement here (`ct/acc 0.777`) — but `c2st/graph_auc 1.0` shows the joint
  expression–position structure is still wrong, the gap the graph metric is built to expose.
- **OT-CFM gene-only is the aspatial floor**: random coordinates (`moran/gen_mean ≈ 0`, `corr 0.10`)
  and trivially separable expression.

_Sources: `reports/otcfm_1025/metrics.csv`, `reports/otcfm_spatial_1025/metrics.csv`,
`reports/nicheflow_cfm_unaligned/metrics.csv`, `reports/nicheflow_vfm/metrics.csv`. The
`c2st/graph_*` rows for OT-CFM gene-only were computed on its saved generated artifact
(`--groups c2st_graph --n_pcs 50`, seed 0) and added to its CSV._
