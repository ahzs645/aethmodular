# Chemometrics package trials — 2026-08-17

Evidence-based verdicts on the section-2 recommendations of
[`docs/package-survey-2026-08-17.md`](../../../docs/package-survey-2026-08-17.md),
tested against the repo's validated PLS/CV/Deming machinery on the
lowest-OC/EC 800 cohort (X 800×2722 raw spectra, y = TOR EC loading µg/filter,
126 IMPROVE sites).

## Environment

Anaconda python 3.13.9 (`/Users/ahmadjalil/anaconda3/bin/python`, the
interpreter the phase-3 scripts run under), numpy 2.3.5, scipy 1.16.3,
scikit-learn 1.7.2. Installed for this trial:

| package | version | notes |
|---|---|---|
| ikpls | 6.1.2 | pulls in cvmatrix 3.2.2 (the fast-CV kernel) |
| astartes | 1.3.3 | |
| kennard-stone | 3.0.1 | |
| methcomp | 1.0.0 | installed fine on 3.13 — but see the bug below |

## Scripts (run from this directory)

- `trial_common.py` — loads/caches the 800 cohort (chunk-scans the 760 MB pool
  CSV once → `cache/lowest_ocec_800.npz`, git-ignored). Cross-checks the
  canonical `load_pool_metadata() ⋈ load_tor_loadings()` construction against
  the columns `lowest_ocec_800_cohort.csv` carries.
- `validate_ikpls.py` — fit equivalence, CV-curve equivalence, timing.
- `validate_deming.py` — scipy.odr wrapper vs `calibration_explorer/app.py::deming`
  vs methcomp, synthetic data with known slope.
- `demo_sampling.py` — Kennard-Stone / SPXY vs random 190-sample subsets.

## 1. ikpls vs the repo machinery

**Fit equivalence (k = 6, 800 cohort).** `ikpls.numpy.PLS` vs
`sklearn.PLSRegression(scale=False)` predictions on the cohort:

| ikpls variant | max \|Δpred\| | median | 
|---|---|---|
| algorithm #1 | 1.14e-12 | 2.4e-14 |
| algorithm #2 | 1.42e-12 | 3.7e-14 |

(y scale ≈ 8 µg/filter, so this is machine precision.) **The only
preprocessing needed to make them match:** construct
`PLS(center_X=True, center_Y=True, scale_X=False, scale_Y=False)`.
ikpls *defaults to scaling on*, which sklearn's `scale=False` does not do —
forget those two kwargs and nothing matches. No manual centering is needed;
both center internally, and `cross_validate` re-centers per fold from training
statistics exactly like the repo's per-fold refits.

**CV-curve equivalence (k = 1..30).** Per-sample fold labels reproduce both
repo protocols exactly:

| protocol | fold labels handed to `ikpls.cross_validate` | aggregation | max \|ΔRMSECV\| vs repo curve |
|---|---|---|---|
| site-grouped 5-fold (`component_cv_curve`, GroupKFold shuffle seed 42) | fold id per sample from the same GroupKFold splits | mean of per-fold RMSEs | 2.27e-13 |
| interleaved 10-fold (`interleaved_cv_curve`) | `np.arange(n) % 10` | pooled PRESS → √(ΣSSE/n) | 1.89e-13 |
| interleaved, via `ikpls.fast_cross_validation` | same | pooled | 2.85e-12 |

Curve floors: 4.99 (site-grouped) / 4.90 (interleaved) µg/filter. Note the two
protocols aggregate differently (mean-of-fold-RMSEs vs pooled PRESS); ikpls's
`metric_function` hook expresses both, so neither convention is forced.

**Wall-clock, full 1..30-component curve (single warm run, this Mac):**

| protocol | engine | seconds |
|---|---|---|
| site-grouped 5-fold | repo `component_cv_curve` | 1.45 |
| site-grouped 5-fold | ikpls `cross_validate` (n_jobs=-1) | 1.65 |
| site-grouped 5-fold | ikpls `cross_validate` (n_jobs=1) | 4.14 |
| site-grouped 5-fold | ikpls `fast_cross_validation` | 1.23 |
| interleaved 10-fold | repo `interleaved_cv_curve` | 5.29 |
| interleaved 10-fold | ikpls `cross_validate` (n_jobs=-1) | 7.47 |
| interleaved 10-fold | ikpls `cross_validate` (n_jobs=1) | 5.10 |
| interleaved 10-fold | ikpls `fast_cross_validation` | **0.44** |

The repo's prefix-truncation trick (one max-component fit per fold) is already
near-optimal, so plain `ikpls.cross_validate` buys nothing at n = 800 — process
spawn overhead even makes it slower. The real win is
`ikpls.fast_cross_validation` (cvmatrix cross-product updates): **12× on the
interleaved curve**, and it too matches the repo curve to ~3e-12.

**Verdict: keep the repo machinery as the default; ikpls is validated as a
numerically interchangeable drop-in, and `ikpls.fast_cross_validation` is the
engine to reach for when CV cost actually bites** — the 13,010-filter
IMPROVE-network cohort curves, protocol sweeps, or permutation tests. A 12×
speedup on the cohort that "dominates the runtime" of ftir_21 is worth having;
rewriting `component_cv_curve`/`interleaved_cv_curve` for a sub-2-second call
is not. The component-selection rules (first-major-minimum, within-5%) stay
ours either way — ikpls only returns curves.

## 2. Deming: scipy.odr wrapper vs the closed form vs methcomp

Synthetic EIV data, true slope 2.0, intercept 0.5, n = 4000, generated with the
error-variance ratio matching each tested λ:

| λ | app closed-form slope | scipy.odr slope | \|Δslope\| | \|Δintercept\| | methcomp slope |
|---|---|---|---|---|---|
| 0.50 | 1.99149297 | 1.99149298 | 9.2e-09 | 5.8e-08 | 1.264718 |
| 1.00 | 1.99078843 | 1.99078844 | 5.4e-09 | 4.1e-08 | 1.436193 |
| 2.96 | 1.98947058 | 1.98947058 | 2.4e-09 | 1.1e-08 | 1.847034 |

scipy.odr matches `calibration_explorer/app.py::deming` to **5.8e-8 worst
case** (target 1e-6) — but only after tightening `sstol=1e-14, partol=1e-14`;
ODR's default stopping tolerance leaves ~1e-5 residual disagreement. Axis-swap
sanity at λ=1: slope(x,y)·slope(y,x) = 1.00000000.

**methcomp 1.0.0's Deming is numerically wrong** (slopes 1.26–1.85 against a
truth of ~1.99). Its `_Deming._derive_params` computes
`spdxy = np.cov(x, y)[1][1] * (n-1)` — `[1][1]` is **var(y)**, not the
cross-product — and puts `4*lamb*(ssdy**2)` under the square root where the
Deming formula requires the squared cross-product. A third bug: its `sdr`
("known standard deviations") parameter is used as λ directly without
squaring. Do not use, do not vendor.

### λ-convention mapping

| implementation | parameter | meaning | to get repo behaviour |
|---|---|---|---|
| `calibration_explorer/app.py::deming(x, y, lam)` | `lam` | **λ = σy²/σx²** (error-variance ratio, y relative to x) | reference |
| `scipy.odr` | `RealData(sx=, sy=)` | per-axis error SDs; only the ratio matters | `sx=1, sy=sqrt(λ)` (generally λ = (sy/sx)²) |
| methcomp `deming(vr=)` | `vr` | *documented* as the same λ = var(y)/var(x) | n/a — implementation broken |

Caution for future cross-checks: R packages disagree on direction — R
`MethComp::Deming(vr=)` documents the same y-over-x ratio, while others (e.g.
`mcr`'s `error.ratio`) parameterize the reciprocal. Always verify on synthetic
data with a known slope, as `validate_deming.py` does, before trusting an
external Deming.

**Verdict: keep the repo's closed form for point estimates (it is exact and
three lines); keep a thin scipy.odr wrapper** (as in `validate_deming.py::deming_odr`)
as the independent numerical cross-check and the route to standard errors
(`output.sd_beta`) / bootstrap CIs if the deck ever needs them. Drop methcomp
from the survey's "vendor/cross-check" list — it is the thing that needs
cross-checking.

## 3. astartes / kennard-stone (brief)

190-sample subsets of the 800 cohort (y range 0.09–335.25 µg/filter):

| selection | y range covered | deciles hit | subset mean y | n sites |
|---|---|---|---|---|
| full cohort (800) | 100% | 10/10 | 8.07 | 126 |
| astartes Kennard-Stone (X) | 99.98% | 10/10 | 19.75 | 67 |
| astartes SPXY (X, y) | 99.96% | 10/10 | 21.49 | 68 |
| kennard-stone pkg (KS) | 99.96% | 10/10 | 20.22 | — |
| random, mean of 20 seeds | **46.1%** | 10/10 | 8.59 | — |

KS/SPXY subsets span essentially the full loading range — random 190-sample
draws cover only ~46% of it on average because they miss the sparse
high-loading tail — but they do so by preferentially selecting extremes: the
subset mean is ~2.5× the cohort mean. Both packages work out of the box on
2722-point spectra and agree with each other.

**Verdict: adopt for calibration-*design* questions** (does a
maximally-spanning training subset beat the lowest-OC/EC filter?, transfer-set
selection), **never for evaluation splits** — a KS test set is
distribution-shifted by construction, and the site-held-out protocol already
answers the honest-evaluation question.

## Summary of verdicts

1. **ikpls** — validated equivalent (≤1.4e-12); *don't* replace the repo CV
   code for its own sake; *do* use `ikpls.fast_cross_validation` for
   big-cohort/many-protocol CV (12× on interleaved). Gotcha: must pass
   `scale_X=False, scale_Y=False`.
2. **scipy.odr Deming wrapper** — keep (with `sstol=1e-14, partol=1e-14`), as
   cross-check + CI provider; closed form stays the point estimator.
3. **methcomp** — reject; Deming implementation has three distinct bugs.
4. **astartes / kennard-stone** — adopt for training-set design experiments
   only.
