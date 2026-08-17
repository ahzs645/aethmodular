# Calibration Iteration Explorer

A standalone local Flask app (repo root) for iterating the phase-3 FTIR-EC calibration
space interactively. Independent of `research/spartan_ec_2026_06_16/recreation_app`
(which recreates the AQRC Shiny tool); this app drives the **phase-3** machinery — the
locked cohorts, both component-selection protocols, and the AIRSpec caches — and reads
everything out against a configurable **evaluation target** (Addis/ETAD built in).

## Evaluation targets (generic readout)

The readout is no longer Addis-specific. "Evaluate on" picks a target; Addis (ETAD)
ships built-in with its Fabs reference, seasons, fixed-190 cohort, dates, and deployed
EC. To evaluate against **your own site**, drop a folder into
`calibration_explorer/targets/<name>/` (gitignored) containing:

- `spectra.csv` — first column an id, then the IMPROVE wavenumber columns (the same
  2722-column grid as the `local_db` spectra export; the app checks and tells you
  what's missing).
- `reference.csv` — first column the same id, plus:
  - `Fabs` (Mm⁻¹, HIPS-style: x-axis becomes Fabs/MAC and the MAC 10/6 toggle
    applies, Deming uses λ\*) **or** `EC_ugm3` (direct EC reference: single-row
    metrics, MAC toggle disabled, Deming at λ=1);
  - `Volume_m3` (required — predictions are µg/filter ÷ volume);
  - optional `Date` (enables the Series tab) and `Group` (colours the crossplot,
    e.g. seasons).

The target appears in the dropdown on the next status poll; presets store the target,
so an "ETBI crossplot" becomes a preset the moment that export exists. Note the
*selection* reference (Ethiopia-shaped/analog matching) intentionally stays Addis —
targets change what you evaluate against, not the research design. A
`targets/demo_addis_subset/` example (60 ETAD filters with deployed EC as the
reference) shows the format.

## Run
```
python calibration_explorer/app.py
# → http://127.0.0.1:5058
```
Data loads in the background on startup (~1–2 min; the 13k-pool spectra CSV dominates).
The page polls until ready.

## Layout

```
calibration_explorer/
├── app.py            # Flask backend: data loading, cohort resolution, fits, caching
├── static/
│   ├── index.html    # markup only — preset bar, config bar, toggles, four tabs:
│   │                 #   Calibrate · Addis readout · Selection · Compare
│   ├── style.css     # theme + responsive breakpoints (≤700px single-column mobile,
│   │                 #   700–1100px two-column, ≥1100px full desktop grid)
│   └── app.js        # all frontend logic (plots via Plotly, pins + presets in
│                     #   localStorage)
└── cache/            # per-configuration CV curves + fits (gitignored)
```
The page is responsive: on phones the controls stack full-width and every panel is a
single column; tables scroll horizontally inside their own containers.

**Presets**: the six setup-matrix rows ship as built-in presets; "Save as…" stores the
full current configuration (cohort, cutoff, both spectra spaces, protocol, k mode/value)
as a named custom preset in the browser. Export downloads custom presets as JSON;
Import merges a JSON file back in (so presets can be shared between machines or
committed alongside results).

**Tabs**: Calibrate (run summary + CV curve + k sweep), Addis readout (crossplot +
metrics incl. the x-intercept **c = −b/m** per estimator + residuals), Selection
(cutoff diagnostic with slider + pool-distribution view, reference-data
characteristics, **composition ruler** (ftir_30: cohort vs pool OC/EC with the
FTIR-derived Addis marker 1.34), overlap, spectra), **Series** (ftir_29 generalized:
the run's dated EC record with 45-day rolling median, a plausibility card — median/
IQR, negative days, days >8 µg/m³, group medians — and the vs-deployed-SPARTAN
crossplot), Compare (pinned runs + intercept ladder). Plot captions live in HTML
above each figure so Plotly legends never collide with titles.

## Roadmap (mined from ftir_29–33; not yet implemented)

From the notebook survey, in priority order: **weighted + robust Deming** estimators
(ftir_31 §1.10 — ~30 lines to lift from `run_ftir_31.py` into `pls_transfer`; would
show OLS is the conservative bound on every crossplot), **fold-count toggle** (ftir_32
— the curve functions already take `folds`/`n_splits`; closes the last hardcoded
protocol dimension), **%RMSECV axis toggle** + interleaved fold-spread ribbon,
**residual-vs-D² extrapolation gauge** (% of target beyond training p95 D²),
**held-out TOR scatter** per run, and **site-cluster bootstrap CI** on the intercept
ladder (ftir_22 machinery, ~200 refits per run, on-demand button).

## What it sweeps (mapped to the July-17 meeting items)

| Control | Meeting item |
|---|---|
| Cohort + **cutoff N**, with the metric-vs-rank plot to eyeball jumps (click it to move the cutoff) | *Selection cutoffs: "try somewhat more and somewhat less… look for where the distance metric jumps"* |
| **Select on** raw vs AIRSpec-corrected spectra — re-runs the Ethiopia-shaped band-feature selection on corrected spectra (exact committed metric, corrected space), while **Calibrate on** stays independently raw/corrected (Satoshi: select corrected, calibrate raw) | *Baseline correction before spectral matching* |
| **Calibrate on** raw vs AIRSpec-corrected (df1=6 caches), for *any* cohort | *Baselined vs non-baselined calibrations* |
| Protocol: **Site-held-out** (site-grouped 5-fold CV, first-major-minimum, disjoint fit, seed 20260717) vs **Calibration app** (interleaved 10-fold, within-5%, fit on all) | *Cross-validation: keep both running* |
| k: rule choice, **click the CV curve**, or **Sweep k** (~8 points from the rule choice to ~double, capped at 20) plotting intercept + held-out R² vs k | *Number of components: scan rather than trusting the 5% rule* |
| Spectra panel: cohort median + IQR vs Addis, in the chosen spectra space — plus an **"all selection cohorts"** mode overlaying Ethiopia-shaped, spectral analogs and lowest-OC/EC against the Addis median | *Spectra comparison plots: "plot the 800 lowest OC/EC spectra, the Ethiopia-shaped selection, and the spectral analog selection side by side against the Addis spectra"* |
| **Selection overlap table** — pairwise shared filters between smoke, Ethiopia-shaped (raw *and* corrected space), analogs and lowest-OC/EC at the current cutoffs | *"Are these methods actually picking similar filters?"* + *"see whether the pool of selected samples actually changes"* (raw vs corrected selection) |
| **Residuals panel** under the crossplot (predicted − HIPS EC-equivalent vs HIPS, by season) | the meeting's residual check ("it's definitely removing the curve") |
| Calibrate on **SG 2nd derivative** (ftir_20's parameters: window 11, polyorder 2) as a third spectra option | the second-derivative comparison discussed for the erratic RMSE curves |
| Crossplot: MAC 10/6, OLS and Deming (λ\*=2.96 at MAC 10 from the HIPS_Uncertainty rows, scaled by MAC²), fixed-190 vs all-pairs, points coloured by Ethiopian season | the readout itself (+ Ann's dry/wet colouring) |
| Pin runs → comparison table + intercept ladder + CSV export | the "ten different options" iteration |

## Provenance

- Cohort rankings come from the committed tables: `Addis_band_feature_distance`
  (Ethiopia-shaped), `analog_rank_score` (spectral analogs), and the ftir_11
  eligibility + OC/EC ordering (lowest-OC/EC). At the locked cutoffs (300/500/800) the
  resolved sets are checked against the committed cohorts at startup; ✓/✗ shown in the
  toolbar.
- **All math is the shared script code the notebooks run — nothing is re-implemented
  here.** The two protocols (train mask, CV curve, k rule) come from
  `research/ftir_ec_phase3/scripts/calibration_modes.py`
  (`protocol_train_mask` / `protocol_cv_curve` / `protocol_select_k`, which
  `fit_calibration` itself is built from — refactor verified bit-identical to the
  pre-refactor module, and `fit_calibration` now takes an optional `k_override` for
  notebook k sweeps). The estimators and the selection metric come from
  `research/ftir_hips_chem/scripts/pls_transfer.py`: `regression_metrics` (OLS),
  `deming_regression` (λ = σy²/σx²; λ from `calibration_modes.deming_lambda(mac)`,
  λ\*=2.96 at MAC 10), and `band_feature_distance` (the exact committed
  Ethiopia-shaped metric — the corrected-space selection is the same function on the
  AIRSpec caches).
- The site-held-out path is the ftir_10/11 locked protocol (same seed, folds, rule);
  after the refactor the app re-reproduces the locked numbers from a cold cache
  (k=6 / 1.585x−3.221 / held-out R² 0.9107; AIRSpec k=5 / −1.615 / Deming −2.089).
- CV curves and fits are cached in `./cache/` (gitignored) keyed by the exact resolved
  configuration; the entire-network cohort's first interleaved curve takes several
  minutes, then is instant.

## Known limits

- **Spectral-analog selection is raw-space only**: its rank comes from the ftir_09 PLS
  score-space machinery (D², VIP weights) and must be re-run on corrected spectra in a
  notebook before a corrected-space analog cohort can exist here. The app says so if
  you ask for it.
- Interleaved CV is row-order dependent (ftir_22), so an `app`-protocol k here can
  differ by a component or two from a committed run if the cohort row order differs.
  Site-held-out is order-stable.
- Needs the Google Drive data mounted (same paths as the phase-3 notebooks) and the
  AIRSpec cache npz files under `research/ftir_ec_phase3/output/corrected/`.
- Dev server, localhost only; Plotly loads from CDN.
