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
  - `Fabs` (Mm⁻¹, HIPS-style: x-axis becomes Fabs/MAC and the MAC 10/6/17 toggle
    applies, Deming uses λ\*) **or** `EC_ugm3` (direct EC reference: single-row
    metrics, MAC toggle disabled, Deming at λ=1);
  - `Volume_m3` (required — predictions are µg/filter ÷ volume);
  - optional `Date` (enables the Series tab), `Group` (colours the crossplot,
    e.g. seasons — and drives the Eval season lever) and `LotId` (drives the
    Eval lot lever).

## Evaluation view: lot × season × equal-n half

Three composable levers narrow **what the readout is reported on**, applied in
that order and always *after* the cached fit. Predictions cover every filter and
the curve/fit caches stay view-agnostic, so every view of a fitted configuration
is a cache hit. All three flow through Run, Sweep k, presets, the CSV exports,
the Optimize tab, the server batch, cutoff refinement, and Validate top 5.

**Eval lot** — judge a lot-251 calibration on lot-251 filters (Ann, 2026-08-19).
Available for **any** target whose `reference.csv` carries a `LotId`, not just
Addis: `build_spartan_target.py` now writes the column, and
`scripts/add_target_lots.py` backfills it into targets exported before that by
joining the SPARTAN HIPS table on `ExternalFilterId`. Current coverage — Addis
191/34/14 (lots 251/248/253), Beijing 164/28 (251/248), Delhi 76/61/15
(251/253/248), Pasadena 144/14 (251/248), Bishoftu 26 (251). Beijing is the clean
lot-effect experiment: two lots on one site, same period.

**Eval season** — report on one `Group` only (Ethiopian sites: Bega dry, Belg,
Kiremt; SPARTAN quarter scheme elsewhere). Selection and fit are untouched, so a
calibration can be read out on the season that did not shape it — the wet/dry
split Navid's source work motivates.

**Eval half** — split the evaluation set into two **equal-n** halves and report on
one (Ann, 2026-08-27). `early`/`late` are the date-ordered halves: pick a
configuration on one, read it out on the half that never guided the choice.
`odd`/`even` interleave in date order, so both halves share the same temporal and
seasonal coverage — the control that separates a real time trend from ordinary
sampling scatter. The split is **stratified by the fixed/all evaluation subset**:
an unstratified date halving puts 119 of Addis's fixed-set filters in the early
half against 70 in the late one, and comparing R² across unequal n is exactly what
the split exists to prevent. Stratified, every half reports n=119 (95 on the fixed
set).

Every run returns `split_check`: the same fitted model scored on *all* the halves
of the current lot/season view, which costs one regression each and drives the
**Blind-half check** panel in the Target readout tab (intercept vs slope per half,
with the between-half deltas spelled out underneath). In the server batch, **score
both blind halves** sweeps `early` and `late` in one pass, so "how far does the
winner move between halves" needs no second batch.

`POST /api/eval_view_options {"target": "chts"}` returns the lots, seasons and
half size a given target supports; the page asks per target rather than carrying
one global (Addis-only) list.

The Selection tab's ranking view also
gained a **raw vs corrected** mode overlaying the selection metric in both spectra
spaces, with shared membership at the current cutoff in the caption.

The target appears in the dropdown on the next status poll; presets store the target,
so an "ETBI crossplot" becomes a preset the moment that export exists. Note the
*selection* reference (Ethiopia-shaped/analog matching) intentionally stays Addis —
targets change what you evaluate against, not the research design. A
`targets/demo_addis_subset/` example (60 ETAD filters with deployed EC as the
reference) shows the format.

`target_registry.json` assigns every shipped target a role. The five primary site
targets are screening targets. Reconstructed holdouts and augmented references are
provisional confirmation/readout targets: the app deliberately blocks browser,
batch, and cutoff-refinement optimization against them so a nominal holdout cannot
silently become tuning data.

## Run

Quickest — the launcher script (from the repo root):

```bash
./calibration_explorer/run.sh          # background start → http://127.0.0.1:5058
```
```bash
./calibration_explorer/run.sh status   # is it up, and has the data finished loading?
```
```bash
./calibration_explorer/run.sh stop     # stop it
```
(`run.sh fg` runs in the foreground; log for background runs: `/tmp/calibration_explorer.log`.)

Manual, canonical environment:

```bash
uv sync --extra explorer --python 3.13
```
```bash
uv run --extra explorer python calibration_explorer/app.py
```

The `explorer` extra is required on a clean checkout — the repo's *base* uv venv
deliberately excludes Flask, so plain `uv run python calibration_explorer/app.py`
fails with `No module named 'flask'`. Any other interpreter that carries
flask + pandas + scikit-learn also works (historically `~/anaconda3/bin/python`);
point `EXPLORER_PYTHON` at one to make `run.sh` use it.

After starting: data loads in a background thread (~1–2 min; the 13k-pool spectra
CSV from Google Drive dominates) and the page polls until ready. Optionally pre-warm
every configuration so all clicks resolve from cache in <1 s:

```bash
uv run --extra explorer python calibration_explorer/warm_cache.py
```

## Layout

```
calibration_explorer/
├── app.py            # Flask backend: data loading, cohort resolution, fits, caching
├── hips_lab.py       # York/EIV and deployed-calibration-line HIPS diagnostics
├── target_registry.json # target role, site identity and optimization policy
├── static/
│   ├── index.html    # markup only — controls and all explorer tabs
│   ├── style.css     # theme + responsive breakpoints (≤700px single-column mobile,
│   │                 #   700–1100px stacked panels + two-column config, ≥1100px
│   │                 #   full desktop grid)
│   └── app.js        # all frontend logic (plots via Plotly, pins + presets in
│                     #   localStorage)
└── cache/            # per-configuration CV curves + fits (gitignored)
```
The page is responsive: below ~1100px the side-by-side panels stack so plots and the
fit-metrics table get the full width; on phones the controls stack full-width, the
tabs and view toggles wrap instead of scrolling off-screen, wide tables scroll
horizontally inside their own containers with edge shadows marking hidden columns,
and the configuration card collapses behind a one-line summary of the current setup.

**Exhaustive batch (Optimize tab)**: "Run batch" executes the checked cohort ×
cutoff-ladder × selection-space × spectra × protocol grid **inside the server**
(one background thread; interactive runs interleave), scoring every
(configuration, k) row exactly like the leaderboard and appending it to
`cache/batch_results.jsonl`. It survives closing the page — reopen later and
"Load saved results" merges everything into the leaderboard + Pareto view.
Saved-result loading is server-filtered by target/held-out and checked-grid status.
Normal loads are capped at 10,000 rows; robust mode requests a bounded, filtered
slice per site (up to 50,000 each) so an Addis-heavy file prefix cannot masquerade
as a multi-site ranking. The page reports matched, loaded, and truncated counts.
`POST /api/batch_start` / `batch_stop`, `GET /api/batch_status` /
`batch_results` are the API. `batch_start` also takes `eval_group` and
`eval_splits` (a list, e.g. `["early", "late"]` — the "score both blind halves"
checkbox), and resolves the evaluation view **per target**: a lot or season a
given site does not have falls back to `all` instead of erroring that row out. The Colab notebook
(`colab/Calibration_Explorer_Colab.ipynb`, regenerated by
`colab/build_colab_bundle.py` — edit the template there, not the .ipynb; build, ship, and
verification notes incl. the 2026-09-01 end-to-end run live in `colab/README.md`) has an
optional cell that runs the full grid in Colab against Drive-mounted data and
zips the cache back to Drive; unzip it into `cache/` locally (content-keyed, so
merging is safe) and every batch row loads and every configuration is a cache
hit.

**Refine cutoffs ±10 (Optimize tab)**: hill-climbs each ranked cohort's cutoff
in ±10-filter steps from its default, following score = |intercept| +
w·|slope − 1| (Deming MAC 10, fixed set; held-out floor as a penalty) downhill
until it stops improving. Server-side like the batch; refined rows append to
`cache/batch_results.jsonl` and appear via "Load saved results".

**Analogs tab (the analog lab)**: compares the committed spectral-analog
selection against literature similarity metrics — cosine/SAM, Pearson-to-median
(LOCAL's metric), normalized Euclidean, nearest-neighbour cosine, and
Mahalanobis-to-Addis-centroid in PCA-10 score space (Reggente et al. 2016's
extrapolation diagnostic turned selector) — in raw, AIRSpec-corrected, or SG
2nd-derivative space (the LOCAL-classic representation). One payload per space
carries full per-filter rank arrays, so the cutoff slider recomputes overlap
matrices, rank-agreement scatter, and the PCA score-space map instantly
client-side. Literature grounding:
`research/ftir_ec_phase3/SPECTRAL_SIMILARITY_LITERATURE.md`.

**Sites tab (cross-site evaluation)**: one click evaluates the current
configuration against the five primary site targets — the five-site
table (Addis / Bishoftu / Beijing / Delhi / Pasadena / custom) with an
intercept-by-site ladder, following the MAC · Fit · target-set toggles. Every
fit now also computes two complementary domain diagnostics: the **Reggente et
al. (2016) score-space distance** and the **PLS Q residual**. The stats card,
Sites table, optimizer guardrails, exports, and saved batch rows carry the share
of target filters above the training p95 for each. Score distance catches an
unusual latent-space location; Q catches spectral structure the retained PLS
subspace cannot reconstruct. Rows above ~30% on either gauge are provisional.
Old cached fits upgrade in place on first access. Target spectra for new SPARTAN sites are built with
`research/ftir_ec_phase3/scripts/build_spartan_target.py` from Networks_1_0
exports.

**Cross-site spectra (Sites tab)**: median (+ optional IQR) spectra for every
evaluation target, with a **Baseline** switch — `raw` / `AIRSpec` / `neutral`.
The switch is the point: APRLssb/AIRSpec anchors segment 2 at the minimum over
**1520–1600 cm⁻¹** (`scripts/airspec_baseline.py::find_min_pos`, and that window
is zero-weighted in the spline fit), which sits directly under a ~1617 cm⁻¹ band
and suppresses it. `neutral` runs `pybaselines.spline.pspline_arpls` with no
anchor window and PTFE-saturated regions masked, as an independent opinion.
The same representation is now available under **Calibrate on** and in the
optimizer after running
`research/ftir_ec_phase3/scripts/build_neutral_baseline_cache.py`; its manifest
records λ, mask, source SHA-256 values, row counts, and output coverage.
Under `neutral` the 1617 peak is present at **Addis and Bishoftu only**
(Delhi/Beijing peak at ~1679, the carbonyl edge; Pasadena at 1634) — see
`research/ftir_ec_phase3/BAND1617_LEAD_2026-08-23.md`. Any band claim in
1500–1650 cm⁻¹ should be shown under at least two baselines.
`POST /api/site_spectra {"space": "raw"|"airspec"|"neutral"}`. Note the neutral
path imports **pybaselines** lazily, so it is a soft runtime dependency.

**Auto-run**: the "auto" toggle next to Run re-runs the calibration ~0.6s after any
configuration change (debounced, one run in flight at a time; a change landing
mid-run queues exactly one follow-up). Cached configurations make this feel live;
uncached ones still compute at their usual cost, so leave it off when stepping
through expensive cold configurations. The choice persists in the browser.
Without auto-run, any configuration change visibly marks the displayed result as
stale and disables Sweep/Pin until a matching run succeeds.

**Presets**: the six setup-matrix rows ship as built-in presets; "Save as…" stores the
full current configuration (cohort, cutoff, both spectra spaces, protocol, k mode/value)
plus the MAC/estimator/evaluation-set readout toggles as a named custom preset in
the browser. The ⋯ menu next to it holds the rarer
actions — Delete, and Export/Import as JSON (so presets can be shared between
machines or committed alongside results).

**Tabs**: Calibrate (run summary + CV curve + k sweep), Target readout (crossplot +
metrics incl. the x-intercept **c = −b/m** per estimator + residuals + the
**Blind-half check**), Selection
(cutoff diagnostic with slider + pool-distribution view, reference-data
characteristics, **composition ruler** (ftir_30: cohort vs pool OC/EC with the
FTIR-derived Addis marker 1.34), overlap, spectra), **Series** (ftir_29 generalized:
the run's dated EC record with 45-day rolling median, a plausibility card — median/
IQR, negative days, days >8 µg/m³, group medians — and the vs-deployed-SPARTAN
crossplot), Compare (pinned runs + intercept ladder), **Optimize** (staged search
over the whole configuration space — see below). Plot captions live in HTML
above each figure so Plotly legends never collide with titles.

**Optimize tab**: a staged search for a small target intercept with a slope near 1.
It drives the existing `/api/run` and `/api/sweep` endpoints from the browser, so
every number is the shared-script math, cached configurations are effectively free,
and no separate job runner is needed (it runs while the page is open; sequential,
Stop keeps partial results). Stages: (1) screen every selected cohort ×
selection-space × calibrate-on × protocol × training-lot combination at its
default cutoff and rule k; (2) re-evaluate the leading ranked cohorts at
×0.5/×1.5/×2 their cutoff; (3) scan every integer k in a configurable range
(default 1–30) on the leaders. Lot-specific Addis rows can automatically pair
train lot 251 with evaluation lot 251. The objective is
`score = |intercept| + w·|slope − 1|` read at the current MAC · Fit toggles on the
fixed evaluation set, with hard guardrails for **held-out TOR R², slope range,
score-space extrapolation, Q residual, and negative predictions**. A robust leaderboard ranks
each configuration by its worst loaded site; the leaderboard and Pareto view re-rank live
when the toggles or weights change, and any run can be loaded straight back into
the explorer. Caveat printed in the tab: screening many configurations against the
same fixed target pairs overfits the readout — winners are candidates to confirm on
the locked protocol, not conclusions.

**Validate top 5** freezes the leading passing candidates and k values, then
reports selection frequency under stratified target-filter resampling and under
site-cluster resampling of the IMPROVE training pool with a full PLS refit on
every draw. The former is descriptive because the screening target has already
guided the search; the latter measures sensitivity to training-site composition.
Neither is labelled external validation. The frozen result and untouched
IMPROVE-lot protocol are documented in
`research/ftir_ec_phase3/VALIDATION_LAYER_2026-08-24.md`.

The full discussion-to-control audit, including why MAC 17 is a sensitivity
readout rather than an optimization dimension and what still requires nested
validation, is in `CONVERSATION_OPTIMIZATION_AUDIT_2026-08-24.md`.

**HIPS tab**: reports per-filter York/EIV fits using HIPS uncertainty models and
recomputes Fabs under deployed, linear-refit, and quadratic-refit blank lines. Blank
refits are grouped by `(LotId, deployed Intercept, deployed Slope)`, not lot alone:
the current batch has four lots but nine deployed calibration lines. Stability
across these forms rules out those forms only; the UI does not claim that stability
proves an aerosol cause.

## Roadmap

From the notebook survey, in priority order: **weighted + robust Deming** estimators
(ftir_31 §1.10 — ~30 lines to lift from `run_ftir_31.py` into `pls_transfer`; would
show OLS is the conservative bound on every crossplot), **fold-count toggle** (ftir_32
— the curve functions already take `folds`/`n_splits`; closes the last hardcoded
protocol dimension), **%RMSECV axis toggle** + interleaved fold-spread ribbon,
**held-out TOR scatter** per run and a fold-count toggle. The Q-residual domain
gauge, neutral training cache, and on-demand site-cluster finalist stability are
now implemented. The highest-value
new science view is a paired HIPS−MA350 versus FTIR−MA350 residual-attribution panel;
see `research/ftir_ec_phase3/DATA_OPPORTUNITY_AUDIT_2026-08-24.md`.

## What it sweeps (mapped to the July-17 meeting items)

| Control | Meeting item |
|---|---|
| Cohort + **cutoff N**, with the metric-vs-rank plot to eyeball jumps (click it to move the cutoff) | *Selection cutoffs: "try somewhat more and somewhat less… look for where the distance metric jumps"* |
| **Select on** raw vs AIRSpec-corrected spectra — re-runs the Ethiopia-shaped band-feature selection on corrected spectra (exact committed metric, corrected space), while **Calibrate on** stays independently raw/corrected (Satoshi: select corrected, calibrate raw) | *Baseline correction before spectral matching* |
| **Calibrate on** raw vs AIRSpec-corrected (df1=6 caches), for *any* cohort | *Baselined vs non-baselined calibrations* |
| Protocol: **Site-held-out** (site-grouped 5-fold CV, first-major-minimum, disjoint fit, seed 20260717) vs **Calibration app** (interleaved 10-fold, within-5%, fit on all) | *Cross-validation: keep both running* |
| k: rule choice, **click the CV curve**, or **Sweep k** (every integer from the rule choice through 30); optimizer finalist and batch ranges explicitly include 21 | *Number of components: scan rather than trusting the 5% rule* |
| Spectra panel: cohort median + IQR vs Addis, in the chosen spectra space — plus an **"all selection cohorts"** mode overlaying Ethiopia-shaped, spectral analogs and lowest-OC/EC against the Addis median | *Spectra comparison plots: "plot the 800 lowest OC/EC spectra, the Ethiopia-shaped selection, and the spectral analog selection side by side against the Addis spectra"* |
| **Selection overlap table** — pairwise shared filters between smoke, Ethiopia-shaped (raw *and* corrected space), analogs and lowest-OC/EC at the current cutoffs | *"Are these methods actually picking similar filters?"* + *"see whether the pool of selected samples actually changes"* (raw vs corrected selection) |
| **Residuals panel** under the crossplot (predicted − HIPS EC-equivalent vs HIPS, by season) | the meeting's residual check ("it's definitely removing the curve") |
| Calibrate on **SG 2nd derivative** (ftir_20's parameters: window 11, polyorder 2) as a third spectra option | the second-derivative comparison discussed for the erratic RMSE curves |
| Crossplot: MAC 10/6/17 sensitivity, OLS and Deming (λ\*=2.96 at MAC 10 from the HIPS_Uncertainty rows, scaled by MAC²), fixed-190 vs all-pairs, points coloured by Ethiopian season | the readout itself (+ Ann's dry/wet colouring); MAC is not auto-selected |
| Pin runs → comparison table + intercept ladder + CSV and replayable manifest export | the "ten different options" iteration |

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
- CV curves and fits are cached in `./cache/` (gitignored) using a cache-schema,
  source-data/code fingerprint, resolved-cohort hash, and target-content fingerprint.
  Run responses include those fingerprints, the git commit + dirty-worktree flag,
  and a stable run id.
  Pinned runs can be loaded/re-run and exported as replayable JSON manifests.

## Known limits

- **Corrected spectral-analog selection is available but expensive on a cold cache**:
  it reruns the ftir_09 score-space/VIP ranking on the AIRSpec caches and stores the
  result in `cache/analog_corrected_ranking.npz`. See
  `ANALOG_CUTOFF_AUDIT_2026-08-18.md` for the eligibility-before-cutoff audit.
- Interleaved CV is row-order dependent (ftir_22), so an `app`-protocol k here can
  differ by a component or two from a committed run if the cohort row order differs.
  Site-held-out is order-stable.
- Needs the Google Drive data mounted (same paths as the phase-3 notebooks) and the
  AIRSpec cache npz files under `research/ftir_ec_phase3/output/corrected/`.
- Dev server, localhost only; Plotly loads from CDN.
