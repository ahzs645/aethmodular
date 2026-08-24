# Selection validation, domain checks, and external-lot protocol

## Decision

Use **IMPROVE lot 253** as the first untouched spectral-lot challenge and keep
**lot 255 sealed** for replication. Lot 253 contributes 5,050 catalog rows with
QC code 1 and matched TOR EC, across 83 sites, sampled 2023-04-12 through
2024-04-15. Its spectra are absent from `spectra_248_251.csv`, the only raw
training export used by the explorer. All 83 sites occur in the older export,
so this challenge tests **substrate/lot and time transfer**, not new-site transfer.

The exact pull manifests already exist:

- `output/tables/improve_pull/ftir_list_253.csv` — all 7,631 lot-253 scans;
- `output/tables/improve_pull/ftir_list_253_255_259_264.csv` — 26,965 rows,
  including the 5,050 lot-253 and 6,047 lot-255 QC/TOR-eligible rows.

Spectral scans have not yet been pulled. Do not inspect lot-253 prediction
performance until the protocol below and the primary model are frozen.

## What is already nested and what is not

`ftir_36_domain_invariant_pls.ipynb` is the current leakage-safe nested layer for
the three frozen finalists. Component count is selected inside the 80% IMPROVE
source-training partition with site-grouped CV; the domain-penalty multiplier is
selected with held-out IMPROVE sites acting as pseudo-targets; an untouched 20%
of source sites is used only for outer evaluation. Addis/Delhi response values
are not read until the protocol is frozen.

That procedure validates the **selection algorithm on source sites**. It does
not turn the repeatedly inspected Addis or Delhi screening readout into an
independent estimate. This distinction follows Varma and Simon (2006), who show
that tuning and reporting on the same cross-validation criterion is biased and
that all tuning steps must be repeated inside the inner loop, and Cawley and
Talbot (2010), who show that the finite-sample variance of a selection criterion
can itself be overfit.

- Varma & Simon: https://doi.org/10.1186/1471-2105-7-91
- Cawley & Talbot: https://www.jmlr.org/papers/v11/cawley10a.html

## Frozen-finalist stability result

The explorer now implements `POST /api/stability` and a **Validate top 5**
control. It freezes candidate definitions and k, then reports two complementary
winner frequencies:

1. stratified resampling of target filters within target group/season;
2. site-cluster resampling of the IMPROVE training sites, with a full PLS refit
   at the frozen k on every draw.

The 200-draw audit (seed 20260717, fixed Addis set, Deming MAC 10,
`|intercept| + 5|slope-1|`) compared the same three finalist definitions used by
`ftir_36`:

| frozen candidate | target-filter wins | source-site wins | source-site slope, median [95%] | source-site intercept, median [95%] |
|---|---:|---:|---:|---:|
| OCEC-440 × AIRSpec, k=5 | **62.5%** | **63.0%** | 1.03 [0.65, 1.50] | -2.23 [-2.90, -0.99] |
| analog-440 corrected selection × deriv2, k=15 | 37.5% | 31.5% | 0.74 [0.48, 0.95] | -1.49 [-2.24, -0.70] |
| analog-530 corrected selection × deriv2, k=12 | 0.0% | 5.5% | 0.51 [0.41, 0.64] | -0.58 [-1.04, -0.21] |

Interpretation: OCEC-440 is the most stable of the frozen finalists, but a 63%
source-site win rate is not decisive. The wide slope interval also says that
the apparent Addis optimum remains sensitive to which IMPROVE sites define the
training reference. This is a stability result, not an unbiased performance
estimate.

## PLS applicability domain

Every explorer fit now reports two p95 diagnostics:

- **score-space %** — whitened distance inside the retained PLS latent space;
- **Q %** — orthogonal spectral residual, i.e. failure of the retained PLS
  subspace to reconstruct the target spectrum.

Both are exposed in the run card, cross-site table, optimizer guardrails,
exports, and saved batch rows. Score distance follows Reggente, Dillner, and
Takahama (2016), who used squared Mahalanobis distance in PLS score space as a
crude predictor of transfer error for IMPROVE FTIR/TOR models:
https://doi.org/10.5194/amt-9-441-2016. Q is complementary: a target can sit near
the score cloud while carrying strong structure orthogonal to the fitted model.

## Neutral-baseline cache

`scripts/build_neutral_baseline_cache.py` now deterministically processes all
13,634 IMPROVE pool spectra, all 319 ETAD spectra, and every registered target
with `pybaselines.Baseline.pspline_arpls(lam=1e6)`. It masks below 700 cm-1 and
1100–1300 cm-1 and carries no AIRSpec band anchor. The generated cache contains
2,410 wavenumbers and a manifest with source SHA-256 values, exact mask, λ,
row counts, and target outputs. The local full build completed in 64.9 s with
zero skipped pool or ETAD rows.

The explorer now offers **neutral pspline-arPLS** under *Calibrate on* and as an
optional optimizer dimension. It is not checked by default because adding it to
every exhaustive run increases the grid by one third.

## Locked lot-253 analysis protocol

Primary model, frozen before scan access:

- cohort: lowest OC/EC, top 440;
- selection space: raw (irrelevant to this chemistry-ranked cohort);
- calibration spectra: AIRSpec df1=6;
- protocol: site-held-out;
- components: k=5, selected by the source-only rule in `ftir_36`;
- training lots: 248/251 only.

Primary endpoint on all lot-253 filters with finite scans, QC code 1, positive
TOR EC, and no duplicate AnalysisId/FilterId:

- direct predicted EC loading versus TOR EC loading;
- slope, intercept, R2, one-to-one RMSE, and mean bias;
- 95% site-cluster bootstrap intervals;
- score-space and Q p95 exceedance shares overall and by site;
- comparison of site-level bias against the frozen 248/251 outer-site results.

Predeclared failure flags:

- either applicability-domain exceedance share >30%;
- slope outside 0.7–1.3;
- materially worse one-to-one RMSE than the frozen outer 248/251 source test;
- site-level bias that changes systematically with loading or lot-253 scan date.

No cohort, baseline, k, cutoff, or exclusion may be changed after inspecting
lot-253 outcomes. If the primary model fails, alternative frozen finalists may
be reported as sensitivity analyses, explicitly labelled post-primary. Lot 255
must remain unopened until that report is complete.

