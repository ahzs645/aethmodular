# Phase 3 results summary (July 2026)

Follow-ups from the July 2026 meeting with Ann and Satoshi. All Addis metrics below are on the
same fixed 190-filter cohort as `ftir_10`, HIPS EC-equivalent (Fabs/MAC) on the x-axis,
MAC = 10 headline. Nothing here uses Addis data for cohort selection or fitting unless
explicitly stated.

## ftir_11 — Ann's lowest-OC/EC cohort: best locked result so far

Cohort: the 800 lowest TOR-OC/EC IMPROVE filters (OC/EC ≤ 2.27; pool median 5.54), locked
disjoint-site TOR test before fitting, first-major-minimum k = 6.

| Model (MAC = 10, n = 190) | slope | intercept | R² | RMSE |
|---|---:|---:|---:|---:|
| Deployed SPARTAN FTIR EC | 1.90 | −4.17 | 0.764 | 1.49 |
| Smoke IMPROVE (906) | 2.65 | −6.91 | 0.685 | 2.85 |
| Ethiopia-shaped smoke (300, ftir_10) | 1.75 | −3.69 | 0.742 | 1.36 |
| **Lowest-OC/EC (800, k=6)** | **1.59** | **−3.22** | **0.774** | **1.16** |

- Held-out TOR (disjoint sites): R² **0.911** vs **0.594–0.775** for the **five** size-matched
  random cohorts at the same k (`N_RANDOM_CONTROLS = 5`, `run_ftir_11.py:55`).
  **Do not quote held-out RMSE as the discriminator**: it is 3.41 vs 3.10–5.00, and
  `random 800 #3` reaches **3.097**, beating the selected cohort. Each cohort is also scored on
  its own disjoint-site split (n_test 114–194), so RMSE is not like-for-like across rows.
  The like-for-like comparison is the Addis crossplot, where all six models are scored on the
  same 190 filters: RMSE **1.16** vs **1.48–2.24**. That, plus the held-out R², is what makes
  the advantage compositional rather than statistical.
- **VIP convergence:** the low-OC/EC model's VIP profile is reported to correlate
  **ρ = 0.74** with the Addis-only HIPS model from `ftir_08`, against **ρ = 0.12** for the
  906-smoke EC model — two independent routes (Addis optical training vs IMPROVE composition
  selection) pointing at the same spectral features. **Provenance caveat:** this number is not
  produced by any executed notebook here. `ftir_11` computes no VIP correlation; per
  `README.md`, VIP-overlap diagnostics came from the independent replication, which "left no
  scripts" and whose tables are not committed. Treat as uncorroborated until re-derived.
- Caveats: intercept still ≈ −3 µg m⁻³; slope MAC-dependent; most random nulls reach
  less-negative intercepts — but only with degraded slopes and worse held-out TOR — so
  intercept alone is not an acceptance criterion.

## ftir_12 — the ~1600 cm⁻¹ band: not amine; carboxylate and/or aromatic C=C

- Addis peak center: median **1618–1619 cm⁻¹** (IQR 1616–1620), and unchanged on
  AIRSpec-corrected spectra (ftir_13). Every IMPROVE cohort sits higher — medians
  **≥ 1633 cm⁻¹** under the canonical edge-corrected window (ftir_12), ~1650 under the
  replication's convention — with essentially no distributional overlap.
- Within Addis the band co-varies with CH (ρ = 0.88) and carbonyl (ρ = 0.93) but **not** with
  the 3100–3400 cm⁻¹ N–H/O–H window (ρ = 0.17; IMPROVE groups: 0.62–0.67). An N–H bend
  without its stretch is implausible → **amine assignment rejected**. On corrected spectra,
  the 1520–1560 cm⁻¹ feature is anticorrelated with the 1600-band raw height (r = −0.62) and
  remains weakly anticorrelated after CH normalization (Spearman −0.29), so it does not
  support a companion-band assignment.
- Remaining candidates — carboxylate COO⁻ asym and aromatic C=C ring stretch (both
  charcoal-consistent) — cannot be separated on Teflon spectra: the ~1400 cm⁻¹ symmetric
  partner sits below the trusted range (AIRSpec segment 2 ends at 1425 cm⁻¹). Needs the
  sub-1500 model Satoshi's group is developing, or lab charcoal spectra.

## AIRSpec port + ftir_13 — EDF 6–8 on real baselines

R is gone from this machine, but the exact APRLssb sources (`reference/APRLssb/`, verified
byte-identical to the Drive backup) and the R run's ETAD output survive, so the Python port in
`scripts/airspec_baseline.py` is validated against real R ground truth (DF1 = 6, DF2 = 4)
before being applied to the 13,634-spectrum IMPROVE pool at DF1 = 6 and 8.

Port validation: all 319 R-corrected ETAD scans reproduced to ≤6×10⁻⁷ absorbance (worst
relative error 5.6×10⁻⁴ of signal RMS) — this is a validated stand-in for AIRSpec, not an
approximation. All 13,634 pool spectra + Addis corrected at DF1 = 6 and 8 (DF2 = 4).

Results on the fixed 190-filter Addis cohort (MAC = 10):

| Model | slope | intercept | R² | RMSE |
|---|---:|---:|---:|---:|
| Lowest-OC/EC raw (ftir_11) | 1.59 | −3.22 | 0.774 | 1.16 |
| **Lowest-OC/EC corrected EDF6** | **0.86** | **−1.62** | 0.657 | 2.41 |
| Smoke 906 raw (ftir_10) | 2.65 | −6.91 | 0.685 | 2.85 |
| Smoke 906 corrected EDF6 | 0.37 | −0.67 | 0.458 | 3.83 |

- **Correction halves the low-OC/EC intercept** (−3.22 → −1.62) and yields a held-out TOR
  slope of 1.01 (R² 0.90) — the best locked intercept-with-defensible-slope so far, at the
  cost of Addis precision (RMSE 1.16 → 2.41).
- **The smoke calibration collapses on corrected spectra** (slope 0.37): its raw-spectrum
  response was substantially carried by the broad baseline — the component the correction
  removes and the component EC's sloping absorption lives in.
- **EDF 6 vs 8 is indistinguishable** (Δslope < 0.01): the EDF choice inside Satoshi's range
  is not a sensitive parameter.
- **The MAC = 6 vs 10 fork is now the deciding unknown**: raw low-OC/EC is self-consistent at
  MAC = 6 (slope 0.95), while corrected low-OC/EC **lands closest** at MAC = 10 (slope 0.86).
  Say "lands closest", not "self-consistent", for that second one — `ftir_19`'s dumbbell
  figure bolds only |slope − 1| ≤ 0.1, a bar 0.86 does not clear. An independent Addis TOR/EC
  reference or a HIPS MAC/protocol bridge resolves it; further cohort engineering will not.

## ftir_15 — uncertainty, residual structure, and the end of cohort engineering

- **Bootstrap (B = 200, site-cluster, fixed Addis cohort, MAC = 10):** corrected OCEC-800
  intercept 95% CI **[−1.78, −1.06]** — excludes zero and is fully disjoint from the raw
  model's **[−4.88, −2.81]**. The AIRSpec improvement is statistically solid; so is the
  remaining offset.
- **Residual structure:** the raw model's Addis residuals track score-space extrapolation
  (D² r = 0.71) and flip sign by season (+0.49 Kiremt → −1.25 Dry); the corrected model's are
  D²-independent and sit at a **season-stable −2.0 to −2.6 µg/m³** with |residual| growing
  with loading. A constant offset of that shape points at a missing constant absorption
  component or a HIPS MAC/protocol mismatch — an external anchor question, not a cohort one.
- **Hybrid cohort (low-OC/EC ∩ spectral similarity, corrected spectra): negative.** Held-out
  TOR collapses (R² 0.19, slope 0.20). Combined with the bootstrap, this closes out cohort
  engineering: OCEC-800 + AIRSpec is the terminal candidate from IMPROVE-only data.

## Post-meeting context: Adama TOR and the ETBI (Bishoftu) site

Two additions from newly synced data (`output/tables/context/`, plot
`output/plots/context/adama_etbi_context.png`):

- **Adama TOR (Batch 54, 5 quartz filters, July 2024)**: OC/EC = **4.6–7.2** (TR basis,
  median ≈ 6.1) — squarely at the IMPROVE pool median (5.5), *not* in the low-OC/EC tail the
  Addis offset implies for the region. This sharpens an uncomfortable question: Addis's
  "extreme low OC/EC" ranking is computed from the SPARTAN FTIR/HIPS measurements under
  suspicion, while the one same-corridor TOR measurement we have looks ordinary. Either the
  Addis urban mix genuinely differs from the nearby corridor, or part of the OC/EC-extreme
  signal is the measurement artifact itself. (Quartz filters — no FTIR/HIPS on the same
  filters, so context only.)
- **ETBI = Bishoftu, Ethiopia** (8.76°N, 39.00°E — between Addis and Adama): a second
  Ethiopian SPARTAN site nobody has looked at in this project. 32 filters (Oct–Dec 2025),
  26 with HIPS Fabs, median **26.9 Mm⁻¹** (EC-equivalent ≈ 2.7 µg/m³ at MAC = 10) — lower
  than Addis but far above IMPROVE. **Quote the seasonal contrast, not the annual one**:
  ETBI's window is Oct–Dec, i.e. Dry only, and Addis's Dry-season median Fabs is **43.2**
  (`ftir_17`), not the 47.1 all-season figure. The like-for-like statement is 26.9 vs 43.2.
  If ETBI FTIR spectra are pulled alongside INDH/CHTS, it is an in-country, dry-season test
  set for every Addis conclusion.
  Caveat: these counts come from `run_context_addenda.py`, whose output table is not
  committed and whose source lives on Drive, so they cannot be re-derived from the repo alone.

## ftir_14 — Delhi/Beijing score-space comparison: blocked on data

The local DB pull covers IMPROVE only (169,566 analyses, no SPARTAN sites); ETAD spectra came
from a separate site-specific pull. Doing the same for INDH/CHTS (adapt
`etad_lots_query.sql` / `pull_scans.ps1`) unblocks a one-notebook analysis using the
already-fitted models. Second-order per the meeting: Delhi/Beijing are test sets, not
calibration inputs.

## ftir_16 — MAC decision prep: ChemSpec debunk, Adama bridge, campaign spec

- **Neither ChemSpec column is an independent EC reference — they are circular in opposite
  directions.** `ChemSpec_EC` for ETAD closely tracks, but is not identical to, HIPS Fabs / 10
  (median ChemSpec/Fabs ratio 0.101, r = 0.89, n = 175 base-joined filters) — which is *not*
  why it fails. It fails because it reproduces `EC_ftir` at **r² = 0.999693** (ratio median
  1.0000, median |Δ| 0.0030 µg/m³ — the 2-dp rounding half-width): it **is** the FTIR-EC
  product routed through SPARTAN's speciation table, i.e. **y-circular** (`ftir_25`). Its
  companion `ChemSpec_BC` is Fabs / 10 rounded (R² 0.9982 against Fabs / 10, implied MAC
  median 10.0003, 86.7% of filters within 0.005) — **x-circular**. So ftir_16's conclusion
  holds for a stronger reason than it gave: neither column can arbitrate the MAC = 6 vs 10
  fork, and not-x-circular does not imply independent.
- **Adama-composition bridge**: if Addis aerosol had Adama's TOR OC/EC (median 6.1), the MAC
  reconciling Addis HIPS Fabs with FTIR OC would be **≈47 m²/g (IQR 36–56)** — unphysical for
  EC (~4–13). So at least one of: (a) Addis OC/EC really is ~5–8× below Adama's (extreme EC),
  (b) a large share of Addis Fabs is **non-EC absorption** (BrC/dust/artifact — the natural
  reading of ftir_15's season-stable −2 to −2.6 µg/m³ offset), or (c) FTIR OC is badly low at
  Addis.
- **Decision instrument**: a co-located quartz TOR campaign at ETAD (or ETBI) — each day
  separates MAC = 6 from MAC = 10 by ~3σ; **11–13 days per season (~36 total)** gives 5σ per
  season at half signal. TOR needs quartz; archived Teflon cannot substitute. Adama Batch-54
  proves the sampling/analysis chain works.
- **IMPROVE implied-MAC bridge (run 2026-07-19 after the FTIR folder was found relocated to
  `University/Research/Grad/Data/FTIR`)**: across 151,843 matched IMPROVE filters, implied
  MAC = Fabs/TOR-EC has median 11.96 (IQR 9.0–15.7); in the **Addis-like OC/EC ≤ 2.27 subset
  (n = 6,503) the median is 10.05** (IQR 6.9–13.1). MAC = 10 is centered at Addis-like
  composition; MAC = 6 sits in the lower tail. The high-OC/EC tail's inflated implied MAC
  (median 19.8) independently shows Fabs carries organic-scaling non-EC absorption.
  **Net phase-3 read: MAC ≈ 10 + AIRSpec-corrected low-OC/EC model + a genuine non-EC
  absorption component at Addis**, with the quartz campaign as the direct confirmation.

## ftir_17 — deck cross plots, side-by-side spectra, and Naveed's seasons

The three figure-level meeting to-dos, drawn (plots in `output/plots/ftir17/`, deck copies
in `output/plots/deck/`):

- **All-data cross plot, new orientation (HIPS on x)**: deployed SPARTAN FTIR EC on the
  fixed cohort (deployed predictions exist only for those 190 filters) reads
  **y = 1.90x − 4.17** at MAC = 10 and **y = 1.14x − 4.17** at MAC = 6.
- **A protocol-matched no-cleaning calibration** (all 13,010 eligible lot-248/251 filters,
  158 sites, locked split, fixed k) is the full-pool version of the random-cohort trap:
  intercepts −1.33 (raw) / −0.61 (corrected) but Addis slopes **0.66 / 0.40** and the worst
  held-out TOR tests of phase 3 (R² 0.53 / 0.63). OCEC-800 is now defended from both
  directions — better than smoke-906 *and* better than no selection.
- **Side-by-side full-range spectra**: Addis's signature is a **deficit, not an exotic
  peak** — per unit CH it carries roughly half the broad O–H/N–H (3000–3600 cm⁻¹) and
  carbonyl absorption of any IMPROVE cohort and sits below the 13.6k pool's IQR, the
  spectra-level face of the low-OC/EC ranking.
- **Seasonal split (Dry Oct–Feb / Belg Mar–May / Kiremt Jun–Sep)**: season modulates
  **loading, not shape** — CH, carbonyl, 1600-band, deployed EC (3.3 → 7.2 µg/m³) and Fabs
  (43 → 56 Mm⁻¹) all peak in Kiremt, while the 1600-band center is 1617–1619 cm⁻¹ in every
  season (the band identity is not a seasonal artifact; consistent with ftir_15's
  season-stable offset). One shape effect: the broad O–H band is relatively strongest in
  the Dry season. 43 of 296 spectra lack sampling dates and are excluded.

## ftir_18 — transfer roundup: every training-set choice on the same Addis axes

Follow-up to the deck-review question "ftir_08 trains on the 916 HIPS-matched filters — did
we also test a model trained on *all* sites, since TOR exists for the whole pool?" All six
calibration families on identical axes (239 Addis filters, Fabs/10 on x in the figure; both
MAC = 10 and MAC = 6 in the tables;
figure `output/plots/deck/transfer_roundup.png`):

- **Lineage audit**: ftir_09's "Current IMPROVE TOR EC" is **byte-identical to smoke-906**
  (max |Δ| = 0; deployed SPARTAN EC is a different model, corr 0.963) — the full-pool TOR
  transfer had never been drawn per-filter before this notebook.
- **Yes, and it doesn't help**: the no-cleaning full-pool model (13,010 filters, 158 sites,
  k = 6 raw; retrain reproduces ftir_17's held-out metrics to 1e-9) reads **0.60x − 1.09**,
  R² 0.691, bias −3.0 µg/m³ on the 239 pairs. 13k TOR filters buy tracking (R² 0.69 vs 0.26
  for the HIPS-916 transfer) but not calibration.
- **Three distinct transfer failure modes**: flat (HIPS-916: 0.22x, R² 0.26), steep with
  deep offset (smoke-906: 2.30x − 5.38; the deck's −6.91 is the fixed-cohort row), and
  compressed low (full pool: 0.60x − 1.09) — versus deployed 1.90x − 4.17, OCEC-800 +
  AIRSpec 0.78x − 1.28, and the local ceiling (Addis nested CV **0.91x + 0.43**, R² 0.883,
  RMSE 0.38). Every TOR transfer is roughly linear at Addis with wrong gain/offset —
  correctable with a small local anchor — so any domain-adaptation effort should start from
  the TOR-target models, not the HIPS-916 transfer.

## ftir_19 — the HIPS MAC fix applied across every calibration setup

Deck follow-up to `calibration_setup_matrix`: what happens to each setup when HIPS
EC-equivalent is computed at MAC = 6 instead of MAC = 10? Structural answer, shown
per-filter for all six matrix setups on the fixed 190-filter cohort
(`output/plots/deck/mac_effect_all_calibrations.png`, `mac_slope_pivot.png`):

- **The MAC choice cannot move any intercept.** x = Fabs/MAC, so switching MAC rescales x
  by a constant: every setup keeps its intercept and R² *exactly* and its slope scales by
  exactly 0.6 (audited to 1e-9 against the committed phase-2/ftir_13 metrics). The matrix's
  intercept column (−4.17 / −6.91 / −3.69 / −3.22 / −1.62) is MAC-proof; each calibration
  just pivots around its intercept.
- **The MAC fork is a slope contest**: MAC = 6 makes the raw models self-consistent
  (OCEC-800 **0.95**, Ethiopia-shaped **1.05**, deployed 1.14); MAC = 10 is where the
  AIRSpec model lands (**0.86**) — the raw-at-MAC6 vs corrected-at-MAC10 fork of
  ftir_13/ftir_16, now visible setup by setup. R² is MAC-invariant, so fit quality cannot
  arbitrate MAC; RMSE/bias swings are re-expressions of the slope change.
- **Deck erratum fixed**: the matrix quoted the AIRSpec intercept as −1.61; the committed
  value is −1.6151 → **−1.62** (ftir_13's tl;dr had it right). `build_deck_figures.py`,
  `calibration_setup_matrix.png`, and `intercept_ladder.png` corrected and regenerated.

## Deck: the three AIRSpec slides — what the "+ AIRSpec" half of the setup name does

Companion to `filtering_by_ocec.png` (which explains the "Lowest-OC/EC" half), as three
standalone slides — `airspec_1_baseline.png`, `airspec_2_corrected.png`,
`airspec_3_background_gap.png` — built by `scripts/build_deck_figures.py`
(`fig_airspec_1_baseline` / `_2_corrected` / `_3_background_gap`), baselines cached under
`output/corrected/deck_airspec_explainer.npz`. Spoken talk track with caveats:
`deck_notes_airspec.md`.

- **The mechanism**: ~**91%** of a raw Addis spectrum's absorbance at the CH band is smooth
  background, not band signal. A PLS model on raw spectra is therefore free to regress partly
  on that background. **Read this as an illustration, not a population statistic**: the figure
  is `baseline/raw` at 2920 cm⁻¹ for the single representative filter chosen by
  `_airspec_representative`, computed at render time in `build_deck_figures.py`
  (`fig_airspec_1_baseline`) and annotated onto the PNG. There is no committed cohort median
  or distribution behind it, so "91% of what, across how many filters?" has one honest answer:
  of one filter. The cohort-level version of the argument is the baseline-height comparison
  below, which is the one to lean on if pressed.
- **Addis rides a higher background than its calibration cohort**: median AIRSpec baseline
  at 2920 cm⁻¹ is **0.170** at Addis vs **0.101** in the lowest-OC/EC 800 (overlapping but
  clearly offset distributions) — so background structure does not transfer, which is the
  qualitative reason baselining moves the intercept.
- **The payoff, already in ftir_13/ftir_19**: intercept −3.22 → **−1.62**, slope 1.59 →
  **0.86** at MAC = 10.

## ftir_20 — component selection: the AQRC app's protocol vs phase 3's, on all six setups

Raised by reading the AQRC **FTIR Calibration** Shiny app (`R/calibrateServer.R`) next to
the phase-3 code. The app fits `pls::plsr(ncomp = 80, validation = "CV", segments = 10,
segment.type = "interleaved")` and the operator reads k off the RMSEP curve; phase 3 uses
site-grouped 5-fold CV with the first-major-minimum rule. Both differences (CV scheme and
stopping rule) crossed with raw vs 2nd-derivative spectra, on all six matrix cohorts
(`output/plots/deck/component_selection_all_setups.png`, `k_by_rule_ladder.png`):

- **k disagrees by up to ~3×, in both directions.** Raw spectra, app protocol vs phase-3
  protocol: whole IMPROVE network 27→10, smoke-906 19→7, Ethiopia-shaped 17→**21** (up),
  analogs 9→9 (tie), lowest-OC/EC 17→9, lowest-OC/EC + AIRSpec 19→6. "Their k is always bigger" is
  wrong; report the CV scheme with every k.
- **The load-bearing result is the error floor, not k.** Holding whole sites out raises the
  %RMSECV floor by **×1.59 (smoke-906)** and **×1.41 (Ethiopia-shaped smoke)** but only
  ×1.07 / ×1.02 / ×1.01 for the full pool, lowest-OC/EC and lowest-OC/EC + AIRSpec.
  Interleaved folds flatter exactly the smoke-selected cohorts the deployed family is built
  from (repeat sampling concentrated at few sites), while the composition-selected cohort
  is indifferent to fold structure — a protocol-level reason to distrust smoke-906 at an
  unseen site, independent of ftir_13's collapse-on-corrected-spectra result.
- **2nd-derivative preprocessing is not a free win.** Under site-grouped CV it cuts the
  floor for the baseline-dominated cohorts (smoke-906 155%→100%, full pool 110%→96%,
  Ethiopia-shaped 53%→43%) but degrades the targeted ones (lowest-OC/EC 62%→77%) — the
  derivative and the AIRSpec baseline fix the same problem, so they are alternatives, not
  additives.
- **The leaked quantity is baseline, not chemistry.** On 2nd-derivative spectra every
  optimism ratio collapses to 0.89–1.06 (smoke-906 ×1.59 → ×1.00): removing the smooth
  baseline removes the advantage of having seen a site before. What an interleaved fold
  leaks is the site's characteristic background — the same structure ftir_13/ftir_19
  identify as the reason raw-spectra models fail to transfer to Addis.

## ftir_21 — every setup run twice: Calibration app vs site-held-out protocol

Follow-on to ftir_20. `scripts/calibration_modes.py` defines the two protocols as
switchable modes so any notebook can re-run a cohort either way:

| | `app` — Calibration app | `site_heldout` — site-held-out |
|---|---|---|
| CV folds | 10-fold interleaved, no site grouping | 5-fold site-grouped |
| component rule | first k within 5% of the minimum | first major minimum |
| final fit | all cohort filters | training side of a site-disjoint 80/20 split |
| held-out TOR test | none, by construction | yes |

**Provenance check passes**: `site_heldout` mode reproduces the committed ftir_11/ftir_13
calibrations exactly (k = 6 / 5, Addis slope and intercept to < 1e-6), asserted in the
notebook — so this is a like-for-like re-run, not a re-derivation.

- **The protocol moves the Addis answer more than the MAC fork does.** Same 800
  lowest-OC/EC filters: **2.15x − 4.59 (RMSE 2.03)** under the Calibration app protocol vs
  **1.59x − 3.22 (RMSE 1.16)** site-held-out — a 1.4 µg/m³ intercept swing from the component choice alone, where
  ftir_19 showed MAC cannot move an intercept at all. Biomass-smoke swings hardest
  (2.43x − 6.35 vs 0.50x − 0.99 at k = 4); the whole IMPROVE network 1.95x − 4.05 vs
  1.65x − 3.44.
- **Lowest-OC/EC + AIRSpec is protocol-robust**: −1.65 vs −1.62, slope 0.96 vs 0.86. Its
  Addis answer does not depend on who processed it — with ftir_20's optimism ×1.01, the
  strongest robustness claim in the deck.
- **Ethiopia-shaped smoke fails the held-out TOR test** (R² **0.00**, slope **−2.20**).
  It is still carried in the matrix at intercept −3.69 but has no site-disjoint skill; it
  should be asterisked alongside the spectral analogs. Passing cleanly: lowest-OC/EC +
  AIRSpec (R² 0.90, slope 1.01) and lowest-OC/EC (0.91 / 0.87).
- **Confounds inherent to an end-to-end comparison** (stated, not corrected): the app mode
  fits on the whole cohort (800) vs the site-held-out training part (606), so training size travels
  with the protocol; and the app mode yields no site-disjoint test by construction.
  Figures: `both_modes_crossplots.png` (per-panel square axes — predictions reach
  ~17 µg/m³ against a HIPS axis topping at 8.7, so one shared range would clip),
  `intercept_slope_by_mode.png`.

### Which earlier figures depend on the component choice?

`K_SENSITIVITY_AUDIT.md` classifies every committed figure. Summary: the band-identity
(ftir_12), implied-MAC (ftir_16) and spectra-comparison (ftir_17) figures involve no
calibration and are **k-free**; ftir_19's MAC figures are **structurally invariant** (the
pivot-on-the-intercept result is algebra, true for any k); the ftir_11/13 crossplots and the
intercept ladder are **superseded** by ftir_21's both-protocol versions; and
`calibration_setup_matrix.png` now carries **both** intercept columns. Still conditional on
the site-held-out component choice and **not re-run**: ftir_15's bootstrap CI and residual
structure, and ftir_17's no-cleaning full-pool crossplots. Second-order caveat: for the
spectral-analog and hybrid cohorts, k changes cohort *membership* (selection runs through a
fitted PLS score space + VIP weights), so those rows vary only the fit.

### Standalone per-protocol figure set

`scripts/build_protocol_variants.py` → `output/plots/deck/by_protocol/{calibration_app,
site_held_out}/`: each k-dependent figure written twice, one folder per protocol with
matching file names, for one-protocol-per-slide use (the ftir_21/22 versions overlay both).
Nine figures in each folder: `calibration_setup_matrix` (that protocol's intercept column
only), `component_selection` (the CV curve and the rule that chose k),
`crossplots_all_setups`, `intercept_slope_ladder`, `mac_effect_all_setups`,
`mac_slope_pivot`, `bootstrap_intercept_ci`, `residual_vs_d2`, `cohort_size_sweep`. Colour encodes the
calibration setup (matching the combined matrix/ladder), not the protocol — the folder and
subtitle carry that. Read from committed tables, so regeneration is instant and cannot
drift from the notebooks.

### ftir_22 — the k-dependent figures re-derived under both protocols

- **ftir_15 survives**: the raw-vs-corrected residual distinction holds under both
  protocols (D² r = 0.87 app / 0.71 site-held-out for raw; 0.33 / 0.21 corrected), and the
  corrected bootstrap CIs overlap ([−2.07, −1.03] app vs [−1.78, −1.17] site-held-out, both
  excluding zero). The site-held-out numbers reproduce ftir_15's committed 0.71 / −0.05.
- **ftir_11's "N = 800 is the sweet spot" is protocol-dependent.** Site-held-out picks 800
  (intercept −3.22, RMSE 1.16, held-out TOR R² 0.911, vs 0.793 at 1600); under the
  Calibration app protocol intercept and RMSE improve monotonically to N = 1600
  (−5.56 → −4.59 → −4.17). What selects 800 is the held-out TOR test, which only one
  protocol produces — state the protocol whenever the cohort size is defended.
- **Interleaved CV is order-dependent, so the app protocol is not reproducible.** The same
  800 filters give k = 18 ranked by OC/EC, 19 in the committed CSV order, 15 shuffled;
  site-grouped CV gives k = 5 in all three. This also explains the k = 18 vs 19 difference
  between ftir_21 and ftir_22 on identical data.

## ftir_23 — how each protocol picks its components, protocol by protocol

Shows the decision, not just its result: each cohort's CV curve under one protocol with
that rule's own machinery drawn on it (`output/plots/ftir23/selection_curves_app.png`,
`selection_curves_site_heldout.png`), then the Addis crossplot the chosen model produces
(`selection_and_consequence_*.png`). Rule internals are re-derived and asserted equal to
the production selectors, so the drawings cannot drift from what runs.

- **Both rules stop well short of the curve minimum.** Selected k vs where the curve
  bottoms — Calibration app: 27/30, 19/21, 17/19, **9/26**, 17/20, 19/25; site-held-out:
  15/17, 4/4, **10/26**, **4/29**, 6/9, 5/5. The spectral-analog curve falls monotonically
  to k ≈ 26–29 and both rules refuse to follow it — the cohort with no held-out TOR skill,
  so chasing the CV minimum would have selected the worst calibration in the set.
- **The two curves are different objects.** The app's pooled RMSECV (√(PRESS/n) over
  position-based folds) carries no fold-to-fold spread and so cannot support an error band;
  the site-grouped curve averages per-fold RMSEs and keeps one. On the IMPROVE-network and
  smoke cohorts that ±1 SE ribbon spans several µg/filter — "the minimum is at k = 17" is
  not supported by those data. The app curve looks decisive because it discards the
  information that would say otherwise.
- **Where the curve genuinely bottoms early, the protocols agree**: biomass-smoke (4/4) and
  lowest-OC/EC + AIRSpec (5/5) take their true global minimum under the site-held-out rule.
  The disagreement is specifically about long flat tails, where the 5% band admits far more
  components than the evidence separates.

## ftir_41–43 — the carbon-definition thread (Sept 2026): what is FTIR failing to predict?

Reframing follow-ups from the 2026-09-01 external research review: split the FTIR-vs-HIPS
disagreement into (a) the definition of the carbon being measured, (b) carbon recovery, and
(c) what a residual model actually demonstrates.

- **ftir_41 — Adama three-method reconciliation.** The Adama PTFE twins *do* exist (CSU AMOD
  Batch 54: 5 PTFE filters with FTIR OC/EC + HIPS, date-paired with the 5 quartz TOR/TOT
  filters; older notes said "no FTIR/HIPS"). TOR→TOT moves EC by −15.4 to −20.6% (median
  −19.0%) while OC+EC is conserved to ≤1e-13 — the convention is a split question, never a
  recovery question. The FTIR sum is 0.40–0.74 of thermal TC (median 0.51) and the deficit is
  **OC-dominated**: FTIR EC vs EC_TOT has median ratio 0.86 (unflagged 0.79–1.02), while
  OC_ftir/OC_TOR ≈ 0.46 (quartz adsorption artifact uncorrected — bound it before reading
  this as FTIR under-recovery). Same-corridor implied MAC: 16.5 (vs EC_TOR), 20.8 (vs
  EC_TOT), 23.5 (vs FTIR EC) m²/g — above the physical 4–13 band but nowhere near the ≈47
  the Addis composition bridge implies. Two pairs carry comparability flags (Jul 9: 39.7-min
  quartz start offset, and the one anomalous EC pair; Jul 30: PTFE volume 0.46× quartz) —
  flagged in every figure, not excluded, not corrected.
- **ftir_42 — target-definition experiment.** Same corrected spectra, same locked protocol
  and folds on the IMPROVE mirror; only the target changes. TC is FTIR's **easiest** target
  (held-out-site R² 0.897 / %RMSE 39.7 vs 0.870 / 43.8 for EC_TOR on the lowest-OC/EC
  cohort); EC_TOT is the hard one (0.764 / 69.4; full-pool collapse to k = 2, R² 0.18) —
  and pool-wide EC_TOT/EC_TOR median is **0.540** (IQR 0.41–0.68), a far larger
  redistribution than Adama's 0.81. The OC/EC partition itself has ~zero out-of-site skill
  (predicted-vs-observed EC/TC R² 0.007 cohort / 0.001 pool); direct-TC and OC+EC-sum agree
  to 3% RMSE. The convention changes the Addis transfer beyond a rescale: EC_TOT-target
  model reads 0.63x−1.24 vs 0.81x−1.53 (intersection n = 728; the full-800 pipeline
  reproduces the locked 0.857/−1.615, asserted). Consequence: any "FTIR EC vs thermal EC"
  claim is underspecified without the convention named.
- **ftir_43 — residual-learner null control.** With f(X) = the committed corrected
  prediction and b = Fabs/10: b = 0.802·f + 2.851, so r = f − b mechanically contains
  0.198·f. Time-blocked controls: f(X)-only residual learner R² 0.064 —
  indistinguishable from a proportional-disagreement null (median 0.077, p = 0.67);
  spectra learner 0.244 vs null q95 0.113 (0/200 null runs) — **genuine incremental
  spectral signal**; but season+volume alone reach 0.184. RMSE ladder: no correction 2.46
  → constant bias 0.72 → spectra residual learner 0.62 → direct spectra→HIPS 0.80. The
  constant offset does 71% of the deployable work. Trap documented: HIPS τ as "metadata"
  manufactures residual R² ≈ 0.94 by arithmetic circularity (Fabs ∝ τ·A/V) — ftir_27's
  lesson in a new costume.

**Caveat added 2026-09-01 (ftir_46) to the ftir_43/45 entries above:** "genuine incremental
spectral signal" holds only against the proportional-disagreement null. Paired against the
metadata baseline (f(X) + season + volume + lot) on the same eight time blocks the spectra
increment is ΔRMSE −0.06 µg/m³ with block-bootstrap CI [−0.13, +0.02] (5/8 blocks; forward-
chained the inner CV picks k = 0), and mean-of-folds R² is ≈ 0 for every non-mass model.
The worklog's 0.32–0.40 was a best-of-grid k read off the test curve on raw spectra
(pathway figure 08): ≈0.10 optimism + ≈0.05 raw-vs-corrected + ≈0.02 f(X) choice.
Gravimetric mass supersedes spectra as the lead (below).

Gated (not built, data missing): full chemical mass closure (local mirror ions = sulfate
only, XRF = Fe/S/Si, grav = PM2.5 — no nitrate/ammonium/full elements); PurpleAir
composition-dependent error (sensor 93783's colocation with the "Jacros BAM" file is
unverified — the BAM names its site Addis Ababa Central).

### ftir_44 / ftir_45 — the in-house follow-ups (2026-09-01)

- **ftir_44 — Adama PTFE spectra through the phase-3 calibrations (Ann's ask #3).** The
  five CSU AMOD spectra, AIRSpec-baselined on the pool grid (id → filter map inferred from
  FilterId order; CH-band vs OC-loading rank ρ = 1.00, vs 0.10 for analysis-time order),
  predicted with the locked and sweep-winner models. **They over-read thermal EC and land
  inside the HIPS band**: locked 800 + AIRSpec **1.40× EC_TOR** (1.76× EC_TOT), winner 440
  k=8 **2.00×** (2.51×), EC_TOT-target model 1.01× (1.27×); deployed SPARTAN 0.69× (0.86×).
  Adama's OC/EC ≈ 6 sits far outside the cohort's ≤ 2.27 domain — the Addis-tuned
  calibrations carry the Addis offset with them and are not general calibrations. OC and
  TC refits leave the Adama carbon sum at 0.47 of thermal TC (deployed 0.44): ftir_41's
  carbon deficit is upstream of any calibration.
- **ftir_45 — where ftir_43's spectral increment lives.** Stepwise on identical folds:
  season+volume 0.184; +f(X)² 0.195; +char_06 flag 0.176; +neutral band heights −0.006;
  all 0.089; spectra learner 0.244. The dry-season anomaly class has the same
  season-adjusted residual as normal filters (0.004 vs 0.024 µg/m³) — **not the missing
  absorber**. The learner's coefficient spectrum is derivative-shaped across
  1560–1750 cm⁻¹: band position/shape, not height, is the lead.

External dependencies are consolidated in `EXTERNAL_ASKS_2026-09-01.md`.

## ftir_46–49 — the pathways from the 2026-09-01 audit: mass, blank lines, the split, the MA350 chain

- **ftir_46 — paired increment and the mass lead.** Same 239 filters and eight contiguous
  blocks as ftir_43. Spectra vs metadata base: ΔRMSE −0.060 [−0.134, +0.020] — not a
  paired result. Mass (`MassCollectedOnFilter`) as a covariate: −0.036 [−0.099, +0.022].
  Mass in the base, then spectra: **−0.126 [−0.199, −0.032]**, 7/8 blocks, residual R²
  0.24 → **0.52**, HIPS-prediction RMSE 0.658 → **0.497**; forward-chained −0.137
  [−0.229, −0.044]. At fixed f(X) and season the residual falls **−0.068 µg/m³ per µg/m³**
  of mass [−0.097, −0.045] (raw slope ≈ 0 because mass and f(X) correlate at 0.84).
  Cross-site, Fabs ~ f(X) + mass with the locked 800 + AIRSpec model, per deposit (µg
  EC-eq per µg mass): Addis **0.064** [0.037, 0.091], Bishoftu **0.071** [0.022, 0.116],
  Beijing 0.019 [0.013, 0.027], Delhi 0.010 (0.032 inside the Addis loading range),
  Pasadena 0.006 [−0.006, 0.021]; pooled non-Ethiopian in the Addis range **0.033**
  [0.024, 0.044]. Reading: a generic loading term exists off-Ethiopia (instrument side,
  ≈ half the Addis coefficient) and an Ethiopian excess of ≈ 0.03–0.04 µg EC-eq per µg,
  the same at two sites 45 km apart and absent at Pasadena on the same lot 251, is the
  aerosol-side lead. Falsifiers: a HIPS loading experiment on non-absorbing deposits
  returning ≥ 0.06 (all instrument), or the Addis coefficient failing to track
  composition within Addis (not aerosol). Note the "loading is not the mechanism" line
  in the worklog tested filter darkness, not mass — different quantities.
- **ftir_47 — blank lines belong to deployed calibration lines, not lots.** Lot 251's 373
  blanks sit on three deployed lines (2423.6−6.033R, R1 224–265; 1416.2−2.783R, 134–206;
  1397.0−2.687R, 138–232) with no common R1 interval; per-line rms 5–10 counts, pooled
  quadratic 32. Keyed per line the shipped Fabs reproduce to 0.0000 Mm⁻¹ (3,047 samples).
  **Pasadena's 3.15x does not dissolve** (per-line quadratic 3.04 ± 0.19; the 0.91x is
  the pooled artifact, +80% on 56/158 filters); **Addis's −1.27 ± 0.17 was the same
  artifact** (per-line −1.48 ± 0.18 vs deployed −1.51 — blank-line share ≤ 2%, not 15%);
  44% of Addis filters (not 36%) below their line's blank R1 range. Network (27 sites):
  14.9% of samples outside their own line's blank range; extrapolation ranking BDDU 49%,
  **ETAD 44%**, CAHA 39% (above), IDBD 30%, INDH 18%; the pooled quadratic shifts Fabs a
  median 0.67 Mm⁻¹ (p90 4.2) vs 0.29 (p90 0.83) per line. Unauditable: the ETBI
  reconstructed holdout (line inferred from dates) and Adama's lot-245 line (zero blanks
  in any local file). OFFSET_ADJUDICATION §5/§5b corrected in place.
- **ftir_48 — the pyrolysis split as a target.** Same rows/folds/protocol as ftir_42.
  Δ = OPTT − OPTR is 46% of EC_TOR at the median and, on the 800 cohort, predictable
  (held-out R² **0.770**, k = 4; OPTT carries it, OPTR 0.46, Δ/EC_TOR 0.00); on the full
  pool unlearnable (0.004). EC_TOT routes on identical held-out sites: direct 0.764 /
  %RMSE 69; composed EC_TOR_hat − Δ_hat 0.774 / 67; **oracle EC_TOR_hat − Δ_true 0.697 /
  78** — worse, so EC_TOT is hard because of the EC_TOR error against a target 57% the
  size, not because of the correction; the Δ and EC_TOR errors co-vary (r 0.55) so
  composing cancels shared error. Full pool: composing rescues EC_TOT from the k=2
  collapse (0.18 → 0.68). Adama: measured Δ is only 0.15–0.21 of EC_TOR and the cohort Δ
  model reads it 0.40×. Addis: composed route 0.68x − 1.38 vs direct 0.63x − 1.24 — the
  negative offset is route-independent.
- **ftir_49 — MA350 raw chain (fallback run).** The raw 1-min CSV (556 MB) would not
  hydrate from Drive (~2 MB/min); the notebook ran on `df_Jacros_9am_resampled.pkl` and
  every minute-only cell prints SKIPPED; **rebuild after `cat FILE > /dev/null`
  completes** and rewrite its tl;dr. From daily means: the DualSpot term K·ATN1 is 37% /
  43% / 48% of IR / Red / Blue BCc (K 0.011 / 0.010 / 0.009); a 10% K error moves IR BCc
  5.9%, Red 7.4%; spot-2-compensated BC2 is **0.925 × BCc at IR, 0.944 at Red** and the
  closure K* is 16% above the reported K — a 6–8% spot-disagreement term at the two
  channels the calibration uses. The ±1-day tolerant mean of `match_aeth_filter_data`
  differs from the 9am–9am filter-day mean by median +3.7% (IR), **53% of 193 ETAD filter
  days move > 10%, 21% > 25%, 32 days > 2,000 ng/m³**. Quirk: the processed pickle's
  Green channel has ATN1 ≈ 0 / K ≈ 0 / BC1 = BCc on all 1,047 days (broken columns).

## ftir_50 — spectral comparison beyond one similarity number (2026-09-01)

`scripts/spectral_similarity.py` adds the comparisons the explorer's Analogs tab cannot
make, and ftir_50 runs them over the whole 13,634-spectrum lot-248/251 library against all
five SPARTAN targets (AIRSpec-corrected, shared 1425–3998 cm⁻¹ grid). Everything is
matmul/PCA, so the same sweep runs on Colab against the entire database (launcher cell
added; module bundled).

| Method | Adds |
|---|---|
| Hotelling T² + Q residual | splits "distance to the library" into in-plane vs off-plane; Q sees chemistry the library never held |
| Band-resolved / moving-window r | localizes disagreement; the module refuses bands the grid covers <60% |
| Spectral information divergence | shape metric weighting relative band differences |
| Neighbourhood redundancy + selectivity-matched mutual k-NN | are N analogs N independent spectra? |
| Borda rank fusion | one ordering when metrics disagree; the disagreement is the uncertainty |

Results. (1) The in-domain claim was one number doing two jobs: 22% of Addis filters
exceed the library 95th percentile on T² or Q, 7.3% on both; the two axes flag different
filters. (2) Addis's analog neighbourhoods are the most redundant of the five sites
(distinct neighbours per neighbour slot 0.19 vs 0.34 Delhi, 0.40 Beijing, 0.46 Pasadena,
0.48 Bishoftu) and 36% of Addis filters have no mutual analog at all (Delhi 18%, others
4–7%) — a top-N analog cohort at Addis is not N independent analogs, which is a concrete
mechanism for analog cohorts winning screening and failing the held-out floor (cf. the
ftir_46–49 audit note that analog/Ethiopia-shaped cohorts reach −0.6 to −1.3 within
AIRSpec rows but fail the floor). (3) Similarity is band-dependent: Addis holds r 0.98 in
the O–H window, 0.67 at aliphatic C–H, 0.78 at 1560–1680 where the marker band sits;
whole-spectrum 0.948 hides both. Moving-window r shows every site collapsing in its own
window (Addis alone dips inside the 1617 band).

Negative result kept in the notebook: the standard equal-k mutual-NN test passes nearly
everything at this target size (a library row picking 50 of ~250 targets is a 20% cut
against 0.4% on the target side) and must be selectivity-matched before it means
anything. Scope: spectral geometry only — no calibration is fit and a library neighbour
is a spectral analog, not a composition match. Obvious next methods, both an afternoon on
the same module: Ward clustering of pool + targets (the field-standard source-class
comparison, Russell 2009 / Takahama 2011) and a validated r threshold instead of a top-N
rank (Open Specy's discipline).

## ftir_52 — the IMPROVE network as a spectral map (2026-09-01)

Ward clustering of the full lot-248/251 library plus all five SPARTAN targets on
AIRSpec-corrected spectra — the field-standard comparison (Russell 2009, Takahama 2011)
that the analog thread had never run — with a site-similarity map and a validated match
threshold. PCA-30 (99.9% of standardized variance) fitted on the library alone, Ward on
library+target scores together, k=4 by silhouette (0.268; the scan is flat, 0.15–0.27).

**The top-level split is deposit and SNR, not composition.** Median peak absorbance by
class: 0.020 (class 1, n=8066), 0.044 (class 2, n=3603), 0.003 (class 3, n=434), 0.004
(class 4, n=1531). Row standardization removes amplitude but not signal-to-noise, so
low-deposit spectra cluster by their noise shape. Class 3 in particular (peak absorbance
0.0025 at a median EC loading of 3.8 µg) looks like a baseline/correction-quality class
and deserves a QC look.

**A real smoke-associated class survives loading-conditioning.** Class 2 is Jul–Oct
weighted (57%), heavily loaded (EC 8.1 µg), OC/EC 5.6, top sites ATLA1/BOND1/LASU2, and
holds 3.3× its share of the deployed model's smoke lineage. Because smoke filters are
heavily loaded, the enrichment was tested *within* EC-loading quantile bins: it holds at
11.5×, 6.2×, 3.3×, 1.8×, 1.3× from the lightest bin up (median 3.3×), so the class carries
smoke information beyond loading. Target membership: Delhi 68%, Beijing 56%, Pasadena 28%,
Bishoftu 15%, Addis 5% — Addis is emphatically not in the network's smoke class, which
sharpens the charcoal framing rather than supporting it. Caveat: there is no fire label in
the local database (the 36 fire-related filter comments do not intersect the spectral
pool), so "smoke" here means the deployed EC model's biomass lineage, nothing stronger.

**Site similarity.** Nearest IMPROVE sites by median corrected spectrum: Addis → NOGA1
(0.990), CHAS1, LTCC1, PUSO1; Bishoftu → BIBE1 (0.996), MELA1, LYEB1; Delhi → PITT1
(0.995), BIRM1, MACA1; Beijing → VILA1 (0.997), QUCI1, GRRI1; Pasadena → PACK1 (0.992),
MOMO1, ACAD1. All five are least like TOOL1 (Arctic Alaska), Addis most extremely
(r 0.793). A class-mix (Jensen–Shannon) distance is reported alongside as a check that
the median is not hiding heterogeneity.

**The threshold test fails, and that is the point.** Same-Ward-class agreement is ~99% at
any cutoff (k=4, one class holds 59%), so it cannot calibrate anything. On the
decision-relevant task — does the nearest neighbour carry EC within 25%? — precision never
reaches 75%: the best is 64% at r ≥ 0.9999, keeping 0.5% of the library, against a 31%
base rate. **Spectral near-identity is not calibration equivalence.** This is the
quantitative form of ftir_50's redundancy result and of the microplastics community's
finding that a high hit-quality index does not imply a correct match. The encouraging
half: same-site precision rises only 12% → 68%, so spectral twins are mostly *not*
same-site — the library does generalize geographically.

Target coverage above the strictest cutoffs (median best-match r): Bishoftu 0.9988,
Beijing 0.9986, Pasadena 0.9966, Delhi 0.9964, Addis 0.9938 — Addis has the weakest best
matches of the five, consistent with its neighbourhood redundancy in ftir_50.
