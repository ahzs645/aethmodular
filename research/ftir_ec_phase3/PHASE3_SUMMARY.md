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

- Held-out TOR (disjoint sites): RMSE **3.41 µg/filter** vs **4.39–7.31** for ten size-matched
  random cohorts at the same k — the advantage is compositional, not statistical.
- **VIP convergence (key finding):** the low-OC/EC model's VIP profile correlates
  **ρ = 0.74** with the Addis-only HIPS model from `ftir_08`, while the current 906-smoke EC
  model correlates only **ρ = 0.12** with it. Two independent routes (Addis optical training
  vs IMPROVE composition selection) point to the same spectral features.
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
| **Lowest-OC/EC corrected EDF6** | **0.86** | **−1.61** | 0.657 | 2.41 |
| Smoke 906 raw (ftir_10) | 2.65 | −6.91 | 0.685 | 2.85 |
| Smoke 906 corrected EDF6 | 0.37 | −0.67 | 0.458 | 3.83 |

- **Correction halves the low-OC/EC intercept** (−3.22 → −1.61) and yields a held-out TOR
  slope of 1.01 (R² 0.90) — the best locked intercept-with-defensible-slope so far, at the
  cost of Addis precision (RMSE 1.16 → 2.41).
- **The smoke calibration collapses on corrected spectra** (slope 0.37): its raw-spectrum
  response was substantially carried by the broad baseline — the component the correction
  removes and the component EC's sloping absorption lives in.
- **EDF 6 vs 8 is indistinguishable** (Δslope < 0.01): the EDF choice inside Satoshi's range
  is not a sensitive parameter.
- **The MAC = 6 vs 10 fork is now the deciding unknown**: raw low-OC/EC is self-consistent at
  MAC = 6 (slope 0.95), corrected low-OC/EC at MAC = 10 (slope 0.86). An independent Addis
  TOR/EC reference or a HIPS MAC/protocol bridge resolves it; further cohort engineering
  will not.

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
  than Addis (47.1) but far above IMPROVE. If ETBI FTIR spectra are pulled alongside
  INDH/CHTS, it is an in-country, dry-season test set for every Addis conclusion.

## ftir_14 — Delhi/Beijing score-space comparison: blocked on data

The local DB pull covers IMPROVE only (169,566 analyses, no SPARTAN sites); ETAD spectra came
from a separate site-specific pull. Doing the same for INDH/CHTS (adapt
`etad_lots_query.sql` / `pull_scans.ps1`) unblocks a one-notebook analysis using the
already-fitted models. Second-order per the meeting: Delhi/Beijing are test sets, not
calibration inputs.

## ftir_16 — MAC decision prep: ChemSpec debunk, Adama bridge, campaign spec

- **`ChemSpec_EC` for ETAD closely tracks, but is not identical to, HIPS Fabs / 10**
  (median ChemSpec/Fabs ratio 0.101, r = 0.89, n = 175 base-joined filters). This
  cross-check does not establish an independent EC reference, so it cannot arbitrate the
  MAC = 6 vs 10 fork.
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

## Deck: `airspec_explainer.png` — what the "+ AIRSpec" half of the setup name does

Companion to `filtering_by_ocec.png` (which explains the "Lowest-OC/EC" half). Built by
`scripts/build_deck_figures.py::fig_airspec_explainer`, baselines cached under
`output/corrected/deck_airspec_explainer.npz`:

- **The mechanism**: ~**91%** of a typical raw Addis spectrum's absorbance at the CH band is
  smooth background, not band signal. A PLS model on raw spectra is therefore free to
  regress partly on that background.
- **Addis rides a higher background than its calibration cohort**: median AIRSpec baseline
  at 2920 cm⁻¹ is **0.170** at Addis vs **0.101** in the lowest-OC/EC 800 (overlapping but
  clearly offset distributions) — so background structure does not transfer, which is the
  qualitative reason baselining moves the intercept.
- **The payoff, already in ftir_13/ftir_19**: intercept −3.22 → **−1.62**, slope 1.59 →
  **0.86** at MAC = 10.
