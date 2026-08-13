# FTIR EC phase 3 — post-meeting follow-ups (July 2026)

Follow-up analyses from the July 2026 meeting with Ann and Satoshi and the
subsequent email thread. Phase 2 lives in `research/ftir_hips_chem/`
(notebooks `ftir_07`–`ftir_10`); this folder continues the numbering.

## Contents

| Item | What it answers | Result |
|---|---|---|
| `ftir_11_ocec_ratio_cohort.ipynb` | Ann's idea: does a calibration built from the lowest-OC/EC IMPROVE samples (Addis is below the entire IMPROVE OC/EC range) fix the Addis intercept without sacrificing held-out TOR performance? | Best locked cohort so far (intercept −3.22 vs −4.17 deployed; held-out TOR R² 0.91), but not a fix |
| `ftir_12_band_1600_identity.ipynb` | Is the elevated ~1600 cm⁻¹ band in Addis spectra carboxylate, amine, or nitro? Peak-position and band co-variation diagnostics. | Amine rejected; carboxylate / aromatic C=C remain; definitive partner bands sit below 1425 cm⁻¹ |
| `scripts/airspec_baseline.py` | Python port of the APRLssb/AIRSpec segmented smoothing-spline baseline (Kuzmiakova, Dillner, Takahama 2016), validated against the R ground-truth output for the ETAD spectra (DF1=6, DF2=4). | Validated to ≤6×10⁻⁷ absorbance vs the R run; applied to all 13,634 pool + ETAD spectra at DF1 = 6 and 8 |
| `ftir_13_airspec_corrected_calibrations.ipynb` | Satoshi's EDF 6–8 instruction: rebuild the key calibrations on AIRSpec-baselined spectra (both IMPROVE pool and Addis) and test whether the intercept story changes. | HIPS transfer gap survives baselining; low-OC/EC intercept halves to −1.62; smoke model collapses (slope 0.37) |
| `ftir_14_delhi_beijing_feasibility.md` | Satoshi's score-space comparison of Delhi/Beijing vs Addis — data availability assessment. | Blocked: no SPARTAN spectra in the local pull; needs an INDH/CHTS export like the ETAD one (add ETBI = Bishoftu) |
| `ftir_15_uncertainty_and_hybrid.ipynb` | Is the −1.6 intercept statistically solid? Where does the corrected model's scatter come from? Does a hybrid OC/EC + spectral cohort beat OCEC-800? | Bootstrap CI [−1.78, −1.06] (disjoint from raw); corrected residuals = season-stable constant offset; hybrid cohort fails → cohort engineering closed |
| `scripts/run_context_addenda.py` | Post-meeting context: Adama TOR OC/EC and the ETBI site vs ETAD. | Adama OC/EC 4.6–7.2 (≈ pool median) challenges the OC/EC-extreme premise; ETBI = untouched in-country test set |
| `ftir_25_intercept_invariant.md` | What is the intercept worth in absorption units, and what is invariant about it? Pure algebra on committed constants — no data, hence markdown + script rather than a notebook. | C = \|intercept\|·MAC/slope is **exactly MAC-invariant**; 18.8–26.1 Mm⁻¹ across six setups (median 21.5 ≈ 46% of median Addis Fabs) while slopes span 3.4× |
| `ftir_26_improve_hips_origin.ipynb` | Does IMPROVE HIPS itself read nonzero Fabs at EC = 0 — i.e. is any of the Addis offset a generic instrument zero (attack-plan item 1)? | **No.** Pooled intercept +1.345 Mm⁻¹ as an upper bound, +0.097 in the Addis-like subset; at EC ≤ 0 median Fabs +0.12 with 23.8% negative. Bound ≤1% of C — but raises curve geometry and loading dependence as non-absorber explanations |
| `ftir_27_chemspec_circularity.ipynb` | Can either SPARTAN speciation carbon column serve as an independent EC reference? | **Neither.** `ChemSpec_BC` = round(Fabs/10, 2) (x-circular, bit-exact on 163/188); `ChemSpec_EC` = round(EC_ftir, 2) (y-circular, r² 0.9997). No reference in hand can arbitrate → quartz TOR becomes decisive |
| `ftir_28_ma350_brc_falsification.ipynb` | Can the MA350's wavelength spread supply the ~20 Mm⁻¹ of non-EC absorption the intercept implies (attack-plan items 4–5)? | **No — and the instrument cannot be asked in either direction.** AAE(625,880) = 0.944 ± 0.060 (below the AAE_BC ≈ 1 anchor), implied Babs_BrC −2.06 Mm⁻¹ vs +21.7 needed; only the IR channel is trustworthy. Closed. |
| `ftir_31_deck_figure_regeneration.ipynb` | Where does every image in the 12–13 Aug briefing decks come from? | **One executed provenance record for all of them.** Re-runs the by_protocol and deck-root builders and rebuilds the implied-MAC bridge, deployed crossplot, seasonal corrected spectra, peak-center panel and Adama/ETBI context from committed tables, headline numbers asserted; manifest in `output/plots/ftir31/figure_manifest.csv`. |
| `ftir_32_cv_scheme_ablation.ipynb` | Is the protocol gap driven by the site grouping or the fold count? | **The grouping.** Interleaved 5- and 10-fold floors agree to 1.00–1.04 on all three cohorts (the leak is structural), the committed ×1.59 smoke inflation reproduces exactly, and one new nuance: grouped floors are count-sensitive on site-concentrated cohorts, making 5-fold grouping the strictest, most deployment-like scheme of the four. |
| `ftir_33_shape_cohort_explainers.ipynb` | How were the Ethiopia-shaped-smoke and spectral-analog cohorts selected — the filtering_by_ocec treatment for the shape-based setups? | **Selection explained, failure explained.** Ethiopia-shaped = top 300 smoke filters by band-feature distance to Addis (only 4/906 sit inside the Addis 5–95% box on all three features); analogs = top 500 of the pool by mean percentile of nearest-Addis D² + VIP-weighted RMSE. On the OC/EC ruler both sit at/above the pool median (7.5 and 6.1 vs cut 2.27; overlap with OCEC-800: 1/300 and 17/500) — shape does not find composition, and both fail the locked TOR test. |

`PHASE3_SUMMARY.md` condenses everything; `draft_email_ann_satoshi.md` is the update email draft.

## Reproducibility

Each notebook is generated and executed from a percent-format script:
`scripts/run_ftir_11.py` / `run_ftir_12.py` / `run_ftir_13.py` via
`python scripts/build_notebooks.py 11 12 13` (run from this directory). Canonical outputs
land in `output/tables/ftir11|ftir12|ftir13` and `output/plots/…`.

`output/tables/replication_ftir_11|12|13` (and matching plot dirs) hold an **independent
replication** of each analysis by a separate implementation (gpt-5.6 Codex agent, same
protocol/seed, its own code); headline metrics agree with the canonical runs to the last
digit, and its extra VIP-overlap diagnostics are cited in `PHASE3_SUMMARY.md`. The
replication left no scripts, so treat those directories as corroboration, not as a pipeline.

## Shared infrastructure

Notebooks import the phase-2 modules from `../ftir_hips_chem/scripts`
(`pls_transfer.py`, `config.py`, …). External data paths resolve through
`pls_transfer.FTIRTransferPaths.defaults()` (Google Drive).

`reference/APRLssb/` holds the exact R sources (from
https://gitlab.com/aprl/APRLssb) that produced the ETAD ground-truth
baselines in `…/ETAD FTIR/baseline_plots_AIRSPEC/spectra_baselined_AIRSPEC.csv`;
the Python port is validated against that file, not just against the paper.
