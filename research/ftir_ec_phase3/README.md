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
| `ftir_29_provisional_addis_ec_series.ipynb` | What does the locked corrected calibration say Addis EC *is*? The provisional series, its seasonal structure, the deployed comparison, and the offset `c` with a bootstrap CI from ftir_15's committed draws. | Median 2.32 µg/m³ (Dry 1.67 / Belg 2.39 / Kiremt 3.62), zero negatives; provisional = 0.45 × deployed (R² 0.85); c = 1.82 [1.60, 1.97] µg/m³ corrected, 2.06 [1.93, 2.24] raw |
| `ftir_31_deck_figure_regeneration.ipynb` | Where does every image in the 12–13 Aug briefing decks come from? | **One executed provenance record for all of them.** Re-runs the by_protocol and deck-root builders and rebuilds the implied-MAC bridge, deployed crossplot, seasonal corrected spectra, peak-center panel and Adama/ETBI context from committed tables, headline numbers asserted; manifest in `output/plots/ftir31/figure_manifest.csv`. |
| `ftir_32_cv_scheme_ablation.ipynb` | Is the protocol gap driven by the site grouping or the fold count? | **The grouping.** Interleaved 5- and 10-fold floors agree to 1.00–1.04 on all three cohorts (the leak is structural), the committed ×1.59 smoke inflation reproduces exactly, and one new nuance: grouped floors are count-sensitive on site-concentrated cohorts, making 5-fold grouping the strictest, most deployment-like scheme of the four. |
| `ftir_33_shape_cohort_explainers.ipynb` | How were the Ethiopia-shaped-smoke and spectral-analog cohorts selected — the filtering_by_ocec treatment for the shape-based setups? | **Selection explained, failure explained.** Ethiopia-shaped = top 300 smoke filters by band-feature distance to Addis (only 4/906 sit inside the Addis 5–95% box on all three features); analogs = top 500 of the pool by mean percentile of nearest-Addis D² + VIP-weighted RMSE. On the OC/EC ruler both sit at/above the pool median (7.5 and 6.1 vs cut 2.27; overlap with OCEC-800: 1/300 and 17/500) — shape does not find composition, and both fail the locked TOR test. |

| `ftir_41_adama_three_method_reconciliation.ipynb` | The Adama PTFE twins exist (CSU AMOD Batch 54: FTIR + HIPS) — put all three methods and both thermal conventions on one table, with a comparability ledger. | TOR→TOT moves EC −15 to −21% while conserving OC+EC to 1e-13; FTIR OC+EC = 0.40–0.74 of thermal TC (OC-dominated; FTIR EC vs EC_TOT median 0.86); same-corridor implied MAC 16.5–20.8 m²/g — above physical EC but far below Addis's ≈47 |
| `ftir_42_target_definition_experiment.ipynb` | Hold spectra/cohort/protocol fixed on the IMPROVE mirror and swap only the target: TOR EC vs TOT EC vs OC vs TC. | TC is the *easiest* target (held-out R² 0.897 vs 0.870 EC_TOR); EC_TOT is the hard one (0.764; full-pool collapse k=2, R² 0.18); the partition EC/TC itself has ~zero out-of-site skill (R² ≤ 0.007); Addis transfer changes by convention (0.63x−1.24 vs 0.81x−1.53) |
| `ftir_43_residual_learner_null_control.ipynb` | Does predicting the FTIR−HIPS residual from spectra demonstrate information beyond f(X)? The f(X)-preserving null and blocked controls. | Spectra learner reaches time-blocked residual R² 0.244 vs a no-chemistry null q95 of 0.113 (p<0.005) — real incremental signal — but season+volume alone reach 0.184, and in RMSE the constant offset does 71% of the work (2.46→0.72; spectra 0.62). τ-as-metadata gives R²≈0.94 by circularity — excluded by design |
| `ftir_44_adama_through_locked_calibrations.ipynb` | Ann's ask #3: push the five Adama PTFE spectra (AIRSpec-baselined on the pool grid) through the locked and sweep-winner calibrations, against quartz EC_TOR/EC_TOT/OC/TC. | **Over-reads.** Locked 800 + AIRSpec 1.40× EC_TOR (1.76× EC_TOT), winner 440 k=8 2.00×, both inside the HIPS band; deployed 0.69× (0.86×). Adama's OC/EC ≈ 6 is outside the cohort's ≤ 2.27 domain — the Addis-tuned models are Addis instruments. OC/TC refits leave the carbon sum at 0.47 of thermal TC (deployed 0.44): the deficit is upstream of calibration. Spectra id map inferred from FilterId order (ρ = 1.00 CH-vs-OC) |
| `ftir_45_residual_increment_attribution.ipynb` | Where does ftir_43's ≈0.06 R² beyond season live — char_06 anomaly, band heights, or loading curvature? | **None of them.** char_06 flag adds nothing (0.176 vs 0.184; both classes ≈0 after season adjustment), neutral band heights hurt (−0.006), f(X)² adds 0.01; spectra learner 0.244. Coefficient spectrum is derivative-shaped across 1560–1750 cm⁻¹ — band position/shape, not height |

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
