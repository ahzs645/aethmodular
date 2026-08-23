# Untapped analyses possible with data already in hand — survey 2026-08-20

A very-thorough repo sweep (agent-assisted) of what data exists vs what has
been analyzed. Ranked by value; every item needs NO new data or external
access. Companion docs: `SPECTRAL_SIMILARITY_LITERATURE.md` (methods survey
behind the explorer's Analogs lab), `calibration_explorer/VARIANT_RESULTS_2026-08-19.md`.

## Ranked list

1. **The offset across all four SPARTAN sites — is 21.5 Mm⁻¹ Addis-specific?**
   Compute C = |intercept|·MAC/slope for CHTS (Beijing), INDH (Delhi), USPA
   (JPL) alongside ETAD from the same-filter Fabs↔EC_ftir pairs (546 pairs:
   ETAD 190 / CHTS 163 / USPA 130 / INDH 63) in
   `research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl`, on
   phase-3 conventions (Deming, both MACs, bootstrap CI). The committed
   manuscript table already hints (Addis intercept 2.83 vs 0.06–0.93
   elsewhere) but at native conventions. The single cleanest discriminator
   among the three intercept explanations. ~1–2 days.
2. **HIPS reflectance channel — separate scattering from absorption.**
   `HIPS_R1/T1/r/t` exist for 224 ETAD filters and are referenced by ZERO live
   code; Fabs correlates 0.91 with tau but only 0.22 with −ln r. Invert (R,T)
   for a per-filter scattering term; regress phase-3 residuals on it. The only
   in-hand route to test the HIPS-artifact branch from inside the instrument.
   ~2–3 days.
3. **Mine the 3,664-row batch grid** (functional ANOVA / permutation
   importance of cohort/cutoff/space/spectra/mode/lot/k on intercept, slope,
   held-out R²; Pareto stability). Turns "baselining is the big lever" into a
   variance share. Also close two grid holes: eval_lot never varied, target
   never left addis. ~0.5–1 day.
4. **Lot-specific HIPS blank line (x-axis lot test).** ETAD blank-line coeffs
   differ by lot (248: slope −2.544/int 1383.2, n=40; 251: −2.783/1416.2,
   n=184). Set B tested lots only on the FTIR training side; recompute Fabs
   under a common blank line to test whether the eval-lot-248 anomaly is
   optical calibration rather than aerosol/season. ~1 day.
5. **Recover the 34 orphan ETAD filters** (224 have full HIPS optics, only 190
   have published Fabs): back out Fabs from tau + lot blank line → +18%
   evaluation set, disproportionately lot 248. Reconcile with the newer Drive
   export (280/239). ~1 day.
6. **ftir_24 — dry vs non-dry with a season×x interaction** (Ann's explicit
   ask; spec exists in ANN_DECK_TASKSPEC_2026-08-12.md slide 07; committed
   numbers suggest her hypothesis is inverted for the raw model). ~1 day.
7. **Four-site FTIR functional-group comparison** — control ftir_17's "Addis
   is a deficit" against CHTS/INDH/USPA on identical SPARTAN protocol (not
   just vs IMPROVE). Pairs with #1. ~1 day.
8. **Join the charcoal typology to the calibration residual** (char_06
   per-filter anomaly class + char_12 extrapolation fraction × ftir_22
   residuals — never joined; also supplies ftir_24's split justification).
   ~1–2 days.
9. **MA350 raw dual-spot reanalysis** — recompute b_abs from raw
   ATN1/ATN2/Sen1/Sen2/Ref (1,047 Jacros days, 320 cols) bypassing DualSpot;
   tests whether AAE=0.944 (unphysical) is instrument or algorithm.
   Highest upside, most likely to end negative. ~3–4 days.
10. **Per-filter weighted/robust Deming with real HIPS uncertainties**
    (HIPS_Uncertainty populated for 190 ETAD filters, 1.82–5.68 Mm⁻¹,
    heteroscedastic; replaces pooled λ*=2.96). Already item 1 on the explorer
    roadmap. NOTE: docs/open-items.md "obtain HIPS uncertainties" is STALE —
    they are populated. ~0.5–1 day.
11. **Source-apportionment factors vs residual** — regress residuals on the
    5-factor ETAD contributions (102 dated days, never touched by phase 3) and
    ChemSpec dust proxies (Al/Si/Fe/Ti, K⁺; 188 filters) → upper bound on
    dust's share of the offset. Exploratory. ~0.5–1 day.
12. **Assemble the Adama-vs-Addis 2–3 slides** — all numbers committed
    already (adama_tor_ocec_summary, ftir16 adama tables,
    ftir_ec_calibration_2026_06_25 tables). ~0.5 day.

## Orphaned / unused assets flagged

- **BRANCH DIVERGENCE (action needed):** ftir_25–28, quartz_tor_campaign
  one-pager, run_ftir_25 script, and the Ann/Satoshi decks exist ONLY on
  `origin/claude/data-analysis-u9r3h7`; main has ftir_29/30, AQRC notes,
  package_trials that the branch lacks. Neither is a superset; PHASE3_SUMMARY
  on main doesn't mention 24–28 (the 21.5 Mm⁻¹ reframe is invisible from
  main). Merge or cherry-pick soon.
- HIPS raw photometer columns (R1/T1/r/Slope/Intercept): 224+459 filters, no
  live code touches them.
- `df_Jacros_9am_resampled.pkl`: 1,047 days × 320 raw MA350 columns, used once
  for an assertion.
- `ETAD Factor Contributions .csv`: 5-factor apportionment, 102 days, unused.
- CHTS/INDH/USPA unified-dataset rows (33k) used only for manuscript figures.
- `ftir_34` is referenced in talking_points as "next commit" but doesn't exist.
- ftir_14's Delhi/Beijing comparison is only *half*-blocked: chemistry + FTIR
  products for INDH/CHTS are local; only raw spectra are missing.
