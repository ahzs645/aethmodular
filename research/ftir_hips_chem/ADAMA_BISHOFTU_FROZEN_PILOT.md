# Frozen AIRSpec/VIBES pilot on Adama and Bishoftu

Executed 2026-09-22. Reproduce from the repository root:

```bash
uv run --locked --no-sync aeth doctor
uv run --locked --no-sync python research/ftir_hips_chem/workflows/run_adama_bishoftu_frozen_pilot.py
uv run --locked --no-sync python gallery/data/export_baseline_external_pilot.py
```

The workflow applies the saved **full-pool** IMPROVE AIRSpec and VIBES PLS models from the completed 12,808-case Colab run. It reuses that run's frozen blank background and model coefficients. No Adama or Bishoftu outcome was used to fit, select, or tune a model. The model's source-domain held-out test is 2,327 IMPROVE filters; the new site results below are external diagnostics, not independent site validation.

## Adama: five provisional date-paired thermal comparisons

Five Adama PTFE spectra were corrected with each frozen method and their predicted EC loadings divided by **each PTFE filter's own sampled volume**. The thermal EC reference is measured on a separate, date-candidate quartz co-sample, reported as both TOR and TOT concentrations. The July 9 pair retains its 39.7-minute start-offset flag; July 30 retains its PTFE/quartz volume-mismatch flag. The other three are *sampling-unflagged*, not confirmed co-samples.

| Frozen model | Median prediction / TOR, all 5 | Median prediction / TOR, 3 sampling-unflagged | Median prediction / TOT, 3 sampling-unflagged |
|---|---:|---:|---:|
| AIRSpec full pool | 0.866 | 0.819 | 1.031 |
| VIBES full pool | 0.644 | 0.557 | 0.702 |

These values use the earlier **provisional** assignment of sorted `SampleAnalysisId` 4744–4748 to sorted PTFE FilterIds. The spectra export does not contain an authoritative ID crosswalk. To show how much this matters, the workflow tries all 120 possible assignments while leaving date-pairing and sampler volumes fixed. Across those assignments, the sampling-unflagged median prediction/TOR ranges from **0.658–1.077 for AIRSpec** and **0.485–0.733 for VIBES**. AIRSpec's median ratio is closer to unity in all 120 assignments, but this exercise cannot establish which assignment is true or whether the candidate samplers were equivalent. In the provisional assignment, July 9 is above TOR under both methods (AIRSpec 1.65×, VIBES 1.28×); it should not be silently removed to improve a score.

The earlier ftir_44 locked lowest-OC/EC AIRSpec calibration gave a different direction (1.40× TOR on its sampling-unflagged subset). These are **different fixed training cohorts**, so the contrast is a calibration-selection sensitivity, not a contradiction in the measured quartz values. No five-filter model ranking should be promoted to a site-wide accuracy claim.

## Bishoftu: same-lot, same-month optical-loading comparison

All 26 available Bishoftu PTFE spectra with HIPS Fabs and volume were processed; each is lot 251 and from October–December 2025. Across those 26, median predicted EC concentrations are **1.006 µg/m³ AIRSpec** and **0.901 µg/m³ VIBES**. Median prediction divided by HIPS/MAC-10 optical EC-equivalent is **0.382** and **0.344**, respectively. HIPS/MAC is an optical conversion, not a thermal EC reference.

For a narrower Addis comparison, the workflow selected Addis lot-251 October–December records, restricted both sites to their shared HIPS Fabs range, and made unique nearest-neighbor pairs within a fixed 20% Fabs caliper. This left **five** pairs. In every pair Bishoftu's frozen FTIR-derived EC prediction is higher than Addis's at near-equal HIPS Fabs. Median paired Bishoftu-minus-Addis differences are **+0.683 µg/m³ AIRSpec** and **+0.628 µg/m³ VIBES**. Both corrected 1617 and 2920 cm⁻¹ local-band features are higher on the Bishoftu side of these five pairs. This is a small, non-concurrent, optical-loading-matched spectral observation. It does not show which site has accurate EC, and the bands are not causal chemical identifications.

The earlier spectral exclusion of 1800–2500 cm⁻¹ and optional >3600/>3500 cuts changed **analog selection**. The frozen full-pool models here still use all 2,002 channels from about 1426–3998 cm⁻¹, as did those historical PLS fits. This pilot is not a new masked-analog calibration test.

## Audit and next evidence

All 31 external baseline corrections converged without retry. The workflow checked the exact spectral grid, replayed one previously saved Addis prediction from the portable coefficients, enforced unique physical-filter and pair IDs, and wrote source hashes. The per-filter and matching tables live in [the pilot output directory](output/tables/adama_bishoftu_frozen_pilot/): `adama_candidate_pairs.csv`, `adama_id_mapping_sensitivity.csv`, `bishoftu_frozen_predictions.csv`, `addis_frozen_predictions.csv`, `addis_bishoftu_loading_matched.csv`, `external_fit_diagnostics.csv`, and `manifest.json`.

The gallery's AIRSpec/VIBES tab displays every Adama candidate, an all-filter optical scatter (**233 Addis plus 26 Bishoftu** with HIPS), and an all-spectrum 1617/2920 descriptor scatter (**253 Addis plus 26 Bishoftu**). Its separate paired chart shows only the five strict lot/month/loading matches. The missing 20 Addis HIPS values are disclosed on the optical chart rather than silently counted as optical pairs. The gallery data export includes every plotted filter row and its ID.

The decisive next input for Adama is the lab's authoritative 4744–4748 spectrum-to-FilterId map plus sampler logs for July 9 and July 30. For Bishoftu, thermal EC on co-located quartz filters and later-season FTIR/HIPS coverage are needed before drawing an EC-transfer or seasonal conclusion.
