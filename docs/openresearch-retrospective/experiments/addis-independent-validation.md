# Addis: independent thermal EC validation

## Question
How accurately do frozen FTIR EC predictions transfer to independent, identity-confirmed Addis thermal EC measurements?

## Current evidence and blocker
No independent Addis thermal EC reference was established in the audited catalog. ChemSpec EC mirrors the FTIR EC product; ChemSpec BC mirrors HIPS/MAC. Neither is independent validation. The five Adama date pairs are different quartz/PTFE filters, have two comparability flags and an analysis-ID mapping awaiting authoritative confirmation. They are a limited external diagnostic, not a substitute for Addis observations.

## Fixed inputs before unblinding
Use the existing ftir_40 proposal as a starting sampling design: 36 primary pairs (13 Dry, 12 Belg, 11 Kiremt), six field blanks and six collocated duplicates. This is a proposed design, not acquired data or a guarantee of power. Freeze the `dry_feb` calendar from config and report it explicitly.

Populate [addis-pairing-template.csv](addis-pairing-template.csv) with a stable pairing ID, separate physical quartz/PTFE filter IDs, authoritative FTIR analysis IDs and provenance, sampling overlap and time zone, instrument/site, flow/volume, laboratory method, blanks, MDLs and uncertainties. Confirm sampling equivalence and identity before pairing any numerical results. Date matching or ranked chemical concentrations alone are insufficient. Keep every record with explicit eligibility flags and reasons; apply canonical exclusion helpers when the data are integrated.

Hash the eligible pairing manifest and frozen deployed, AIRSpec and VIBES predictions before thermal results are unblinded. Hash returned laboratory tables separately. TOR EC is the primary reference; TOT is a method-sensitivity comparison, not interchangeable ground truth. HIPS is a secondary optical comparison. Keep loading (µg/filter) and concentration (µg/m³) conversions explicit with filter-specific volume.

## Comparison method
No calibration fitting or sample selection using the thermal test outcomes. Compare each frozen predictor to TOR using RMSE, MAE, bias and predictive R², with date-block paired bootstrap intervals and seasonal summaries. For separately sampled PTFE and quartz filters, use µg/m³ based on each filter's confirmed volume as the primary paired metric; retain both original masses in µg/filter. Report paired differences against the frozen deployed prediction, sample counts, uncertainty and below-MDL handling. Use an errors-in-variables comparison alongside OLS where displaying a 1:1 line; derive its variance ratio from documented laboratory/prediction uncertainty and include sensitivity if uncertainty is not known.

## Success criteria and decision
A completed experiment requires independently measured thermal EC, authoritative IDs, comparable sampling, frozen predictions and eligibility rules before unblinding, reproducible unit conversions, all declared metrics/intervals and an auditable exclusion trail. Missing identity or independent reference fails the validity gate.

Evidence of improved predictive RMSE requires the entire 95% paired interval for candidate-minus-deployed RMSE to lie below zero; otherwise improvement remains inconclusive. Report season-specific limitations. Application-specific absolute accuracy requirements still need scientific agreement before unblinding; this protocol does not invent a validated accuracy threshold or promise favorable results.

## Reproduction and readiness
The existing 253-filter ETAD cohort now has a frozen, hash-checked AIRSpec/VIBES prediction table. It includes predictions in µg/filter for all 253 filters, recovered from the saved corrected spectra and full-pool coefficients; the 247 previously reported concentration predictions reconcile with the completed Colab export. Six filters lack a usable original sample volume and therefore still have no concentration prediction. This is **not** a new validation result. [Freeze procedure, files and limits](../../addis-validation-freeze.md).

From the repository root, reproduce the freeze with `uv run --locked --no-sync python research/ftir_hips_chem/workflows/freeze_addis_validation.py freeze`. Once authoritative pairing metadata exist, fill every row of the generated template with eligibility and reason, then run `lock-pairs --pairs /path/to/pairing_before_TOR.csv` **before** entering thermal outcomes. Add TOR results to an unchanged copy and run `check-pairs --pairs /path/to/pairing_with_TOR.csv`. The evaluator rejects changed predictions or eligibility, unconfirmed identities or sampling, missing laboratory provenance, a non-TOR primary protocol, missing uncertainty/MDL, and below-MDL pairs without a fixed handling rule. With valid independent input it writes paired metrics and a date-block interval; without input it computes no accuracy.

The freeze applies only to these **historical 253 PTFE filters**. A future 36-pair campaign needs its own prediction freeze on those new spectra before TOR results are opened. No laboratory contact, acquisition or new sampling is implied here. Application-specific absolute-accuracy limits remain to be agreed before unblinding.

## Local data search completed

The local Davis-data inventory covered 753 files, and seven relevant exports were read and hashed. The IMPROVE TOR table contains zero ETAD rows. No authoritative mapping from the five Adama spectral row IDs to physical filter IDs was located. [Readiness evidence and unsent data request](../../../research/ftir_hips_chem/output/tables/addis_validation_readiness/report.md). Independent Addis validation remains blocked; five date-paired Adama candidates cannot substitute for it.
