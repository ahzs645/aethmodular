# Research follow-through — 2026-09-21

The four requested workstreams have been worked through locally and their results published in OpenResearch. Independent Addis validation remains blocked on measurements and authoritative identities.

| Workstream | Result | Evidence |
|---|---|---|
| AIRSpec full reproduction | Unchanged ftir_13 runner completed in an isolated checkout. Five saved result tables pass all 43 column checks at 1e-6 tolerance; largest numeric difference 2.47e-13. Both OCEC fits recover k=5. | [Report](../../research/ftir_hips_chem/output/tables/airspec_locked_reproduction/report.md) |
| VIBES loading discrepancy | Twelve predeclared cases inspected and their corrections repeated with original settings. The two largest penalty contributors are already severe negative predictions under AIRSpec, and VIBES makes them more negative. All repeated corrections converge; maximum prediction shift is about 0.002 µg/filter. | [Report and plots](../../research/ftir_hips_chem/output/tables/vibes_case_investigation/report.md) |
| Independent Addis validation | Inventoried 753 local data files and inspected seven relevant exports. No independent ETAD thermal reference or authoritative Adama spectral-ID crosswalk established. Prepared a five-row confirmation queue and an unsent data request. | [Readiness report](../../research/ftir_hips_chem/output/tables/addis_validation_readiness/report.md) |
| Historical gaps | Repaired the smooth/raw notebook; all 13 code cells execute without errors. Triaged all 25 sources without an executed companion: three P1, fourteen P2, eight P3. Those 25 remain unexecuted. | [Repair and priority queue](../../research/ftir_hips_chem/output/tables/historical_gap_followthrough/report.md) |

## What the results mean

AIRSpec's calibration calculations are reproducible in the current environment using the frozen corrected caches. This does not rerun full-pool baseline correction or establish independent Addis accuracy. The previous 319-scan R-port check separately supports the baseline implementation.

The VIBES investigation does not support a gross convergence failure explaining the two leading cases. The spectra and fitted coefficients produce large opposing contributions, and the overall result is a more negative prediction. A chemical cause is not established. Keep both filters in evaluation; no parameters were tuned, predictions clipped or exclusions added to improve performance. Any improvement needs training-only selection and new evaluation data.

The local Addis search does not establish that thermal data cannot exist in laboratory systems or correspondence. The prepared request specifies the missing measurements, uncertainties, physical-filter crosswalk and sampling logs. It has not been sent. The July 9 Adama pair has a 39.73-minute start offset; the July 30 pair has a PTFE/quartz volume ratio of 0.456. Neither flag has been resolved by assuming that dates alone establish equivalence.

The repaired smooth/raw notebook uses the canonical import/style, explicit exploratory threshold flags, paired finite-value handling and Deming slope sensitivity. FTIR-as-thermal and causal interpretation overstatements were corrected. Its legacy ±1-day matching remains clearly labelled; successful execution does not establish true active sampling overlap. Historical snapshots and their saved error remain intact.

## Verification

The AIRSpec output comparison, all observed input hashes, VIBES source identity and repeatability, and the final archived notebook's 13/13 executed cells were checked. The dashboard's 192 UI tests, localization/style checks, typecheck and production build passed. The live AIRSpec entry displays “Local reproduction verified.” Existing OpenResearch run records remain separate from these evidenced local executions; no run rows were fabricated.

Current scripts and generated evidence are in the workspace. The original research checkout was not switched; the detached AIRSpec worktree remains at `/tmp/aeth-airspec-reproduction-20260921`, with durable result copies in the repository output directory. The OpenResearch UI remains a local customization.
