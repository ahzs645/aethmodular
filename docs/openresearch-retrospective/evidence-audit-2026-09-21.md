# Historical evidence audit and next experiments

The audit is published in the local OpenResearch dashboard: 101 historical records, three next-experiment records and six family headings. Historical notebook execution, errors, external runs, local diagnostics and planned/blocked work have distinct labels. No OpenResearch run rows were fabricated; the existing verified release retains its command, branch, parent and completed run.

## Evidence recovered

All 638 indexed snapshot files match their recorded hashes. The 101 primary sources contain 55 fully counted saved notebook executions without recorded errors, 32 without saved execution, four partial/untracked executions, one saved error and nine documents/manifests. Seven of the 32 unexecuted sources have fully counted archived companions, leaving 25 without a known executed companion in this catalog. Saved counts are not proof of valid results or historical source/environment identity.

Recovered historical narrative appears in 75 records. Another 38 current artifacts were recovered from source-declared output paths, copied and hashed; their historical generation is not established merely by existence. The HIPS smooth/raw notebook stores an ImportError. ftir_49 records skipped minute-resolution work and a daily fallback, so those missing checks cannot support negative scientific findings. Dynamic input expressions and incomplete environment provenance remain explicit in each audit card.

The reproduction inventory now includes 48 located runner recipes, 47 candidate notebook commands, one existing verified release reproduction, one R-reference source, one completed external notebook protocol, one verified local saved-prediction audit and two document/notebook records without standalone commands. Candidate commands have not been executed by this audit.

[All audit cards and saved errors](../../research/ftir_hips_chem/output/tables/openresearch_evidence_audit/report.md) · [Full structured audit and hashes](../../research/ftir_hips_chem/output/tables/openresearch_evidence_audit/audit.json).

## Research connections and claims

The historical graph grew from 31 documented links to 72: 42 source-supported data dependencies, 22 related-work links, five qualifying links and three revisions. The three new experiments add ten links, including the completed VIBES trace's actual saved-data dependency: 82 overall. Multiple relationship types can coexist between the same pair. Registration parents remain unchanged.

[The scoped claim ledger](claim_updates.json) preserves earlier prose while recording later findings: ChemSpec's circularity, the per-deployed-line blank correction, the paired spectral-increment uncertainty, Adama data availability and transfer limits, the limits of fuel attribution at 1617 cm⁻¹, and the cohort-specific pyrolysis/target-difficulty interpretation. A revised claim does not invalidate an entire earlier experiment.

## Three concrete experiments

| Experiment | Completed here | Still required |
|---|---|---|
| [VIBES loading trace](experiments/vibes-loading-trace.md) | Frozen-input diagnostic reconciles all 2,464 saved held-out rows, with no refit or new exclusions | Physical mechanism review using selected cases and solver diagnostics; any method improvement needs training-only selection and new evaluation |
| [AIRSpec reproduction](experiments/airspec-reproduction.md) | All 34 identified files hashed; 319-scan R-port check passes both historical absolute and current relative criteria | Isolated staging and full locked-calibration rerun; this has not run |
| [Independent Addis validation](experiments/addis-independent-validation.md) | Written protocol, validity gates, comparison rules and empty pairing schema | Independent thermal EC, authoritative physical-filter/analysis identities and laboratory uncertainties |

The AIRSpec R-port check reproduced the reported numerical scale: worst absolute error 6.014×10⁻⁷ absorbance. It does not establish Addis prediction accuracy. Adama's five date-paired comparisons and HIPS/ChemSpec products cannot fill the independent Addis reference gap.

## Verification and limits

Seven focused Python tests passed; 191 OpenResearch UI tests, typecheck, localization/style checks and production build passed. The live dashboard displayed all new status labels and the bounded relationship view. Reapplying the next-experiment registration preserved IDs and created no duplicates. Every pre-existing experiment's branch, parent and run command, and all existing run records, were checked unchanged.

[Final catalog verification](../../research/ftir_hips_chem/output/tables/openresearch_evidence_audit/final_verification.json) · [Experiment registration receipt](../../research/ftir_hips_chem/output/tables/openresearch_next_experiments/registration_receipt.json).

This is a saved-evidence audit and two bounded local diagnostics, not a rerun or scientific verification of all 101 historical analyses. No archive-only work outside the imported catalog is claimed audited. OpenResearch's status/relationship UI is a local customization and may be overwritten by an upstream update.
