# Coverage of the frozen baseline and consolidated report

The baseline represents the completed **filter-only** results and their audit
trail. It does not mean every part of the earlier measurement-comparison roadmap
has been validated.

## Three different meanings

| Term | Meaning in this release |
| --- | --- |
| Frozen study baseline | Saved physical-filter identities, reported values, flags, population memberships and specified analysis rules. These are kept fixed across the reported comparisons. |
| Prediction baseline | HIPS training median, estimated within each training fold. The proportional and intercept-bearing fits are separate comparisons. |
| FTIR spectral baselining | Upstream spectral preprocessing. This release does not reconstruct or verify it; the exact FTIR product definitions and processing mappings still need authoritative evidence. |

## Agreed work and where it appears

| Part of the work | Representation in this release | Status and limit |
| --- | --- | --- |
| Physical-filter identities and selection | [Frozen points](data/diagnostic/analysis_points.parquet), source-row links, manuscript Methods/Table 1 | Frozen and reproduced for the filter-only analysis. These identities do not establish active collection intervals. |
| Diagnostic relationships and denominator sensitivity | [Diagnostic report](phase_reports/diagnostic_results_report.md), Figure 1, saved MDL memberships and figures | Included; diagnostic and ratio populations remain distinct. |
| Withheld-quarter and later-period stability | [Stability report](phase_reports/stability_filter_relationship_stability_report.md), Results and block tables | Included; the two schemes revisit overlapping records and are not independent replications. |
| Proportional versus intercept-bearing prediction | [Proportionality report](phase_reports/proportionality_proportionality_temporal_transfer_report.md), Table 2 and Figures 2–3 | Completed under the frozen specification; site differences and failures are retained. |
| ID-11 model form and training composition | [ID-11 summary](data/proportionality/id11_training_summary.parquet), Results and Figure 4 | Both findings included separately; identifier definitions remain unresolved. |
| Weighting, bias and influence | Table 3, full block ledgers and the stability influence outputs | Included; no further error-driven exclusion or weighting selection. |
| Graphs and detailed notes | [Portable notebook](notebooks/filter_only_results.ipynb), main figures and all three phase figure folders | Included. Earlier slide decks remain historical outputs; they have not been refreshed by this consolidation. |
| Upstream EC roles, FTIR processing definitions and training membership | [Existing packet](phase_reports/proportionality_upstream_questions_v2_draft.md), manuscript scope discussion | Reviewed but unsent; recipient/channel and authoritative responses are still needed. No new metadata correction is made. |
| Active collection, clocks, session-46 processing and export scaling | [USPA-0257 evidence package](phase_reports/proportionality_USPA-0257_evidence_package.md) | Candidate review remains unresolved. The limited status screen is not a quality approval or proof of observed coverage. |
| Aethalometer/filter interval and absorption comparison | Candidate evidence and preserved numerical modules | Not a completed scientific comparison. A qualifying interval still requires the documented evidence gates; optical conversion must be established for an absorption comparison. |
| Independent EC validation and instrument calibration | Explicit scope limits in the manuscript | Not established by this filter-only analysis or by a future single accepted interval. |
| Reproduction outside the original checkout | [Entry point](reproduce.py), [pinned environment](requirements.lock), manifests and release checks | Reproduces frozen downstream analyses; it does not recreate FTIR predictions from raw spectra. |

The consolidated narrative is the [scientific draft](manuscript.md). The
[claim ledger](claim_ledger.csv) and [figure ledger](figure_ledger.csv) identify
the frozen tables, selections and hashes supporting the reported results.
