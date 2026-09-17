# Filter-only scientific results

The frozen inputs reproduce **545 HIPS/FTIR-predicted-EC diagnostic pairs and 480 ratio-eligible pairs**.
Every ratio point is a diagnostic point. The 65-point difference is accounted for below by complete,
mutually exclusive reason strings. Low and nonpositive predictions are retained in diagnostic plots;
their denominators are never replaced. These results require neither ChemSpec adjudication nor aethalometer matching.

## Within-site results

| site        |   diagnostic_n |   ratio_n |   descriptive_r_squared |   descriptive_ols_slope |   descriptive_ols_intercept |   ratio_median | ratio_IQR   |
|:------------|---------------:|----------:|------------------------:|------------------------:|----------------------------:|---------------:|:------------|
| Beijing     |            163 |       150 |                   0.543 |                   5.590 |                       6.212 |         10.149 | 7.84–11.99  |
| Delhi       |             62 |        56 |                   0.688 |                   6.337 |                      13.559 |          8.610 | 7.60–11.66  |
| JPL         |            130 |        84 |                   0.522 |                   8.135 |                       0.692 |          9.507 | 7.70–11.57  |
| Addis_Ababa |            190 |       190 |                   0.764 |                   4.023 |                      28.324 |         10.158 | 8.14–12.69  |

Within-site descriptive R² ranges from 0.522 (JPL) to
0.764 (Addis_Ababa). The fits are unweighted OLS with an intercept,
using each site's complete diagnostic population. Site-specific axes expose each concentration range;
R² comparisons do not isolate instrument performance from concentration range, population or sampling differences.
No pooled cross-site line, 1:1 line, independent EC validation claim or uncertainty-weighted fit is used.
OLS slopes are descriptive, and measurement error in both quantities can affect them.

![Within-site relationships](../../plots/filter_diagnostics/01_site_relationships.png)

HIPS is retained in Mm⁻¹ and the denominator is FTIR-predicted EC in µg m⁻³.
The derived quantity is labeled **HIPS/FTIR-predicted-EC ratio**, with units (Mm⁻¹)/(µg m⁻³).
Although those units are dimensionally m² g⁻¹, this analysis does not establish an independently validated BC absorption efficiency.
The site medians (8.61–10.16) sit within substantially overlapping IQRs; a shared ratio is not established by those summaries.

![Ratio distributions](../../plots/filter_diagnostics/02_ratio_distributions.png)

## Denominator eligibility and sensitivity

| site    | reason                                                       |   count |
|:--------|:-------------------------------------------------------------|--------:|
| Beijing | ftir_ec_below_reported_mdl                                   |      10 |
| Beijing | ftir_ec_denominator_not_positive; ftir_ec_below_reported_mdl |       3 |
| Delhi   | ftir_ec_below_reported_mdl                                   |       4 |
| Delhi   | ftir_ec_denominator_not_positive; ftir_ec_below_reported_mdl |       2 |
| JPL     | ftir_ec_below_reported_mdl                                   |      46 |

The 65 exclusions comprise 60 positive predictions below their reported MDL and 5 nonpositive predictions
that also fall below MDL. These categories are verified from the retained flags, rather than assumed.
The baseline requires a resolved, finite, positive FTIR EC and a nonconflicting reported MDL, with EC ≥ MDL.
HIPS/FTIR ratios are absent outside that population. The full per-point decisions and original row links remain available.

![Ratio versus denominator](../../plots/filter_diagnostics/03_ratio_denominators.png)

The ring annotation describes EC between 1× and <2× its own MDL; it is not a new exclusion.
At JPL, 77 of 84 ratio points fall in this range; a 2× rule leaves only 7. At Addis, it leaves 189 of 190.
The largest Addis ratio (55.41) is its one near-MDL point. The Addis median is stable under stricter rules,
while Beijing's median and cohort size decline. Those changes reflect different retained populations, not a chosen calibration threshold.
This panel shares EC algebraically between the horizontal axis and the ratio denominator, so an inverse pattern
alone is not evidence for a physical mechanism. Group summaries are:

| site        | denominator_group    |   n |   median |   maximum |
|:------------|:---------------------|----:|---------:|----------:|
| Addis_Ababa | 1 to <2 times MDL    |   1 |   55.407 |    55.407 |
| Addis_Ababa | at least 2 times MDL | 189 |   10.147 |    18.652 |
| Beijing     | 1 to <2 times MDL    |  37 |   11.820 |    29.139 |
| Beijing     | at least 2 times MDL | 113 |    9.444 |    23.044 |
| Delhi       | 1 to <2 times MDL    |   4 |    8.843 |    17.499 |
| Delhi       | at least 2 times MDL |  52 |    8.610 |    22.830 |
| JPL         | 1 to <2 times MDL    |  77 |    9.651 |    17.013 |
| JPL         | at least 2 times MDL |   7 |    8.547 |    11.403 |

![Reported-date patterns](../../plots/filter_diagnostics/04_ratio_dates.png)

Dates are reported sample dates. They do not assert active-interval timing; the dashed line is the site's overall median.
There is no temporal interpolation, date adjustment, seasonal-calendar substitution or temporal trend fit.
Site/cohort date ranges, medians and IQRs for HIPS, FTIR EC and ratios are exported in `distribution_summary.parquet`.

![Denominator sensitivity](../../plots/filter_diagnostics/05_denominator_sensitivity.png)

The fixed sensitivity grid is 1, 1.5, 2, 3 and 5 times each filter's MDL, applied within the existing ratio population.
It describes changes in counts and medians; it does not select a new threshold from measurement agreement.
Every retained filter at every threshold is linked in `sensitivity_point_links.parquet`.

| site        |   minimum_ec_mdl_multiple |   n |   ratio_median |
|:------------|--------------------------:|----:|---------------:|
| Beijing     |                     1.000 | 150 |         10.149 |
| Beijing     |                     1.500 | 128 |          9.857 |
| Beijing     |                     2.000 | 113 |          9.444 |
| Beijing     |                     3.000 |  72 |          8.336 |
| Beijing     |                     5.000 |  30 |          7.599 |
| Delhi       |                     1.000 |  56 |          8.610 |
| Delhi       |                     1.500 |  53 |          8.620 |
| Delhi       |                     2.000 |  52 |          8.610 |
| Delhi       |                     3.000 |  44 |          8.564 |
| Delhi       |                     5.000 |  32 |          8.226 |
| JPL         |                     1.000 |  84 |          9.507 |
| JPL         |                     1.500 |  28 |          8.668 |
| JPL         |                     2.000 |   7 |          8.547 |
| JPL         |                     3.000 |   0 |        nan     |
| JPL         |                     5.000 |   0 |        nan     |
| Addis_Ababa |                     1.000 | 190 |         10.158 |
| Addis_Ababa |                     1.500 | 189 |         10.147 |
| Addis_Ababa |                     2.000 | 189 |         10.147 |
| Addis_Ababa |                     3.000 | 189 |         10.147 |
| Addis_Ababa |                     5.000 | 183 |         10.040 |

![Registry context](../../plots/filter_diagnostics/06_registry_context.png)

The existing Delhi registry exclusion is shown for audit, with its original value unchanged. It does not enter the 545 diagnostics.
HIPS uncertainty/MDL records are retained, but their unresolved semantics are not replaced by standard deviations or AIRSpec RMSE.

## Evidence-recovery priorities

| site        |   1 |   2 |   3 |   4 |   5 |
|:------------|----:|----:|----:|----:|----:|
| Beijing     |   0 |   0 |   0 | 163 | 242 |
| Delhi       |   0 |   0 |   0 |  62 |  58 |
| JPL         |  49 |   0 |  81 |   0 | 176 |
| Addis_Ababa | 181 |   3 |   6 |   0 |  34 |

Priority 1: usable HIPS plus hour-consistent reported bounds overlapping an available export.
Priority 2: usable HIPS with overlap but active periods or timing requiring reconciliation.
Priority 3: missing bounds or no overlap; more timing/instrument evidence is needed first.
Priority 4: no staged timestamped export for that site. Priority 5: HIPS/filter identity or registry gating.
These are retrieval priorities, not primary eligibility or observed-coverage estimates.
No filter-linked run log has been located; the queue states that explicitly rather than inventing an evidence source.

First retrieval candidates by site (chronological within priority):

| site        | base_filter_id   |   priority | reported_start_utc        | reported_end_utc          | missing_evidence                                                                      |
|:------------|:-----------------|-----------:|:--------------------------|:--------------------------|:--------------------------------------------------------------------------------------|
| Addis_Ababa | ETAD-0017        |          1 | 2022-12-07 06:00:00+00:00 | 2022-12-08 06:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |
| Addis_Ababa | ETAD-0018        |          1 | 2022-12-10 06:00:00+00:00 | 2022-12-11 06:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |
| Addis_Ababa | ETAD-0019        |          1 | 2022-12-13 06:00:00+00:00 | 2022-12-14 06:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |
| Addis_Ababa | ETAD-0020        |          1 | 2022-12-16 06:00:00+00:00 | 2022-12-17 06:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |
| JPL         | USPA-0257        |          1 | 2023-06-23 16:00:00+00:00 | 2023-06-24 16:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |
| JPL         | USPA-0258        |          1 | 2023-06-26 16:00:00+00:00 | 2023-06-27 16:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |
| JPL         | USPA-0259        |          1 | 2023-06-29 16:00:00+00:00 | 2023-06-30 16:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |
| JPL         | USPA-0260        |          1 | 2023-07-02 16:00:00+00:00 | 2023-07-03 16:00:00+00:00 | filter-linked continuous-operation evidence and scoped observation/processing history |

**ETAD-0243 is priority 3:** its reported start is 2024-08-24 06:00:00+00:00,
after the current Addis export ends at 2024-08-20 09:01:00+00:00. Recovering its schedule alone cannot unlock a comparison with that export.
The queue retains each filter's source bounds, instrument-file hash, serial/firmware metadata and the precise evidence gate.
Where the frozen catalog had no portal start, recovered 2025 exports now supply separately labeled retrieval bounds;
they do not change diagnostic dates, eligibility or active schedules. The earlier collection-bound table is preserved in full.
A reviewed real interval subset remains unavailable; no affirmative provenance booleans or inferred on/off periods were supplied.
An additional `aethalometer_combined.db` was found in the Drive instrument directory. Its schema was readable, but
the read-only grouped range query did not complete within roughly four minutes and was canceled. Its coverage and processing history
remain unverified; no claim of missing later records is inferred from that access limitation.

## ChemSpec source-to-output trace

The earlier local exports in `research/filter_combine/` are labeled File Updated 2025-07-29, Data version 3.0.
All **1,043 current ChemSpec EC rows** match an earlier row's value and MDL, with filter identity and method preserved.
Their parameter code is 28203, analysis description FTIR, concentration units µg m⁻³ and conditions Ambient local.
Method codes 217/218 are retained; their detailed calibration definitions remain unrecovered.

The recovered historical importer at commit `c91542c705254dd7e7e7271bc36789388d225dc3` maps `Value` directly to `Concentration`,
`MDL` separately to `MDL`, and `Parameter_Name` to a `ChemSpec_` label. It appends one output row per input row;
the inspected importer does not melt MDL into additional concentration rows. It does discard some source metadata,
which this trace now preserves. No parser correction is justified by the observed duplication at this stage.

For CHTS-0658, earlier source rows 5883 and 5884 already contain 0.93 and 0.06 in the **Value** field under
the same code, FTIR description and method. They map to current unified rows 5859 and 5860, respectively.
Thus the competing values predate this importer. Their upstream export-generation code or role definitions
are still needed to decide whether one Value is actually an MDL. Neither value is selected by magnitude or MDL equality.

The historical code is saved read-only as `historical_filter_integrator.txt`, with its commit and content hash in
the row trace. Original earlier rows and all source-to-output links are exported. This supersedes the previous
inventory gap about EC parameter codes; it does not establish an authoritative concentration within each conflicting group.

## Reproduce and inspect points

Run `uv run python research/ftir_hips_chem/workflows/analyze_filter_diagnostics.py` from the repository root.
`analysis_points.parquet` provides one row per physical filter with point IDs, original HIPS/FTIR row links,
reported dates, flags, ratios and denominator multiples. `original_measurements.parquet` preserves the source rows
and unified-source hash. All six figure families use these same points; the registry panel additionally shows its retained exclusion.
`manifest.json` records input, code, table and figure hashes. No calibration is fitted.

## Narrow implementation review

Reviewed processing evidence now requires an explicit full-export or bounded-period scope and a scope justification.
Bounded assertions cannot certify a filter interval extending beyond their UTC bounds; optional session-ID restrictions
must cover all input rows in the interval. Out-of-scope observations retain input availability but unknown observed coverage and eBC means.
This is a scope guard, not evidence that any current research stream has been reviewed.

The 75% overall / 50% per-segment policy is unchanged. Successive quarters of each active interval now receive
descriptive input and valid-observed coverage summaries. Their values do not affect eligibility.
For a single continuous interval, the per-segment rule is redundant with the total rule; quarters expose concentrated gaps.
Real quarter coverage remains unavailable while active schedules and observation histories remain unresolved.
