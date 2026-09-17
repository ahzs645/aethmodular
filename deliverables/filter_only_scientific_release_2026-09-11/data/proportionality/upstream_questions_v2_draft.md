# Draft upstream question — not sent

In the data-version-3.0 export updated 29 July 2025, CHTS-0658 has two rows for EC
parameter 28203 with the same method and MDL, but Value entries of 0.93 and 0.06.
Both precede our importer. What distinguishes the result roles of these rows,
which field or key identifies that distinction, and what export-generation rule
produces them? Please provide the result-role definition and any key omitted from
this export, without assuming that either row is the preferred concentration.

## Exact evidence attached

- [Original source rows](frozen_inputs/CHTS-0658_original_source_rows.csv), with original preamble and header.
- [All source metadata](frozen_inputs/CHTS-0658_source_metadata.csv).
- [Mapping to unified measurement rows](frozen_inputs/CHTS-0658_unified_row_links.csv).
- [Frozen original measurements](frozen_inputs/original_measurements.parquet), indexed by source_row.

Source: [FilterBased_ChemSpecPM25_CHTS.csv](/Users/ahmadjalil/github/aethmodular/research/filter_combine/FilterBased_ChemSpecPM25_CHTS.csv)

SHA-256: `ec9a6c24308cb050a58472350a75689bc865751b1ac784a2f02e7bd56cedae61`.

| Source data row (zero based) | File line (one based) | Value | MDL | Unified source_row |
|---:|---:|---:|---:|---:|
| 5883 | 5888 | 0.93 | 0.06375 | 5859 |
| 5884 | 5889 | 0.06 | 0.06375 | 5860 |

Both: Method_Code 217; Parameter_Code 28203; EC PM2.5; Analysis_Description FTIR;
units µg/m³; Conditions Ambient local. Reported start/end: 5–13 July 2022 at
09:00 local, Hours_sampled 24. These bounds do not establish an active schedule.
Blank Analytical_MDL, UNC and Flag fields do not resolve result roles.

The recovered importer assigns Value to Concentration once per source row. The
competing values already occur in the upstream Value field; it did not create the
second value by flattening MDL. Historical commit:
`c91542c705254dd7e7e7271bc36789388d225dc3`; recovered importer SHA-256:
`a6fda4ea34e7c6d5f3c058ccd56e23d36c2afe07fee3dd8ee9072a80eeae197f`.
Both values remain preserved; no active parser correction or authoritative-value
selection is proposed. ChemSpec's FTIR description does not establish an independent EC reference.

## Additional FTIR prediction provenance questions — draft, not sent

For the source EC_ftir predictions associated with CalibrationSetId 11 and 17,
please provide:

1. The definition of each identifier and the dates/filters to which it applies.
   Are these calibration sets, model versions, reporting branches or another concept?
2. The prediction model/version mapping for each ID, including any preprocessing,
   analytical method changes and the exact upstream file/version containing these predictions.
3. The reference EC target and its measurement method; the training population,
   training dates, model fitting/tuning/selection procedure and evaluation design.
4. The original training, tuning and independent evaluation membership for each
   attached physical filter, including whether replicate spectra or related filter
   records cross those roles. Please distinguish downstream use here from original
   FTIR model evaluation membership.
5. Whether LotId denotes a physical filter lot, analytical batch or another field;
   if a true analytical-batch key exists, provide its definition and row linkage.

[Exact filter and original EC source-row links](FTIR_calibration_identifier_filter_links.csv)
identify the 545 diagnostic filters. The IDs below are reported identifiers; their
observed date ranges do not establish official applicability periods.

| site | ftir_CalibrationSetId | n | first_reported_date | last_reported_date |
| --- | --- | --- | --- | --- |
| Addis_Ababa | 17 | 34 | 2022-12-07 00:00:00 | 2023-03-22 00:00:00 |
| Addis_Ababa | 11 | 156 | 2023-03-29 00:00:00 | 2024-09-21 00:00:00 |
| Beijing | 17 | 20 | 2022-07-05 00:00:00 | 2023-04-09 00:00:00 |
| Beijing | 11 | 143 | 2023-01-10 00:00:00 | 2024-12-08 00:00:00 |
| Delhi | 17 | 15 | 2022-07-17 00:00:00 | 2023-02-12 00:00:00 |
| Delhi | 11 | 47 | 2023-03-08 00:00:00 | 2024-06-30 00:00:00 |
| JPL | 11 | 116 | 2022-07-22 00:00:00 | 2023-11-14 00:00:00 |
| JPL | 17 | 14 | 2023-02-23 00:00:00 | 2023-04-03 00:00:00 |

We have not treated IDs 11/17 as documented model versions, mapped ChemSpec
217/218 to them, subtracted group residual means or relabeled any predictions
corrected. The downstream proportionality and ID-11 training comparisons do not
establish original FTIR training independence. The CHTS-0658 result-role question
above remains open and both source values remain preserved.
