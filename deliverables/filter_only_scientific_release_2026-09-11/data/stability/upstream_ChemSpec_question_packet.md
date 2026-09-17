# Draft upstream question — not sent

In the data-version-3.0 export updated 29 July 2025, CHTS-0658 has two rows for EC
parameter 28203 with the same method and MDL, but Value entries of 0.93 and 0.06.
Both precede our importer. What distinguishes the result roles of these rows,
which field or key identifies that distinction, and what export-generation rule
produces them? Please provide the result-role definition and any key omitted from
this export, without assuming that either row is the preferred concentration.

## Exact evidence attached

- [Original source rows](CHTS-0658_original_source_rows.csv), with original preamble and header.
- [All source metadata](CHTS-0658_source_metadata.csv).
- [Mapping to unified measurement rows](CHTS-0658_unified_row_links.csv).
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
