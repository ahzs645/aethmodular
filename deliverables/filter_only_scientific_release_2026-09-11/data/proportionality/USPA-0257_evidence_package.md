# USPA-0257 / session 46: candidate-specific evidence package

No verified interval eBC or absorption comparison is produced. The existing SQLite
retrieval is retained; no new database query or broad inventory was run.

| requirement | status | evidence | remaining |
| --- | --- | --- | --- |
| Physical filter and reported collection envelope | source_linked_reported_bounds | USPA-0257; 23 June 2023 16:00 to 24 June 2023 16:00 UTC; 24 reported hours; SS5i sampler | Confirm continuous operation or obtain filter-linked active segments and clock basis |
| Contemporaneous instrument identities | matched_records | 1,440 timestamp/datum matches and exact IR BCc values; MA350-0229 session 46, format 3, firmware 1.12, app 1.6 | Record agreement alone does not certify observed-data provenance |
| Observation/correction history | unresolved_for_session | DualSpot on is reported; cleaned CSV and SQLite agree | Producer trace from original download to cleaned CSV and database; transformations, removed/inserted records, time corrections and applicable dates/sessions |
| Applicable quality decision | candidate_screen_not_approved | Status 131648 on all rows; zero triggers in existing local startup/tape/optical/timing/flow-status screen | Confirm session-specific applicability; check original flow/optics and exclusions; absence of status flags does not prove overall validity |
| Timestamp alignment | manual_time_source_reported | Status decomposes to 64+512+131072, matching saved labels | Clock setting, drift/offset corrections and alignment with sampler timestamps |
| Exported IR BCc units | nominal_manufacturer_definition_identified | Indexed official May 2023 manual defines format-3 IR BCc as DualSpot-compensated mass concentration in ng/m3 | Confirm that the specific cleaned export retained native scale; full manual fetch returned 404, so the PDF was not archived |
| Absorption conversion and wavelength treatment | not_verified | No reviewed conversion applied | Document optical conversion and wavelength treatment after the eBC evidence chain is supported |

[Filter identity, reported bounds and original portal row links](USPA-0257_filter_evidence.json)
connect the physical filter to its source records. [The retained crosswalk](frozen_inputs/USPA-0257_record_crosswalk.parquet)
and [original targeted retrieval](frozen_inputs/USPA-0257_candidate_retrieval.json)
contain source hashes and database row IDs.

## Status and quantity definitions

The source labels on every row are DualSpot on, Time source manual and Ext. power.
Their summed code is 131648 = 64 + 512 + 131072. The manual's indexed status table
identifies those as active second-spot operation, manual/computer time source and
external power. It identifies format-3 IR BCc as mass concentration with DualSpot
loading compensation in ng/m³. [AethLabs operating manual, May 2023, sections 6.1/6.3](https://aethlabs.com/sites/all/content/microaeth/maX/MA200%20MA300%20MA350%20Operating%20Manual%20Rev%2005%20May%202023.pdf).
The full PDF returned 404 during direct retrieval; these are retrieved official
index excerpts, corroborated for status by the local code, not an archived PDF.
The vendor also documents DualSpot compensation on its [MA350 product page](https://aethlabs.com/products/ma350).

The existing local `remove_tape_advance` and `remove_concerning_statuses` functions
screen startup, tape advance, flow instability, optical saturation and sampling
timing errors. [Local source](/Users/ahmadjalil/github/aethmodular/src/external/calibration.py:104); SHA-256 `d18ed9c836f965a19221b2da3bce30538adcd08fbf7638e1df476e88b02854e0`.
All 1,440 retained rows have zero triggers in this **candidate screen**. No rows
were removed. Its session-specific applicability and the treatment of flow/optics
outside status flags are not established. The local routine's treatment of missing
status as manual time is also not evidence of original measurement validity.

[Per-record flags](USPA-0257_candidate_status_screen.parquet) retain timestamps,
datum identities and raw values. Passing this screen does not prove that rows
were observed, that clocks agree, or that upstream removals/interpolation are known.

## Exact remaining request — not sent

For physical filter USPA-0257 (alias USPA-0257-1), provide sampler continuous-mode
confirmation or active on/off records for its reported envelope and the clock/timezone
basis. For MA350-0229 session 46, provide the original download and the producer's
processing log linking it to the cleaned CSV and database: time corrections,
calibration/flow changes, smoothing, deleted/inserted records and quantity scaling.
Identify the approved quality rule and its applicability to this session; supply
supporting flow/optical checks and any previous exclusions. Confirm whether IR BCc
retained the manufacturer's native ng/m³ scale.

Valid observations can include documented, appropriate corrections. The target is
supported correction history, not an assumption that every valid value is uncorrected.
The evidence is scoped to this candidate; no whole-export certification follows.
