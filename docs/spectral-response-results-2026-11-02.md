# Which response does the spectral association track?

## Result

On **232 Addis physical filters with current, directly exported HIPS absorption
and reported FTIR EC**, the raw and removed signal near 1617 cm⁻¹ are much more
strongly associated with **reported FTIR EC** than with **measured HIPS Fabs**,
after the same metadata adjustment.

For the removed component at 1617 cm⁻¹, the partial rank correlations are:

- Reported FTIR EC: **r = +0.841, r² = 0.707**.
- Measured HIPS Fabs: **r = +0.483, r² = 0.233**.
- Reported FTIR EC minus HIPS/10: **r = +0.582, r² = 0.339**.

This gives a concrete reason to avoid treating a spectral association with
FTIR–optical disagreement as independent proof of a baseline error. The
FTIR-derived response already contains information from spectra. The screen
can recover that estimator's behavior alongside aerosol and background
structure. It does not establish which instrument is correct or whether the
removed signal should be restored to an EC model.

## Data, processing and the common cohort

The calculation uses the same four original Addis files as the native audit:
`ETAD_FTIR_spectra.csv`, `ETAD_metadata.csv`, `etad_spectra_lotmap.csv` and
`spectra_baselined_AIRSPEC.csv`. It reuses `load_native` and the prespecified
`features` definitions from
[`run_native_addis_baseline_audit.py`](../research/ftir_hips_chem/workflows/run_native_addis_baseline_audit.py).

There are **319 scans of 296 physical filters**: 253 PM2.5 filters, 37 field
blanks and six PM10 filters. Twenty-two physical filters have repeat scans,
giving 23 acquisitions beyond one scan per filter. All scan identities remain
in the provenance output.

The saved signed baseline correction is evaluated per acquisition before
physical-filter averaging. Raw spectra are averaged by the same physical
identity. No spectra are clipped or area-normalized, no baseline is fitted
again, and no EC coefficients are refitted. Known review items, including the
nearly flat ETAD-0037 acquisition, the discordant ETAD-0139 repeat and the
documented ETAD-0158 damage question, remain explicitly flagged. This screen
does not select scans using their agreement with HIPS.

Physical-filter IDs are joined exactly to the current committed PM2.5 chemistry
export. Source eligibility and published exclusion reasons are preserved. The
native and current collection dates must agree exactly. Public-HIPS extension
values cannot substitute for directly exported HIPS Fabs.

| Native-ledger outcome | Physical filters |
|---|---:|
| Eligible direct-HIPS, current-EC PM2.5 pairs with matching dates | 232 |
| Public-HIPS extension pairs retained separately | 14 |
| Current measurement pair ineligible | 1 |
| Native physical filters without a current PM2.5 chemistry match | 49 |
| Total retained native physical filters | 296 |

All eight indicators and all three responses have complete data on the same
**232-filter cohort**. The workflow nevertheless calculates completeness
separately for each indicator and requires identical membership across its
three targets. Every membership decision and a sorted-ID cohort hash are
written to the output, so future data updates cannot silently create
different comparison populations.

The current 232-filter lot categories are 248 (34 filters), 251 (154 filters)
and unknown (44 filters). Unknown is an explicit category rather than an
inferred lot assignment.

## Indicators

These indicators were specified from the existing investigation before running
this response screen. They are exploratory choices informed by prior results,
not a new independent hypothesis-confirmation set.

| Indicator | Definition |
|---|---|
| Raw 1617 | Raw absorbance at the nearest native channel, 1617.369 cm⁻¹ |
| Corrected 1617 | Saved signed spline-corrected absorbance at that channel |
| Removed 1617 | Raw minus saved corrected absorbance at that channel |
| Raw broad contrast | Mean in 1580–1650 cm⁻¹ relative to local flanks 1500–1540 and 1760–1800 cm⁻¹ |
| Raw narrow contrast | Mean in 1600–1635 cm⁻¹ relative to flanks 1540–1570 and 1660–1690 cm⁻¹ |
| Raw C–H contrast | Mean in 2850–2950 cm⁻¹ relative to flanks 2750–2800 and 3000–3040 cm⁻¹ |
| Raw O–H/N–H contrast | Mean in 3200–3400 cm⁻¹ relative to flanks 3080–3130 and 3500–3550 cm⁻¹ |
| Raw high-wavenumber continuum | Mean raw absorbance in 3500–3900 cm⁻¹ |

The local contrasts use the existing line-through-flanks calculation; their
names describe spectral regions, not uniquely identified compounds or sources.
The saved diagnostic baseline representation is not proof that the current
reported EC product used those exact preprocessing settings.

## Identical adjustment for three separate responses

The responses are reported FTIR EC in µg/m³, directly measured HIPS Fabs in
Mm⁻¹, and FTIR EC minus HIPS Fabs divided by the configured MAC of 10 m²/g.
The last response is an optical disagreement in µg/m³.

For each indicator, both the indicator and each response are ranked. Their
ranks are separately residualized against **the same design matrix**:

1. Ranked current PM2.5 concentration.
2. Ranked current public sampled volume.
3. Monthly sine and cosine terms.
4. Ranked elapsed collection day.
5. Drop-first indicators for filter-lot category, including unknown.
6. An intercept.

Nonconstant design columns are centered and scaled for numerical conditioning.
The correlation between the residual ranks is reported as signed partial rank
**r**, together with **r²**. This is an association in adjusted rank space,
not a statement that the corresponding percentage of physical EC mass is
explained. Neither HIPS nor FTIR EC is included as a nuisance control.

The fitted design has **eight columns and rank eight** for every comparison;
the condition number is **5.164**. No collinearity warning or degenerate
residual occurs in the actual run. The implementation records constant,
rank-deficient and ill-conditioned controls explicitly and suppresses
correlations for constant or fully adjusted-away features/targets. At least
20 complete physical filters and ten residual degrees of freedom are required.

The controls differ from the earlier native association calculation, which
included the optical comparator as a control and used collected mass/native
volume. Consequently, the earlier removed-component r² near 0.599 and the new
disagreement r² of 0.339 are **different estimands**, not a replication failure
or a measured reduction in baseline error.

## Results

Each cell is **signed partial rank r (r²)**. All cells use 232 physical filters.

| Spectral indicator | Reported FTIR EC | Measured HIPS Fabs | FTIR EC minus HIPS/10 |
|---|---:|---:|---:|
| Raw 1617 | +0.846 (0.716) | +0.444 (0.197) | +0.603 (0.364) |
| Corrected 1617 | +0.202 (0.041) | −0.137 (0.019) | +0.268 (0.072) |
| Removed 1617 | +0.841 (0.707) | +0.483 (0.233) | +0.582 (0.339) |
| Raw broad contrast | +0.352 (0.124) | −0.124 (0.015) | +0.451 (0.203) |
| Raw narrow contrast | −0.106 (0.011) | +0.146 (0.021) | −0.305 (0.093) |
| Raw C–H contrast | +0.727 (0.529) | +0.508 (0.258) | +0.378 (0.143) |
| Raw O–H/N–H contrast | −0.289 (0.084) | +0.247 (0.061) | −0.483 (0.233) |
| Raw 3500–3900 continuum | +0.344 (0.119) | +0.078 (0.006) | +0.394 (0.155) |

Several distinctions matter:

- Raw and removed 1617 absorbance are strongly associated with the reported
  FTIR product; the corrected single-channel value is much less associated.
  This does not determine whether broad information was treated appropriately.
- The broad and narrow contrasts have different signed associations. Their
  common location near 1600 cm⁻¹ does not make them interchangeable diagnostics.
- C–H contrast has appreciable association with both measurement responses,
  so its association with their difference is not the whole relationship.
- The O–H/N–H contrast has modest associations of opposite sign with the two
  responses, producing a larger association with the difference. That pattern
  alone does not identify water interference or an aerosol source.

These three rank-correlation columns are **not an additive variance
partition**. The rank of a difference is not the difference of ranks, and the
two measurement responses share real aerosol variation and possible errors.
The screen does not estimate how much of the disagreement is caused by each
measurement or by baseline handling.

## Reproduction, validation and outputs

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/run_spectral_response_screen.py \
  --native-dir "$AETHMODULAR_ETAD_NATIVE_DIR"
uv run pytest -q tests/test_spectral_response_screen.py
```

Set `AETHMODULAR_ETAD_NATIVE_DIR` to the folder holding the four native audit
source files. The source filenames and SHA-256 hashes are recorded without
embedding a private provider URL or machine-specific data path. The current
public chemistry SHA-256 is
`11d545419f0d56259304fefa71c313f2f3cb53125cb8c87066424e927c1c92be`.
All five input hashes are unchanged after execution.

The checked environment uses Python 3.12.14, NumPy 2.4.2 and pandas 2.3.3.
The repository recommends Python 3.13. **Eight focused tests pass**, covering
physical-filter averaging and repeat provenance, identical response cohorts,
absence of a HIPS response control, signed/squared correlations, explicit
collinearity warnings, degenerate residual handling, unchanged inputs,
duplicate identity rejection, direct-source separation, collection-date
matching and preservation of exclusion reasons. Ruff passes.

Generated outputs are under
`research/ftir_hips_chem/output/tables/spectral_response_20261102/`:

- `native_scan_provenance.csv`: all scan IDs, physical identities, metadata and
  retained review flags.
- `physical_filter_response_ledger.csv`: averaged native features joined to
  current products, with flags and match/eligibility reasons.
- `indicator_cohort_membership.csv`: each filter's membership/reason for each
  indicator and the shared three-response cohort hash.
- `response_associations.csv`: all 24 associations, signed and squared,
  together with unadjusted rank correlations and design diagnostics.
- `partial_rank_r_squared_comparison.csv`: the compact eight-by-three table.
- `receipt.json`: source and code hashes, parameters, versions, counts and
  scientific interpretation limits.

The next useful application is to keep these three-response views alongside
baseline preservation tests and calibration-domain checks. An improved
association with the FTIR–HIPS difference alone should not choose the baseline
or justify a numerical EC correction.
