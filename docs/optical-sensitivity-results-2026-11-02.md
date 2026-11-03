# Optical comparator sensitivity — 2 November 2026

## What changed in our interpretation

**The current Addis–Bishoftu contrast is strongly dependent on collection-date
coverage.** Across all available direct-HIPS pairs, median FTIR EC minus HIPS/10
is −0.329 µg/m³ at Addis and −1.266 µg/m³ at Bishoftu. On their shared
**5 November–29 December 2025** collection-date window, the medians are
**−1.393 µg/m³ at Addis (12 filters)** and **−1.266 µg/m³ at Bishoftu (21)**.
Every directly paired filter at both sites in that shared window has reported
FTIR EC below HIPS/10.

This supports comparing the same dates before interpreting a site difference.
It does **not** establish a common error, or isolate a seasonal, background,
filter-lot or calibration effect. The filters are not collocated or exactly
date-matched; acquisition batches and product versions remain incompletely
documented. A collection-date window is also not an acquisition-date window.

## Reproduce the experiment

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/run_optical_sensitivity.py
uv run pytest -q tests/test_optical_sensitivity.py
```

Reusable calculations are in
[`optical_sensitivity.py`](../research/ftir_hips_chem/scripts/optical_sensitivity.py).
The workflow uses the committed
`gallery/app/public/data/spartan_reference.json` through the existing
`load_public_chemistry` and `optical_pair_ledger` functions. Its SHA-256 is
`11d545419f0d56259304fefa71c313f2f3cb53125cb8c87066424e927c1c92be`.
The source hash is checked again after calculation; it is unchanged.

The checked environment is Python 3.12.14, NumPy 2.4.2 and pandas 2.3.3;
the repository recommends Python 3.13. All **nine focused tests pass**, covering
unit conversion, signed values, source separation, preservation of source rows
and exclusions, zero and missing denominators, inclusive denominator thresholds,
duplicate physical-filter rejection, scenario direction, and date-window rules.
Ruff also passes for the three added Python files.

## Population and measurement provenance

The input contains **5,069 PM2.5 physical-filter records at 38 sites**. The
scenario calculations use **2,500 eligible direct-HIPS pairs** and retain
**873 eligible public-HIPS extension pairs** in the ledger without using them
as directly exported absorption measurements. Published exclusion flags and
reasons remain unchanged. The source loader's product and volume caveats still
apply.

This experiment keeps the quantities distinct:

- FTIR EC is the reported FTIR-derived product.
- Direct HIPS Fabs is an optical absorption measurement in Mm⁻¹.
- Fabs divided by an assumed MAC is an optical equivalent-BC comparator in
  µg/m³.
- Fabs divided by FTIR EC is an **effective absorption-to-reported-EC ratio**
  in m²/g. It inherits any FTIR estimation error and optical measurement effects;
  it is **not an independently measured MAC**.

The numerical unit conversion is simply

\[
\mathrm{eBC}\;(\mu\mathrm{g\,m}^{-3})
=\frac{\mathrm{Fabs}\;(\mathrm{Mm}^{-1})}
{\mathrm{MAC}\;(\mathrm{m}^{2}\mathrm{\,g}^{-1})}.
\]

The two factors of 10⁻⁶ cancel. No additional factor of 10⁶ is applied.
Negative and zero finite measurements remain in the disagreement ledger.
Ratio calculations require positive finite EC and positive Fabs, with separate
denominator and eligibility counts.

## 1. Assumed-MAC scenarios

These four values are **illustrative scenarios at the HIPS wavelength of
633 nm**. They are not fitted to the data and are not validated Ethiopia MACs.
In particular, the often-cited **7.5 ± 1.2 m²/g** value in Bond and Bergstrom
(2006) refers to uncoated particles at **550 nm**. That result cannot be moved
to 633 nm without a wavelength-dependent absorption assumption. Their article
is [*Light Absorption by Carbonaceous Particles: An Investigative Review*](https://doi.org/10.1080/02786820500421521).

The reported difference is FTIR EC minus Fabs/MAC. A negative difference means
reported FTIR EC is below the optical comparator.

| Site | Assumed MAC, m²/g | Direct pairs | FTIR below optical | Median difference, µg/m³ | Median absolute disagreement, µg/m³ |
|---|---:|---:|---:|---:|---:|
| Addis | 7.5 | 232 | 196 | −1.710 | 1.712 |
| Addis | 10.0 | 232 | 134 | −0.329 | 1.033 |
| Addis | 12.5 | 232 | 74 | +0.465 | 0.851 |
| Addis | 15.0 | 232 | 31 | +1.063 | 1.073 |
| Bishoftu | 7.5 | 21 | 21 | −2.065 | 2.065 |
| Bishoftu | 10.0 | 21 | 21 | −1.266 | 1.266 |
| Bishoftu | 12.5 | 21 | 21 | −0.712 | 0.712 |
| Bishoftu | 15.0 | 21 | 18 | −0.291 | 0.291 |

Raising the assumed MAC mechanically lowers the equivalent-BC comparator. The
change in disagreement is a sensitivity result, not evidence that the larger
MAC is correct. Choosing the MAC that minimizes the displayed difference would
fit one uncertain measurement to another; it would not establish EC accuracy.

## 2. Effective Fabs/FTIR EC ratio and denominator sensitivity

The thresholds below are separately labelled analytical cohorts, not new
quality exclusions. Every original record stays in the physical-filter ledger.

| Site | Minimum reported FTIR EC, µg/m³ (inclusive) | Ratio count | Median effective ratio, m²/g | Interquartile range, m²/g |
|---|---:|---:|---:|---:|
| Addis | 0.1 | 232 | 10.969 | 8.496–13.001 |
| Addis | 0.5 | 232 | 10.969 | 8.496–13.001 |
| Addis | 1.0 | 231 | 10.948 | 8.473–12.997 |
| Bishoftu | 0.1 | 21 | 17.815 | 16.446–20.914 |
| Bishoftu | 0.5 | 21 | 17.815 | 16.446–20.914 |
| Bishoftu | 1.0 | 18 | 17.613 | 15.533–18.830 |

The Ethiopian full-period difference is not driven solely by the handful of
EC denominators below 1 µg/m³. Coverage remains a larger issue: on the shared
November–December window, Addis's median effective ratio rises to
**16.396 m²/g (12 filters)**, while Bishoftu remains **17.815 m²/g (21)** at the
0.1 and 0.5 thresholds. This further limits any full-period site interpretation.

Denominator thresholds also change other sites' sample populations substantially.
For Pasadena/JPL, 154 ratios are available at the 0.1 threshold, 66 at 0.5 and
only two at 1.0. A higher threshold does not produce a uniformly comparable
network cohort. The complete per-site counts are written alongside each result.

## 3. Shared-date Ethiopia comparison

The bounds come from the overlap of the sites' eligible direct-pair collection
date ranges. The inputs are not selected using the sign or size of disagreement.

| Site, 5 November–29 December 2025 | Direct pairs | Median difference at MAC 10, µg/m³ | FTIR below optical | Median effective ratio, EC ≥0.5 µg/m³ |
|---|---:|---:|---:|---:|
| Addis | 12 | −1.393 | 12 of 12 | 16.396 m²/g |
| Bishoftu | 21 | −1.266 | 21 of 21 | 17.815 m²/g |

These results motivate a specific next comparison: inspect signed spectra,
removed backgrounds and product provenance for those same **33 filters**.
The present experiment does not attribute their shared direction to baseline
removal. Similar discrepancies could arise from different combinations of
spectral calibration, aerosol composition and optical behavior.

## Outputs and claim limits

The workflow writes the following to
`research/ftir_hips_chem/output/tables/optical_sensitivity_20261102/`:

- `physical_filter_ledger.csv`: all 5,069 filters, eligibility flags, original
  exclusions, denominator flags and effective ratios.
- `scenario_filter_ledger.csv`: all original filters at each of the four
  scenarios, with ineligible optical comparisons left missing.
- `site_mac_scenarios.csv`: network-wide source-separated scenario summaries.
- `effective_ratio_threshold_sensitivity.csv`: denominator thresholds and
  missing/nonpositive/below-threshold counts.
- `ethiopia_shared_date_window.csv`: the shared-date scenario comparison.
- `ethiopia_shared_date_filter_panel.csv`: the 33 filters in that collection-date
  window, including their original fields and all diagnostic flags.
- `ethiopia_shared_date_ratio_thresholds.csv`: ratio thresholds in that window.
- `receipt.json`: source and code hashes, parameters, software versions,
  population counts and interpretation limits.

Nothing here recalculates FTIR EC, replaces a baseline, changes the configured
MAC, or validates a new correction. The useful result is a reproducible check
on **which scientific conclusions survive comparator and cohort choices**.
