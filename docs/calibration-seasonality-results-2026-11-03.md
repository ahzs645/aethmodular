# Matched calibration transfer: Ethiopian seasonality

## Scope and calendar

The matched preprocessing experiment was evaluated on the **same 232 physical
Addis PM2.5 filters with direct HIPS measurements**, separately for Bega/dry,
Belg and Kiremt. Twelve recalibrated methods are included, plus the current
published FTIR product as context. The models were already frozen from IMPROVE
source calibration; Ethiopian optical agreement did not choose preprocessing,
PLS components, or a season-specific model.

For new reporting this analysis follows [site-seasonality.md](site-seasonality.md):
**`belg_feb` is primary** (Bega/dry October–January, Belg February–May,
Kiremt June–September). Historical **`dry_feb`** is an explicit sensitivity
(dry October–February, Belg March–May, Kiremt June–September). Both use the
registered functions in `config.py`; no new month calendar is defined here.
Seasons are assigned by native sampling start date. End dates crossing the
season boundary are flagged. These are fixed calendar bins, not measured
rainfall classifications.

HIPS Fabs / MAC is an optical BC equivalent. The differences below describe
**optical agreement, not Ethiopian EC accuracy**: these SPARTAN filters do not
have independent TOR EC. Current published FTIR is contextual and was not
recalibrated by this experiment.

## Seasonal result

Median **FTIR prediction minus HIPS Fabs / 10**, in µg/m³; negative means FTIR
is below the optical comparator. All methods use the identical filter panel.

| Method | Bega/dry, n=81 | Belg, n=81 | Kiremt, n=70 |
|---|---:|---:|---:|
| Raw spectra, full native grid | −2.176 | −2.011 | −1.482 |
| Raw spectra, shared 1425–4000 cm⁻¹ window | −0.896 | −0.268 | +1.420 |
| Spline baseline, df 6/4 | −3.428 | −3.728 | −3.823 |
| Spline baseline, df 6/3 | −3.424 | −3.706 | −3.778 |
| Spline baseline, df 6/6 | −3.477 | −3.779 | −3.926 |
| arPLS, original 50-iteration limit | −3.718 | −3.681 | −3.856 |
| arPLS, 500-iteration check | −3.710 | −3.689 | −3.850 |
| Removed spline background alone | −0.889 | −0.182 | +2.215 |
| Raw without 1580–1650 cm⁻¹ | −0.839 | −0.196 | +1.473 |
| Spline without 1580–1650 cm⁻¹ | −3.251 | −3.336 | −3.494 |
| Spline, 1580–1650 cm⁻¹ alone | −3.870 | −4.123 | −5.041 |
| Second derivative | −2.425 | −2.685 | −2.816 |
| Current published FTIR, context | −1.046 | −0.510 | +1.547 |

The shared-window raw model and published FTIR context change from negative
median differences in Bega/Belg to positive differences in Kiremt. In contrast,
the spline and arPLS representations remain well below the optical comparator
in all three seasons. That finding should be considered together with the
independent source-domain TOR validation. Selecting the method closest to HIPS
would not establish EC accuracy.

The arPLS iteration check barely changes the seasonal medians. It improves
numerical convergence, but six of the 905 source spectra still miss the
specified convergence tolerance at 500 iterations; all 319 target scans meet
it. The method is labelled by its iteration limit rather than called wholly
converged. No failed record or negative prediction is silently discarded.

## February is a material reporting choice

Twenty-one direct-pair February filters move between dry and Belg. The
historical `dry_feb` counts are **102 dry, 60 Belg, 70 Kiremt**; the primary
`belg_feb` counts are **81 Bega, 81 Belg, 70 Kiremt**. Kiremt is unchanged.

For current published FTIR, the Belg median optical difference changes from
**−0.236** under `dry_feb` to **−0.510 µg/m³** under `belg_feb`. For the
shared-window raw calibration it changes from −0.175 to −0.268 µg/m³. For
the 6/4 spline it changes from −3.764 to −3.728 µg/m³. This is a cohort
change caused by the calendar boundary; no model is refitted in this check.

## Coverage and potential confounding

All native data remain in an audit ledger: **296 physical filters**, comprising
253 PM2.5, 37 field blanks and six PM10 filters. Six PM2.5 filters lack native
dates and valid native volumes. They remain undated and cannot enter a named
seasonal concentration comparison. All 37 field blanks lack collection dates;
no season-specific blank bank is identifiable from this metadata.

The native PM2.5 population by lot is:

| Lot | Bega/dry | Belg | Kiremt | Undated |
|---|---:|---:|---:|---:|
| 248 | 18 | 16 | 0 | 1 |
| 251 | 37 | 53 | 65 | 5 |
| Unknown | 32 | 20 | 6 | 0 |

Lot 248 has **no Kiremt filters**. The two known lots share only one observed
calendar month, March 2023, and no Bega month. Therefore a raw lot contrast
cannot separate season, time and lot effects.

The seasonal pattern persists within lot 251. Its direct-pair counts are
37 Bega, 53 Belg and 64 Kiremt. Published FTIR median optical differences are
−0.695, −0.080 and +1.519 µg/m³; the shared-window raw model gives −0.777,
−0.217 and +1.367 µg/m³. Thus mixing known filter lots alone cannot explain
the full seasonal pattern. This remains an observational comparison, with
other temporal differences unresolved.

Partial years also matter: the direct pairs cover December 2022 through
January 2026, but only 2023 and 2024 include observations in all twelve months.
The direct-pair season counts by calendar year are:

| Year | Bega/dry | Belg | Kiremt |
|---|---:|---:|---:|
| 2022 | 9 | 0 | 0 |
| 2023 | 35 | 35 | 38 |
| 2024 | 20 | 34 | 26 |
| 2025 | 15 | 12 | 6 |
| 2026 | 2 | 0 | 0 |

A coverage-only sensitivity selects years with direct measurements in all
twelve months, independently of prediction values. On those **188 filters in
2023–2024** (55 Bega, 69 Belg and 64 Kiremt), published FTIR differences remain
−0.847, −0.450 and +1.519 µg/m³; shared-window raw differences remain −0.853,
−0.226 and +1.367 µg/m³. This does not balance sample counts within months,
but it shows that the sign reversal is not created by the partial 2022/2026
dry-season tails.

## Source calibration season coverage

The 905-filter source calibration is heavily concentrated in June–October.
Source counts by calendar month are Jan 8, Feb 5, Mar 11, Apr 11, May 2,
Jun 93, Jul 160, Aug 125, Sep 259, Oct 168, Nov 50 and Dec 13. The workflow
reports the already held-out IMPROVE predictions by month and quarter. These
are **source calendar bins**, not Ethiopian wet/dry seasons and not new
leave-season-out validation.

Sparse source winter/spring coverage and absent Ethiopian TOR mean these
results do not support fitting a separate target-season calibration solely to
match the optical measurements. The valid use here is to expose seasonal
transfer behavior, retain the calendar sensitivity, and compare it with
source-domain validation and baseline diagnostics.

## MAC sensitivity

The workflow reports all predeclared 633 nm MAC scenarios (7.5, 10, 12.5 and
15 m²/g) separately. They are not fitted or independently measured MAC values.
For published FTIR context, the Bega/Belg/Kiremt median differences are:

| Illustrative MAC | Bega/dry | Belg | Kiremt |
|---|---:|---:|---:|
| 7.5 m²/g | −2.427 | −1.947 | −0.279 |
| 10 m²/g | −1.046 | −0.510 | +1.547 |
| 12.5 m²/g | −0.124 | +0.363 | +2.640 |
| 15 m²/g | +0.463 | +0.904 | +3.388 |

The sign depends on the optical conversion assumption. The commonly cited
7.5 m²/g literature value is at 550 nm and is not adopted as an estimated
633 nm MAC here.

## Reproduce and inspect

The reusable logic is `research/ftir_hips_chem/scripts/calibration_seasonality.py`;
the workflow is `research/ftir_hips_chem/workflows/run_calibration_seasonality.py`.
Pass existing completed calibration directories explicitly:

```bash
uv run python research/ftir_hips_chem/workflows/run_calibration_seasonality.py \
  --calibration-dir "$CALIBRATION_RESULTS_DIR" \
  --additional-calibration-dir "$ARPLS500_RESULTS_DIR"
```

Tables are generated under
`research/ftir_hips_chem/output/tables/calibration_seasonality_20261103/` and
figures under the corresponding `output/plots/calibration_seasonality_20261103/`.
The receipt records input/code hashes, conventions, cohort counts and claim
limits. The physical eligibility, February-mover and noneligible-filter
ledgers preserve the records behind every count. Figures include the seasonal
optical-difference heat map and target-lot/source-month coverage panel.

Ten focused tests cover calendar boundaries, missing dates, duplicate physical
identities, cross-arm metadata consistency, identical comparison panels,
signed predictions, direct optical provenance, contextual FTIR identity,
source calendar labels, and coverage-only year selection. An independent
recomputation verified every original seasonal summary number against the
frozen prediction inputs.
