# HIPS bend checks after the 19 November meeting (20 November 2026)

What Ann asked for on 19 Nov, before the 24 Nov meeting, and what the checks give. Gallery: Meeting tab, Nov 24,
**"HIPS bend checks"** (`#tab=meeting&meeting=hips-checks`, `pages/HipsBendChecks.tsx`). Nothing here changes the curve,
the ceiling or the statuses of [hips-loading-protocol.md](hips-loading-protocol.md).

## 1. The bend on every IMPROVE filter, not only the group medians

The ceiling τ∞ = 2.26 [1.82, 2.70] is fitted through 17 group medians, grouped by the previous sample's TOR EC loading
at the same site. Ann's concern: a grouping is a choice, so the curve should also be shown on all the data.
`gallery/data/export_hips_improve_density.py` writes every one of the same 347,936 IMPROVE filters (210 sites,
2005-01-04 to 2025-12-30) as a count grid, and fits the same equation to single filters, with 90 % intervals from
200 site resamples.

| Fitted to (all sites) | Filters | τ∞ | 90 % | × on the median Addis filter (τ 0.97) |
|---|---|---|---|---|
| Group medians (the curve used) | 347,936 | 2.26 | 1.82–2.70 | 1.75 |
| Single filters, τ residuals | 347,936 | 2.68 | 2.20–3.04 | 1.56 |
| Single filters, log τ residuals | 347,936 | 1.72 | 1.48–1.98 | 2.28 |
| From 10 µg, τ residuals | 59,692 | 2.82 | 2.35–3.21 | 1.52 |
| From 10 µg, log τ residuals | 59,692 | 2.39 | 1.92–2.75 | 1.68 |
| From 20 µg, τ residuals | 17,112 | 2.89 | 2.42–3.27 | 1.50 |

(5 µg rows, and the OC/EC < 4 and six urban populations, are in the gallery table.) Every fit bends, so the bend is
not from the grouping. How far it bends depends on the choice: τ∞ 1.7 to 2.9 over all sites, and most single-filter
fits lean higher than the group fit (a smaller correction). At the OC/EC < 4 sites single-filter fits give 2.2 to 4.9.
Single-filter fits carry TOR's error on the x axis, and a lowest loading selects on it; the group fit avoids both,
which is why it stays the one used. **Say: the bend is robust, its size is known to about ±25 %.**

## 1b. How the ceiling is decided, and how much the grouping matters

The published ceiling is the curve fitted (least squares on log τ) through the medians of 17 groups, grouped by the
**previous** sample's TOR EC loading at the same site (within 7 days), with hand-set edges about 1.3 to 1.5 times apart
(0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 50, 70, 100 µg) and groups under 150 filters left out. The previous
sample tracks how loaded this filter is likely to be, but carries none of this filter's measurement errors.
`gallery/data/export_hips_improve_groupings.py` refits it under 70 other groupings (100 site resamples each):

| Grouped by | Edges | τ∞ |
|---|---|---|
| Previous sample's loading | published, 17 groups | 2.26 [1.85, 2.64] |
| Previous sample's loading | equal count or log-spaced, 6 to 40 groups | 2.15 to 2.83, mostly 2.25 to 2.45 |
| The filter's own TOR EC loading | equal count or log-spaced | 1.1 to 1.9 (the bend overstated by TOR's error) |
| The filter's own HIPS τ | any | no ceiling (the bend hidden by HIPS's error) |

So the edges barely matter once the grouping is by the previous sample; the two naive groupings are biased in opposite
directions, which is why the previous sample is used. With four groups or fewer the fit has too few points (3.24).
The darkest few groups decide τ∞; the light ones barely move it.

Gallery (Meeting tab, Nov 24, part 2): Page options, **IMPROVE grouping** (group by, edges, how many, smallest group,
residuals; `hc.g.*` in a link) drives three charts: the medians with both curves and the fit's error at every ceiling
(τ∞ is its lowest point), the ceiling under every grouping of the sweep with intervals, and τ∞ with each group left out.
"Linearise this page with τ∞ …" makes a refitted ceiling drive every chart on the page (`hl.ceil=custom&hl.ceilv=`).
The bend chart now shades the published 90 % interval and names the active ceiling. The browser reads medians from
4 % log bins and lands within 2 % of the exporter's exact-median fits (`lib/hipsGroupings.test.mjs`).

## 1c. Is the ceiling defensible: every other way of estimating it

`gallery/data/export_hips_ceiling_checks.py` (100 site resamples; ~10 min with `--cache`) asks what a reviewer would. The
published set is rebuilt exactly (347,936 of 357,928 IMPROVE filters with TOR EC and HIPS have a previous sample).

| Method | τ∞ [90 %] | Addis fit at it | MAC for slope 1 |
|---|---|---|---|
| Published: previous sample, 17 group medians | 2.26 [1.85, 2.64] | 0.55x + 0.01 | 18.3 |
| Instrument: the sample two back / three back / next | 2.12 / 2.34 / 2.31 | 0.47x + 0.41 / 0.59x − 0.19 / 0.57x − 0.12 | 21 / 17 / 18 |
| Instrument: previous sample only across a HIPS lot change | 7.6 [1.8, no ceiling], 2,632 filters | not informative | |
| Within sites (each site its own absorption per EC, one ceiling) | 1.45 [1.10, 1.76] | 0.04x + 4.00 | > 200 |
| One period: 2005–2014 / 2015–2025 | 2.53 / 1.71 | 0.68x − 0.63 / 0.20x + 2.25 | 15 / 50 |
| Single filters, errors in both axes (measurement + real spread) | 1.63 [1.48, 1.84] | 0.13x + 2.91 | 75 |
| Single filters, measurement error only / τ only | 2.28 / 1.46 | | |
| Exponential approach (ΔAIC +0.5 on the medians) | 1.33 | 16 of 189 Addis filters past it | |
| Patterned deposit, fitted / fully pixelated 0.638 (White et al. 2025; ΔAIC +6 / +7.5) | 1.08 / 1.02 | 60 / 76 of 189 past it | |
| As measured | | 2.31x − 6.17 | 4.3 |

Errors for the single-filter fits come from IMPROVE's only collocated pair, Phoenix (PHOE1 and PHOE5 POC 2, 1,118 days,
2015–2026): TOR EC 5.9 % + 0.025 µg/m³, Fabs 4.4 % + 0.04 Mm⁻¹; IMPROVE's reported uncertainties (15 %, 6 %) run a little
high (robust z spread 0.78, 0.90). On light filters (2–8 µg) absorption per EC spreads 35 % from filter to filter, of which
measurement explains 22 %; the real part, 27 %, is as large as the errors, so single-filter fits swing with the error model.

What this supports:

- **Defensible**: HIPS bends on heavy filters. Every method finds it, it is stronger within sites (so not heavy filters
  coming from other sites), and the grouping's instrument does not matter (two back, three back, next all within 0.15).
  Patterned and exponential curves fit IMPROVE nearly as well but put the ceiling below filters Addis has measured (τ to 1.51).
- **Defensible**: HIPS as measured cannot set a level at Addis; every ceiling moves the intercept away from −6.
- **Not yet defensible**: one ceiling. Within-site fits, the 2015–2025 period and errors-in-variables fits sit at 1.45–1.7,
  below the published interval, and at those values Addis over-corrects (positive intercepts, MACs of 50 to over 200).
  Across the published interval alone the Addis intercept runs +1.6 to −1.0 and the MAC 13 to 35, so "MAC about 18" is a
  central value, not a result, and the zero intercept at 2.26 is partly where 2.26 sits.
- **What settles it**: a ceiling fitted on Addis's own filters against thermal EC (the Addis TOR filters); until then,
  report the bend, the published ceiling with its interval, and the Addis level as a range. The period difference (2.53
  against 1.71) is the largest open question for SPARTAN, whose filters are mostly 2015 on.

Gallery: Meeting tab, Nov 24, part 3: every estimate side by side (`#hc-methods`), and the Addis intercept and MAC at every
ceiling with the full table (`#hc-addis-ceiling`). `lib/hipsCeilingChecks.test.mjs` holds the exporter's Addis fits to the browser's.

**Correction (20 Nov):** the MAC for slope 1 was first computed by bisection with λ held fixed (Addis 17.6). The repo's
convention moves λ with MAC², under which it is exactly MAC ÷ slope: Addis 18.3, Delhi below τ 0.45 11.1.

## 2. The same curve in Fabs against EC

Fabs carries HIPS's own corrections, so Ann asked whether the bend survives in Fabs. On IMPROVE it does: Fabs against
TOR EC (µg/m³) bends the same way, since IMPROVE's volume barely varies (median 32.6 m³). On a concentration axis the
bend sits at a different place for each sampled volume, because it is a filter-loading effect: a 20 % under-read is
reached at about 4.9 Mm⁻¹ at IMPROVE's volume and 45 Mm⁻¹ at Delhi's 3.5 m³. IMPROVE and a SPARTAN site therefore
cannot share one curve on a concentration axis; that is the reason to draw it against optical depth.

## 3. Addis and Delhi with only the filters below τ 0.45

The crossplot (FTIR EC on HIPS BC, Deming) can now keep only filters below an optical-depth cut, and every fit reports
the MAC at which its slope is 1: MAC ÷ slope, the λ ∝ MAC² convention of `pls_transfer.deming_regression` (ftir_31). The table
covers every SPARTAN site with 20 or more filters.

| Site | Past τ 0.45 | All, as measured | Below τ 0.45, as measured | All, linearised (MAC for slope 1) |
|---|---|---|---|---|
| Addis Ababa | 100 % (189) | 2.31x − 6.17 | none below the cut | 0.55x + 0.01 (18.3) |
| Delhi | 62 % (109) | 1.33x − 1.66 | 0.90x + 0.14, n 41 (MAC 11.1) | 0.82x − 0.30 (12.2) |

- **Delhi**: the 41 light filters alone give an intercept near zero with no correction, and a slope of 1 at MAC 11.
  Its share past the cut is about two thirds of filters with FTIR EC (54 % of all 170 HIPS filters in the 11 Nov table),
  not "about a tenth" as said in the meeting.
- **Addis**: no filter is below 0.45, so the cut cannot be tested there. Linearised, MAC 18.3 gives slope 1, close to
  the "about 17" Ann remembered for global-south sites.

## 4. Linearising only filters past a τ

Side panel, Plot, HIPS, Linearised, **Apply to: Filters past a τ** (`hipsat=past`, cut `hipscut`, default 0.45):
every site, only filters at or past the cut, lighter filters as measured; units read ", linearised past τ 0.45".
The Calibration tab's saved runs follow it. It is offered because Ann asked for it; per filter stays the default, since
a cut puts a step of × 1.25 in the response at 0.45 ([hips-linearise-rules-2026-11-19.md](hips-linearise-rules-2026-11-19.md)).

## 5. Answers and corrections from the meeting

- The curve is τ = aL / (1 + aL/τ∞), a = 0.0372 µg⁻¹ (13.1 m²/g), saturating, not quadratic. Now written under the bend chart.
- The six urban IMPROVE sites: Atlanta (ATLA1), Birmingham (BIRM1), Fresno (FRES1), Phoenix (PHOE1), Pittsburgh (PITT1),
  Puget Sound (PUSO1). Birmingham was the one missing. A citation for the list is still needed.
- Warren's range: the fit uses 2005–2025; White et al. (2016) treat HIPS as comparable from 2003. Whether White et al.
  (2016, 2025) describe or exclude the heaviest filters is open and needs the papers read.
- Wording: filters past the cut have compromised Fabs; nothing is lost. HIPS is the check on FTIR EC.

## Rebuild and storage

```bash
uv run python gallery/data/export_hips_improve_density.py            # ~3 min to read the tables from Drive, ~1 min to fit
uv run python gallery/data/export_hips_improve_density.py --cache /tmp/improve.parquet   # reuse a read (25 MB parquet)
```

Steps `hips-improve-density` and `hips-improve-groupings` in `gallery/data/rebuild.py`
(`export_hips_improve_groupings.py` takes the same `--cache`; ~6 min for the resamples). Storage: `improve_density.json`
50 kB (15 kB gzipped) and `improve_groupings.json` 232 kB (63 kB gzipped), both committed; the page's code is a lazy
chunk of about 30 kB. No new gitignored export; shipping the 347,936 samples themselves instead would have been 2 to 5 MB.
