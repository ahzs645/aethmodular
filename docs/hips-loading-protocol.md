# HIPS on loaded filters: the protocol

Date: 11 November 2026. The procedure that came out of [addis-reference-closure-2026-11-10.md](addis-reference-closure-2026-11-10.md)
and [hips-network-2026-11-11.md](hips-network-2026-11-11.md). Code: `research/ftir_hips_chem/scripts/hips_loading.py`.
Gallery: Meeting tab, Nov 11, "HIPS: the procedure".

HIPS reads low as a filter darkens. This is how to find the filters and sites it affects, see what it does to the numbers
read off HIPS, correct it with a stated range, and check that the correction worked. It runs on any set of filters with
HIPS Fabs and a sampled volume.

```bash
aeth hips screen                               # every SPARTAN site in the gallery's table
aeth hips screen --site ETAD INDH --out OUT    # chosen sites, tables written to OUT
aeth hips screen --input batch.csv --fabs Fabs --volume Volume --site-column Site \
    --reference EC --instrument PM25 --date SampleDate --out OUT
```

## The rule behind it

On IMPROVE filters, against thermal (TOR) EC, HIPS optical depth follows `tau = L / (1 + L / ceiling)`, with a ceiling of
2.26 [1.82, 2.70] (347,936 samples, 210 sites). So:

| Quantity | Formula |
|---|---|
| Optical depth of the filter | `tau = Fabs × volume ÷ (100 × 3.53 cm²)`; this is the laboratory's own HIPS tau |
| Expected under-read | `tau ÷ ceiling` |
| Linearised Fabs | `Fabs ÷ (1 − tau ÷ ceiling)` |
| Its range | the same at both ends of the ceiling's interval |

## Step 1. Identify

**Each filter**, from Fabs and volume alone, gets a status:

| Status | Expected under-read | Optical depth | Correction known to (median filter) | Use for |
|---|---|---|---|---|
| use, linearised | up to 20 % | up to 0.45 | ±2 % | level and shape |
| caution | 20 to 30 % | 0.45 to 0.68 | ±7 % | shape; level only with the range stated |
| shape only | over 30 % | over 0.68 | ±13 % or worse | rank order, timing, season; not a level |

A filter darker than 1.18 is also marked `beyond_reference`: no IMPROVE filter the curve was fitted on is that dark.

**Each site**, from its own data and with no curve assumed, gets two tests (40 filters with FTIR EC and PM2.5 needed):

- **Exponent.** b in `Fabs ∝ (FTIR EC)^b`, with weighed PM2.5 as the instrument so that error in FTIR EC does not flatten it.
  Below 1, HIPS is compressed.
- **Heavy over light.** Fabs ÷ FTIR EC on the most loaded third of filters over the least loaded third. The thirds are chosen by
  PM2.5 on the filter, never by HIPS or FTIR EC, which would select on their own errors. Below 1, HIPS falls behind.

Each is read beside what the curve predicts for that site's filters. Readings on the SPARTAN network:

| Reading | Sites |
|---|---|
| expected and seen | Addis Ababa, Dhaka, Johannesburg, Delhi, Santiago, Pretoria, Kaohsiung, Beijing |
| expected, cannot be tested | Kanpur, Ilorin, Bujumbura, Bandung, Singapore |
| expected, not seen | Jaipur, Abu Dhabi, Bishoftu, Mexico City |
| not expected, but compressed | Talca |
| HIPS rises faster than the reference | Taipei, Melbourne, Haifa, Fajardo |

A third sign, in any straight-line fit with HIPS on the x-axis: a slope above 1 with a negative intercept. The published
Addis fit, 2.31x − 6.17, is this.

## Step 2. Trace

Every number read off HIPS is recomputed on linearised HIPS and shown beside the reported one.

| Read off HIPS | Addis Ababa | Delhi |
|---|---|---|
| Median BC (µg/m³) | 4.7 → 8.0 [7.2, 9.8] | 5.9 → 8.0 [7.7, 8.5] |
| Mean BC (µg/m³) | 4.8 → 9.0 [7.8, 11.9] | 6.5 → 8.9 [8.4, 9.8] |
| BC share of PM2.5 | 0.22 → 0.39 [0.35, 0.49] | 0.09 → 0.12 [0.11, 0.13] |
| Three highest months ÷ three lowest | 1.36 → 1.76 [1.64, 2.05] | 6.71 → 8.14 [7.83, 8.69] |
| Fabs ÷ FTIR EC (m²/g) | 11.4 → 19.3 [17.0, 24.9] | 10.0 → 12.9 [12.2, 14.1] |
| Slope, FTIR EC on HIPS BC | 2.29 → 0.54 [0.29, 0.75] | 1.27 → 0.80 [0.68, 0.88] |
| Intercept (µg/m³) | -6.22 → -0.18 [-1.10, 1.29] | -1.77 → -0.41 [-0.68, 0.08] |

What to expect in general:

- **Levels rise**, means more than medians, because a few very dark filters carry most of the correction.
- **Seasons widen.** The bend takes most from the darkest months, so reported HIPS flattens the seasonal cycle.
- **Sites change order.** Reported, Delhi has the highest median BC and Addis is third; linearised they are level.
- **Fits straighten.** Slopes fall towards or below 1 and negative intercepts move towards zero.
- **A change of sampled volume shows as a step** in reported Fabs that the linearised series should not have.

## Step 3. Correct

1. Compute the linearised Fabs for every filter, with its low and high. Delete nothing.
2. Let the status decide the use. For a level (a mean, a MAC, a comparison with FTIR EC), use only `use, linearised` filters
   and quote the range. For shape, all filters can be used.
3. Where a site fails step 4, treat its HIPS as shape only for level, whatever its filters' statuses.
4. Where sampling can still be changed, lower the volume first. Less air through the filter lowers the optical depth itself;
   Addis at 3.6 m³ would lose 27 % on its median filter and at 1.8 m³ 15 %, against 42 % now.

## Step 4. Check

After linearising, HIPS should be proportional to the reference: the step 1 exponent, recomputed, should include 1.

| Site | Before | Curve alone | After | Verdict |
|---|---|---|---|---|
| Addis Ababa, Ethiopia | 0.46 | 0.55 | 0.84 [0.77, 0.91] | under-corrected |
| Dhaka, Bangladesh | 0.68 | 0.67 | 1.01 [0.64, 1.43] | corrected |
| Johannesburg, South Africa | 0.88 | 0.79 | 1.10 [1.01, 1.21] | over-corrected |
| Delhi, India | 0.80 | 0.86 | 0.92 [0.87, 0.98] | under-corrected |
| Santiago, Chile | 0.56 | 0.78 | 0.72 [0.68, 0.75] | under-corrected |
| Pretoria, South Africa | 0.90 | 0.81 | 1.10 [0.99, 1.21] | corrected |
| Kaohsiung, Taiwan | 0.83 | 0.90 | 0.92 [0.83, 1.01] | corrected |
| Beijing, China | 0.93 | 0.88 | 1.06 [0.92, 1.29] | corrected |
| Talca, Chile | 0.56 | 0.86 | 0.65 [0.61, 0.68] | under-corrected |
| Taipei, Taiwan | 1.06 | 0.93 | 1.14 [1.02, 1.28] | over-corrected |
| Melbourne, Australia | 0.80 | 0.98 | 0.81 [0.60, 1.13] | corrected |
| Pasadena, United States | 1.04 | 0.96 | 1.08 [0.87, 1.36] | corrected |

Before ÷ curve alone = after. Eleven more sites have intervals too wide to read, and seven too few filters.

A failed check means the site's filters bend differently from IMPROVE's, or FTIR EC is not proportional to the absorber
there. With FTIR EC as the only reference, the check cannot say which.

**Run it within each season too.** The correction is per filter, but a site's seasons can sit at different levels of HIPS
per unit FTIR EC for reasons that are not loading. Pooled, that reads as a failed correction. At Addis the pooled check
gives 0.84 while Bega, Belg and Kiremt each pass (1.10, 1.04, 1.13). At Johannesburg the site passes on average while
winter is under-corrected and spring over-corrected. `aeth hips screen` reports both and writes `seasons.csv`; a file takes
`--season COLUMN`. See [hips-seasons-2026-11-11.md](hips-seasons-2026-11-11.md).

## Per site, per season or per filter

Correct **per filter**. At the heavy sites one factor per site is wrong by 7 % on the typical filter and one per site and
season by 5 %, and a season factor misses more than a quarter of filters by over a tenth. Only 15 % of the spread in the
correction is between seasons; 44 % is between filters of the same site and season. Use **seasons to check** the
correction, and the **site** to decide whether its HIPS can carry a level at all.

## What it writes

| File | Rows | Use |
|---|---|---|
| `filters.csv` | one per filter: tau, under-read, status, `beyond_reference`, factor, linearised Fabs with low and high | join to any analysis that uses HIPS |
| `sites.csv` | one per site: shares by status, both tests beside their predictions, the reading, the check's verdict | which sites are corrected and which are shape only |
| `propagation.csv` | one per site and quantity: reported, linearised, low, high | what to quote, and how far a reported result moves |
| `seasons.csv` | one per site and season: the check within that season | which seasons pass, and where a pooled verdict misleads |

## How to report it

- Give the optical depth, or the status, beside any HIPS value from a heavy site.
- Quote a linearised value with its range, and say the ceiling is IMPROVE's.
- Call HIPS optical. Do not call linearised HIPS thermal EC, and do not call FTIR EC thermal EC.
- For Addis: no filter on file is `use, linearised`, so HIPS does not set a level there.

## When to run it again

| When | What to do |
|---|---|
| A new HIPS delivery | Run `aeth hips screen`; new filters get a status the day they arrive |
| A sampled volume changes at a site | Rerun; expect a step in reported Fabs that the linearised series should not show |
| Thermal EC arrives for a heavy site (the Addis TOR filters) | Fit that site's own ceiling against thermal EC and compare with IMPROVE's |
| The IMPROVE ceiling is re-estimated | Change `IMPROVE_CEILING` in `hips_loading.py`; `tests/test_hips_loading.py` fails until it matches the summary |

## Limits

- The ceiling is IMPROVE's. The same laboratory and filter type serve both networks, but SPARTAN has no thermal EC to test it.
- IMPROVE's two half-periods give ceilings of 2.53 and 1.71. For `shape only` filters that is a 30 % difference in the factor.
- The site tests and the check use FTIR EC. They cannot separate a bend in HIPS from one in FTIR EC.
- Thresholds of 20 % and 30 % are choices. At those marks the correction factor is uncertain by about ±5 % and ±9 %.
- The procedure was checked on made-up sites with a known bend (`tests/test_hips_loading.py`): it recovers the true median
  within 3 % and calls a stronger or weaker bend under- or over-corrected.
