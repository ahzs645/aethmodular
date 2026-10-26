# Notebook coverage in the gallery

The **Sites → Data availability** page now incorporates the existing analyses,
alongside the filter inventory, recency threshold (default two years), monthly
gaps, count jumps and measurement coverage from the initial implementation.

## Views and sources

| View | Existing work incorporated |
| --- | --- |
| Network coverage | `research/spartan/spartan_all_sites_hips_overview.ipynb`; saved `coverage.csv`, `monthly_samples.csv`, HIPS bridge, HIPS inventory and all-site inventory tables. Site × product matrix, monthly sampling heatmap, cadence and longest-gap table. |
| Study overlap | `notebooks/analysis/data_availability/filter_data_availability_strip_chart.ipynb` and `research/ftir_hips_chem/Multi_Site_Analysis.ipynb`. Day strips, exact calendar-day intersections, physical-filter overlap, completeness thresholds and sample-date audit. |
| Filter explorer | Current gallery SPARTAN/IMPROVE exports: searchable site inventory, editable reference date, inactivity threshold, sampling spans, per-site monthly counts, gaps, jumps and finite field coverage. |
| Prior comparison | `research/spartan/spartan_improve_orientation.ipynb` and its saved shared-coverage table. Explicitly identifies the four-site SPARTAN study and historical cleaned IMPROVE population. |

The first three views answer different questions: product presence across the
network, overlapping measurement days in the study, and availability in the
gallery's per-filter exports. Their totals describe different populations.
The global chart subset does not alter these inventories.
Site labels prefer the current public-header/map names over the older lookup
labels, and expand known compact place names (for example, Addis Ababa and
Mexico City). Site codes stay unchanged.

## Counting rules

- Public chemistry and reconstruction counts use unique filter IDs within
  each product; nephelometer and estimated-PM counts use distinct timestamps.
  Product columns overlap and must not be summed into physical filter totals.
  IMPROVE columnar exports are decoded into site/date/POC sample keys, retaining
  distinct sampler POCs on the same date. Its current inventory describes the
  validated cleaned population with positive EC and absorption, not every
  downloaded measurement.
- Monthly cells retain every calendar month. Zero observations inside a
  reported span have a pale fill; months outside it have a white fill.
- HIPS monthly coverage is recomputed from finite, dated bridge records,
  including zero/negative Fabs. Blank-filter counts stay separate.
- Study FTIR-EC and HIPS Fabs require finite `Concentration` values. No fitting
  threshold or manual-outlier removal is applied to availability counts.
- Study aethalometer coverage uses finite `IR BCc` and the canonical processed
  local `day_9am`. Optional 50%/80% thresholds use `data_completeness_pct`.
  Addis lacks that field; thresholded availability is **unavailable**, not zero.
- Exact same-day overlap is distinct from physical-filter matching. The
  latter uses the shared helper's base filter ID, collapsing replicate suffixes.
  It does not use ±1-day matching. Multiple filters can share a calendar day.
- Nominal sampling ratios use the existing pipeline's schedule assumptions.
  Values above 100% are preserved; they are not instrument uptime percentages.

## Reconciled differences

The saved network cadence/monthly tables agree with one another, but predate
the public inventory refreshed on **2026-10-15**. That inventory advertises
later valid endpoints for **28** site/product series. Two ILHA endpoints,
**2026-11-17**, occur after the release date and are omitted from that comparison.
Repackaging these tables does not refresh the public observations.

The HIPS bridge contains **3,033** finite dated physical-filter records. Its
separate inventory reports **3,178** sample filters and **686** blanks. Those
snapshots remain distinct. The notebook's saved monthly heatmap reconciles
exactly with the bridge after its original positive-Fabs restriction; the
gallery includes the **14** finite nonpositive PRFJ values as available data.

The original day-strip notebook counted the presence of parameter rows,
including missing concentrations. It also supplied aethalometer paths only
for Addis, using a different processed population. The gallery recomputes
from the canonical loaders/configuration for all four sites:

| Site | FTIR reported / finite days | HIPS finite days | Aeth finite days | FTIR + HIPS days | All three days | Same FTIR/HIPS physical filters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beijing | 190 / 184 | 161 | 222 | 161 | 65 | 163 |
| Delhi | 97 / 63 | 63 | 113 | 63 | 24 | 63 |
| JPL | 158 / 158 | 130 | 489 | 130 | 65 | 130 |
| Addis Ababa | 189 / 189 | 189 | 515 | 189 | 172 | 190 |

JPL's last finite FTIR/HIPS sample date is **2023-11-14**. FTIR/HIPS analysis
dates in February 2024 do not extend sampling coverage. One stored JPL day
reports minute completeness above 100%; that source value is retained.

The prior IMPROVE comparison ends in July 2025 and covers 214 sites for most
parameters. It is explicitly historical; the current gallery inventory has
217 sites, including one without map coordinates (216 mapped). SPARTAN EC
remains FTIR-derived; IMPROVE EC is thermal-optical.

## Refresh and validation

```bash
uv run aeth doctor
uv run python gallery/data/export_availability.py
cd gallery/app
node --test src/lib/availability.test.mjs src/lib/notebookAvailability.test.mjs
python scripts/check-responsive.py
npm run build
```

The exporter writes `gallery/app/public/data/availability.json`, with relative
source paths, SHA-256 hashes, source roles and packaging time. It asserts
unique bridge IDs, checks the original positive-only HIPS heatmap, and
reconciles every monthly sum against the corresponding product count.
The page offers this JSON as a download. Neither notebook files nor scientific
inputs are modified.

Verified in Chromium at 14 widths from 320–2560 px, including all availability
views, both filter networks and the 80% completeness state. All 8 availability
tests pass. The broader suite has 123 passes and the same 8 missing/stale-export
failures documented in [the gallery review](gallery-responsive-review.md).
