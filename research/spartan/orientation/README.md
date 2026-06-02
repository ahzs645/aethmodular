# SPARTAN & IMPROVE — orientation overview

An intro deck for someone seeing the two networks **for the first time**. Built by
[`../spartan_improve_orientation.ipynb`](../spartan_improve_orientation.ipynb) —
re-run that notebook top to bottom to regenerate everything here.

Three parts:

1. **SPARTAN on its own** — what it is, a world map of its sites, the public product
   structure, and overall coverage (drawn from the full-network inventory under
   `research/spartan/inventory/`).
2. **IMPROVE on its own** — the same orientation for the US IMPROVE network, from the
   pre-cleaned pull `improve_valid_cleaned.csv` (US map, parameters, coverage, dates).
3. **Where they meet** — the quantities both networks carry (EC, OC, fAbs, Fe, PM₂.₅),
   shown as side-by-side distributions, cross-plots (fAbs~EC ≈ MAC, OC~EC, Fe~PM₂.₅),
   and derived ratios. The networks share no sites or dates, so this is a
   distributional + relationship comparison, not a row-level join.

This complements the deeper [`../improve_comparison/`](../improve_comparison/) analysis;
the orientation notebook is the gentler, presentation-first entry point.

## Figures (`figures/`)

| File | What |
|---|---|
| `fig_spartan_map.png`  | SPARTAN sites worldwide, size ∝ observations, colour = coverage score |
| `fig_improve_map.png`  | IMPROVE sites (US-centric), size/colour ∝ filter count |
| `fig_distributions.png`| Network-level boxplots (log y) per shared parameter |
| `fig_cross_plots.png`  | fAbs~EC, OC~EC, Fe~PM₂.₅ scatter + OLS per network |
| `fig_ratios.png`       | OC/EC, fAbs/EC (≈ MAC), Fe/PM₂.₅ ratios by network |

## Tables (`tables/`)

| File | What |
|---|---|
| `improve_parameter_coverage.csv` | IMPROVE per-parameter n / n_sites / percentiles |
| `shared_coverage_summary.csv`    | Per-network n, n_sites, date range, median for each shared parameter |
| `relationship_fits.csv`          | Per-network OLS n/slope/intercept/R² for the three relationships |
| `ratio_stats.csv`                | Per-network median/IQR for OC/EC, fAbs/EC, Fe/PM₂.₅ |

## Notes

- The maps fall back to a plain lon/lat scatter if `cartopy` isn't installed (it isn't,
  in the current env). Install `cartopy` for coastlines/borders.
- The SPARTAN side of Section 3 is the **4-site FTIR/HIPS subset** in
  `unified_filter_dataset.pkl`, not the full 40-site network shown in Section 1.
- MAC (fAbs/EC) pairs SPARTAN **FTIR** EC with HIPS fAbs vs IMPROVE thermal-optical EC —
  the main caveat to firm up next.
