# AGENTS.md

Guidance for AI agents (and new humans) working in this repo. Lives at the
repo root so it's discoverable from any working directory. Read this before
touching notebooks or `scripts/`.

## Scope

This repo has several subprojects; the active analysis area is
**`research/ftir_hips_chem/`** — multi-site aethalometer + filter chemistry
across Beijing, Delhi, JPL, and Addis Ababa. The conventions below apply
there. Other areas (`src/`, top-level `scripts/`, `notebooks/archive/`) use
different patterns — do not mix them.

## Python environment

Use `/opt/anaconda3/bin/python3.13`. The system `python3` lacks pandas.

## Where things live (paths from repo root)

```
aethmodular/
├── AGENTS.md                            # <-- this file
├── plotting_gaps_scenarios.ipynb        # gallery of inline patterns to fix
├── README.md
├── docs/                                # repo-layout.md, research-workflow.md, etc.
├── src/                                 # production visualization system (separate)
├── research/ftir_hips_chem/             # <-- the main active workspace
│   ├── scripts/                         # all reusable logic lives here
│   │   ├── __init__.py                  # re-exports everything; one-stop import
│   │   ├── config.py                    # SITES, paths, MAC_VALUE, flow periods
│   │   ├── outliers.py                  # EXCLUDED_SAMPLES + exclusion API
│   │   ├── data_matching.py             # load + match aeth/filter/ETAD data
│   │   ├── flow_periods.py              # before/after flow-fix period helpers
│   │   ├── plotting/                    # standardized plot package
│   │   │   ├── __init__.py              # PlotConfig (global state)
│   │   │   ├── crossplots.py            # scatter + regression
│   │   │   ├── timeseries.py            # time-series BC/wavelength
│   │   │   ├── distributions.py         # histograms, boxplots, CDFs
│   │   │   ├── comparisons.py           # before/after, threshold tiles
│   │   │   └── utils.py                 # regression stats, axis helpers
│   │   └── plotting_legacy.py           # DEPRECATED — do not use
│   ├── addis_01..05_*.ipynb             # Addis analyses (use scripts/)
│   ├── Filter Data/                     # raw + unified filter datasets
│   ├── processed_sites/                 # resampled aethalometer pickles
│   └── output/                          # generated figures (git-ignored usually)
```

All paths in this doc are **relative to repo root** unless they appear inside
a notebook setup cell (where they're relative to the notebook that contains
them).

## Standard notebook setup cell

Every notebook under `research/ftir_hips_chem/` should start with this. Do
not invent your own plot style or redefine season colors inline.

```python
import sys
# For notebooks inside research/ftir_hips_chem/:
sys.path.insert(0, './scripts')
# For notebooks at repo root (rare — e.g. plotting_gaps_scenarios.ipynb):
# sys.path.insert(0, './research/ftir_hips_chem/scripts')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Config + data
from config import SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH, MAC_VALUE

# Exclusions (use these — do not hand-filter)
from outliers import (
    EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
    apply_exclusion_flags, apply_threshold_flags,
    get_clean_data, print_exclusion_summary,
)

# Data loading / matching
from data_matching import (
    load_aethalometer_data, load_filter_data,
    match_aeth_filter_data, match_all_parameters,
    load_etad_factor_contributions, match_etad_factors,
)

# Plotting — importing the package auto-applies the white-background default
# style (apply_default_style()). Do NOT call plt.style.use('seaborn-v0_8-darkgrid')
# afterwards — it re-adds the grey axes facecolor we don't want.
from plotting import PlotConfig, crossplots, timeseries, distributions, comparisons
from plotting.utils import calculate_regression_stats

PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)
```

### Plot style — why the background is white

The plotting package applies a white-background default the moment you
`import plotting` (via `apply_default_style()` in `plotting/__init__.py`).
This matches the look in `Analysis_Tasks_Jan2025.ipynb` and prints/publishes
cleanly. If a notebook upstream called `plt.style.use('seaborn-v0_8-darkgrid')`
and you want to reset mid-session:

```python
from plotting import apply_default_style
apply_default_style()
```

If you genuinely want a different style (e.g. ggplot for one figure), call
`plt.style.use(...)` **after** importing plotting, and call
`apply_default_style()` when you're done.

## The standard analysis recipe

Load → match → flag exclusions → get clean data → plot. Every analysis
notebook should follow this flow unless there's a documented reason not to.

```python
# 1. Load
aeth = load_aethalometer_data()            # dict[site] -> DataFrame
filters = load_filter_data()                # DataFrame

# 2. Match per site
matched = match_aeth_filter_data(
    'Beijing', aeth['Beijing'], filters, SITES['Beijing']['code']
)

# 3. Flag exclusions (adds is_excluded / exclusion_reason columns)
matched = apply_exclusion_flags(matched, 'Beijing')
matched = apply_threshold_flags(matched, 'Beijing')  # optional

# 4. Audit
print_exclusion_summary(matched, 'Beijing')

# 5. Clean subset for stats & plots
clean = get_clean_data(matched)

# 6. Plot — use the module, not inline matplotlib
crossplots.bc_vs_ec(clean)
timeseries.bc(aeth['Beijing'], wavelength='IR')
```

**Never** drop rows by hand (`df = df[df['col'] < x]`) for exclusion purposes.
Add the sample to `outliers.EXCLUDED_SAMPLES` with a reason, or add a rule to
`MANUAL_OUTLIERS`. The exclusion system is *flagging-based* (non-destructive):
it adds `is_excluded` / `is_outlier` columns so you can always inspect what
was removed and why.

### Adding a new exclusion

Edit `scripts/outliers.py`, append to `EXCLUDED_SAMPLES[<site>]`:

```python
{
    'date': 'YYYY-MM-DD',
    'filter_id': 'SITE-XXXX' or None,
    'aeth_bc_approx': <value>,
    'filter_ec_approx': <value>,
    'reason': 'Why this sample is excluded (be specific — EC contamination,
               flow anomaly, instrument malfunction, etc.)',
}
```

Every exclusion needs a `reason`. Reviewers read these.

## Plotting module API

### PlotConfig (global state)

Set once at the top of a notebook. All plotting functions read from it.

| Key | Default | Meaning |
|---|---|---|
| `sites` | `'all'` | `'all'`, a list, or a single site name |
| `layout` | `'individual'` | `'individual'`, `'grid'`, `'combined'` |
| `figsize` / `figsize_grid` | `(10,8)` / `(14,12)` | figure dims |
| `show_stats` | `True` | overlay R²/slope/n box |
| `show_1to1` | `True` | 1:1 reference line |
| `equal_axes` | `True` | 1:1 aspect ratio |
| `marker_size` | `80` | scatter marker size |
| `font_size` / `title_size` | `11` / `13` | typography |
| `dpi` | `100` | figure dpi |

Helpers: `PlotConfig.get_site_color(site)`, `PlotConfig.get_sites_list()`,
`PlotConfig.reset()`, `PlotConfig.show()`.

### What's available

| Module | Functions | Use for |
|---|---|---|
| `crossplots` | `scatter`, `bc_vs_ec`, `hips_vs_ftir`, `hips_vs_aeth`, `with_iron_gradient` | scatter + regression with per-site or combined layout |
| `timeseries` | `bc`, `bc_multiwavelength`, `data_completeness`, `filter_vs_aeth`, `flow_ratio` | time-series BC, completeness, flow QC |
| `distributions` | `bc_boxplot`, `wavelength_boxplot`, `smooth_raw_histogram`, `uv_ir_ratio_histogram`, `correlation_matrix` | distributions & correlation heatmaps |
| `comparisons` | `before_after_outliers`, `threshold_analysis`, `flow_periods`, `summary_bars` | before/after panels, threshold sweeps |
| `utils` | `calculate_regression_stats`, `add_regression_line`, `add_one_to_one_line`, `add_stats_textbox`, `style_axes`, `create_grid_layout` | low-level helpers if extending a plot |

### Layouts

`layout='individual'` → one figure per site.
`layout='grid'` → all sites in a 2×2 grid.
`layout='combined'` → all sites overlaid on one axes (color by site).

### Outlier visualization in plots

`crossplots.scatter(..., outlier_col='is_excluded')` highlights excluded points
as red X's on the "before" panel in `comparisons.before_after_outliers`. For
standalone scatters the default is to plot clean data only.

## Known gaps (where you still need inline matplotlib)

The module covers the common cases but not everything. When you hit these,
add inline matplotlib **on top of** a standard plot (retrieve the figure/axes
via `plt.gcf()` / `plt.gca()`) rather than rebuilding from scratch. If you
find yourself doing any of these in more than one notebook, raise it — it's
a candidate for promotion into `plotting/overlays.py`.

Current inline patterns used across notebooks:

1. **Seasonal/period shading** — Ethiopia seasons (`Dry`, `Belg Rainy`, `Kiremt
   Rainy`) shaded via `ax.axvspan`. Colors currently redefined per notebook as:
   ```python
   SEASON_COLORS = {
       'Dry Season':         '#E67E22',  # orange
       'Belg Rainy Season':  '#27AE60',  # green
       'Kiremt Rainy Season':'#3498DB',  # blue
   }
   ```
   If you add a notebook that needs these, **copy this dict exactly** so output
   stays consistent. Planned: `plotting.overlays.add_seasonal_shading(ax)` +
   `config.ETHIOPIA_SEASONS`.

2. **Reference / threshold lines** — `ax.axhline`, `ax.axvline` for flow ratio
   ideals (1.0, 2.0), smooth/raw thresholds (1, 2.5, 4, 5 %), AAE source-region
   boundaries (fossil fuel / mixed / biomass). Some are hardcoded inside
   `timeseries.flow_ratio` and `distributions.smooth_raw_histogram` — don't
   duplicate those calls on top.

3. **Event date markers** — flow-fix dates are in
   `config.FLOW_FIX_PERIODS[<site>]`. Draw with
   `ax.axvline(pd.to_datetime(FLOW_FIX_PERIODS[site]['before_end']), ...)`.
   Do **not** hardcode the date.

4. **Uncertainty bands** — `ax.fill_between(x, q25, q75, alpha=0.08)` for
   percentile ranges. Used in `smoothing_comparison.ipynb` and `addis_05*`.

5. **PMF region highlights** — `ax.axvspan(pmf_min, pmf_max, alpha=0.08)`
   for date-range emphasis (ETAD factor analysis).

## Notebook coverage (current state)

| Notebook | Plotting module | Exclusions | Notes |
|---|---|---|---|
| `Analysis_Tasks_Jan2025` | ✅ | ✅ | |
| `Example_Modular_Analysis` | ✅ | ✅ | template for new analyses |
| `FlowFix_BeforeAfter_Analysis` | ✅ | ✅ | inline flow-fix date markers |
| `Task_Analysis_Notebook` | ✅ | ✅ | |
| `primary_tasks_notebook` | ✅ | ✅ | |
| `addis_01_source_apportionment` (+`_daily`) | ✅ | ✅ | inline AAE region shading |
| `addis_02_temporal_patterns` (+`_daily`) | ✅ | ✅ | inline seasonal shading |
| `addis_03_meteorology` (+`_daily`) | — | ✅ | custom met plots |
| `addis_05_diurnal_wavelength_analysis` | — | ✅ | inline uncertainty bands |
| `multisite_diurnal_wavelength_analysis` | ✅ | ✅ | |
| `HIPS_Aeth_SmoothRaw_Analysis` | ✅ | ❌ | smooth/raw compare — exclusions skipped by design |
| `addis_04_aeronet` (+`_daily`) | ❌ | ❌ | AERONET-only; not wired in yet |
| `source_apportionment_regression` | ❌ | ❌ | legacy — migrate before extending |

When creating a new notebook, follow the `Example_Modular_Analysis.ipynb` or
`addis_01_source_apportionment.ipynb` pattern.

## Data join quirks (easy to get wrong)

- **PMF GF fractions need normalization.** Raw `GF1`–`GF5` values in
  `ETAD Factor Contributions .csv` are PM2.5 mass fractions (sum ≈ 0.03–0.46
  per row), not relative source contributions. Divide each GF by its row sum
  before using as fractions. Without this, `dominant_fraction` max is ~0.24
  and no samples cross a 30 % threshold. With normalization, mean ≈ 46 %.
- **ETAD join chain:** `ETAD Factor Contributions .csv` (oldDate `M/D/YYYY`) →
  `ETAD Filter ID.csv` (oldDate `YYYY-MM-DD`) → merged on parsed `date`. Use
  `load_etad_factors_with_filter_ids()`. `base_filter_id` strips the `-N`
  suffix to match `unified_filter_dataset.pkl` FilterId format
  (`ETAD-0035-3` → `ETAD-0035`).
- **Date-column notebooks** (`addis_02.5_temporal_patterns_filter`) use
  `pd.merge(df, factor_merge, on='date')`. All other notebooks map via
  datetime index: `df.index.normalize().tz_localize(None)`.

## Agent rules

Do:

- Always start from the standard setup cell. Import from `scripts/`, not by
  retyping paths or redefining `MAC_VALUE`, site colors, etc.
- Use `apply_exclusion_flags` + `get_clean_data`. Document new exclusions
  with a `reason`.
- Use `PlotConfig.set(...)` once per notebook rather than passing `figsize`
  to every call.
- Prefer `crossplots.scatter` and friends over raw `plt.scatter`.
- When an inline overlay is needed (seasonal shading, event line, etc.),
  copy the canonical pattern from this file's "Known gaps" section so
  colors and semantics stay consistent.

Don't:

- Don't hand-filter bad samples with `df[df['col'] < x]` for exclusion —
  that silently removes the audit trail.
- Don't `import matplotlib` and build a plot from scratch when a module
  function covers it. If a function is close-but-not-quite, post-process the
  returned axes rather than bypassing.
- Don't use `plotting_legacy.py`. It's deprecated.
- Don't redefine site colors — read from `SITES[site]['color']` or
  `PlotConfig.get_site_color(site)`.
- Don't commit generated PNGs unless asked.
- Don't hardcode flow-fix dates — read from `config.FLOW_FIX_PERIODS`.

## Quick reference

```python
# Set up
PlotConfig.set(sites='all', layout='individual')

# Standard exclusion flow
matched = apply_exclusion_flags(matched, site_name)
matched = apply_threshold_flags(matched, site_name)
print_exclusion_summary(matched, site_name)
clean = get_clean_data(matched)

# Standard plots
crossplots.bc_vs_ec(clean)
crossplots.hips_vs_ftir(clean)
timeseries.bc(aeth[site], wavelength='IR')
distributions.bc_boxplot(clean)
comparisons.before_after_outliers(matched, x_col='aeth_bc', y_col='ec')
```
