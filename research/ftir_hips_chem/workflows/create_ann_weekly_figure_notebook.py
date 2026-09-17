"""Create and execute the weekly figure notebook without refitting models."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "research/ftir_hips_chem"
SOURCE = ACTIVE / "ann_weekly_20260910_figures.ipynb"
EXECUTED = ACTIVE / "notebooks/archive/executed/ann_weekly_20260910_figures.ipynb"

setup = """import sys
sys.path.insert(0, './scripts')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
    AERONET_DATA_DIR, WEATHER_DATA_DIR, MAC_VALUE,
)
from outliers import (
    EXCLUDED_SAMPLES, MANUAL_OUTLIERS,
    apply_exclusion_flags, apply_threshold_flags,
    get_clean_data, print_exclusion_summary,
)
from data_matching import (
    load_aethalometer_data, load_filter_data,
    match_aeth_filter_data, match_all_parameters,
)
from etad_factors import load_etad_factor_contributions, match_etad_factors
from aeronet import load_aeronet, aeronet_dir, COLS as AERONET_COLS
from improve_io import load_improve_clean
from plotting import PlotConfig, crossplots, timeseries, distributions, comparisons
from plotting.utils import calculate_regression_stats

PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True,
               figsize=(14.8, 5.6), font_size=18, title_size=22)

from pathlib import Path
from IPython.display import display
from plotting.ann_weekly_figures import AnnWeeklyFigures
"""

pages = [
    (2, "Spectral exclusion regions", "spectral_regions()"),
    (3, "Changes in membership and seasonal overlap", "membership_changes()"),
    (4, "Seasonal regressions", "seasonal_crossplots()"),
    (5, "PMF comparison", "pmf_comparison()"),
    (6, "Training membership and Bishoftu provenance", "provenance()"),
    (7, "Proposed Addis selection / validation split", "proposed_split()"),
    (8, "Deming 95% confidence intervals", "uncertainty()"),
    (9, "Calibration screening comparison", "benchmark()"),
    (10, "Dry: every training and Addis spectrum", "full_spectra(report.seasons[0])"),
    (11, "Belg: every training and Addis spectrum", "full_spectra(report.seasons[1])"),
    (12, "Kiremt: every training and Addis spectrum", "full_spectra(report.seasons[2])"),
]

nb = nbf.v4.new_notebook(
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
)
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Addis FTIR weekly figures\n\n"
        "Graphs for the September 10 weekly presentation, generated from the completed analysis tables. "
        "No model refitting or held-out scoring happens here.\n\n"
        "Run the source notebook from `research/ftir_hips_chem/` using the repository's uv environment. "
        "This executed copy is archived under `notebooks/archive/executed/`. "
        "The active source notebook is `research/ftir_hips_chem/ann_weekly_20260910_figures.ipynb`.\n\n"
        "The primary selection uses the median spectrum and removes 1800–2500 cm⁻¹. "
        "Seasons use `dry_feb`. Addis results are exploratory. The 166/73 split is an unscored "
        "retrospective proposal. Deming intervals condition on the chosen model, MAC and λ. "
        "TOR tests differ between cohorts. Full methods and audit trails are in the analysis report."
    ),
    nbf.v4.new_code_cell(setup),
    nbf.v4.new_code_cell(
        "report = AnnWeeklyFigures(Path.cwd().parents[1])\n"
        "print(f'Input tables: {report.tables}')\n"
        "print(f'PNG exports: {report.output}')\n"
        "print(f'Addis filters: {len(report.addis)}; MAC: {MAC_VALUE}')"
    ),
]
for slide, title, call in pages:
    nb.cells.append(nbf.v4.new_markdown_cell(f"## Slide {slide}: {title}"))
    nb.cells.append(
        nbf.v4.new_code_cell(
            f"fig = report.{call}\n"
            f"png = report.save(fig, {slide})\n"
            "display(fig)\n"
            "plt.close(fig)\n"
            "print(png)"
        )
    )
nbf.write(nb, SOURCE)
client = NotebookClient(
    nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ACTIVE)}}
)
client.execute()
EXECUTED.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, EXECUTED)
assert (
    sum(
        any(
            o.output_type == "display_data" and "image/png" in o.get("data", {})
            for o in c.get("outputs", [])
        )
        for c in nb.cells
    )
    == 11
)
print(f"Executed and saved {EXECUTED}")
