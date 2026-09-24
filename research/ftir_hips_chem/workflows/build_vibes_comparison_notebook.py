"""Build the reviewable, executable VIBES/AIRSpec comparison notebook."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research/ftir_hips_chem"
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# Addis FTIR: VIBES versus AIRSpec baseline correction

Compare the downloaded **pyvibes 1.0.0** with the existing validated AIRSpec port
(**DF1=6, DF2=4**) on identical Addis/ETAD PM2.5 spectra.

This is a **spectral preprocessing evaluation**, not a new EC calibration.
AIRSpec is a reference method, not known chemical truth. No target concentrations,
HIPS measurements, or AIRSpec outputs are used to train VIBES.

Run from the repository environment: `uv sync --extra vibes`, then select its
Python kernel and **Run All**. Raw inputs resolve through the existing Drive-path
configuration. The bundled upstream source is preserved under `vendor/pyvibes/`
with its MIT license and source hashes; the Downloads folder is no longer needed.
"""),
    code("""import sys
from pathlib import Path
# Standard analysis setup, also works from the archived executed-copy directory.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'pyproject.toml').exists())
AREA = ROOT / 'research/ftir_hips_chem'
sys.path.insert(0, str(AREA / 'scripts'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from config import (SITES, PROCESSED_SITES_DIR, FILTER_DATA_PATH,
                    AERONET_DATA_DIR, WEATHER_DATA_DIR, MAC_VALUE)
from outliers import (EXCLUDED_SAMPLES, MANUAL_OUTLIERS, apply_exclusion_flags,
                      apply_threshold_flags, get_clean_data, print_exclusion_summary)
from data_matching import (load_aethalometer_data, load_filter_data,
                           match_aeth_filter_data, match_all_parameters)
from etad_factors import load_etad_factor_contributions, match_etad_factors
from aeronet import load_aeronet, aeronet_dir, COLS as AERONET_COLS
from improve_io import load_improve_clean
from plotting import PlotConfig, crossplots, timeseries, distributions, comparisons
from plotting.utils import calculate_regression_stats
PlotConfig.set(sites='Addis_Ababa', layout='individual', show_stats=True, show_1to1=True)
from vibes_comparison import run_comparison
from plotting.vibes_comparison import comparison_figures
TABLES = AREA / 'output/tables/vibes_vs_airspec'
PLOTS = AREA / 'output/plots/vibes_vs_airspec'
PLOTS.mkdir(parents=True, exist_ok=True)
"""),
    md("""## Protocol fixed before evaluating samples

- Average replicate scans by **MediaId (physical filter)** before either method.
- Use the same 2,002 retained channels in the AIRSpec analysis window, approximately
  1426–3998 cm⁻¹. No normalization, clipping of negatives, or grid interpolation.
- Apply the canonical exclusion registry and keep an inclusion ledger. PM10 is
  outside scope; this spectrum-only comparison does not require HIPS availability.
  Aethalometer-specific threshold flags are inapplicable here.
- Identify field blanks from **ExternalFilterType = FB**, never low sample loading.
  Hold out approximately 25% within each known/unknown lot with seed 20260920.
  All scans of a physical blank stay together. Blanks are pooled for the background
  PCA; lot-specific residual diagnostics show where that assumption may matter.
- Select PCA rank with upstream blank-only leave-one-out one-standard-error rule,
  scanning up to `n_training_blanks − 2`. Fit VIBES with **PB loss, τ=0.1** and
  10,000 L-BFGS-B iterations, then its upstream MAP/CLARABEL solver. A failed
  optimizer receives one logged continuation (up to 50,000 additional iterations,
  maxls=50). This adapter robustness step changes no objective or data.
  τ is fixed to the upstream script default; it is not optimized against AIRSpec.
- AIRSpec retains its existing DF1=6, DF2=4 settings. Each method gets identical
  raw sample rows. Failed VIBES fits are retained as explicit failures with NaN
  output; paired summaries use the same valid rows for both methods.

Field blanks may contain handling contamination. Their residual from zero is a
background diagnostic, not proof of analyte accuracy. Unknown lot IDs are retained
and labeled; no convenient blanks or spectra are silently discarded.
"""),
    code("""# Recompute from raw inputs every run (usually a few minutes).
summary = run_comparison(TABLES)
ledger = pd.read_csv(TABLES / 'inclusion_and_blank_split.csv')
metrics = pd.read_csv(TABLES / 'paired_sample_metrics.csv')
diagnostics = pd.read_csv(TABLES / 'vibes_fit_diagnostics.csv')
display(ledger.groupby(['role', 'LotId'], dropna=False).size().rename('filters').to_frame())
display(pd.Series({k: v for k, v in summary.items()
                   if k not in ('input_sha256', 'vendor_manifest', 'versions')}, name='Run summary').to_frame())
display(diagnostics.groupby('role').agg(attempted=('success', 'size'),
                                      successful=('success', 'sum'),
                                      max_iterations=('iterations', 'max')))
assert not set(ledger.loc[ledger.role.eq('blank_train'), 'MediaId']) & set(
    ledger.loc[ledger.role.isin(['blank_test', 'sample']), 'MediaId'])
figures = comparison_figures(TABLES)
def show_next():
    name, fig = next(figures)
    fig.savefig(PLOTS / (name + '.png'), bbox_inches='tight')
    plt.show()
    plt.close(fig)
"""),
    md("""## 1. Blank background and component selection

Bands show the interquartile range, not confidence intervals. The training and
held-out blanks are distinct physical filters. Component selection uses only the
training blanks; the rank curve is not a sample-accuracy curve."""),
    code("show_next()"),
    md("""## 2. Same-filter baseline and corrected-spectrum overlays

Examples are selected reproducibly near the 10th, 50th, and 90th percentiles of
paired RMS disagreement, so the plots show low, typical, and high disagreement.
Each row uses the same raw filter spectrum in both methods. The third column
zooms into the carbonyl / 1600 cm⁻¹ region."""),
    code("show_next()"),
    md("""## 3. Population spectra and channel-wise differences

Median and interquartile range use only paired valid spectra. Differences can
reflect background removal, analyte removal, or retained interference; the raw
spectra alone cannot identify which interpretation is correct."""),
    code("show_next()"),
    md("""## 4. Identical band features for both methods

The existing `ftir_source_band_features` function supplies local-continuum CH,
carbonyl and 1600 cm⁻¹ shoulder heights. These are spectral features, not chemical
concentrations. The shared crossplot helper reports OLS and Deming slopes on the
1:1 panels. Deming uses λ=1 as a descriptive equal-error assumption because neither
method has an independently measured band-height uncertainty here."""),
    code("""show_next()
valid = metrics.paired_valid
band_summary = []
for band in ('CH_peak', 'carbonyl_peak', 'shoulder_1600_peak'):
    a, v = metrics.loc[valid, 'AIRSpec_' + band], metrics.loc[valid, 'VIBES_' + band]
    band_summary.append({'band': band, 'n': len(a), 'AIRSpec_median': a.median(),
                         'VIBES_median': v.median(), 'median_paired_delta': (v-a).median(),
                         'median_absolute_delta': (v-a).abs().median(),
                         'correlation_R2': a.corr(v)**2})
band_summary = pd.DataFrame(band_summary)
band_summary.to_csv(TABLES / 'band_summary.csv', index=False)
display(band_summary)
"""),
    md("""## 5. Agreement distribution and lot dependence

Squared spectral correlation measures shape agreement; it is not held-out
predictive R². RMS retains sensitivity to offsets and amplitude. Lot labels come
from the existing ETAD lot map; missing mappings are explicitly shown."""),
    code("show_next()"),
    md("""## 6. Independent blank and known-addition tests

Held-out field blanks are processed by both methods. For recovery, add the same
four Gaussian peaks (1610, 1720, 2920, 3400 cm⁻¹; widths 30, 22, 35, 130 cm⁻¹) to
**each held-out blank** at three predetermined amplitudes (0.01, 0.05, 0.15).
The recovered increment is:

`correct(blank + added peaks) − correct(blank)`.

Compare that increment with the known added peaks. This cancels the unverified
assumption that field blanks have exactly zero analyte. It tests incremental
signal preservation on blank backgrounds, not recovery of all real aerosol
chemistry. The exported absolute RMSE against the added signal is secondary
because it includes each field blank's own residual. Repeated injections of a
blank are correlated and are not counted as independent physical filters."""),
    code("""show_next()
blanks = pd.read_csv(TABLES / 'heldout_blank_metrics.csv')
injections = pd.read_csv(TABLES / 'injection_metrics.csv')
blank_pairs = blanks.pivot(index='sample_id', columns='method', values='rms_from_zero').dropna()
injection_pairs = injections.pivot(index=['sample_id', 'added_amplitude'], columns='method', values='increment_rmse').dropna()
display(blank_pairs.agg(['count', 'median', 'max']))
display(injection_pairs.groupby(level='added_amplitude').median())
"""),
    md("""## 7. Runtime and convergence

Serial wall-clock correction time is reported on the same machine and sample set.
VIBES blank-PCA/rank-selection training is reported separately. Import time and
file I/O are excluded. Warnings and failures remain in `vibes_fit_diagnostics.csv`."""),
    code("""show_next()
display(pd.read_csv(TABLES / 'timing.csv'))
issues = diagnostics.loc[(~diagnostics.success) | diagnostics.warnings.fillna('').ne('') | diagnostics.retry_count.gt(0)]
display(issues if len(issues) else 'No retries, failed fits, or captured numerical warnings.')
"""),
    md("""## Interpretation and limits

The summary below is generated from the run, rather than a prewritten winner.
Smaller blank residual and better known-addition recovery are useful evidence,
but neither establishes that real-sample EC is more accurate. Field-blank
contamination and mixed/unknown lots remain limitations. The default asymmetry
parameter has not been tuned, and the component-selection rule measures blank
reconstruction, not analyte preservation.

**Do not feed VIBES-corrected spectra into an AIRSpec-trained PLS calibration.**
A downstream EC comparison requires applying each preprocessing method to its
calibration pool, refitting on identical training filters, selecting components
without the test set, and evaluating on the same locked TOR test split. This
notebook evaluates the baselining strategy and leaves existing calibrations intact.
"""),
    code("""blank_med = blank_pairs.median()
recovery_med = injection_pairs.median()
text = (f"Compared **{summary['paired_count']}/{summary['sample_count']}** paired PM2.5 filters. "
        f"Median whole-spectrum RMS difference: **{summary['median_rms_difference']:.5f} absorbance**; "
        f"median squared spectral correlation: **{summary['median_spectral_R2']:.3f}**.\\n\\n"
        f"Held-out blank median RMS: AIRSpec **{blank_med['AIRSpec']:.5f}**, "
        f"VIBES **{blank_med['VIBES']:.5f}**. Median incremental recovery RMSE: "
        f"AIRSpec **{recovery_med['AIRSpec']:.5f}**, VIBES **{recovery_med['VIBES']:.5f}**. "
        "These are diagnostic comparisons, not validation of a new EC calibration.")
display(Markdown(text))
(TABLES / 'summary.md').write_text(text + '\\n')
display(pd.Series(summary['versions'], name='Version').to_frame())
print('Tables:', TABLES)
print('Figures:', PLOTS)
"""),
]
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (aethmodular)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.13"},
}
nbf.write(nb, AREA / "ftir_vibes_vs_airspec.ipynb")
print(AREA / "ftir_vibes_vs_airspec.ipynb")
