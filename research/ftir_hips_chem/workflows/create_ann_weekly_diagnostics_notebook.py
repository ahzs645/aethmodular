"""Execute additional analyses and export presentation-ready notebook figures."""

import ast
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "research/ftir_hips_chem"
NAME = "ann_weekly_20260910_diagnostics.ipynb"
# Reuse the established standard setup without importing the executing builder.
tree = ast.parse((ACTIVE / "workflows/create_ann_weekly_figure_notebook.py").read_text())
setup = next(
    ast.literal_eval(n.value)
    for n in tree.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "setup" for t in n.targets)
)
setup += "\nfrom plotting.ann_weekly_diagnostics import AnnWeeklyDiagnostics\n"

pages = [
    (
        13,
        "Paired comparison: same Addis filters",
        "paired_comparison()",
        "Each candidate is compared with the historical calibration on identical filters. Positive differences mean greater RMS discrepancy from HIPS Fabs/MAC. Error bars are 95% percentile intervals from 4,000 paired whole-month resamples. These are conditional, exploratory discrepancies, not chemical-EC validation errors.",
    ),
    (
        14,
        "Cross-season transfer",
        "transfer_heatmap()",
        "Each column uses the same Addis filters for all five calibration choices. Rows differ only in the already fitted calibration. The left panel measures agreement with the HIPS proxy; the right measures squared correlation. This distinguishes a model effect from season composition without treating correlation as accuracy.",
    ),
    (
        15,
        "Monthly discrepancy",
        "monthly_bias()",
        "Monthly mean differences are filter-weighted within each observed month. The season-specific series applies the corresponding seasonal model. Missing months break lines; month-level sample counts are saved. A pattern here is descriptive and does not establish a physical mechanism.",
    ),
    (
        16,
        "Analog selection stability",
        "stability_plot()",
        "For each target group, resample complete observed year-month blocks 200 times, recompute its median spectrum, and rerank all eligible source spectra using signed Pearson correlation with CO₂ excluded. Select 500 unique physical filters each time. The baseline must exactly match the original saved cohort. Boxes span quartiles; whiskers span the 5th–95th percentiles of resamples, not confidence intervals. The 80% line is a descriptive stability marker, not an acceptance standard.",
    ),
    (
        17,
        "Spectral shape and reconstruction",
        "shape_map()",
        "Center and unit-normalize every spectrum over the retained channels as in Pearson matching. Fit a fixed ten-component PCA to all unique TOR-eligible IMPROVE filters and project Addis without refitting. The map shows only two PCs. The right panel counts Addis spectra whose ten-PC squared reconstruction residual exceeds the source 95th percentile. This is a descriptive source-relative threshold, not a calibrated rejection rule or a prediction-error bound. No chemical identities are assigned to PCs.",
    ),
    (
        18,
        "Source-site concentration",
        "diversity_plot()",
        "Each primary cohort has 500 unique filters. Distinct site count is compared with the inverse-Simpson effective number of sites, 1/Σp², where p is a site's fraction of filters. This indicates concentration, not an effective independent sample size. The curve retains all source sites and the underlying CSV names each site.",
    ),
]

nb = nbf.v4.new_notebook(
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
)
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Additional Addis FTIR diagnostics\n\n"
        "Six additional calculations and figures for the weekly discussion. Run this active source notebook from `research/ftir_hips_chem/` in the repository's uv environment. "
        "The executed copy is archived in `notebooks/archive/executed/`. Inputs are the completed September 10 analysis tables and corrected spectral caches. "
        "Seasons use `dry_feb`; exclusions follow the canonical registry. No model is refit or promoted, and the proposed Addis holdout is not scored.\n\n"
        "Methods references: [SciPy paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) "
        "describes keeping paired resamples aligned; here whole year-month clusters are resampled together. "
        "[scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) "
        "documents learning source components and projecting other observations. PCA input normalization is explicitly defined below."
    ),
    nbf.v4.new_code_cell(setup),
    nbf.v4.new_code_cell(
        "report = AnnWeeklyDiagnostics(Path.cwd().parents[1])\nreport.calculate()"
    ),
]
for number, title, method, explanation in pages:
    nb.cells.extend(
        [
            nbf.v4.new_markdown_cell(f"## {title}\n\n{explanation}"),
            nbf.v4.new_code_cell(
                f"fig = report.{method}\npng = report.save(fig, {number})\ndisplay(fig)\nplt.close(fig)\nprint(png)"
            ),
        ]
    )
nbf.write(nb, ACTIVE / NAME)
NotebookClient(
    nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ACTIVE)}}
).execute()
destination = ACTIVE / "notebooks/archive/executed" / NAME
nbf.write(nb, destination)
assert (
    sum("image/png" in out.get("data", {}) for c in nb.cells for out in c.get("outputs", [])) == 6
)
print(f"Executed notebook: {destination}")
