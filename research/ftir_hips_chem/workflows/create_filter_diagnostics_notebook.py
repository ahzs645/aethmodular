"""Create an executed notebook for the six frozen-cohort scientific figure families."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
NAME = "filter_only_diagnostics.ipynb"


def main():
    guide = (ROOT / "AGENTS.md").read_text()
    setup = (
        guide.split("## Standard notebook setup cell", 1)[1]
        .split("```python\n", 1)[1]
        .split("```", 1)[0]
    )
    setup = setup.replace(
        "PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)",
        "PlotConfig.set(sites='all', layout='individual', show_stats=False, show_1to1=False)",
    )
    pages = [
        (
            "01_site_relationships",
            "Within-site HIPS versus FTIR-predicted EC",
            "The diagnostic population contains 545 eligible physical-filter pairs. Open rings mark the 65 predictions that remain diagnostic but cannot be divided into HIPS under the baseline rule. Fits are descriptive unweighted OLS with an intercept; their R² values do not establish independent EC validation. Site-specific axes retain negative predictions. Both variables have measurement error, so slopes should not be read as unbiased calibration coefficients.",
        ),
        (
            "02_ratio_distributions",
            "Ratio distributions",
            "All 480 eligible ratios are shown, with medians and interquartile boxes. Ratios use the original HIPS value in Mm⁻¹ divided by the original FTIR-predicted EC in µg m⁻³. No denominator is substituted. The site distributions overlap substantially; differences in their medians are descriptive and do not isolate source, seasonal or instrument effects. All individual points, including the large Addis ratio, remain visible.",
        ),
        (
            "03_ratio_denominators",
            "Denominator behavior",
            "Open rings identify EC from 1 to less than 2 times its own reported MDL. This annotation is not an added exclusion. An inverse relationship with the denominator can arise algebraically; no regression or causal interpretation is applied to this panel. The report and denominator-strata table give counts and medians near and above that range.",
        ),
        (
            "04_ratio_dates",
            "Reported-date patterns",
            "These are reported filter dates, not verified active periods. No observations are interpolated or connected through gaps. Dashed lines are overall site medians. The graph suggests date structure, especially at Addis, but the analysis does not infer a mechanism or fit a temporal trend. Full date ranges and point links are retained in the tables.",
        ),
        (
            "05_denominator_sensitivity",
            "Sensitivity to stricter denominators",
            "The declared descriptive grid is 1, 1.5, 2, 3 and 5 times the filter-specific MDL. The original 480-pair eligibility is unchanged. Each stricter scenario reports its own count and median, with every selected filter linked in sensitivity_point_links.parquet. A missing median means no retained points, not a zero ratio. No threshold is optimized for measurement agreement.",
        ),
        (
            "06_registry_context",
            "Registered exclusion",
            "The original Delhi extreme point remains visible in red and traceable to INDH-0172. Its registered exclusion predates this report. The fit uses only the 62 eligible Delhi diagnostic pairs and ends within their EC range. This audit view does not reintroduce the point into the 545-pair population.",
        ),
    ]
    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (aethmodular)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        }
    )
    nb.cells = [
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell(
            "# Filter-only scientific diagnostics\n\nSix reproducible figure families. The workflow verifies frozen input hashes, reapplies the canonical exclusion registry to original replicate IDs, and uses `get_clean_data` before diagnostic statistics. The 480 ratio points must be a subset of the 545 diagnostic points. ChemSpec and instrument timing gaps do not alter these populations. No uncertainty-weighted or independent EC calibration fit is performed."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nfrom IPython.display import display, Image\nsys.path.insert(0, './workflows')\nfrom analyze_filter_diagnostics import main\nfrom plotting.filter_diagnostics import FIGURES\nTABLES = main()\nPLOTS = Path('output/plots/filter_diagnostics')\npoints = pd.read_parquet(TABLES / 'analysis_points.parquet')\ndisplay(pd.read_parquet(TABLES / 'site_results.parquet'))"
        ),
    ]
    for key, title, note in pages:
        nb.cells.extend(
            [
                nbf.v4.new_markdown_cell(f"## {title}\n\n{note}"),
                nbf.v4.new_code_cell(
                    f"display(Image(filename=str(PLOTS / '{key}.png')))\n# Regenerated by main() using the corresponding function in plotting.filter_diagnostics."
                ),
            ]
        )
    nb.cells.extend(
        [
            nbf.v4.new_markdown_cell(
                "## Traceability and retrieval priorities\n\nThe queue uses reported envelope overlap for retrieval only. The earlier ChemSpec CSVs already contain the conflicting EC values; the recovered historical importer copies Value and MDL separately. Run logs and scoped instrument history are still required for a real interval subset."
            ),
            nbf.v4.new_code_cell(
                "queue = pd.read_parquet(TABLES / 'evidence_recovery_queue.parquet')\ndisplay(queue.loc[queue.priority <= 2, ['site', 'base_filter_id', 'reported_start_utc', 'missing_evidence']].groupby('site').head(4))\ndisplay(pd.read_parquet(TABLES / 'ratio_exclusion_reasons.parquet')[['site','reason','count']])\nprint(TABLES / 'results_report.md')"
            ),
        ]
    )
    nbf.write(nb, AREA / NAME)
    NotebookClient(
        nb, timeout=240, kernel_name="python3", resources={"metadata": {"path": str(AREA)}}
    ).execute()
    executed = AREA / "notebooks/archive/executed" / NAME
    nbf.write(nb, executed)
    images = sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", []))
    assert images == 6, images
    print(f"Executed notebook with {images} figure families: {executed}")


if __name__ == "__main__":
    main()
