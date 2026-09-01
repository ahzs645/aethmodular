"""Build and execute the ftir_37-onward reader-facing notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


SCRIPTS = {
    "37": ("run_ftir_37.py", "ftir_37_hips_loading_and_curvature.ipynb"),
    "38": ("run_ftir_38.py", "ftir_38_1617_and_dust_attribution.ipynb"),
    "39": ("run_ftir_39.py", "ftir_39_pls_applicability_domain.ipynb"),
    "40": ("run_ftir_40.py", "ftir_40_quartz_tor_campaign_design.ipynb"),
    "41": ("run_ftir_41.py", "ftir_41_adama_three_method_reconciliation.ipynb"),
    "42": ("run_ftir_42.py", "ftir_42_target_definition_experiment.ipynb"),
    "43": ("run_ftir_43.py", "ftir_43_residual_learner_null_control.ipynb"),
    "44": ("run_ftir_44.py", "ftir_44_adama_through_locked_calibrations.ipynb"),
    "45": ("run_ftir_45.py", "ftir_45_residual_increment_attribution.ipynb"),
    "46": ("run_ftir_46.py", "ftir_46_paired_increment_and_mass.ipynb"),
    "47": ("run_ftir_47.py", "ftir_47_blank_lines_per_deployed_line.ipynb"),
    "48": ("run_ftir_48.py", "ftir_48_pyrolysis_split_target.ipynb"),
    "49": ("run_ftir_49.py", "ftir_49_ma350_raw_chain_diagnostics.ipynb"),
    "50": ("run_ftir_50.py", "ftir_50_spectral_comparison_methods.ipynb"),
    "51": ("run_ftir_51.py", "ftir_51_priority_figure_export.ipynb"),
    "52": ("run_ftir_52.py", "ftir_52_network_spectral_map.ipynb"),
}


def script_to_cells(text: str) -> list:
    cells, kind, lines = [], None, []

    def flush() -> None:
        nonlocal kind, lines
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if body.strip():
            if kind == "markdown":
                body = "\n".join(
                    line[2:] if line.startswith("# ") else ("" if line == "#" else line)
                    for line in body.splitlines()
                )
                cells.append(nbformat.v4.new_markdown_cell(body))
            else:
                cells.append(nbformat.v4.new_code_cell(body))
        kind, lines = None, []

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind = "markdown"
        elif line.startswith("# %%"):
            flush()
            kind = "code"
        else:
            lines.append(line)
    flush()
    return cells


def build(number: str) -> None:
    script, notebook = SCRIPTS[number]
    cells = script_to_cells((Path("scripts") / script).read_text())
    first_code = next(cell for cell in cells if cell.cell_type == "code")
    first_code.source = "%matplotlib inline\n" + first_code.source
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(nb, timeout=1800, resources={"metadata": {"path": "."}}).execute()
    nbformat.write(nb, notebook)
    print(f"executed and wrote {notebook}")


if __name__ == "__main__":
    for number in (sys.argv[1:] or SCRIPTS):
        build(number)
