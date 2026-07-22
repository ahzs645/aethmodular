"""Convert the percent-format run_char_*.py scripts into executed notebooks.

Usage: python scripts/build_notebooks.py [01 02 ...]
Run from research/ftir_etad_char/. Same convention as phase 3: the committed
notebook is always the product of a real top-to-bottom run, so every number and
figure in it came from the code above it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SCRIPTS = {
    "01": ("run_char_01.py", "char_01_reference_spectra_survey.ipynb"),
    "02": ("run_char_02.py", "char_02_addis_vs_charcoal.ipynb"),
    "03": ("run_char_03.py", "char_03_burned_vs_unburned.ipynb"),
    "04": ("run_char_04.py", "char_04_spectral_typology.ipynb"),
    "05": ("run_char_05.py", "char_05_addis_vs_unburned_feedstock.ipynb"),
    "06": ("run_char_06.py", "char_06_dry_season_anomaly.ipynb"),
    "07": ("run_char_07.py", "char_07_char05_without_dry_season.ipynb"),
    "08": ("run_char_08.py", "char_08_fire_vs_furnace_char.ipynb"),
    "09": ("run_char_09.py", "char_09_char08_dry_vs_nondry.ipynb"),
    "10": ("run_char_10.py", "char_10_seasonal_spectra.ipynb"),
    "11": ("run_char_11.py", "char_11_normalization_alternatives.ipynb"),
}


def script_to_cells(text: str) -> list:
    cells = []
    kind, lines = None, []

    def flush():
        nonlocal kind, lines
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if not body.strip():
            kind, lines = None, []
            return
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
    script_name, notebook_name = SCRIPTS[number]
    text = (Path("scripts") / script_name).read_text()
    cells = script_to_cells(text)
    # nbclient runs headless (Agg): switch the first code cell to the inline
    # backend so plt.show() embeds figures instead of warning.
    first_code = next(c for c in cells if c.cell_type == "code")
    first_code.source = "%matplotlib inline\n" + first_code.source
    notebook = nbformat.v4.new_notebook(cells=cells)
    client = NotebookClient(
        notebook, timeout=3600, resources={"metadata": {"path": "."}}
    )
    client.execute()
    nbformat.write(notebook, notebook_name)
    print(f"executed and wrote {notebook_name}")


if __name__ == "__main__":
    for number in sys.argv[1:] or list(SCRIPTS):
        build(number)
