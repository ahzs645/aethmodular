#!/usr/bin/env python
"""Build the portable scientific handoff; preserve all completed analysis outputs."""

from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
import subprocess
import sys

import nbformat as nbf
from aethmodular_cli.env import display_path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

AREA = Path(__file__).resolve().parents[1]
REPO = AREA.parents[1]
RELEASE = REPO / "deliverables/filter_only_scientific_release_2026-09-11"
GROUPS = {
    "filter_diagnostics": "diagnostic",
    "filter_relationship_stability": "stability",
    "filter_proportionality": "proportionality",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_dependencies():
    seeds = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pyarrow",
        "seaborn",
        "nbformat",
        "nbclient",
        "ipykernel",
        "scikit-learn",
        "openpyxl",
        "statsmodels",
    ]
    seen = {}
    pending = list(seeds)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        d = metadata.distribution(name)
        seen[name] = d.version
        for spec in d.requires or []:
            r = Requirement(spec)
            if r.marker is None or r.marker.evaluate({"extra": ""}):
                pending.append(r.name)
    (RELEASE / "requirements.lock").write_text(
        "\n".join(f"{name}=={version}" for name, version in sorted(seen.items())) + "\n"
    )


def relative_links(text, source, destination, pathmap):
    """Convert known file links once in the release generator; retain external provenance as text."""
    import os

    def replace(match):
        prefix, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "#")):
            return match.group(0)
        clean = target.strip("<>")
        # Strip optional line locator only for resolving a real local file.
        clean = re.sub(r":\d+$", "", clean)
        resolved = (
            (source.parent / clean).resolve() if not Path(clean).is_absolute() else Path(clean)
        )
        mapped = pathmap.get(str(resolved))
        if mapped:
            return prefix + os.path.relpath(mapped, destination.parent) + ")"
        return prefix.split("](")[0] + "] (external source recorded in provenance)"

    return re.sub(r"(!?\[[^\]]*\]\()([^\n)]+)\)", replace, text)


def build():
    RELEASE.mkdir(parents=True, exist_ok=True)
    pathmap = {}
    originals = []
    for dirname, alias in GROUPS.items():
        src = AREA / "output/tables" / dirname
        manifest = json.loads((src / "manifest.json").read_text())
        for r in manifest["outputs"]:
            assert sha(Path(r["path"])) == r["sha256"], r["path"]
        target = RELEASE / "data" / alias
        target.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if not p.is_file():
                continue
            dest = target / p.name
            shutil.copy2(p, dest)
            pathmap[str(p.resolve())] = dest
            originals.append(
                {
                    "release_path": str(dest.relative_to(RELEASE)),
                    "original_path": display_path(p),
                    "sha256": sha(p),
                }
            )
        plotdir = AREA / "output/plots" / dirname
        for p in plotdir.glob("*"):
            dest = RELEASE / "figures" / alias / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            pathmap[str(p.resolve())] = dest
    # Both later phases refer to byte-identical diagnostic frozen inputs. Map those
    # known aliases to the one diagnostic copy; copied archival manifests are not executable paths.
    for dirname in ["filter_relationship_stability", "filter_proportionality"]:
        for p in (AREA / "output/tables" / dirname / "frozen_inputs").glob("*"):
            for target in (RELEASE / "data").glob("*"):
                candidate = target / p.name
                if candidate.exists() and sha(candidate) == sha(p):
                    pathmap[str(p.resolve())] = candidate
                    break
    for p in (AREA / "scripts").rglob("*.py"):
        dest = RELEASE / "scripts" / p.relative_to(AREA / "scripts")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    shutil.copy2(Path(__file__).parent / "release_assets/reproduce.py", RELEASE / "reproduce.py")
    authoring = RELEASE / "authoring"
    authoring.mkdir(exist_ok=True)
    for name in ["write_manuscript.py", "build_manuscript_docx.py"]:
        shutil.copy2(Path(__file__).parent / "release_assets" / name, authoring / name)
    (RELEASE / "requirements-documentation.txt").write_text("python-docx==1.2.0\n")
    for p in (REPO / "docs").glob("filter*spec*.md"):
        dest = RELEASE / "specifications" / p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dest)
        pathmap[str(p.resolve())] = dest
    # Keep historical source reports byte-for-byte under data; publish working,
    # relative-link copies separately. Evidence source rows remain immutable.
    docdir = RELEASE / "phase_reports"
    docdir.mkdir(exist_ok=True)
    for group, alias in GROUPS.items():
        src = AREA / "output/tables" / group
        for p in src.glob("*.md"):
            dest = docdir / (alias + "_" + p.name)
            pathmap[str(p.resolve())] = dest
    for group, alias in GROUPS.items():
        for p in (AREA / "output/tables" / group).glob("*.md"):
            dest = docdir / (alias + "_" + p.name)
            dest.write_text(relative_links(p.read_text(), p, dest, pathmap))
    (RELEASE / "release_config.json").write_text(
        json.dumps({"release": "filter_only_scientific_release", "schema": 1}) + "\n"
    )
    locked_dependencies()
    prior = nbf.read(AREA / "filter_proportionality.ipynb", as_version=4)
    setup = next(c.source for c in prior.cells if c.cell_type == "code")
    bootstrap = """import sys
import os
from pathlib import Path
hint = os.environ.get('FILTER_ONLY_RELEASE_ROOT')
search = [Path(hint).expanduser().resolve()] if hint else [Path.cwd(), *Path.cwd().parents]
RELEASE_ROOT = next((p for p in search if (p / 'release_config.json').exists()), None)
if RELEASE_ROOT is None:
    raise RuntimeError('Set FILTER_ONLY_RELEASE_ROOT to the extracted release directory.')
sys.path.insert(0, str(RELEASE_ROOT / 'scripts'))"""
    setup = setup.replace(
        "import sys\n# For notebooks inside research/ftir_hips_chem/:\nsys.path.insert(0, './scripts')",
        bootstrap,
    )
    cells = [
        nbf.v4.new_markdown_cell(
            "# Site dependent proportionality and temporal transfer\n\nA portable notebook for the completed filter-only analyses. It reads the included frozen tables and reproduces the specified fits; it does not search external data or fit another model."
        ),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell(
            "The exclusion flags and physical-filter memberships were frozen in the completed analysis. This release verifies their hashes rather than applying new exclusions."
        ),
        nbf.v4.new_code_cell(
            "import importlib.util\nfrom IPython.display import display\nspec = importlib.util.spec_from_file_location('release_reproduction', RELEASE_ROOT / 'reproduce.py')\nreproduction = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(reproduction)\ntables = reproduction.run()\nfrom plotting import filter_proportionality as charts\nb = tables['proportionality_blocks']\nib = tables['id11_training_blocks']"
        ),
    ]
    for title, code, note in [
        (
            "Reported products on the same physical filters",
            "from plotting.filter_diagnostics import relationship\npoints = pd.read_parquet(RELEASE_ROOT / 'data/diagnostic/analysis_points.parquet')\nfig = relationship(points)",
            "The diagnostic cohort includes 545 filters; ratios use 480. Below-MDL predictions remain flagged observations of a reported product, not substituted values.",
        ),
        (
            "Proportionality is site dependent",
            "fig = charts.paired_blocks(b)",
            "Positive differences favor the intercept. Addis wins all eight withheld quarters; the other sites do not support a uniform preference.",
        ),
        (
            "Limits of later-period prediction",
            "fig = charts.forward_errors(b)",
            "These folds use only earlier training data and revisit the same record as withheld-quarter evaluation. They are not independent replications.",
        ),
        (
            "ID-11 common-support sensitivity",
            "fig = charts.id11_errors(ib)",
            "Pooling IDs 11 and 17 is not required for the aggregate intercept advantage: within ID 11, proportional MAE is 9.968 versus 3.884 Mm⁻¹. Restricting training gives a separate, smaller OLS gain on 127 common filters.",
        ),
        (
            "Weighting and directional errors",
            "fig = charts.weighting(tables['proportionality_summary'])",
            "The Delhi final quarter supplies 68.4% of later-period test filters and 76.6% of intercept-model absolute error. Those are different quantities.",
        ),
    ]:
        cells += [
            nbf.v4.new_markdown_cell("## " + title),
            nbf.v4.new_code_cell(code + "\ndisplay(fig)\nplt.close(fig)"),
            nbf.v4.new_markdown_cell("**Notes.** " + note),
        ]
    cells.append(
        nbf.v4.new_markdown_cell(
            "Read the [scientific draft](../manuscript.md), [claim ledger](../claim_ledger.csv), [extended questions](../phase_reports/proportionality_upstream_questions_v2_draft.md) and [candidate evidence](../phase_reports/proportionality_USPA-0257_evidence_package.md)."
        )
    )
    cells.insert(
        1,
        nbf.v4.new_markdown_cell(
            "**What baseline means.** The frozen study baseline is the saved filter population, flags and analysis rules. The prediction baseline is the training-median HIPS value; proportional prediction is a separate comparison. Neither verifies upstream FTIR spectral baseline correction. See the [coverage map](../coverage_map.md) for the completed reports and pending evidence."
        ),
    )
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (RELEASE / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, RELEASE / "notebooks/filter_only_results.ipynb")
    (RELEASE / "README.md").write_text("""# Filter only scientific release

The modeling milestone is complete. Start with [the scientific draft](manuscript.md),
[coverage map](coverage_map.md), [claim ledger](claim_ledger.csv), and [figure ledger](figure_ledger.csv).

## Reproduce in a fresh environment

Python 3.13 is required for the tested release. With uv installed, run from the
extracted release directory:

```sh
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/python reproduce.py
```

The numerical source modules and all frozen analysis tables are included. No
original Google Drive mount, database, source checkout or network data retrieval
is required after installing dependencies. Outputs go to `reproduced/`; frozen
`data/` are verified and never overwritten. The numerical comparison uses 1e-10
relative/absolute tolerance, and exact string/physical-filter/split membership.
Original byte hashes are retained; path changes are not treated as new data.

Open `notebooks/filter_only_results.ipynb` using this environment. Its setup finds
the release marker from the current directory or its parents. If the kernel starts
elsewhere, set `FILTER_ONLY_RELEASE_ROOT` to the extracted release path. All reader
links in the draft and portable notebook are relative. `data/*/manifest.json` and
historical reports retain original source paths as archival provenance only; the
portable entry point never attempts to resolve them.

The release reproduces the completed filter-only analyses starting from frozen,
identity-linked analysis inputs. It does not recreate upstream FTIR predictions
from spectra, certify instrument sampling intervals, or imply independent EC
validation. Source records and source-row links are included for audit.

## Regenerate the draft

The authoring modules are included alongside the numerical modules. Run from the
release root after reproducing the analysis:

```sh
.venv/bin/python authoring/write_manuscript.py .
uv pip install --python .venv/bin/python -r requirements-documentation.txt
.venv/bin/python authoring/build_manuscript_docx.py .
```

The Word export was visually checked using the bundled document rendering tools.
The source notebook contains the same figure calls and detailed interpretive
notes. Prior slide decks remain historical deliverables; this release does not
silently relabel their older findings as the consolidated result.

## Pending external work

The existing upstream packet has been reviewed but remains unsent until a
recipient and delivery channel are supplied. No authoritative response has yet
been received. USPA-0257 remains ineligible for a reviewed interval comparison
because active collection/clock evidence, session-specific processing history,
export scaling and applicable quality approval are unresolved. See
[handoff status](handoff_status.json). No new model or metadata correction is proposed.
""")
    (RELEASE / "source_origins.json").write_text(json.dumps(originals, indent=2) + "\n")
    subprocess.run(
        [sys.executable, str(authoring / "write_manuscript.py"), str(RELEASE)], check=True
    )
    (RELEASE / "handoff_status.json").write_text(
        json.dumps(
            {
                "filter_only_modeling": "complete",
                "scientific_draft": "written; Word visual review recorded separately",
                "upstream_packet": "reviewed_existing_packet; awaiting recipient and delivery channel; not sent; no response received",
                "USPA_0257": "not qualified on current evidence; awaiting collection/clock, session processing, scaling and quality records",
                "independent_EC_validation": False,
                "instrument_calibration": False,
            },
            indent=2,
        )
        + "\n"
    )
    immutable = [
        p
        for folder in ["data", "scripts", "specifications", "authoring"]
        for p in (RELEASE / folder).rglob("*")
        if p.is_file()
    ] + [
        RELEASE / "reproduce.py",
        RELEASE / "requirements.lock",
        RELEASE / "requirements-documentation.txt",
    ]
    (RELEASE / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "immutable_files": [
                    {"path": str(p.relative_to(RELEASE)), "sha256": sha(p)}
                    for p in sorted(immutable)
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(RELEASE)


if __name__ == "__main__":
    build()
