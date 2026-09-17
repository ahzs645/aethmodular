# Filter only scientific release

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

[Release check results](release_checks/README.md) and the [executed notebook](notebooks/filter_only_results.executed.ipynb) are included.
