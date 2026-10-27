# Repository Commands

`aeth` is the stable command interface for routine repository work. With the
canonical environment, prefix examples with `uv run`:

```bash
uv run aeth --help
```

The command package uses only the Python standard library, so environment
diagnosis can also run before the scientific dependencies are available:

```bash
python -m aethmodular_cli doctor
```

## Environment and validation

```bash
aeth doctor                 # interpreter, modules, data root, core inputs
aeth doctor --json          # machine-readable diagnosis
aeth check                  # pytest + Ruff
aeth check --notebooks      # also enforce notebook path portability
```

## Notebooks

```bash
aeth notebook list
aeth notebook list --all
aeth notebook check
aeth notebook check --changed
aeth notebook check path/to/notebook.ipynb
aeth notebook run
aeth notebook run path/to/notebook.ipynb --timeout 600
```

`notebook check` reads source cells only. It detects machine-specific absolute
paths and pre-reorganization path names without requiring Jupyter or executing
the notebook. `--changed` provides an incremental quality gate while the legacy
notebook backlog is migrated. `notebook run` delegates to the existing nbclient
smoke runner.

## Diagnostics and data preparation

```bash
aeth diagnose etad
aeth diagnose matching
aeth diagnose compare-pkl
aeth diagnose etad-stats
aeth diagnose flow

aeth data resample
aeth data resample --list-sites
aeth data resample --site ETAD
aeth data resample --site ETAD --site Delhi
```

The resampling pipeline accepts either site codes or configured site names. Its
default remains all configured sites.

## SPARTAN workflows

```bash
aeth spartan pull --check-updates
aeth spartan pull
aeth spartan pull --skip-download
aeth spartan coverage
aeth spartan coverage-plots
aeth spartan bridge
aeth spartan connections
aeth spartan extras
```

Arguments after the workflow name are passed through to the implementation
script. The crawler reads SPARTAN's current public S3 bucket. `--check-updates`
streams files already in the local raw cache and compares SHA-256 content
without changing files or reports. It reports filename and S3 ETag changes
against the saved inventory. With no raw cache, unchanged ETags provide a
metadata comparison; files without an earlier saved ETag remain unverified.
Plain `pull` now refreshes changed CSVs
atomically and regenerates the inventory only after every transfer succeeds.
Both network modes require the public SPARTAN host to be reachable;
`--skip-download` is offline and needs a populated raw cache.

## MAIA catalogue

```bash
aeth maia catalog                 # refresh research/maia/inventory/ from the portal and CMR
aeth maia catalog --offline       # rebuild the tables from the cached responses in data/maia/meta/
aeth maia pull --collection MAIA_ANC_SURFACEMONITOR_PM_TOTAL --target ETH-AddisAbaba \
    --start 2023-01-01 --end 2023-03-31 --dry-run
```

`catalog` reads two public sources with no login: the MAIA MMGIS portal
(target areas, map layers, surface-monitor lists) and NASA CMR (every MAIA
collection at the ASDC and every granule's target, day, version and size). It
writes `CATALOG.md` and CSVs, and matches each SPARTAN site to its
target area and filter days. It takes about two minutes. `pull` downloads one
collection's granules into `data/maia/raw/<collection>/<target>/`, keeping the
newest version of each target-day unless `--all-versions` is given. Downloads
need `EARTHDATA_TOKEN` (see `.env.example`); `--dry-run` lists URLs without it.

## Gallery data

```bash
aeth gallery plan                 # every step, what it reads and writes, and whether its inputs are here
aeth gallery rebuild              # run the export steps in order (add --fetch to refresh AERONET/MAIA first)
aeth gallery rebuild --only aeronet,aeronet-sites   # or --from <step>, --skip <step>, --dry-run
aeth gallery check                # chart index, typecheck, tests, the public-build leak check, sizes
aeth gallery check-public         # only the check that private sections stay out of the public build
aeth gallery sizes                # bytes per data folder, public and private
aeth gallery markdown --write     # refresh the step table in docs/gallery-data-pipeline.md
aeth drive mirror --dest <dir> --link-root <dir>    # copy the "Davis Data" Drive folder down
```

`gallery/data/rebuild.py` is the one list of how `gallery/app/public/data/`
is made: each step names its script, inputs and outputs, and a step whose
inputs are missing is skipped with the reason rather than failing the run.
Each run writes
`research/ftir_hips_chem/output/tables/gallery_rebuild/last_run.json`. A
rebuild on unchanged inputs should change only the `generated` stamps. See
[`gallery-data-pipeline.md`](gallery-data-pipeline.md) for the steps, the
Drive mirror, the public flag and the rules the exporters keep.

## Research builders

```bash
aeth build list
aeth build list ftir-calibration
aeth build run ftir-calibration 08_adama_hips_crossplots
aeth build run july07 all
```

Builders stay beside the research project that owns them. The command discovers
only `_build*.py` files inside registered groups; it does not move notebook
generation logic into the production package.

Registered groups:

- `ftir-calibration`
- `spartan-ec`
- `july07`
- `addis-deming`

(`catch-up` retired 2026-07-26; see `research/archive/catch_up`.)

## Calibration Iteration Explorer (Flask app)

The interactive phase-3 calibration app at `calibration_explorer/` is not part of
`aeth`; it has its own launcher (see `calibration_explorer/README.md` for details):

```bash
./calibration_explorer/run.sh           # start in the background → http://127.0.0.1:5058
./calibration_explorer/run.sh status    # up? data loaded?
./calibration_explorer/run.sh stop
```

Manual start with the canonical environment (the `explorer` extra is required —
the base venv has no Flask):

```bash
uv run --extra explorer python calibration_explorer/app.py
```

Optional pre-warm so every configuration is a cache hit:

```bash
uv run --extra explorer python calibration_explorer/warm_cache.py
```

Backfill the per-filter `LotId` column into explorer targets exported before
`build_spartan_target.py` started writing it (needed for the Eval lot lever on
sites other than Addis; idempotent, skips targets that already have it):

```bash
uv run python research/ftir_ec_phase3/scripts/add_target_lots.py
```
