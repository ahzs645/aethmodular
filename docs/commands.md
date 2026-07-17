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
aeth diagnose system

aeth data resample
aeth data resample --list-sites
aeth data resample --site ETAD
aeth data resample --site ETAD --site Delhi
```

The resampling pipeline accepts either site codes or configured site names. Its
default remains all configured sites.

## SPARTAN workflows

```bash
aeth spartan pull --skip-download
aeth spartan coverage
aeth spartan coverage-plots
aeth spartan bridge
aeth spartan connections
aeth spartan extras
```

Arguments after the workflow name are passed through to the implementation
script.

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
- `catch-up`
