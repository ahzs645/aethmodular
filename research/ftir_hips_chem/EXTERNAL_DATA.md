# External Data Dependencies

Most active analyses load from repo-local paths defined in `scripts/config.py`:

- `PROCESSED_SITES_DIR`
- `FILTER_DATA_PATH`
- `ETAD_FACTOR_CONTRIBUTIONS_PATH`
- `ETAD_FILTER_ID_PATH`
- `WEATHER_DATA_DIR`
- `AERONET_DATA_DIR`

Raw minute-resolution Addis MA350 files, AERONET exports and the FTIR spectra
are too large for git and stay out of it. Nothing needs a local copy any more:
they are read from the Drive mount through the resolvers below.

## Resolving external data

Do **not** spell out a path to the Drive tree. It has been reorganised more than
once, and every hardcoded copy broke when it moved. Use
[`scripts/data_paths.py`](scripts/data_paths.py) instead:

| Helper | What it returns |
|---|---|
| `maia_data_root()` | the directory holding every raw dataset |
| `aethalometry_dir()` | raw aethalometer exports |
| `etad_dir()` | Addis (ETAD) FTIR spectra and metadata |
| `ftir_spectra_dir()`, `ftir_local_db()` | IMPROVE spectra exports and calibration tables |
| `weather_dir()`, `weather_file(...)` | weather data, preferring the in-repo copy |
| `aeronet.aeronet_dir()`, `improve_io.improve_dir()` | AERONET and IMPROVE, off the same root |

Each honours an `AETHMODULAR_*` environment override, so a different machine or
layout needs no code change; `pls_transfer.drive_root()` finds whichever Google
Drive account is signed in. Candidate layouts are declared once, in
`pls_transfer.MAIA_DATA_CANDIDATES` and `FTIR_DIR_CANDIDATES` — that is the only
place to edit when the tree moves again.

Print `data_paths.describe()` at notebook setup to see what actually resolved.
A load error is then obviously a missing-mount problem rather than a logic bug.

## Local-only working copies

If you do keep a local copy, put it in an ignored folder under
`research/ftir_hips_chem/` and point the matching `AETHMODULAR_*` variable at
it. `Weather Data/` is the one dataset with a small tracked in-repo copy, which
`weather_dir()` prefers over the Drive original; note the two hold *different*
files, so use `weather_file()` when you want a specific one.
