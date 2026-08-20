# IO package trials — 2026-08-19

Follow-up to the [chemometrics trials](../chemometrics/README.md): the
explorer's calculations are already accelerated (ikpls fast CV, one-fit
prefix truncation), so the remaining startup cost is I/O — chiefly the
725 MB `local_db/spectra_248_251.csv` pool-spectra read. This trial tests
**polars** as a reader for it.

## Environment

Anaconda python 3.13.9 (`/Users/ahmadjalil/anaconda3/bin/python`, the
interpreter `calibration_explorer/app.py` runs under). Installed for this
trial: polars 1.43.2.

## Scripts (run from this directory)

- `validate_polars_load.py` — times the app's exact historical pandas read
  vs the polars path now in `calibration_explorer/app.py::_read_pool_spectra`
  vs an npz binary cache, on the **real** pool CSV, and verifies equivalence
  after the app's dedup/set_index steps. Must be run from a shell with
  Google Drive access (sandboxed agent shells on this machine get
  `Operation not permitted` on CloudStorage file contents — metadata works,
  `open()` doesn't).

## Synthetic-trial results (2026-08-19)

The real CSV was unreadable from the agent shell (Drive permissions), so
the engine comparison ran on a same-shape synthetic: 13,000 rows x
(AnalysisId + 2722 wavenumber columns), full-precision floats, 703 MB,
warm file cache, exercising the app's actual `_read_pool_spectra`.

| read | time | vs pandas |
|---|---|---|
| pandas default (`dtype=float32`) — the historical read | 3.1 s | 1x |
| pandas `float_precision='round_trip'` | 10.3 s | 0.3x |
| **polars (Float64 parse → float32 cast)** — the app's new fast path | **0.5 s** | **6x** |

**Precision.** polars parses correctly rounded and is **bit-identical to
pandas `float_precision='round_trip'`**. pandas' *default* C parser is not
correctly rounded: vs polars it differed on **1 cell in 35,386,000**, by
exactly **1 float32 ulp**, on a ~1.1e-5 absorbance value — i.e. polars is
the (very slightly) more correct of the two, and the difference is orders
of magnitude below every precision the explorer reports (4–5 decimals).

**Caveat.** 3.1 s → 0.5 s is the *parse* gain. The app's real-world
"a minute or two" for this step is partly Google Drive streaming, which no
parser fixes; `validate_polars_load.py` measures the real split (its second
polars read is the cache-warm number). If Drive streaming dominates, the
npz binary cache it also times is the follow-up worth wiring in.

## Verdict

Adopted as an optional fast path in `calibration_explorer/app.py`
(`_read_pool_spectra`): polars when importable, byte-for-byte historical
pandas read otherwise or with `CALIB_EXPLORER_CSV=pandas`. Not added to
`pyproject.toml` — same status as ikpls, an optional accelerator the code
must never require.
