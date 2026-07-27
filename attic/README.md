# attic/

Code that works and is tested, but that **no analysis in this repo imports**.

It lives here rather than in `src/` so that `src/` reads as the set of modules
active work actually depends on. Nothing here is deprecated or known-broken —
it simply has no current consumer. Moved 2026-07-26.

## What's in here

| Module | Was | Tested by |
|---|---|---|
| `analysis/advanced/` | `src/analysis/advanced/` | `tests/test_advanced_analytics.py` (10) |
| `analysis/aethalometer/smoothening/` | `src/analysis/aethalometer/smoothening/` | `tests/test_smoothening.py` (28) |
| `core/monitoring.py` | `src/core/monitoring.py` | `tests/test_integration_performance.py` (15) |
| `core/parallel_processing.py` | `src/core/parallel_processing.py` | ″ |
| `utils/memory_optimization.py` | `src/utils/memory_optimization.py` | ″ |
| `utils/logging/logger.py` | `src/utils/logging/logger.py` | ″ |
| `utils/file_io.py` | `src/utils/file_io.py` | `tests/test_attic_file_io.py` (10) |
| `utils/plotting.py` | `src/utils/plotting.py` | — |
| `utils/statistics.py` | `src/utils/statistics.py` | — |
| `config/smoothening_params.py` | `src/config/smoothening_params.py` | — |
| `config/seasonal_config.py` | `src/config/seasonal_config.py` | — |
| `config/analysis_presets.py` | `src/config/analysis_presets.py` | — |

`utils/logging/logger.py` came along because `ETADLogger` had no consumer outside
the three modules above. The rest of `src/utils/` followed on the same day: an
audit found its only importer was `scripts/diagnostics/test_system.py`, which
imported the names but never called them (that script has since been retired).
`src/utils/` no longer exists.

The three `config/` modules followed on 2026-07-27, all with zero importers.
`smoothening_params.py` parameterizes the ONA/CMA/DEMA classes that had already
moved here, so leaving it in `src/` was the tail of an incomplete move;
`seasonal_config.py` encoded a fifth rival Ethiopian calendar superseded by
`research/ftir_hips_chem/scripts/config.ETHIOPIA_SEASONS`; `analysis_presets.py`
holds presets for a preset-consuming runner that does not exist.
`src/config/quality_thresholds.py` was deliberately **kept** and promoted -- it
is now the single source the four completeness classifiers read from.

### Two known defects in `utils/statistics.py`

Recorded rather than fixed, because nothing calls this code. Fix them before
promoting it back:

- `StatisticalAnalyzer.__init__` never calls `super().__init__(name)`, so the
  instance has no `.name` and inherited `BaseAnalyzer` methods raise.
- `detect_outliers` has no guard for `MAD == 0`. On mostly-constant data the
  modified-z-score branch flags ordinary values as outliers, silently — e.g.
  `[5.0]*20 + [1, 99, 3, 7]` flags `3.0` and `7.0`.

### A hazard in `utils/plotting.py`

`AethalometerPlotter.__init__` calls `plt.style.use()`, which reverts the white
background that `research/ftir_hips_chem/scripts/plotting` installs on import.
That is the exact footgun AGENTS.md warns about, triggered by a constructor
rather than an explicit call. The research package already covers five of this
class's six methods, and all 31 active notebooks use it instead.

## Rules

- **Tests still run.** These 53 tests are part of the normal `pytest` run; the
  code is expected to keep passing. If a test here starts failing, that's a real
  signal, not noise to silence.
- **Don't import `attic.*` from `src/`, `scripts/`, `research/`, or notebooks.**
  A new dependency on something in here means it isn't attic material — move it
  back to `src/` in the same commit that adds the importer.
- **Not shipped.** `pyproject.toml` builds only `src` and `aethmodular_cli`, so
  nothing here goes into the wheel.

## Promoting something back

Move the module to its old path under `src/`, restore the re-export in that
package's `__init__.py`, and update the test imports from `attic.*` to `src.*`.
The removed re-exports are recorded in the 2026-07-26 consolidation commit.
