# Repo cleanup & consolidation

Living status of repository consolidation on `main`. Everything below was
found by auditing `src/`, `research/`, and `notebooks/`.

## Unified commands

Routine workflows are now exposed through the dependency-light `aeth` command.
Use `aeth doctor`, `aeth check`, `aeth notebook check`, and `aeth build list`
instead of memorizing script paths. Scientific logic remains in its sanctioned
package or research-script location; the CLI is an orchestration layer only.

The seven temporary root-level Python wrappers have been removed. The 9am
resampling workflow now accepts repeatable `--site` arguments while preserving
its previous all-sites default.

## Current automated cleanup inventory

`aeth notebook check` currently identifies 37 active notebooks with source-level
machine or legacy paths:

| Area | Notebooks |
|---|---:|
| `research/ftir_hips_chem` | 14 |
| `research/improve_hips_offset` | 8 |
| `research/catch_up` | 6 |
| top-level `notebooks` | 8 |
| `research/spartan` | 1 |

Migrate generated families by editing their `_build*.py` source first, then
regenerating and checking the notebook. Hand-authored notebooks should use the
standard setup cell or `prep.find_repo_root`; do not perform a blind JSON text
replacement across executed notebooks.

The ignore rule for downloaded `data/` was anchored to `/data/`. Previously it
also hid source files under `src/data/` and notebook inputs under
`notebooks/analysis/data/`. Generated PNG/CSV analysis artifacts and a
machine-specific `node_modules` symlink were removed from tracking. Two output
presentation decks remain tracked pending a content/retention decision.

## Done on this branch

1. **Git de-bloat** — untracked ~56 MB of committed binaries/generated output
   (kept on disk, now gitignored): `tmp_warren_pages/`, `notebooks/analysis/output/`,
   12 `*_executed.ipynb` twins. Removed scratch/stray files and disk cruft.
   Tracked repo: 271 MB -> 215 MB.
2. **Pruned orphaned `src/`** — deleted `src/visualization/` (2,246 LOC template
   framework, zero imports anywhere) and the unused British-spelling
   `SourceApportionmentAnalyser` alias.
3. **Consolidated helpers into the sanctioned `scripts/` home** — per AGENTS.md,
   the reusable-logic home for the active research area is
   `research/ftir_hips_chem/scripts/` (which every active satellite dir already
   imports), **not** `src/`. Added the missing helpers there and retired an
   earlier `src/common/` attempt (which violated the "do not mix `src/` and
   research" rule). Covered by `tests/test_scripts_helpers.py` (26 tests).
4. **Pruned archived-only `src/` duplicates** — deleted `src/notebook_utils/`
   (779 LOC) and both dual-dataset pipelines (`dual_dataset_pipeline.py` +
   `optimized_dual_dataset_pipeline.py`, 990 LOC); cleaned the guarded import
   block in `src/data/processors/__init__.py`. Only archived notebooks referenced
   these.
5. **Migrated the first consumer** — `research/addis_fabs_ec_deming` build scripts
   now import `deming`/`deming_lambda` from `plotting.utils` (was inline); both
   notebooks regenerated, logic verified identical.

## `scripts/` — what to import instead of redefining

All importable from `research/ftir_hips_chem/scripts/` (already on path in these
notebooks via `sys.path.insert(0, './scripts')` or `'../ftir_hips_chem/scripts'`).

| Was inlined as | Now import | Copies replaced |
|---|---|---|
| `deming(x, y, lam)` | `from plotting.utils import deming, deming_lambda` | 4 (addis_fabs_ec_deming) |
| `regression_stats` / `calculate_regression_stats` | `from plotting.utils import calculate_regression_stats` (already existed) | 8+ (ftir_hips_chem, improve_hips_offset, spartan) |
| `base_filter_id` (scalar) / `normalize_filter_id` | `from data_matching import base_filter_id, normalize_filter_id` | many (ftir_hips_chem) |
| `_to_ugm3` | `from prep import to_ugm3` | 7 (catch_up) |
| `map_ethiopian_seasons` / `get_season_3` | `from config import season_for_month, ETHIOPIA_SEASONS` | 6+5 (notebooks + research) |
| `find_repo_root` / `find_root` | `from prep import find_repo_root` | 5 subdirs / 14 repo-wide |

Everything above is also re-exported from the package root (`from scripts import
deming, season_for_month, ...`).

### Conflicts resolved

- **Regression stats**: `calculate_regression_stats` was extended to a
  backward-compatible superset (DataFrame form, +/-inf drop, opt-in
  `positive_only`, `r2`/`origin_slope` keys) so the divergent inline
  `regression_stats(df, x_col, y_col)` copies can delegate to it. **Migrated +
  independently verified byte-identical**: `anne_spartan`, `etad_vs`,
  `warren_cena` (improve_hips_offset). **Left inline (bespoke, not duplicates):**
  `improve_smoke_event_qc` (also computes Theil-Sen), `improve_addis_analog_audit`
  & `improve_hips_offset_narrative` (`nunique<2` guard + `origin_mac`/capitalized
  keys), `ftir_hips_chem/hips_offset_narrative` (no positive/inf filtering). These
  genuinely differ from the common case; shimming them adds complexity for ~zero
  dedup, so they keep their own defs.
- **Season calendar**: several inline copies disagreed; one
  (`plotting_gaps_scenarios.ipynb`) had scrambled month assignments. Canonical is
  now `config.ETHIOPIA_SEASONS` (Dry Oct-Feb, Belg Mar-May, Kiremt Jun-Sep) —
  matching AGENTS.md's planned `config.ETHIOPIA_SEASONS`.

### Notebook migration recipe

These notebooks already put `scripts/` on `sys.path`, so migration is: replace
the inline `def deming(...)` / season / filter-id / unit helper with an import
from the module above, then re-run. Where a family is generated by a `_build_*.py`
script, edit the generator and regenerate (verifiable without data). Do one
family at a time and confirm outputs are unchanged before deleting inline defs.

## Remaining decisions (not auto-applied — need a call)

These `src/` modules are unused by active code but were left in place because
deleting them is ambiguous:

| Module | Decision |
|---|---|
| `src/notebook_utils/`, dual-dataset pipelines | **Deleted** (see Done #4). |
| `src/data/qc/enhanced_pkl_processing.py` vs `pkl_cleaning.py` | **Kept both — audit was wrong.** `pkl_cleaning.py` imports `EnhancedPKLProcessor` in 7 places, so this is a real dependency, not a droppable duplicate. |
| `src/analysis/ftir/*` (`enhanced_mac_analyzer`, `oc_ec_analyzer`, `fabs_ec_analyzer`) | **Kept.** `EnhancedMACAnalyzer` is coherent 4-method MAC logic that exists nowhere else; unused today but the right home for future MAC work rather than a plain delete. |
| `src/analysis/seasonal/ethiopian_seasons.py` | **Open.** A `src/`-side season module, superseded for research use by `scripts/config.ETHIOPIA_SEASONS`; still imported by a few notebooks. Migrate those first, then delete. |
| `src/core/monitoring.py`, `parallel_processing.py`, `src/analysis/advanced/*` | **Open.** Imported only by the test suite; keep or drop the tests with them. |

### Notebook migration — remaining families

`addis_fabs_ec_deming` is migrated. Still inline (safest via their build scripts
where present, else hand-edit + re-run): `regression_stats`/`calculate_regression_stats`
across ftir_hips_chem/improve_hips_offset/spartan, the `catch_up` loader stack
(`_build_catch_up_notebooks.py`), season helpers in the absorption/meteorology
notebooks, and `find_repo_root` everywhere.

### Large regenerable data (already gitignored, on disk only)

Safe to move off local disk / to external storage if space matters:
`research/improve_hips_offset/output/` (877 MB), `charcoal_ftir/data/raw/`
(308 MB, re-fetchable via `pull_reference_spectra.py`),
`spartan_ec_2026_06_16/data/*.RDS` (335 MB), and a stray `node_modules`
committed under `ftir_hips_chem/output/` (57 MB).

### Stale subdir

`research/filter_combine/` (last touched 2026-02-11) holds three rival
`load_filter_sample_data` implementations (`_simplified`, `_enhanced`,
`enhanced_`) plus three summary markdowns — strongest abandonment candidate.
