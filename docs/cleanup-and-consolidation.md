# Repo cleanup & consolidation

Status of the `cleanup/repo-consolidation` branch and the remaining decisions.
Everything below was found by auditing `src/`, `research/`, and `notebooks/`.

## Done on this branch

1. **Git de-bloat** — untracked ~56 MB of committed binaries/generated output
   (kept on disk, now gitignored): `tmp_warren_pages/`, `notebooks/analysis/output/`,
   12 `*_executed.ipynb` twins. Removed scratch/stray files and disk cruft.
   Tracked repo: 271 MB -> 215 MB.
2. **Pruned orphaned `src/`** — deleted `src/visualization/` (2,246 LOC template
   framework, zero imports anywhere) and the unused British-spelling
   `SourceApportionmentAnalyser` alias.
3. **Added `src/common/`** — one home for helpers that were copy-pasted across
   dozens of notebooks/scripts, with `tests/test_common.py` (30 tests).
4. **Pruned archived-only `src/` duplicates** — deleted `src/notebook_utils/`
   (779 LOC) and both dual-dataset pipelines (`dual_dataset_pipeline.py` +
   `optimized_dual_dataset_pipeline.py`, 990 LOC); cleaned the guarded import
   block in `src/data/processors/__init__.py`. Only archived notebooks referenced
   these.
5. **Migrated the first consumer** — `research/addis_fabs_ec_deming` build scripts
   now import `deming`/`deming_lambda` from `src.common` (was inline); both
   notebooks regenerated, logic verified identical.

## `src/common` — what to import instead of redefining

| Was inlined as | Now import | Copies replaced |
|---|---|---|
| `deming(x, y, lam)` | `from src.common import deming, deming_lambda` | 4 (addis_fabs_ec_deming) |
| `regression_stats` / `calculate_regression_stats` | `from src.common import regression_stats` | 8+ (ftir_hips_chem, improve_hips_offset, spartan) |
| `fit_fabs_ec` / `robust_fabs_ec` | `from src.common import fit_fabs_ec` | ftir_hips_chem, improve_hips_offset |
| `base_filter_id` / `normalize_filter_id` | `from src.common import base_filter_id, normalize_filter_id` | many (ftir_hips_chem) |
| `_to_ugm3` | `from src.common import to_ugm3` | 7 (catch_up) |
| `map_ethiopian_seasons` / `get_season_3` / `assign_season` | `from src.common import season_for_month, assign_season` | 6+5+4 (notebooks + research) |
| `find_repo_root` / `find_root` | `from src.common import find_repo_root` | 5 subdirs / 14 repo-wide |

### Conflicts resolved (most-robust variant kept)

- **`regression_stats`**: the `improve_hips_offset` variant (drops +/-inf, requires
  n>=3) was chosen over the plain `dropna` variant. Its strictly-positive x/y
  filter is domain-specific to fAbs/EC, so it is now the opt-in
  `positive_only=True` flag (default off) to keep the helper safe for general
  regressions. `fit_fabs_ec` passes `positive_only=True`.
- **Season calendar**: several inline copies disagreed; one
  (`plotting_gaps_scenarios.ipynb`) had scrambled month assignments. All are
  superseded by `src/config/multi_site_seasons.SITE_SEASONS`
  (Dry Oct-Feb, Belg Mar-May, Kiremt Jun-Sep), which `season_for_month` wraps.

### Notebook migration recipe

Replace the inline `def deming(...)` (and the others above) with an import cell
at the top of the notebook. `sys.path` already points at repo root in these
notebooks, so `from src.common import deming, regression_stats, ...` works.
Migrate a notebook family at a time and re-run to confirm outputs are unchanged
before deleting the inline defs.

## Remaining decisions (not auto-applied — need a call)

These `src/` modules are unused by active code but were left in place because
deleting them is ambiguous:

| Module | Decision |
|---|---|
| `src/notebook_utils/`, dual-dataset pipelines | **Deleted** (see Done #4). |
| `src/data/qc/enhanced_pkl_processing.py` vs `pkl_cleaning.py` | **Kept both — audit was wrong.** `pkl_cleaning.py` imports `EnhancedPKLProcessor` in 7 places, so this is a real dependency, not a droppable duplicate. |
| `src/analysis/ftir/*` (`enhanced_mac_analyzer`, `oc_ec_analyzer`, `fabs_ec_analyzer`) | **Kept.** `EnhancedMACAnalyzer` is coherent 4-method MAC logic that exists nowhere else; unused today but the right home for future MAC work rather than a plain delete. |
| `src/analysis/seasonal/ethiopian_seasons.py` | **Open.** Superseded by `config/multi_site_seasons` (which `src/common/seasons` wraps), but still imported by a few notebooks. Migrate those notebooks first, then delete. |
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
