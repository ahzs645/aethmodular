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

`aeth notebook check` identifies **49 of 138** active notebooks with source-level
machine, cloud, or legacy paths (2026-07-26, after adding `CLOUD_PATH_MARKERS`):

| Area | Notebooks | Note |
|---|---:|---|
| `research/ftir_ec_calibration_2026_06_25` | 10 | all cloud-path only; newly visible |
| `research/ftir_hips_chem` | 14 | |
| `research/improve_hips_offset` | 8 | |
| top-level `notebooks` | 8 | |
| `research/July07` | 4 | cloud-path only; newly visible |
| `research/spartan` | 1 | |
| `spartan_ec` / `ftir_etad_char` / `ftir_ec_phase3` | 4 | cloud-path only; newly visible |

The previous count was 37 of 147. The denominator fell because `catch_up`'s six
notebooks moved under `research/archive/` (skipped by the checker) and the
scrambled `Multi_Site_Analysis_Modular.ipynb` was deleted; the numerator rose
because the cloud-path blind spot was closed.

**Known remaining backlog** (audited and quantified, deliberately not acted on in
this pass — the dedup scope chosen was "silent-fallback bug only"):

- 11 rival `find_repo_root` / `find_root` / `_root` definitions across 5 subdirs.
  The 4 dangerous ones (silent `cwd()` fallback) are fixed; the rest are
  duplicates that raise correctly. `prep.find_repo_root` has **0 %** notebook
  adoption — 0 of 138 notebooks import it.
- 134 inline helper redefinitions across 86 notebooks: `SEASON_COLORS` ×44,
  `SITE_COLORS` ×23, season function ×17, `regression_stats` ×16, `MAC_VALUE`
  ×15, `find_repo_root` ×10, `_to_ugm3` ×6.
- `research/spartan_ec_2026_06_16/ftir_pls_calibration.py` is reusable logic
  living outside the sanctioned `scripts/` home, imported by 8 files in a
  *different* research subdir.
- `_build_06_tool_style_plots.py` is an orphan builder — it writes a notebook,
  but no `06_*.ipynb` is tracked.
- `research/ftir_hips_chem/output/presentations/*.pptx` (8.8 MB) are tracked
  despite matching `.gitignore`'s `research/**/output/`. Retention decision still
  open.
- Untested in the sanctioned `scripts/`: the entire `outliers.py` exclusion API
  (which AGENTS.md makes mandatory for every analysis), `etad_factors.py`
  (including the GF-normalization quirk AGENTS.md flags as easy to get wrong),
  and `flow_periods.py`.

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

1. **Git de-bloat** — untracked ~56 MB of committed binaries/generated output:
   `tmp_warren_pages/`, `notebooks/analysis/output/`, 12 `*_executed.ipynb`
   twins. Removed scratch/stray files and disk cruft. Tracked repo: 271 MB ->
   215 MB. (These were described as "kept on disk"; as of 2026-07-26 none of the
   three remain on disk — they are gone, not merely untracked.)
2. **Pruned orphaned `src/`** — deleted `src/visualization/` (2,246 LOC template
   framework, zero imports anywhere) and the unused British-spelling
   `SourceApportionmentAnalyser` alias.
3. **Consolidated helpers into the sanctioned `scripts/` home** — per AGENTS.md,
   the reusable-logic home for the active research area is
   `research/ftir_hips_chem/scripts/` (which every active satellite dir already
   imports), **not** `src/`. Added the missing helpers there and retired an
   earlier `src/common/` attempt (which violated the "do not mix `src/` and
   research" rule). Covered by `tests/test_scripts_helpers.py` (34 tests).
4. **Pruned archived-only `src/` duplicates** — deleted `src/notebook_utils/`
   (779 LOC) and both dual-dataset pipelines (`dual_dataset_pipeline.py` +
   `optimized_dual_dataset_pipeline.py`, 990 LOC); cleaned the guarded import
   block in `src/data/processors/__init__.py`. Only archived notebooks referenced
   these.
5. **Migrated the first consumer** — `research/addis_fabs_ec_deming` build scripts
   now import `deming`/`deming_lambda` from `plotting.utils` (was inline); both
   notebooks regenerated, logic verified identical.

## Done 2026-07-26 (four-zone audit pass)

Real defects found and fixed — these were breakage, not cosmetics:

1. **`import src.data.qc` failed outright.** `src/external/calibration.py:1` does
   a top-level `from tqdm import tqdm`, and `tqdm` was declared nowhere.
   `src/data/__init__.py` wrapped the import in a bare `except ImportError`, so
   the failure was invisible and `src.data.__all__` silently evaluated to `[]`.
   Added `tqdm`; `__all__` now has its 7 entries.
2. **`aeth spartan pull` was dead on arrival** — top-level `import requests` in
   `scripts/pipelines/spartan_pull_and_summarize.py`, also undeclared. Added.
3. **225 MB of research data was hidden by a stock packaging rule.** The
   `.gitignore` boilerplate `downloads/` (unanchored) matched
   `research/ftir_hips_chem/charcoal_ftir_sources/downloads/`, whose
   `CHECKSUMS.sha256` manifest *was* tracked. Same class of bug as the earlier
   `data/` → `/data/` fix. Anchored to `/downloads/` and added an explicit,
   commented rule for the payload so the exclusion is deliberate.
4. **Four `find_repo_root` variants silently returned `Path.cwd()`** on failure
   instead of raising, so a run from the wrong directory produced wrong paths
   quietly. Fixed in all four `_build_*.py` generators *and* the three already
   generated notebooks (`05`, `07`, `08` in `spartan_ec_2026_06_16`).
5. **`aeth notebook check` had a blind spot.** Paths built as
   `Path.home() / "Library/CloudStorage/GoogleDrive-<account>"` contain no
   `/Users/` literal, so they passed. Added `CLOUD_PATH_MARKERS`; the failure
   count went 31 → 49, surfacing 18 previously-invisible notebooks (10 in
   `ftir_ec_calibration_2026_06_25`, 4 in `July07`, 4 elsewhere). Covered by two
   new tests in `tests/test_cli.py`.
6. `src/utils/__init__.py` assigned `__all__` twice, so `AethalometerPlotter` and
   `StatisticalAnalyzer` were imported but never exported. Merged.

Structural consolidation:

- **`attic/`** created for working, tested code with no consumer — see the
  decisions table below and [`attic/README.md`](../attic/README.md).
- **`research/archive/`** created for retired workspaces (`catch_up/`,
  `charcoal_ftir/`'s KBr notebooks). Named `archive` so the existing
  `aeth notebook check` skip rule (`"archive" in path.parts`) applies with no
  code change. Dropped the `catch-up` build group from the CLI.
- **`charcoal_ftir` merged** — `sources.json`, `leads.md`, `product_targets.md`,
  and `scripts/pull_reference_spectra.py` moved into the live
  `ftir_hips_chem/charcoal_ftir_sources/`. Note that fetcher writes to
  `data/raw/` and does **not** reproduce the existing `downloads/` tree.
- **`filter_combine` reduced to data.** Deleted three rival dead
  `load_filter_sample_data` implementations plus `main.py`,
  `quick_start_guide.py`, `setup.py`, and 3 summary docs (~2,000 LOC); every one
  hardcoded `/Users/ahzs645/Github/aethmodular-clean/`, a different user's
  checkout, so none could run here. The CSV/PKL data is **kept** — it is read by
  `spartan_ec_2026_06_16/_build_01_carbon_methods_audit.py`.
- **`Multi_Site_Analysis_Modular.ipynb` deleted.** Verified as a block-scrambled
  duplicate of `_Fixed.ipynb`: identical 53-cell multiset, exactly
  `reversed(Fixed[31:]) + Fixed[:31]`, opening on its own closing section and so
  unrunnable top to bottom. `_Fixed` is the intact copy.
- `aeth check` now lints `scripts/` and `manuscript/` (both already passed) plus
  `attic/`, not just `src tests aethmodular_cli`.
- Removed `pathlib2` (a Python-2 backport with zero imports) from all three
  dependency specs, and the ruff config entries pointing at
  `src/visualization/templates/examples.py`, deleted in the previous pass.

Two audit claims were **checked and rejected** rather than acted on:
`enhanced_pkl_processing.py`'s `HAS_CALIBRATION` is `True`, not always `False` —
the sibling `from .pkl_cleaning import ...` on line 9 registers `calibration` in
`sys.modules` first (fragile, but working); and `_Modular`'s execution counts are
not all `None`.

## Done 2026-07-26 (utilities pass)

A second three-zone audit covering `src/utils/`, the sanctioned
`research/ftir_hips_chem/scripts/`, and operational `scripts/`.

### Defects fixed

1. **Three scripts pointed at a pre-`git mv` path.** Commit `01df6ae` moved
   `df_Jacros_9am_resampled.pkl` into `processed_sites/`;
   `compare_pkl_files.py`, `create_9am_resampled_datasets.py`, and
   `test_system.py` were never updated. `aeth diagnose compare-pkl` died with
   `FileNotFoundError` before printing anything, and `aeth data resample --site
   ETAD` failed for the flagship site.
2. **`comparisons.flow_periods` accepted only one label vocabulary.**
   `flow_periods.add_flow_period` emits `before`/`after`;
   `data_matching.add_flow_period_column` emits `before_fix`/`after_fix`. The
   plot matched only the bare form, so every caller using the `data_matching`
   column silently printed "skipping" and drew nothing. Fixed on the consumer
   side — both spellings are accepted and results are keyed canonically — so no
   existing notebook filter breaks.
3. **`add_base_filter_id` corrupted ids already in base form.** Its bare
   `r'-\d+$'` stripped the 4-digit sample number, collapsing `ETAD-0035` to
   `ETAD` and mapping every sample at a site onto one join key. Now anchored on
   the `SITE-NNNN` prefix, matching the scalar `base_filter_id`. The rival copy
   in `spartan_hips_bridge.py` had the opposite bug (`r"-(\d)$"` missed
   two-digit replicates, so `ZAJB-0041-12` never joined); both now use the same
   anchored rule.
4. **`save_results_to_json` crashed on `ndarray`/`Series`.** It tested
   `hasattr(obj, 'item')` before `hasattr(obj, 'tolist')`, but both types expose
   both and `.item()` raises on more than one element — making the `tolist`
   branch unreachable for exactly the inputs the helper exists to serialize.
5. **`tabulate` was undeclared**, so `aeth spartan pull`, `coverage`, and
   `connections` crashed in their report-writing step *after* completing a full
   crawl or scan. `cartopy` is now an opt-in `geo` extra (it needs system
   GEOS/PROJ) and `spartan_extras` skips its world map cleanly without it.
6. **`aeth spartan pull --skip-download` still required the network** —
   `crawl()` ran before the guard. Added `scan_local()`, which rebuilds the file
   list from `RAW_DIR` using the same directory grammar.
7. **`run_notebook_smoke.py` re-opened the cloud-path blind spot.** It kept a
   private copy of the portability markers lacking `CLOUD_PATH_MARKERS`, so
   `aeth notebook run` passed notebooks `aeth notebook check` failed. It now
   imports the markers from `aethmodular_cli.cli` (stdlib-only, no new dep).
8. **`flow_periods.calculate_period_stats` was unimportable** under the
   package-relative path — a bare `from plotting.utils import ...` with no
   `try/except ImportError` fallback, unlike every sibling module.
9. **`scripts/__init__.py` never re-exported `flow_periods`** despite four
   active notebooks importing it and the package claiming to re-export
   everything. Now 82 names, 0 missing, on both import paths.
10. Two unguarded `ZeroDivisionError` paths in `check_matching_statistics.py`,
    and a dead loop in `spartan_extras.site_locations` that fully parsed every
    Nephel CSV and discarded the result.

### Structural changes

- **`plotting_legacy.py` deleted** (664 LOC). AGENTS.md had forbidden it for
  months, but three active notebooks still used it because the modern
  `plotting/` package owns its figures while legacy drew onto a caller-supplied
  axes. New **`plotting/overlays.py`** supplies that missing axes-level layer
  (`scatter_on_axes`, `iron_gradient_on_axes`, `bc_timeseries_on_axes`) with
  styling matched to legacy exactly — verified identical on text, axis limits
  and artists across six option combinations — so the migration changed no
  figure. `print_comparison_table` was ported verbatim into `plotting/utils.py`
  for the same reason. `iron_gradient_on_axes` deliberately keeps legacy's
  stricter mask (iron must be present) rather than adopting
  `crossplots.with_iron_gradient`'s looser one, which would have changed
  reported n/slope/R². The overlays additionally mask with `np.isfinite`, so
  ±inf no longer reaches `polyfit` or the axis-limit computation.
- **`src/utils/` no longer exists** — moved to `attic/utils/`. Its only importer
  was `test_system.py`, which imported the names and never called them.
- **`scripts/diagnostics/test_system.py` retired** (374 LOC). Zero `assert`
  statements and a permanent exit 1 caused by 2025-era layout expectations (a
  `setup.py` this repo deliberately lacks, plus gitignored `data/`/`outputs/`).
  Its one unique contribution — that `src/` imports and the analyzers construct
  — now lives in `tests/test_import_smoke.py` as real assertions that `aeth
  check` gates. The `system` diagnose subcommand is gone.
- **SPARTAN score denominator unified.** `spartan_coverage_plots.py` divided
  breadth by a hardcoded 9 while `spartan_extras.py` divided by
  `nunique()` computed on each *subset*, so `top_sites.csv` and
  `top_sites_nephel.csv` were scored on different scales while the map colorbar
  claimed "breadth / 9". Both now use `MAX_BREADTH = 9`, the full
  `EXPECTED_STEP_H` count. **This changes the published scores in
  `research/spartan/inventory/` — regenerate those tables.**
- **`spartan_pull` gained the 2010-2030 year guard** its siblings always had;
  `overview.csv` had been reporting ILNZ `date_max` as 2166/2167.
- **`scripts/common/` created** for the operational layer. It sits beside
  `scripts/diagnostics/` and `scripts/pipelines/` rather than in `src/` or the
  research package, so the AGENTS.md separation holds — a SPARTAN CSV header
  sniffer is not research logic, and making an operational script depend on
  `research/` would invert the dependency (the mistake the earlier `src/common/`
  attempt made). Scripts reach it with a two-line `sys.path` bootstrap because
  the CLI runs them by path; `aethmodular_cli` was not changed.

  | Helper | Was | Now |
  |---|---:|---|
  | `REPO_ROOT` | 14 copies | `common.paths.REPO_ROOT` |
  | `AETHMODULAR_DATA_ROOT` lookup | 6 copies | `common.paths.data_root()` |
  | SPARTAN header sniffer | 6 copies | `common.spartan_io.find_header_line` |
  | CSV reader wrapper | 5 copies | `common.spartan_io.read_spartan_csv` |
  | site-code-from-filename | 8 copies | `common.spartan_io.site_from_path` |
  | datetime builder | 3 copies | `common.spartan_io.build_datetime` |
  | HIPS loader + blank rule | 2 copies | `common.spartan_io.load_hips` |
  | `EXPECTED_STEP_H` / `NEPHEL_SUBS` | 3 forms | `common.spartan_io` |

  Net −87 lines across `scripts/` with a 220-line shared package replacing the
  copies. Equivalence was checked functionally, not just structurally: all four
  header-sniffer variants return identical results on five edge cases
  (plain / one comment line / two comment lines / no header / leading blank),
  and every migrated symbol was asserted to be the *same object* as the shared
  one.

  **`data_root()` fixes a latent bug.** The six inlined copies used a bare
  `Path(os.environ.get(...))` with no normalization, so the documented
  `AETHMODULAR_DATA_ROOT=~/data` override produced a path with a literal `~`
  component that never existed and every downstream file check silently failed.
  The shared helper applies `.expanduser().resolve()`, matching
  `research/ftir_hips_chem/scripts/config.py` and `src.config.project_paths`.

## Done 2026-07-27 (constants unification)

### Wrong physical constant corrected

`src/analysis/bc/source_apportionment.py`, `src/analysis/bc/black_carbon_analyzer.py`
and `attic/utils/plotting.py` keyed **AE33** wavelengths (370 / 520 / 660 nm)
onto **MA350 microAeth** column names (`'UV BCc'`, `'Green BCc'`, `'Red BCc'`).
Since AAE divides by `ln(w1/w2)`, that inflates AAE(Red,IR) by ~16 %
(`ln(660/880) = -0.288` vs `ln(625/880) = -0.342`), and
`source_apportionment` converts AAE straight into a biomass fraction — a true
AAE of 1.20 was reported as 1.43, roughly **doubling** the biomass share
(0.20 → 0.43).

Not currently affecting published results: the only consumers are archived
notebooks and `plotting_gaps_scenarios.ipynb`. (`figure7_source_contributions.ipynb`
mentions `source_apportionment` in a comment only — it does not import it.)
Corrected anyway, because it sits in shipped package code and would have
corrupted results silently on first real use.

`config.WAVELENGTHS_NM` (MA350: 375/470/528/625/880) is now canonical, with
`config.AE33_WAVELENGTHS_NM` alongside it for genuine AE33 `BC1..BC7` exports —
the `BC1..BC7` entries were already correct and were left alone. Covered by
`tests/test_outliers.py::TestChannelWavelengths`, which guards against the AE33
values creeping back onto BCc-keyed maps.

### Filter-id pattern unified

The replicate-stripping regex existed in three places with two different bugs:
`add_base_filter_id` used a bare `r'-\d+$'` (which also strips the 4-digit
sample number from ids already in base form, collapsing every sample at a site
onto one join key), while `spartan_hips_bridge._normalize_fid` used
`r'-(\d)$'` (which misses two-digit replicates, so `ZAJB-0041-12` never joins).
All three now use `config.BASE_FILTER_ID_PATTERN`. It lives in `config.py`
because `data_matching` already imports `etad_factors`, so `etad_factors` cannot
import back without a cycle — and `config.py` imports nothing local.

### New tests

`tests/test_outliers.py` (40 tests) covers the previously untested exclusion API
that AGENTS.md makes mandatory, including registry integrity (every entry has a
reason and a parseable date), the ±1-day tolerance window, filter-id narrowing,
and that flagging never removes rows.

### GF fraction normalization encapsulated

AGENTS.md documents that raw `GF1`-`GF5` are PM2.5 mass fractions summing to
~0.03-0.46 per row and **must** be divided by their row sum, but the code never
did it — all 18 consuming notebooks re-implemented it inline. Added
`etad_factors.normalize_gf_fractions()` and `add_dominant_source()`, which also
guard the all-zero row (NaN, not inf) and the all-NA `idxmax` that raises in
newer pandas. Existing notebooks are unchanged; this gives new work one correct
path.

### Other

- `pls_transfer.FTIRTransferPaths` no longer hardcodes one person's Google
  account. `drive_root()` resolves `AETHMODULAR_DRIVE_ROOT`, then auto-discovers
  `~/Library/CloudStorage/GoogleDrive-*/My Drive`, then falls back to the
  historical path — so it resolves identically on the original machine.
- `build_warren_meeting_deck.py` moved `scripts/` → `workflows/` per AGENTS.md
  (with `warren_meeting_slides.ipynb`'s hardcoded path updated in the same
  change). Its photo/page directories are now env-overridable, and it exits
  non-zero when figures are missing instead of silently emitting a deck of
  `[FIGURE MISSING]` placeholders.
- `python-pptx` + `pillow` added as a `decks` extra (three deck builders could
  not run at all); `ipython` declared in the notebooks group.

### Follow-up same day — four `src/` bugs fixed and the tiers unified

1. **`identify_excellent_periods` returned only periods that had gaps.** It
   grouped the *missing* timestamps, so a period with perfect coverage never
   entered the index and could never be selected — the inverse of the
   function's purpose. Reproduced on 4 periods with one 5-minute gap: it
   returned exactly 1 (the gappy one). Now reindexes over every period the data
   spans; returns 4, three of them with `missing_minutes == 0`.
2. **Threshold classification was dict-order dependent.** `_get_base_quality`
   and `_classify_period_quality` iterated the threshold mapping in insertion
   order, so `{'Good': 60, 'Excellent': 10}` classified 5 missing minutes as
   *Good*. Both now compare in ascending threshold order.
3. **`period_processor` was missing the `moderate` tier**, so 100 missing
   minutes was `poor` there and `moderate` in every other classifier — and its
   classification logic skipped the tier even after it was added.
4. **The 10/60/240 tiers were hardcoded in four places.** All four now derive
   from `src/config/quality_thresholds.completeness_tiers()`, which reads the
   `CompletenessThresholds` dataclass. A test asserts none of them re-literal a
   threshold, so they cannot drift apart again.

Covered by `tests/test_quality_period_bugs.py` (19 tests).

### Also this pass

- **`wavelength="Red"` → `"IR"`** in `aethalometer_filter_merger` (2 signatures)
  and `notebook_config`, matching `config.DEFAULT_BC_WAVELENGTH`. A run at
  defaults had been producing a different BC-vs-EC regression than every other
  entry point.
- **`config.AAE_REGIONS`** added (`fossil_max` 0.9, `biomass_min` 1.5 — the
  value existing `addis_01` output was produced with). `plotting_gaps_scenarios`
  had proposed shipping 1.4 in a helper that was never written.
- **`SMOOTH_RAW_THRESHOLDS`** is no longer re-hardcoded as a default in
  `plotting/comparisons.py` and `plotting/distributions.py`.
- **`plotting_gaps_scenarios.ipynb` corrected.** Its summary cell presented the
  *scrambled* season months as a recommended `config.py` addition — following
  the notebook's own advice would have overwritten the correct calendar. It now
  shows what actually shipped, and the Scenario 1 demo dict is explicitly
  labelled as a deliberate reproduction of the bug.
- **Three zero-importer config modules moved to `attic/config/`**
  (`smoothening_params`, `seasonal_config`, `analysis_presets` — 1,031 LOC).
  `quality_thresholds` was kept and promoted to canonical instead.
- The stale `from src.utils.plotting import ...` left in
  `enhanced_notebook_loader.py` by the attic move is gone.

### Quality stacks — all four bugs fixed, merge still open

Went through `src/analysis/quality/` vs `src/data/qc/` properly. Every defect the
audit named is real and is now fixed:

1. **`data/qc` labelled 9am periods by their END.** For `ts.hour >= 9` it emitted
   `normalize() + 9h + 1 day`, so every label was a day late — while
   `filter_mapping._convert_filter_dates_to_periods` keys filters on
   `date.normalize() + 9h`, a period START. `FilterSampleMapper` joins those two
   series, so **every filter/quality overlap was computed against the wrong
   period**. Both now use the start convention.
2. **The completeness denominator excluded perfect periods.**
   `_analyze_daily_missing` / `_analyze_9am_missing` grouped only the missing
   timestamps, so a period with full coverage never entered the index.
   Reproduced: 4 periods with one 5-minute gap reported **1** period, 100 %
   Excellent. Both now reindex over the full range (4 periods, 3 with zero gaps),
   and the two stacks finally agree on how many periods exist.
3. `identify_excellent_periods` returned only gappy periods (see above).
4. Dict-order-dependent threshold classification (see above).

Also removed `src/data/qc/example_usage.py` — 240 LOC of demo shipped inside the
package, zero importers, 6 `F821` errors, and ruff-excluded *because* it doesn't
lint. Its five examples are all covered by `src/data/qc/README.md`. Its two dead
ruff entries went with it.

**Correction to the audit:** the three `_analyze_temporal_patterns` copies are
*not* duplicates — they take different input types (`DatetimeIndex` vs
`List[Dict]`). Not a safe merge.

**The merge itself is still open**, and the real target is smaller than "850 LOC":

| Group | LOC | Note |
|---|---:|---|
| Genuine overlap (the merge candidate) | 2,078 | `completeness_analyzer` + `period_classifier` + `missing_data_analyzer` vs `missing_data` + `quality_classifier` |
| `data/qc` delivery layer — no rival | 1,772 | `filter_mapping`, `visualization`, `reports`, `seasonal_patterns` |
| `data/qc` instrument cleaning — different concern, already decided **keep** | 1,331 | `pkl_cleaning`, `enhanced_pkl_processing` |
| `analysis/quality` unique | 388 | `data_quality_assessment` |

`analysis/quality` is the better base: it owns the only correct 9am enumerator
and period denominator, it is the test-covered stack, and it inherits
`core.base.BaseAnalyzer`. A merge would port `QualityVisualizer`,
`QualityReportGenerator` and `FilterSampleMapper` into it and drop
`qc/missing_data.py` + `qc/quality_classifier.py`.

**Why it has not been done:** nothing active consumes either stack, so this is
~2,000 LOC of restructuring on code no science runs — and it needs the test
suite rewritten. The four bugs were the actual value and they are fixed. Whether
to spend the day depends on whether `src/` is meant to stay published API
(`pyproject.toml` does ship it in the wheel).

Covered by `tests/test_quality_period_bugs.py` (29 tests).

### Continued 2026-07-27 — live breakage in four active notebooks

**`AethalometerFilterMatcher` could not be constructed at all.**
`_setup_filter_loader` side-loaded `data_loader_module.py` from a path derived as
`dirname(dirname(filter_db_path))`, which resolved to
`research/ftir_hips_chem/data_loader_module.py` — a file that never existed
there. The only copy was in `notebooks/archive/scratch/`. The resulting
`FileNotFoundError` escaped the surrounding `except ImportError`, so the class
raised on construction, breaking all four notebooks under
`notebooks/analysis/absorption/` that import it.

Fixed by promoting the 193-LOC `FilterDataLoader` out of `archive/scratch` into
`src/data/loaders/filter_data_loader.py` (self-contained: pandas/numpy only, no
equivalent existed in `src/`) and importing it normally. Verified: the matcher
now constructs and reports all four sites.

**The same four notebooks also used the pre-`git mv` pickle path** —
`data_root/df_Jacros_9am_resampled.pkl` instead of `processed_sites/`. Repointed
in those four plus `filter_data_availability_strip_chart.ipynb`,
`notebooks/etad_diagnostic.py`, and a docstring example. No live source
reference to the old path remains.

**Six plots silently drew nothing for unsupported layouts.** `resolve_layout`
accepts `'combined'` as valid, but `smooth_raw_histogram`, `uv_ir_ratio_histogram`,
`correlation_matrix`, `data_completeness`, `filter_vs_aeth` and `flow_ratio` had
if/elif chains with no `else` — so `PlotConfig.set(layout='combined')` globally
made them return `None` having drawn nothing. `resolve_layout` now takes an
optional `supported=` set and warns + falls back to `'individual'`. (An earlier
scan flagged seven; `bc_boxplot` was a false positive — it has an `else` that
handles grid/combined deliberately.)

**The FTIR fallback loader dropped half its columns.**
`load_ftir_hips_fallback` selected 6 columns where `FTIRHIPSLoader.load` selects
12, silently losing `volume_m3`, all three MDLs, `fabs_uncertainty` and
`ftir_batch_id` — and which schema you got depended on an `ImportError` you never
saw. Projections are now identical, with a test asserting parity.

Also: `src/data/qc/example_usage.py` deleted (240 LOC, zero importers, 6 `F821`
errors, ruff-excluded *because* it did not lint; its examples are all in
`src/data/qc/README.md`), and four notebooks had markdown cells carrying
`execution_count`/`outputs` keys — invalid per the nbformat schema and a hard
error in future versions. All 161 notebooks now validate.

### Open — needs a decision, deliberately not changed

- **February boundary — resolved as genuinely ambiguous, documented not changed.**
  `config.ETHIOPIA_SEASONS` now carries a note explaining that Dry Oct-Feb (used
  here) and Dry Oct-Jan / Belg Feb-May (used by `ETAD_Factor_Analysis.ipynb`) are
  both published conventions differing only in which season owns February, that
  Belg onset varies by year and altitude, and that the two notebooks' seasonal
  means are therefore not directly comparable. Neither was changed.
- **`plotting_gaps_scenarios.ipynb:408-410`** repeats the scrambled season map
  inside a block presented as a recommended `config.py` addition. Anyone
  following that notebook's own advice would overwrite the correct calendar.
- **Red vs IR default**: `src/data/processors/aethalometer_filter_merger.py:36,351`
  and `src/config/notebook_config.py:21` default to `wavelength="Red"` where
  `config.DEFAULT_BC_WAVELENGTH` is `'IR'`.
- **Site palette — decided: red means Beijing.** Canonical is
  `config.SITES[...]['color']` (Beijing red `#E74C3C`, Delhi blue, JPL green,
  Addis orange). Three notebooks with **no stored outputs** were migrated to
  `{site: cfg['color'] for site, cfg in SITES.items()}` — `Analysis_Tasks_Jan2025`,
  `flow_fix_explorer`, `warren_cena_improve_prep_analysis` — since changing their
  source cannot desync anything. Six notebooks **do** carry stored outputs and
  still use a rival palette; migrating their source without re-running would make
  the notebook contradict its own figures, so they were left: `Task_Analysis_Notebook`,
  `primary_tasks_notebook`, `hips_offset_narrative`, `improve_hips_offset_narrative`,
  `anne_spartan_improve_ec_mass_fabs`, `improve_spartan_may_full_analysis`, plus the
  generated `spartan_ec_2026_06_16/04_new_plots`. Migrate each when you next re-run it.
  (`Multi_Site_Analysis` cell 17 is a deliberate complementary-contrast set, not a rival.)
- **`src/` quality stacks** — all four bugs fixed; the structural merge remains
  open. See the section above for the measured breakdown.

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
| `src/analysis/seasonal/ethiopian_seasons.py` | **Deleted 2026-07-26.** The stated precondition was already met: a repo-wide search for `ethiopian_seasons` / `EthiopianSeasonAnalyzer` outside `src/` returned zero importers. It also encoded a wrong calendar (Dry = `[10,11,12,1,2,3,4,5]`, overlapping Belg `[3,4,5]`), disagreeing with all five other definitions. |
| `src/core/monitoring.py`, `parallel_processing.py`, `src/analysis/advanced/*` | **Moved to `attic/` 2026-07-26**, along with `analysis/aethalometer/smoothening/`, `utils/memory_optimization.py`, and `utils/logging/logger.py` (the last had no consumer outside the other three). Tests kept and repointed at `attic.*`. See [`attic/README.md`](../attic/README.md). |

### Notebook migration — remaining families

`addis_fabs_ec_deming` is migrated. Still inline (safest via their build scripts
where present, else hand-edit + re-run): `regression_stats`/`calculate_regression_stats`
across ftir_hips_chem/improve_hips_offset/spartan, the `catch_up` loader stack
(`_build_catch_up_notebooks.py`), season helpers in the absorption/meteorology
notebooks, and `find_repo_root` everywhere.

### Large regenerable data — resolved 2026-07-26

All four previously-listed items (`research/improve_hips_offset/output/` 877 MB,
`charcoal_ftir/data/raw/` 308 MB, `spartan_ec_2026_06_16/data/*.RDS` 335 MB, and
the stray `node_modules` under `ftir_hips_chem/output/` 57 MB) are **gone from
disk**. Nothing to reclaim here.

**Consequence to be aware of:** with `spartan_ec_2026_06_16/data/` deleted,
`research/ftir_ec_calibration_2026_06_25/`, `research/July07/`, and
`charcoal_ftir`'s workflow notebook are **no longer re-runnable locally** — their
committed notebook outputs are now the only surviving record of those results.
Treat output-stripping in those directories as destructive.

The one large payload that remains is
`research/ftir_hips_chem/charcoal_ftir_sources/downloads/` (225 MB), verifiable
against the tracked `CHECKSUMS.sha256` and re-downloadable from the source DOIs
listed in that directory's `README.md`.

### Stale subdir

`research/filter_combine/` (last touched 2026-02-11) holds three rival
`load_filter_sample_data` implementations (`_simplified`, `_enhanced`,
`enhanced_`) plus three summary markdowns — strongest abandonment candidate.
