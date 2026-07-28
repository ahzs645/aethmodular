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

`aeth notebook check` reports **0 of 138** active notebooks with source-level
machine, cloud, or legacy paths (2026-07-27, after the migration below).

It was 49 of 138 on 2026-07-26. Every one is now resolved through
`scripts/data_paths.py` and the dataset modules; see "Notebook path migration"
under Resolved 2026-07-27 for what changed, what it repaired, and how it was
verified. The distribution that was fixed:

| Area | Notebooks |
|---|---:|
| `research/ftir_ec_calibration_2026_06_25` | 10 |
| `research/ftir_hips_chem` | 14 |
| `research/improve_hips_offset` | 8 |
| top-level `notebooks` | 8 |
| `research/July07` | 4 |
| `research/spartan` | 1 |
| `spartan_ec` / `ftir_etad_char` / `ftir_ec_phase3` | 4 |

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
- ~~`ftir_pls_calibration.py` outside the sanctioned home~~ — **done
  2026-07-28**; moved to `scripts/pls_calibration.py`, 11 importers updated.
  (The count was 8; it was 11.)
- ~~`_build_06_tool_style_plots.py` orphan builder~~ — **done 2026-07-28**;
  builder run, `06_tool_style_plots.ipynb` now exists, `plotly` declared.
- `research/ftir_hips_chem/output/presentations/*.pptx` (8.8 MB) — **decided
  2026-07-28: keep.** They match the ignore rule but are documented,
  non-regenerable deliverables, not build artifacts. See Structural cleanup.
- Untested in the sanctioned `scripts/`: the entire `outliers.py` exclusion API
  (which AGENTS.md makes mandatory for every analysis), `etad_factors.py`
  (including the GF-normalization quirk AGENTS.md flags as easy to get wrong),
  and `flow_periods.py`.

Migrate generated families by editing their `_build*.py` source **and** the
generated notebook's source cells, keeping the two in sync. Do **not** re-run a
builder to migrate: the builders write notebook JSON with no outputs, so
regenerating destroys the stored output text. Hand-authored notebooks should use
the standard setup cell or `prep.find_repo_root`; do not perform a blind JSON
text replacement across executed notebooks.

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

## Notebook discovery pass, 2026-07-27

Four agents read all 138 active notebooks looking for utilities worth defining
rather than bugs. They found both.

### AAE sign convention was inverted in four notebooks

`addis_01_source_apportionment`, its `_daily` twin, `addis_04_aeronet` and its
`_daily` twin computed

    aae = ln(IR / UV) / ln(880 / 375)

which is the **exact negative** of the standard
`AAE = -ln(b_short/b_long) / ln(wl_short/wl_long)` used by
`addis_05_diurnal_wavelength_analysis` and
`multisite_diurnal_wavelength_analysis`. Because UV absorption exceeds IR for
real aerosol, the inverted form is always ≤ 0, so every sample fell below
`AAE_REGIONS['fossil_max']` and was classified **fossil fuel** — a genuine
biomass sample at UV/IR = 5.5 (true AAE 2.0) was reported as −2.0 and labelled
fossil. The `clip(-1, 3)` those notebooks carried, whose negative lower bound
accommodated the symptom, is the tell.

None of the four has stored outputs, so no published figure carries this.
Fixed by flipping the ratio to UV/IR at all six call sites, and by adding
`scripts/optics.py` — `aae`, `aae_from_columns`, `classify_aae`,
`aae_source_summary` — which raises if the wavelengths are passed in the wrong
order rather than silently flipping the sign, and reads
`config.WAVELENGTHS_NM` so the AE33/MA350 mix-up cannot recur.
Covered by 8 tests.

### `hips_fabs` carried two units under one name

`pivot_filter_by_id` returns HIPS_Fabs as stored — Mm⁻¹, median 47.1 for ETAD.
`match_all_parameters` and `match_hips_with_smooth_raw` return a column of the
**same name** already divided by `MAC_VALUE` — a µg/m³ BC equivalent, ~4.7. A
factor of ten, with nothing in the name to say which you hold; five notebooks
assume one reading and three the other.

`pivot_filter_by_id` now takes `hips_units={'raw'|'ugm3'|'both'}`. Default is
`'raw'`, so existing callers are unaffected; `'both'` adds an unambiguous
`hips_bc_ugm3` and is what new code should use. 5 tests pin the relationship.

### `calculate_regression_stats` is now a superset of `regression_metrics`

The two returned the *same* OLS R² under different key names (`r_squared` vs
`R2`), and only `pls_transfer.regression_metrics` computed RMSE/MAE/bias. That
fork is why roughly 30 phase-3 and absorption call sites hand-rolled a stats box
instead of using `add_stats_textbox` / `add_regression_line` — they held a dict
the plotting helpers could not read. Verified numerically identical, then added
`R2`, `RMSE`, `MAE`, `bias`, `median_bias`. All eight shared keys now agree
exactly. Legacy keys unchanged.

### Other fixes

- **SNV zero-σ guard added to 10 inline copies** across `run_char_02/05/06/07/
  08/09/10/12`. Only `run_char_11` had it; a flat row (masked region, or all-NaN
  after resampling) yielded `nan` instead of `0`, and `nan` propagates through
  every downstream correlation. The generators are fixed; the executed
  `char_*.ipynb` pick it up on their next re-run, which needs the Drive mount.
- **25 builder scripts wrote notebooks to a bare relative filename**, so running
  one from the repo root silently dropped its `.ipynb` in the repo root instead
  of beside its source. All now resolve against `Path(__file__).parent`.
- **`config.SPARTAN_DEPOSIT_AREA_CM2` / `IMPROVE_DEPOSIT_AREA_CM2` /
  `IMPROVE_AREA_SENSITIVITY_CM2` added.** SPARTAN's is 3.53, taken from the data
  (constant across all 11,720 rows). IMPROVE's is 3.5 in six notebooks and 3.53
  in three — so the ETAD-vs-IMPROVE µg/cm² comparison carries a systematic
  0.86 % offset. Default left at 3.5 (what the published envelope figures used);
  **this needs a decision**, and the constant now makes it one.
- **`Seasonalclass.ipynb` cell 4 deleted** — byte-identical to cell 3 (same MD5,
  19,315 chars), writing the same six PNGs twice. Neither had outputs.
- **`hips_two_problem_decomposition.ipynb`** hand-filtered Delhi with
  `valid[valid['ftir_ec'] < 30]`. That exact sample (2024-06-21, INDH-0172-4,
  60,640 ng/m³) is already in `outliers.EXCLUDED_SAMPLES['Delhi']` with a
  reason. Now uses `apply_exclusion_flags` + `get_clean_data`, per AGENTS.md.

### Backlog the pass produced — not yet acted on

Ranked by consuming notebooks. Each is a utility that does not exist yet:

| Proposed | Notebooks | Note |
|---|---:|---|
| `nbsetup.bootstrap(slug)` | ~40 | four rival repo-root idioms, all cwd-relative; 36 of 43 `ftir_hips_chem` notebooks only work when cwd is the notebook dir |
| `prep.add_calendar_columns` | 13 | `season_for_month` ships with **0 adopters** — consumers need the columns wrapper, not the scalar |
| `etad_factors.attach_factors_by_date` | 12 | `match_etad_factors` takes a date *column*, not an index, and loops per row |
| `improve_io.load_improve_clean` | 11 | the canonical IMPROVE frame exists only as one notebook cell plus an untracked CSV that is now gone |
| `spectra/` subpackage (grid, preprocess, bands) | 9–10 | reference implementation is trapped inside `run_char_11.py` |
| `aeronet.py` loader + column aliases | 8 | `AERONET_DATA_DIR` points at an empty dir, so all 8 hardcode a Drive path |
| `prep.output_dirs` | 18 | 14 byte-identical `setup_directories()`, all cwd-relative |
| `plotting.overlays.crossplot_on_axes` | ~24 | now unblocked by the stats superset above |

Also observed: the `_daily` twins are 62–97 % identical (`addis_01` ↔
`addis_01_daily` = 0.97, 244 shared lines) and differ by three parameters;
47 % of `scripts/` imports across the estate are unused, because AGENTS.md's
50-line setup cell is pasted wholesale; and `ETAD_comprehensive` cells 46–49
gate on `valid_aeth_mac_clean`, which is read at 11 sites and assigned at none.

## Backlog utilities built, 2026-07-27

Implemented by gpt-5.6 sol high via `codex exec`, each in an isolated worktree,
then harvested and re-verified independently in the main checkout.

| Shipped | Replaces inline copies in | Tests |
|---|---:|---:|
| `nbsetup.bootstrap` | ~40 notebooks | 10 |
| `prep.output_dirs` | 18 | ″ |
| `prep.add_calendar_columns` | 13 | ″ |
| `etad_factors.attach_factors_by_date` | 12 | 6 |
| `spectra/{grid,preprocess,bands}` | ~10 | 14 |
| `plotting.overlays.crossplot_on_axes` | ~24 | 6 |
| `aeronet.py` (loader + column aliases) | 8 | 11 |

**`nbsetup.bootstrap(slug)`** returns absolute `repo_root` / `data_root` /
`plots_dir` / `tables_dir`, delegating to `prep.find_repo_root` (which raises)
rather than the silent `Path.cwd()` fallback ~17 notebooks carry. 14 notebooks
build cwd-relative output dirs today, so running one from the repo root writes
figures to the wrong place — that is the bug this removes.

**`prep.add_calendar_columns`** adds `Month`/`Hour`/`DayOfWeek`/`DayOfYear` plus
`season`, reading `config.season_for_month`. It raises `TypeError` when given
neither a DatetimeIndex nor a `date_col`, and accepts custom season mappings in
either shape. `season_for_month` previously had **zero** adopters because every
consumer needed this wrapper, not the scalar.

**`etad_factors.attach_factors_by_date`** is the index-friendly counterpart to
`match_etad_factors` (which needs a date *column* and loops per row). It uses
`pd.merge_asof`, applies `normalize_gf_fractions` by default — the step AGENTS.md
marks mandatory and which ten of the twelve notebooks skip — and **guards
`tz_localize(None)`**, fixing a real `TypeError` that two notebooks hit on an
already-naive index. Verified on both tz-aware and tz-naive input.

**`spectra/`** lifts the machinery trapped inside `run_char_11.py` and
`charcoal_spectra.py`. Two deliberate variants were preserved rather than
merged, and both were verified to still differ:
- `normalize_area(..., negatives="min-shift"|"clip")` — min-shift is documented
  as necessary for the already-SNV'd Maezumi/WDG collections, which "would lose
  half their shape to clipping"; clip is the July07 convention.
- `band_mean` vs `band_integral` — `band_area` previously named both a
  `nanmean` and a min-shifted trapezoid integral. `band_integral` now sorts the
  wavenumber axis internally, so the `-np.trapezoid` sign hack is gone; verified
  to give the same positive result on ascending and descending grids.
`snv` carries the zero-sigma guard. `MASK_REGIONS` names the CO2 (2280-2400)
and PTFE/CF (1100-1300) exclusion windows the notebooks derived ad hoc.

### Not built

**`improve_io.load_improve_clean` (11 notebooks) — this entry was wrong, and it
shipped 2026-07-27.** It claimed the FED source workbooks were not on the Drive
mount. They are; the Query Wizard names them `<account>_<timestamp>_<id>.xlsx`,
so searching for `improve*` found nothing. See the Resolved section for what was
built and how it was verified.

**`plotting.overlays.crossplot_on_axes`** is the largest single dedup in the
estate: ~24 notebooks hand-draw the same panel -- 89 hand-written 1:1 lines, 90
inline `linregress` calls, and 111 hand-built stats boxes whose text is
character-identical to what `add_stats_textbox` already emits. It is built from
the existing `plotting.utils` primitives, accepts a precomputed `stats=` dict
(phase-3 notebooks already hold one from `pls_transfer.regression_metrics`)
without mutating the caller's copy, normalises `R2` to `r_squared` at the helper
boundary, and appends RMSE to the stats box when present -- verified rendering
`n = 50 / R2 = 0.965 / y = 1.613x - 0.38 / RMSE = 7.09`. Showing RMSE is exactly
the capability whose absence blocked those notebooks from adopting the helpers.

**`aeronet.py`** replaces four rival loader signatures across 8 notebooks. Two
format traps made a header *sniffer* necessary rather than a fixed `skiprows`:
AOD exports use `Date(dd:mm:yyyy)` while SDA exports use `Date_(dd:mm:yyyy)`
(extra underscore), and the AERONET portal files the notebooks read have a
6-line preamble while the three samples committed at
`notebooks/analysis/data/aeronet_aod_*.csv` have 5. Verified: the sniffer returns
6 and 5 respectively, and parses the real Beijing sample (346 rows, DatetimeIndex,
81 columns). `-999` becomes NaN. `COLS`/`resolve_column` provide the alias layer
for `AOD_500nm` (116 raw literals across the estate) and
`Precipitable_Water(cm)` (47).

`aeronet_dir()` resolves `AETHMODULAR_AERONET_DIR` -> a non-empty
`config.AERONET_DATA_DIR` -> a Drive location via the existing
`pls_transfer.drive_root()`. **No account name is hardcoded** — the module source
contains none; the account only appears in the path resolved at runtime. That is
what lets the 8 notebooks stop hardcoding a personal Drive path.

**Note on the worktrees:** each Codex worktree is a clean checkout of HEAD, not
of the working tree, so implementations were written against the *committed*
API. `crossplot_on_axes` was therefore built without the `calculate_regression_stats`
superset present -- it degrades gracefully (`if 'RMSE' in stats`) and gains the
RMSE line once merged. When harvesting, only the changed module and its own new
test file were copied; a worktree's `tests/test_scripts_helpers.py` is HEAD's
56-test version and would silently revert newer tests.

All agent worktrees have since been removed (5.3 GB reclaimed) along with their
two stale `worktree-agent-*` branches, both of which pointed at `main`'s commit
with zero commits ahead. Before deleting, the abandoned drafts were diffed
against what shipped. Coverage was equivalent (same behaviours, different test
names), and the shipped versions were the later and better ones — they add a
`tolerance_days < 0` guard, and they let a bad date raise instead of
`errors="coerce"` silently producing `NaT`, which is the same anti-silent-fallback
rule applied to `find_repo_root`. One draft difference was a latent breaking
change: it set `DayOfWeek` to `day_name()` strings, but **every** existing
definition in the estate uses integer `.dayofweek`, and consumers filter on
`DayOfWeek >= 5` / `< 5` for weekend/weekday, which raises `TypeError` against
strings. The integer form shipped. The one draft idea worth keeping was adopted:
`nbsetup.bootstrap` now derives `scripts_dir` from `Path(__file__).parent` rather
than rebuilding `repo_root / "research" / "ftir_hips_chem" / "scripts"`, so it
survives a relocation of the package.

## Structural cleanup (2026-07-28)

1. **`ftir_pls_calibration.py` moved to the sanctioned home** -- the one item
   here with live consumers. 342 LOC of reusable PLS calibration logic sat in
   `research/spartan_ec_2026_06_16/` while **11 files across two research
   directories** imported it, the far side via
   `sys.path.insert(0, str(Path("../spartan_ec_2026_06_16")))` -- a cwd-relative
   hop into a sibling workspace, which is both an AGENTS.md violation and the
   same fragility class as the notebook paths fixed earlier that day. It now
   lives at `scripts/pls_calibration.py` (no name collisions with the existing
   `pls_transfer.py`), is exported from the package, and all 16 affected files
   were updated. The three same-directory importers -- which had no `sys.path`
   line at all because the module used to sit beside them -- got a repo-root
   bootstrap; without that they would have broken silently. Verified by the
   module's own `_self_test()`: coefficient export/re-apply round-trips to
   2.84e-14. Notebook outputs preserved (4/7/7/7/5/10).

2. **The two quality stacks: merge NOT justified -- the backlog estimate was
   wrong.** It claimed "~850 LOC saving" from merging `src/analysis/quality`
   (1,744 LOC) into `src/data/qc` (3,916 LOC). Measured, the two share exactly
   **2 name collisions** (`MissingDataAnalyzer` and its `analyze_missing_patterns`)
   out of 23 and 77 public names. They are largely *different* functionality, not
   duplicates, and neither has an active consumer -- only tests and archived
   notebooks. Churning 5,660 LOC of unused code to save a genuine overlap of two
   names is not worth the risk. **Recorded as declined, not deferred.**

   The one real defect found while checking was fixed: `qc/quality_classifier.py`
   hardcoded `{'excellent': 10, 'good': 60, 'moderate': 240}` while
   `quality/period_classifier.py` read the same tiers from
   `src/config/quality_thresholds.py`. That is the rival-copy pattern that had
   already bitten once -- `period_processor` was missing the 240 tier entirely,
   so a 100-minute gap classified as 'poor' there and 'moderate' everywhere else.
   It now calls `completeness_tiers(lowercase=True)`. Behaviour-preserving:
   all four tier boundaries verified unchanged, explicit overrides still win.
   3 tests.

3. **Orphan builder resolved.** `_build_06_tool_style_plots.py` wrote a
   `06_tool_style_plots.ipynb` that was **never** committed (01-05, 07, 08 exist;
   06 was the gap). Ran the builder -- it only writes JSON, no data needed -- so
   the notebook now exists, 22 cells, 0 syntax errors, and the builder is no
   longer an orphan. It surfaced an undeclared dependency: the generated notebook
   imports `plotly`, which was nowhere in `pyproject.toml` (same class as the
   `tqdm`/`requests` gaps fixed 2026-07-26). Added to the `notebooks` extra.
   Note the notebook still cannot *execute*: it reads `rds_EC_X.csv`, the same
   missing file that blocks four notebooks in `ftir_ec_calibration_2026_06_25`.

4. **Tracked presentation decks -- recommend keeping, and the framing was
   misleading.** The two `.pptx` (8.8 MB, 3.2 % of a 273 MB repo) do match
   `.gitignore`'s `research/**/output/`; the rule is simply inert because they
   were committed first (`git check-ignore` stays silent on tracked paths --
   `--no-index` shows the match). But they are **not** build artifacts:
   `COMPLETE_RESEARCH_SUMMARY.md:132,216` documents them as deliverables (an
   18-slide weekly deck and a 15-slide confident-results deck with presenter
   notes), and **no builder in the repo regenerates them** -- the two deck
   builders that exist write different filenames
   (`spartan_ec_weekly_2026_06_16.pptx`, `warren_meeting_slides.pptx`).
   Untracking them would lose content that cannot be rebuilt. Left tracked; this
   is a retention call, not a cleanup defect.

## Plot taxonomy (2026-07-28)

Earlier passes characterised exactly one family (crossplots) and counted inline
helper redefinitions. This is the full census: **3,942 plotting calls across 180
files**, forming **654 figure-producing cells**.

Figure recipes, by primitive combination:

| Recipe | Cells | What it is |
|---|---:|---|
| `plot+scatter` | 216 | the crossplot family (scatter + fit/1:1 line) |
| `plot` | 82 | time series |
| `scatter` | 50 | bare scatter, no fit |
| `fill_between+plot` | 49 | series with an uncertainty/spread band |
| `bar` / `barh` | 46 | summary/count bars |
| `boxplot` | 21 (+20 mixed) | distribution by season/site/wavelength |
| `hist` | 19 (+14 mixed) | distributions |
| `hexbin+plot+scatter` | 8 | density crossplot for the large IMPROVE pool |

Variant axes *within* those families -- these are the knobs that actually differ,
and the reason the panels look repetitive but are not interchangeable:

| Variant feature | Figures |
|---|---:|
| hand-built stats box | 231 |
| 1:1 reference line | 187 |
| a fitted regression | 185 |
| equal/square axes | 140 |
| colour-by-third-variable (loading, iron, biomass %) | 110 |
| uncertainty band | 104 |
| shared axes across a grid | 98 |
| log axis | 31 |
| seasonal shading | 28 |

### Findings

1. **Regression method is the substantive one.** *(Acted on 2026-07-28 --
   `calculate_regression_stats(..., errors_in_variables=True)` now returns
   `deming_slope`, `deming_intercept`, `deming_lambda`, and
   `slope_attenuation_pct`; `crossplot_on_axes` enables it automatically
   whenever `one_to_one and equal_axes` and prints both slopes in the stats box.
   Off by default in `calculate_regression_stats` so the 10+ existing callers'
   boxes are unchanged, and `crossplot_on_axes` had zero notebook adoption, so
   no published figure moved. Validated against a synthetic set with a known
   slope of 1.00 and equal error in both variables: OLS returns 0.859, Deming
   1.004. 10 tests.)* Of the **115** figures that draw
   a 1:1 line -- which asserts both axes measure the same quantity, so both carry
   error -- **114 fit with an OLS-family estimator only** (`linregress` 216 uses,
   `polyfit` 162, `OLS` 135) and just **one** also computes an
   errors-in-variables fit. The project already owns the right tool
   (`plotting.utils.deming`, plus a whole `addis_fabs_ec_deming` research dir),
   so this is under-adoption, not a missing capability.

   Measured on the real filter data, orthogonal (lambda=1) vs OLS:

   | Comparison | n | OLS | Deming | shift |
   |---|---:|---:|---:|---:|
   | ETAD Fabs/MAC vs FTIR EC | 190 | 1.898 | 2.379 | +25.3 % |
   | CHTS Fabs/MAC vs FTIR EC | 160 | 1.079 | 1.428 | +32.4 % |
   | USPA Fabs/MAC vs TOR EC | 114 | 0.651 | 0.890 | +36.9 % |
   | FTIR EC vs TOR EC (all sites) | 149-175 | ~1.00 | ~1.00 | 0.0 % |

   Median shift **24 %** across the ten site-pairs. The zero-shift rows are the
   control: where the two variables track each other almost exactly, OLS and
   Deming agree, so the 24 % is regression dilution and not an artefact of the
   estimator. **Caveat: lambda=1 (orthogonal) is an assumption here** -- the
   `Uncertainty` column in `unified_filter_dataset.pkl` is entirely null for
   `HIPS_Fabs`, `EC_ftir`, and `ChemSpec_EC_PM2.5`, so a data-driven lambda is
   not currently computable. The direction is robust; the exact magnitude is not.

2. **`EC_ftir` and `ChemSpec_EC_PM2.5` agree to about 1 %** -- slope 1.00,
   intercept 0.00, r2 = 0.99992, median ratio 1.000 at every site. For two
   nominally independent analytical methods (FTIR vs thermal-optical) that is not
   physically plausible; published FTIR-vs-TOR EC calibrations reach r2 ~0.8-0.95.
   Investigated 2026-07-28. What the data settles:

   - **Not a copy or join-fill.** 103 filters carry `EC_ftir` with **no** TOR EC
     at all (494 have both, 49 TOR-only). A copied column could not exist where
     the source is absent.
   - **Not an arithmetic restatement of the reported TOR value.** `EC_ftir` is
     stored at full float precision (14-16 decimals, 597 distinct values) while
     `ChemSpec_EC_PM2.5` is a reported figure rounded to 2 decimals (343 distinct
     values). The residual sd between them, 0.0254 ug/m3, is **8.8x** larger than
     the 0.0029 explainable by TOR's rounding, and **0 of 494** pairs are exactly
     equal.
   - So `EC_ftir` is an independently *computed* quantity that nonetheless tracks
     a 2-decimal TOR figure to ~1 %. That is the signature of a calibrated
     prediction whose training target was TOR EC.

   **Where the values come from.** The pickle's builder did exist in this repo:
   `research/filter_combine/` (`main.py` + three `load_filter_sample_data_*.py`
   variants), deleted in `aa9714b` on 2026-07-27. Recovered from git, `main.py`
   is a pure reshaper -- its FTIR branch is commented *"Load FTIR data (already
   in long format)"*, and it only tags `DataSource='FTIR'`, sets units, and
   concatenates. It never computes EC. `EC_ftir` arrives **already computed** in
   `Four_Sites_FTIR_data.v2.csv`, at full float precision
   (`MassLoading_ug = 6.50430770368977`).

   **So the calibration -- and whether these filters were in its training set --
   lives upstream of this repository entirely.** That remains the crux: in-sample
   fit would explain r2 = 0.9999 and would mean the 18 files crossplotting the
   pair report calibration fit rather than independent method agreement. It
   cannot be settled from this checkout. **Ask whoever produced
   `Four_Sites_FTIR_data.v2.csv`**; until then treat FTIR-EC-vs-TOR-EC panels as
   calibration diagnostics, not validation.

5. **`MDL` is populated where `Uncertainty` is not, and it pins lambda.**
   `Uncertainty` is empty for every parameter, but `MDL` is present for `EC_ftir`
   (597/750), `OC_ftir` (597/750), and `ChemSpec_EC_PM2.5` (1000/1043) -- and
   **absent for `HIPS_Fabs` (0/546)**. Using median MDL as the error proxy gives
   lambda = (0.4143/0.0578)^2 = **51** for the TOR-vs-FTIR EC pair: FTIR's own
   stated detection limit is 7.2x TOR's, so nearly all the error is on the FTIR
   axis, lambda >> 1, and OLS-of-y-on-x is already close to correct **for that
   pair**. That independently explains the 0.0 % rows in the table above rather
   than leaving them as a coincidence.

   The consequence is specific and actionable: for the Fabs-vs-EC comparisons --
   the ones showing the 22-37 % attenuation -- `HIPS_Fabs` carries no MDL, so
   lambda = 1 (orthogonal) is the only available assumption and the magnitude of
   that correction stays uncertain. **Obtaining HIPS Fabs uncertainties would
   firm up the largest correction in the estate.** Pass them via
   `calculate_regression_stats(..., sigma_x=, sigma_y=)` when available.

3. **MAC reference lines are healthy** -- worth stating because it looked like a
   risk. 36 files draw MAC=10 only (matching `config.MAC_VALUE`), **zero** draw 6
   alone, and the 14 files carrying both are the deliberate side-by-side protocol
   comparisons (`ftir_19_mac_effect_on_calibrations`,
   `ftir_22_figures_under_both_protocols`). The 10/20 pairs in the IMPROVE
   notebooks are bracketing reference lines, not a rival convention. The open
   MAC 6-vs-10 fork is being handled explicitly rather than drifting.

4. **Cosmetic drift confirms the panels are hand-built.** *(dpi acted on
   2026-07-28: `config.SAVEFIG_DPI` (200) and `config.FIGURE_DPI` (110) are now
   applied by `apply_default_style()`. 462 savefig calls pin a dpi explicitly and
   still win; the 224 that never did were silently getting matplotlib's 100 and
   now match the pinned majority. Resolution only -- no data or layout change.)* R-squared is spelled
   five different ways in stats boxes (`R2` 352, `r2` 167, escaped `R\u00b2` 55,
   `R2` 28, `R^2` 15), and `savefig` uses **13 distinct dpi values** from 120 to
   1000 (160/150/140 most common). Harmless individually; together they are the
   signature of 231 independently written stats boxes.

### Resolved 2026-07-27

- **Notebook path migration — 49 of 138 flagged, now 0.** `aeth notebook check`
  reported 49 notebooks with machine-specific paths; the count is now **0**, and
  no active notebook carries a `/Users/`, `Library/CloudStorage`, or
  `GoogleDrive-` literal in code. 52 notebooks and 9 `_build_*.py` generators
  changed.

  A new `scripts/data_paths.py` resolves every external dataset at runtime
  (`maia_data_root`, `aethalometry_dir`, `weather_dir`, `weather_file`,
  `ftir_spectra_dir`, `ftir_local_db`), each overridable by env var, none naming
  an account. The MAIA prefix had been spelled out separately in `aeronet.py` and
  `improve_io.py`; both now share the one definition, verified to resolve
  unchanged. 25 tests in `tests/test_data_paths.py`.

  **Generated notebooks were never re-run.** The `_build_*.py` scripts write the
  notebook JSON with no outputs, so regenerating would have destroyed the stored
  output text. Both the builder and the notebook's source cells were edited
  instead, keeping them in sync. Output counts were compared before and after for
  every file: **zero drift across all 52.**

  Verified by executing each migrated setup cell from the notebook's own
  directory and comparing against the same cell at HEAD: 31 unchanged and
  working, **4 repaired**, 1 regression found and fixed (an ordering slip that
  used `Path` before `pathlib` was imported), and 4 that were already failing on
  a missing `rds_EC_X.csv` — unrelated to paths and still open.

  Four defects surfaced that the checker alone would not have caught:
  - `diurnal_pub_figures.ipynb` read a Desktop copy of the Jacros pickle that no
    longer exists. The same file is on Drive, so the migration **repaired a
    broken notebook**.
  - Several notebooks hardcoded `My Drive/FTIR/local_db`, which does not exist
    here, printed "BLOCKED: local_db not found" and carried on degraded — while
    the tables were present under `University/Research/Grad/Data/FTIR`.
    `ftir_local_db()` searches both, so those notebooks stop degrading.
  - The in-repo and Drive `Weather Data/Meteostat` directories hold **different**
    files, so a `weather_dir()` that merely prefers the local copy silently
    returns a directory lacking the requested file. `weather_file(*names)`
    searches both and raises listing every location tried.
  - `_build_03_adama_han_char_soot.py` pointed at `<MAIA>/Adama TOR`; the dataset
    has since moved to `<MAIA>/DAVIS/Adama TOR`. The notebook's own stored output
    shows the old location was real when it last ran, so both are searched.

  Migration replaced some absolute paths with a cwd-relative
  `sys.path.insert(0, "../ftir_hips_chem/scripts")`, which merely trades a
  machine dependency for a working-directory one — run from the repo root, the
  import fails outright. All 11 such sites now derive `scripts/` from a
  `pyproject.toml` walk-up instead.


- **`improve_io.load_improve_clean` — shipped; it was never actually blocked.**
  The earlier note said the FED source workbooks were not on the Drive mount.
  That was wrong: they are there, named `<account>_<timestamp>_<id>.xlsx` by the
  FED Query Wizard, so a filename search for `improve*` missed them. Only the
  *derived* `improve_valid_cleaned.csv` was gone, and it is a generated artifact
  under a gitignored `output/` tree — regenerable, not lost.
  `research/ftir_hips_chem/scripts/improve_io.py` now carries the loading and
  cleaning logic that previously existed only inside
  `improve_high_fabs_comparison.ipynb`, so the 11 consuming notebooks can call
  `load_improve_clean()` instead of hardcoding a dead path (several hardcoded an
  absolute `/Users/...` prefix). `improve_dir()` resolves
  `AETHMODULAR_IMPROVE_DIR` -> `config.IMPROVE_DATA_DIR` -> Drive discovery, with
  no account name in the module source. A missing cache with no reachable source
  raises `FileNotFoundError` naming both the path and the env var; it never
  returns an empty frame.
  **Verified against the producer notebook's own recorded output — all five
  statistics match exactly**: 379,697 valid rows, 2003-01-03 to 2025-07-30, 214
  sites, max fAbs 310.95 Mm^-1, 147,380 rows with raw 635 nm RT fields joined.
  The cache has been regenerated.
  One fragility was fixed on top of the port: the module originally rebuilt this
  notebook's deposit-area sweep by reshaping `IMPROVE_AREA_SENSITIVITY_CM2`,
  which belongs to a *different* notebook
  (`improve_white_style_rt_calibration_space`) and differs in its middle value —
  3.53 (SPARTAN's area) vs 3.5 (FED Module A's). The result was correct but
  reads as a typo and would break silently if either constant were edited, so
  `IMPROVE_HIGH_FABS_AREAS_CM2 = (2.2, 3.5, 4.0)` is now declared separately and
  a test pins the two sweeps apart. 9 tests in `tests/test_improve_io.py`.
- **February boundary — resolved by making the choice explicit, not by picking a
  winner.** Dry Oct-Feb and Belg Feb-May are both published Ethiopian calendars;
  forcing one would have silently rewritten a notebook's published seasonal
  means. Instead `config.py` now registers **both** —
  `ETHIOPIA_SEASONS` (default, unchanged) and `ETHIOPIA_SEASONS_BELG_FEB` — in a
  `SEASON_CONVENTIONS` registry, with `season_for_month(month, seasons=...)`,
  `resolve_seasons()` (accepts a name or a mapping; raises on an unknown name
  rather than falling back), and `season_convention_name()` for labelling output.
  A non-registered mapping reports as `'custom'`, so a report can say the
  calendar was non-standard instead of implying it was canonical.
  `ETAD_Factor_Analysis.ipynb` now selects `'belg_feb'` **by name** instead of
  redeclaring the calendar inline, which makes the choice greppable. The
  migration is behaviour-preserving: a test asserts the named convention
  reproduces the notebook's former inline mapping for all 12 months, and the
  notebook has no stored outputs to desync. Both calendars share one palette,
  since colour encodes which season, not which convention. 13 tests in
  `tests/test_season_conventions.py`.
- **Red vs IR default — already fixed; the entry above was stale.** Every
  `wavelength` default in `src/` is now `'IR'`, matching
  `config.DEFAULT_BC_WAVELENGTH` (`aethalometer_filter_merger.py:13,39,361,785`,
  `notebook_config.py:22`, `site_corrections.py:92,161,212,254`), each with a
  comment recording the change. The only remaining `'Red'` defaults are in
  retired `archive/` notebooks, which are a historical record and correctly left
  alone.
- **`plotting_gaps_scenarios.ipynb` scrambled season map — already fixed; the
  entry above was stale.** The scrambled dict now carries an explicit "these
  month lists are SCRAMBLED on purpose … Do not copy this dict" warning naming
  `config.ETHIOPIA_SEASONS` as correct, and the "what to build" summary carries a
  status block pointing at the shipped calendar and `plotting/overlays.py`.

### Open — needs a decision, deliberately not changed

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
