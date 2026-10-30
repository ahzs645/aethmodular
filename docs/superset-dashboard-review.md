# Superset ideas for the Aethmodular dashboard

Reviewed: 2026-10-30.

The best direction is to extend the existing React/D3 gallery with Superset's patterns for filter scope, chart lifecycle, drill-down and reproducible dashboard definitions. The strongest immediate improvement is to stop loading calibration data for dashboard sheets that do not use it.

This review assumes “our dashboard” means `gallery/app/src/pages/DashboardPage.tsx` in this repository. It compares dashboard-related Superset source and documentation with the gallery's implementation. It is a source review and payload inspection, not an exhaustive audit of every Superset file or a browser performance benchmark. No application code or deployment was changed.

## Sources and version boundary

- Local gallery baseline: `bf706c294b76bba84bfd9ae0035f8aa9bf3822ca`.
- Superset checkout: [`1e06f5e83b31998ff73a146de9865cff1b6e57d6`](https://github.com/apache/superset/tree/1e06f5e83b31998ff73a146de9865cff1b6e57d6), the default branch retrieved for this review.
- GitHub's latest-release endpoint returned [6.1.0](https://github.com/apache/superset/releases/tag/6.1.0), published 2026-05-13. Default-branch features and defaults discussed below must not be assumed to exist in that release. The recommendations borrow patterns; they do not require installing Superset.
- Upstream links in this document are pinned to the reviewed commit.

## What the gallery already has

| Capability | Existing implementation | Implication |
| --- | --- | --- |
| Movable/resizable dashboard tiles, collision handling and keyboard arranging | `pages/DashboardPage.tsx`, `lib/dashboard.ts` | Keep the existing layout engine. |
| Named sheets, site/compare/network scopes, focus layout and presentation mode | `DashboardPage.tsx`, `components/dashboard/DashHeader.tsx` | Superset tabs are a useful comparison, but named pages already exist. |
| Independent settings for duplicate chart instances | `lib/url.ts` and tile-specific parameter prefixes | Extend this mechanism for new tile settings. |
| Saved views, compressed links and chrome-free embeds | `lib/views.ts` | Sharing is already implemented; improve reproducibility rather than adding another link format. |
| Undo/redo with grouped URL changes | `lib/undo.ts`, `lib/undoHistory.ts`, `components/UndoToast.tsx` | Do not add Redux or replace this with Superset's undo stack. |
| Linked hover/pin, brush selection and explicit keep/exclude actions | `lib/highlight.tsx`, `App.tsx`'s `BrushBar`, `components/SubsetBar.tsx` | Cross-chart interaction exists. The new opportunity is selecting which tiles a filter affects. |
| Sample detail drawer and per-chart figure exports | `components/SampleDrawer.tsx`, `ChartFrame.tsx`, `ExportDialog.tsx` | Add aggregate drill-down and dashboard-level data exports on top. |
| Fetch deduplication, derived-data caching and failed-request eviction | `lib/data.ts` | Reuse the loader; do not introduce a parallel fetch/cache system. |
| Live SPARTAN refresh, source dates, timeout and bundled fallback | `lib/spartanSource.ts`, `spartanPublic.ts`, `networks.ts` | Freshness exists; make it visible at the relevant tile and include it in exports. |
| Lazy map code, mobile near-viewport mounting and cached field-presence scans | `charts/map/lazy.tsx`, `DashboardPage.tsx`, dashboard registry | Extend these existing optimizations. |
| Shared dense-point rendering and selective hover subscriptions | `components/ScatterLayer.tsx`, `lib/highlight.tsx` | Preserve the performance rules in `AGENTS.md` and `docs/gallery-chart-standards.md`. |

## Recommended functions and interactions

### 1. Scoped filters and an explicit reference tile

**Value: high. Effort: medium–large.**

Superset resolves filter scope to chart IDs, supports cross-filter scopes, and displays applied/incompatible filter indicators. Borrow the interaction without importing its Redux state model.

Add an “Affects” choice for a dashboard filter: all tiles, selected tiles, or a named comparison group. The default remains the existing global behavior. A reference tile can retain a deliberately selected cohort while an adjacent tile changes site, season or dates. This makes before/after and cohort comparisons much easier to read.

Every tile needs a compact, inspectable summary of its **effective** filters, source, measurement method and sample count. A tile that ignores a date filter must say why. Start with one opt-in reference tile and a small set of supported filters; arbitrary scope combinations can follow if useful.

Keep hover, brush highlighting and analytical filtering distinct. The gallery already lets a brush become “Keep only these” or “Exclude these”; do not make an ordinary point click silently change regression statistics. Clear a scoped filter from its own chip, and include its scope in saved views.

Sources: [cross-filter configuration](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/dashboard/util/crossFilters.ts), [scope resolution](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/dashboard/util/getChartIdsInFilterScope.ts), [filter badges](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/dashboard/components/FiltersBadge/index.tsx).

### 2. “Show contributing filters” for aggregate charts

**Value: high. Effort: medium.**

Superset's drill-to-detail pattern is a good fit for the scientific gallery. The existing drawer explains one physical filter; an aggregate mark should open the filters behind that mark.

For example, a monthly median opens its contributing filter IDs, dates, site, measurement values, exclusion flags and reasons. A seasonal box opens that season's records. A regression opens its paired, finite observations, including the effective fields after baseline/unit choices. Reuse the existing drawer when a table row is selected, and offer CSV download of the contributing rows.

The drill-down should use the same row selector as the chart, not independently reconstruct its cohort. For calibration charts, show the relevant calibration/test population and provenance rather than applying a gallery filter to a different scientific population.

Source: [DrillDetailModal](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/components/Chart/DrillDetail/DrillDetailModal.tsx).

### 3. A stronger chart registration contract

**Value: high. Effort: medium. Enables several performance improvements.**

Superset separates chart metadata, component loading, controls, query construction and data-to-props transformation. The gallery already has `DashboardChart`; evolve it rather than building a new plugin framework.

Useful additions are declared data dependencies, a lazy component loader, searchable tags, a small preview, supported interactions, and a method for obtaining contributing rows or an export table. Dependencies should distinguish calibration-grid data from calibration-run data and optional weather files.

This lets the dashboard know what to load before rendering a tile. It also lets the catalog offer a calibration chart before its large data file has loaded: selecting it requests that dependency and shows a loading state. An `available()` check that requires already-loaded calibration data would otherwise make demand loading hide the very chart the reader needs to request.

Source: [ChartPlugin](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/packages/superset-ui-core/src/chart/models/ChartPlugin.ts), [ChartMetadata](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/packages/superset-ui-core/src/chart/models/ChartMetadata.ts).

### 4. A shared scientific metric definition

**Value: high for correctness. Effort: medium.**

Borrow the semantic-layer principle: define a measurement once and reuse its identity, label, units, method and aggregation rules throughout the app. Extend the existing metadata, labels, derived fields and `fieldEquivalents` helpers. A new SQL or semantic-layer service is unnecessary for this.

Include population, physical-filter counting rule, missing-value policy, paired-data requirement, baseline/MAC choices and provenance. Keep a documented stand-in visibly distinct from the requested measurement. In particular, ChemSpec FTIR and in-house FTIR are not independent validation measurements; optical BC and TOR EC must retain their method labels; site-local calendars must remain explicit.

This definition can drive KPI labels, filter compatibility, tooltips, drill-down and export manifests. Do not introduce a general custom-SQL editor for metrics in the public gallery.

### 5. Tile-level loading, failure and refresh states

**Value: high. Effort: small–medium.**

Superset wraps chart rendering in an error boundary and separates loading, failed and rendered chart states. The gallery's dashboard handles unknown chart kinds, missing fields and empty results, but has no tile-level React error boundary around chart rendering.

Give each tile a common boundary with its name, a readable error, a retry/reset action and an optional diagnostic reference. Track data-load failures separately: a React boundary does not catch rejected asynchronous requests. `useGalleryData` catches optional calibration errors as `null`, so the dashboard can currently lose a calibration option without exposing the load failure there.

For live refreshes, preserve the last successful drawing, mark it as refreshing, and show its last successful source timestamp. Do not imply that a frozen calibration export is live because the SPARTAN source refreshed.

Sources: [Chart lifecycle and error boundary](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/components/Chart/Chart.tsx), [refresh state](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/dashboard/contexts/AutoRefreshContext.tsx).

### 6. Reproducible dashboard exports and checkpoints

**Value: high for research sharing. Effort: medium.**

Add “Download this sheet” with its resolved dashboard definition, contributing data tables and a manifest. The manifest should record source/export timestamps or hashes, code revision, site/calendar choices, exclusions, units, baseline/MAC, regression estimator and applicable uncertainty settings. A workbook can come later; CSV tables plus JSON are enough to start.

The current shared URL records view choices, not the underlying dataset bytes. A reference to starter sheets or presets can also change meaning after a code update. Provide two clearly labeled outputs: a view link that opens against available data, and a snapshot bundle that includes the permitted data needed to reproduce it. A hash identifies a snapshot but cannot recreate data unless those bytes remain available.

Keep the existing transport codec version; add a versioned dashboard specification with migrations. Resolve implicit starter tiles when freezing a snapshot. Build on existing undo/redo for session edits and offer named checkpoints or JSON import/export for durable milestones. Superset's server version-history machinery is more infrastructure than this static app needs.

Sources: [dashboard data exports](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/docs/docs/using-superset/exporting-dashboard-data.mdx), [version history](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/features/versionHistory/DashboardVersionHistory.tsx), [layout undo](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/dashboard/reducers/undoableDashboardLayout.ts).

### 7. Better chart discovery and explanatory tiles

**Value: medium. Effort: small–medium.**

The Add chart dialog currently groups text cards. Add search by name, scientific question and tags; small static previews; and a compatibility explanation when a chart needs missing data. Reuse the chart index and existing starter sheets as the basis for question-specific templates: site overview, method comparison, seasonal coverage, or meteorology.

Allow a note tile for a hypothesis, cohort explanation, source citation or caveat. Start with plain text or safely rendered Markdown and reuse the gallery's existing explanatory/presentation components. Event annotations should reference canonical flow-fix dates and season definitions rather than encourage repeated hardcoded dates.

Sources: [chart metadata](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/packages/superset-ui-core/src/chart/models/ChartMetadata.ts), [Markdown dashboard component](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/dashboard/components/gridComponents/Markdown/Markdown.tsx).

## Performance findings and proposed changes

### Confirmed payload opportunity: calibration loads on every dashboard sheet

`App.tsx` calls `useGalleryData(['calibration', 'baseline', 'dashboard'].includes(tab))`. Once core data is ready, that hook loads both calibration exports whenever the dashboard is open, regardless of the active sheet's chart kinds.

| File | Raw bytes | MiB |
| --- | ---: | ---: |
| `calibration.json` | 8,596,517 | 8.20 |
| `calibration_runs.json` | 2,645,648 | 2.52 |
| **Calibration total** | **11,242,165** | **10.72** |
| Core `filters.json` + `meta.json` + `census.json` + `pmf.json` | 857,270 | 0.82 |

These are local serialized file sizes, **not measured production transfer sizes or load times**. A local gzip calculation produced about 1.85 MiB for the calibration pair; actual hosting compression was not checked. The calibration data loads after the initial filter charts, so it does not gate their first render. It still adds network, JSON parsing, memory and subsequent render work when unused.

**Change first:** load dependencies declared by the active sheet, or explicitly requested through the catalog. The calibration tiles use runs and a lambda value; inspect whether a small metadata export can supply the lambda without loading the full calibration grid. Reuse `loadJson`/`loadDerived` and their concurrent-request deduplication.

Both calibration loads also perform `JSON.stringify` → whole-string name replacement → `JSON.parse` to normalize “Pasadena” to “JPL.” Normalize that identity in the exporter or during a targeted transformation rather than copying the entire serialized tree. No runtime speedup is claimed until measured.

**Acceptance:** a fresh non-calibration dashboard requests neither calibration file; adding a calibration tile loads its required files once; failures remain visible and retryable; the catalog still offers the tile before its dependency loads.

### Other optimizations, in order

| Priority | Source observation | Proposed adaptation | How to verify |
| --- | --- | --- | --- |
| 1 | `DashboardPage` updates drag previews on every `pointermove`; most chart kinds are directly imported by the registry. | Update at most once per animation frame, skip unchanged grid coordinates and isolate chart contents from layout/chrome changes. Stabilize data/callback props before adding memoization. | Profile dragging a dense sheet; compare script/style/paint and unchanged chart render counts. |
| 2 | Every `TileView` observes subtree mutations and performs child geometry reads in a layout effect after renders, with bounded correction retries. | Make fit invalidation explicit, observe the relevant size/chrome, and batch measurements. Exclude tooltip/focus mutations that cannot alter layout. Preserve async-loading and legend behavior. | Compare fit passes and forced-layout time while resizing, hovering, opening menus and loading weather. This is a profiling candidate, not a measured bottleneck. |
| 3 | Near-viewport mounting is mobile-only and remains enabled after first approach. Desktop overflow and focus rails render their tiles. | Extend visibility-aware rendering to overflowing desktop grids and rails. Retain state outside any unmounted visualization. Keep focused, dragged and exported tiles active. | Confirm offscreen charts do less work; scroll back without lost settings, brush or pin state. |
| 4 | Map code is lazy, but the main registry statically imports many chart families and adapters. | Extend lazy loading to heavy chart families or pages using the existing map pattern. Preserve the public-build section exclusions. | Compare production initial chunks and navigation behavior; do not assume smaller source imports imply a smaller bundle. |
| 5 | Core startup awaits filters, metadata, census and optional PMF together. | Consider loading census only for its page and PMF for requesting charts; retain a useful loading state when adding those charts. | Compare time to a usable ordinary chart under slow optional-file responses. The byte savings here are much smaller than calibration. |
| 6 | Similar tiles independently build weather groups, distributions or regression inputs. Field-presence scans and loaded files already have caches. | Share expensive derived results only where profiling shows repeated work. Key them by row/data revision and relevant scientific settings. | Identical input/settings reuse results; changing exclusions, estimator, MAC or season convention invalidates the appropriate result. |
| 7 | Some charts can contain dense SVG layers; the project already has specialized point-rendering rules. | Continue using `ScatterLayer`, selective hover subscriptions and overlay focus. Consider a Canvas mark layer only if paint/DOM cost remains dominant. | Profile the built dense chart; retain sample hit testing, linked focus and publication-quality exports. |
| 8 | Data and render lifecycles are distributed across hooks and chart components. | Add local timing marks for dependency load, parse/transform, chart-ready and layout settlement. Expose a development-only diagnostic view. | Compare repeatable cold/warm runs of the same sheet, filters and dataset on desktop and narrow screens. |

Superset distinguishes **render virtualization** from **deferred data fetching**. Its default-branch row virtualization uses near/far thresholds, but hiding a visualization alone does not stop its chart-data request. The gallery must gate auxiliary file loads as well as rendering to gain both benefits. Export mode must deliberately render and settle every requested tile before capture, or exports can contain placeholders.

Superset also does **not** provide a frontend concurrent-chart-request limiter in the reviewed performance guide. A bounded queue for live site downloads could be a separate gallery experiment if traces justify it; it is not a feature to claim as copied from Superset.

Sources: [dashboard performance guide](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/docs/admin_docs/configuration/dashboard-performance.mdx), [row virtualization](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/dashboard/components/gridComponents/Row/Row.tsx), [query deferral](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/components/Chart/Chart.tsx), [render lifecycle](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/components/Chart/ChartRenderer.tsx), [logging events](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/superset-frontend/src/logger/LogUtils.ts).

## What to defer

- **A full Superset migration:** reconsider if the real requirement becomes many SQL data sources, governed multi-user authoring, shared server-side assets or database-enforced access policies. Those requirements have not been established here. Moving now would also require porting scientific chart behavior, uncertainty semantics and sample-level interactions.
- **Redis/Celery and SQL Lab:** useful in Superset's query-serving architecture, but unnecessary for serving this gallery's static exports. The base filter export has 942 rows and 28 listed measurement fields; there is no evidence here that a warehouse or a Web Worker is the first needed optimization.
- **Scheduled reports and alerts:** valuable later for data-ingestion failures, coverage gaps or new-source updates. They require a scheduled runner/service; a closed browser cannot run a reliable schedule. Begin with a defined recipient, trigger, source-refresh process and evidence-preserving report. Nothing was scheduled or sent in this review.
- **Embedding Superset into the gallery:** the gallery already has embed mode. Superset's SDK would require a separate Superset deployment and, for guest-token use, a backend token issuer. It does not turn Superset into a standalone client-side chart package.
- **Client-side roles as data protection:** keep the existing public-build exclusion pipeline. A UI switch cannot protect data that was shipped with a public static build. Private, shared dashboards would need authenticated delivery and server-side authorization.
- **Importing the entire Superset UI:** adopt interfaces and interactions selectively. If actual Apache-licensed code is copied, retain the applicable license and attribution notices. An independent implementation of the pattern can stay much smaller.

References: [caching architecture](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/docs/admin_docs/configuration/cache.mdx), [alerts/reports requirements](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/docs/admin_docs/configuration/alerts-reports.mdx), [embedding](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/docs/docs/using-superset/embedding.mdx), [license](https://github.com/apache/superset/blob/1e06f5e83b31998ff73a146de9865cff1b6e57d6/LICENSE.txt).

## Suggested build sequence

1. **Loading and resilience:** declare tile dependencies, defer unused calibration files, expose load errors, add tile boundaries and baseline timing marks. Keep the current cache and optional-data behavior.
2. **Interaction performance:** profile the production build; address drag rerenders and fit invalidation where traces demonstrate cost; extend code/viewport laziness with export and keyboard checks.
3. **Analytical transparency:** show effective filters/source/method/count per tile, then add contributing-row drill-down and one explicit scoped/reference-tile workflow.
4. **Reproducible sharing:** add a versioned dashboard definition, named checkpoints and a data/definition export with a manifest. Add catalog search, previews and note tiles as smaller usability improvements.

Relative effort labels above are implementation estimates, not calendar commitments. Stage 1 has the clearest confirmed payload reduction. Stages 2–4 should be validated against actual research workflows rather than feature parity with Superset.

## Validation performed

Source checks covered the gallery's dashboard/layout/registry, loaders, URL views and history, linked highlighting, brush actions, chart frame/export behavior, live-source refresh and build configuration. Upstream checks covered dashboard layout, filters/scope/indicators, chart plugins, lifecycle/error handling, virtualization, drill-down, version history, data exports, cache, embedding and report requirements.

Executed in `gallery/app` with Node 24:

```sh
node --test src/lib/dashboard.test.mjs src/lib/undoHistory.test.mjs src/lib/fieldEquivalents.test.mjs src/lib/pointLayers.test.mjs
```

Result: **24 passed, 0 failed**. These check existing layout/state round trips, undo grouping, documented substitutions and shared-rendering rules. They do not establish runtime smoothness or validate proposed features. No production browser profile, full app build, deployed-host transfer measurement or running Superset test suite was performed.
