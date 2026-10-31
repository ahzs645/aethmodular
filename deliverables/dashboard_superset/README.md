# Dashboard implementation and verification

The Superset-inspired improvements are implemented in the Aethmodular gallery. The implementation was verified against baseline `bf706c294b76bba84bfd9ae0035f8aa9bf3822ca`. It is also available in the [published private review Site](https://aeth-dashboard-review.ahzs6455.chatgpt.site); reproducible Site presentation sources are in [gallery/site](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/gallery/site/README.md).

[Open the screenshot gallery](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/index.html). It contains **63 screenshots**, including every one of the **40 registered chart kinds**, desktop/mobile workflows, deliberate failure states and snapshot replay.

> The screenshots, their index page, the review zip and the review Site's sources were not merged into `main`
> (generated PNGs, about 15 MB, and an external Site); the links above and below open them at the PR's commit.

## Implemented

| Review recommendation | Result |
| --- | --- |
| Scoped filtering | Independent site/season/date narrowing per tile; retained physical-ID reference cohorts; clear reset to page filters; URL persistence and duplication support. Reference membership survives page filtering, while source values/methods/units remain explicit. |
| Contributing-sample drill-down | Exact KPI contributors, calendar-month markers with keyboard activation, existing distribution-group contributors and scatter fit observations; paginated/searchable tables, sample drawer and spreadsheet-safe CSV. Generic View data is explicitly labeled as finite inputs rather than pretending to reproduce every chart's later processing. |
| Filter/method/source indicators | Tile strip shows cohort scope and physical-filter count. The information dialog explains finite measurement counts, sites, dates, calendars, methods, MAC/baseline/source mode and available live-source update/retrieval dates. |
| Stronger registration contract | Lazy chart families; declared calibration dependencies and auxiliary-file resources; searchable metadata; schematic previews and explanations for unavailable optional PMF data. |
| Shared scientific definitions | Common method identity and finite/paired physical-filter selectors for information, tables and CSV. FTIR duplicate-product and optical-versus-TOR distinctions remain explicit. Existing scientific fit calculations are retained. |
| Loading and failure handling | Ordinary sheets load no calibration files. Adding a calibration tile requests runs only, with an explicit error/retry flow. React render failures stay inside a tile. Missing local network data offers explicit recovery to bundled focus sites. |
| Reproducible sharing | Existing saved views now support named checkpoints. Versioned, checksummed snapshots contain the resolved active sheet, effective rows/fields, metadata, source settings, source-code fingerprint and available auxiliary files. Replays are isolated from the current-source cache; corrupt files and unsupported versions are rejected. |
| Discovery and explanation | Search by chart name/group/description, schematic previews, plain-text research notes retained in the view, and Add chart access on mobile. |
| Interaction/render performance | Drag previews coalesce to animation frames and skip unchanged grid cells; chart content and cohort inputs are memoized; fit measurements have explicit dependencies; viewport deferral also covers desktop overflow and rails. Loaded charts stay mounted to preserve their state. |

## Test results

- **53 Playwright browser tests passed**, covering the workflows and all 40 registered chart kinds. The test suite runs against the production build with Chromium at desktop and mobile widths.
- **33 focused tests passed** for dashboard layout/state, cohort filtering, finite-row selection, CSV escaping, identity normalization, snapshot validation, undo, field substitutions, shared point layers and floating layers.
- **Production TypeScript/Vite build passed.** Vite still reports large chunks; the optimization reduces their size rather than claiming the bundle is fully minimized.
- **Public-build isolation check passed:** private research paths stay out of the public bundle and publish archive; the existing unlisted instruments policy remains intact.
- **Broad unit suite:** 249 passed, 10 skipped, 7 failed because research export fixtures are absent. The same seven failures were reproduced against an archived copy of the untouched original source. They are not hidden or relabeled as passing. Details are in [full-unit-tests.txt](full-unit-tests.txt) and [original-fixture-failures.txt](original-fixture-failures.txt).

The browser checks verify rendering and interaction, not the scientific validity of every upstream dataset. The unchanged default HIPS/FTIR crossplot was checked at n=544 with Deming slope 1.258; new contributors are selected from the inputs actually used by their corresponding aggregation or fit.

## Measured production-load comparison

Three interleaved cold-context Chromium runs per build used the same real bundled focus-site dashboard, 1440×1000 viewport, disabled browser network cache, and 2.5 seconds of settling after a chart appeared. The original build came from a separate archive of the original Git revision. Raw evidence is in [performance.json](performance.json).

| Measure | Original | Updated |
| --- | ---: | ---: |
| Calibration JSON requested by an ordinary sheet | 11,242,165 bytes | 0 bytes |
| Main JavaScript file (Vite output) | 2,611.72 kB | 1,616.86 kB |
| Main JavaScript gzip estimate (Vite) | 863.17 kB | 536.81 kB |
| Total decoded resource bytes, median | 15,277,433 | 3,049,144 |
| Scripting time, median | 1.841 s | 1.074 s |
| Style recalculation, median | 0.818 s | 0.315 s |
| Layout time, median | 0.173 s | 0.174 s |
| JS heap sampled at settlement, median | 72.1 MiB | 34.8 MiB |

Decoded bytes are not compressed wire transfer. These are local measurements on one machine, with possible background activity and GC variation, not a production latency or frame-rate guarantee. Layout time was essentially unchanged. The stable improvements are avoiding unused calibration payloads and reducing initial code; heap and timing samples describe this test only.

## Trying the features

1. Open Dashboard. Use **Edit → + Add** to search charts or add a research note; on mobile use the dashboard overflow menu.
2. Open a tile's **⋯ → Filters for this tile**. Narrow it, retain its current cohort, or reconnect it with **Use page filters**. Use the information strip to inspect scope/method/source.
3. Click a headline number or monthly aggregate, or use the distribution-group/scatter-fit action, to inspect exact contributors. View data exposes the general finite inputs; download CSV or open an individual record.
4. Open **Aa → Save this sheet** for checkpoints or snapshot download/import. Import displays a saved-data banner; **Return to current data** ends replay.

Snapshots need the app/build assets, retain available data rather than magically recovering missing exports, and have a 25 MiB import/export limit. Imported files are not stored in view links or retained across page reloads. Reference cohorts retain membership, not historical values. The snapshot checksum detects accidental changes rather than proving authorship. The original repository's public build rules continue to control what is published.

The initial default SPARTAN network index is absent from this checkout, so the source-error screenshot shows recovery into the real 942-row bundled focus population (938 after its existing exclusions). No generated data files were hand-edited to disguise missing sources. Tests deliberately inject 503 and chart-module failures; those screenshots are test evidence, not unreported spontaneous failures.

SQL Lab, Redis/Celery, server-side authentication and scheduled delivery remain deferred as in the original review. They require requirements/infrastructure beyond this static-dashboard work. No alerts or messages were scheduled or sent.

## Reproducing checks

```sh
cd gallery/app
npm ci
npm run build
npm run test:dashboard
node --test src/lib/dashboard.test.mjs src/lib/dashboardCohort.test.mjs src/lib/undoHistory.test.mjs src/lib/fieldEquivalents.test.mjs src/lib/pointLayers.test.mjs src/lib/floating.test.mjs
python ../data/rebuild.py check-public
```

Use `CHROMIUM_PATH` or install Chromium with `npx playwright install chromium` where needed. `npm test` also runs the existing research-fixture checks and will report the missing exports described above until those source artifacts are restored.

## Screenshot index

- [Original dashboard](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/00-before-desktop.png)
- [Updated desktop dashboard](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/01-after-desktop.png)
- [Source failure and bundled-data recovery](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/02-source-error.png)
- [Independent tile and reference-cohort controls](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/03-scoped-filter-controls.png)
- [Reference cohort beside changing page filters](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/04-reference-versus-filtered.png)
- [Exact contributors, searchable table and CSV](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/05-contributing-filter-table.png)
- [Opening a physical sample from the table](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/06-sample-drawer.png)
- [Calendar-month contributors](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/07-monthly-drilldown.png)
- [Searchable chart catalog with schematic previews](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/08-chart-catalog.png)
- [Persisted research notes](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/09-note-tile.png)
- [Injected calibration request failure](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/10-calibration-load-error.png)
- [Successful calibration retry](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/11-calibration-loaded.png)
- [Injected chart-render failure isolated from its sibling](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/12-chart-error-boundary.png)
- [Snapshot export and named checkpoints](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/13-snapshot-checkpoints.png)
- [Replaying saved populations](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/14-snapshot-replay.png)
- [Mobile dashboard after scrolling each lazy tile into view](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/15-mobile-dashboard.png)
- [Mobile contributing-row inspection](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/16-mobile-data-table.png)
- [Presentation mode](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/17-presentation.png)
- [Figure export preview](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/18-figure-export.png)
- [Effective filters, methods and source provenance](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/19-methods-and-provenance.png)
- [Availability](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-availability.png)
- [Barplot](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-barplot.png)
- [Calibration crossplot](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-calcross.png)
- [Cross-validation curve](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-calcv.png)
- [Seasonal clock](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-clock.png)
- [Composition](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-composition.png)
- [Connected scatter](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-connected.png)
- [Correlogram](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-correlogram.png)
- [Correlation ranking](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-corrrank.png)
- [Box / violin](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-distribution.png)
- [Functional groups](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-fgroups.png)
- [Density hexbin](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-hexbin.png)
- [Histogram](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-histogram.png)
- [Site map](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-map.png)
- [Monthly band](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-monthlyband.png)
- [Monthly heatmap](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-monthlyheatmap.png)
- [Research note](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-note.png)
- [Parallel coordinates](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-parallel.png)
- [Site radar](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-radar.png)
- [Ridgeline](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-ridgeline.png)
- [Method flow](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-sankey.png)
- [Scatterplot](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-scatter.png)
- [Site cards](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-sitecards.png)
- [Sites ranked](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-siterank.png)
- [Site summary](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-sitesummary.png)
- [PMF sources by season](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-sourceseason.png)
- [PMF sources over time](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-sourcestack.png)
- [SPARTAN composition map](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-spartancomposition.png)
- [Species graph](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-species.png)
- [Headline number](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-stat.png)
- [Time series](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-timeseries.png)
- [Monthly climate](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxclimate.png)
- [Weather correlations](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxcorr.png)
- [Polar plot](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxpolar.png)
- [Pollution rose](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxpollrose.png)
- [Wind roses](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxroses.png)
- [By wind speed](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxspeed.png)
- [Weather by site](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxsummary.png)
- [Weather timeline](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxtimeline.png)
- [Rain washout](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/chart-wxwashout.png)
- [Calibration snapshot replay](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/snapshot-calcross.png)
- [Map snapshot replay](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/snapshot-map.png)
- [Weather snapshot replay](https://github.com/ahzs645/aethmodular/blob/d54daf69b31bec0e2693759ee7e0f2000479f230/deliverables/dashboard_superset/screenshots/snapshot-wxtimeline.png)
