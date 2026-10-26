# Gallery review — 23 October 2026

Implemented the items in the supplied distribution/gallery review notes.

| Area | Result |
| --- | --- |
| Distribution | Season names and month ranges wrap above one shared site label. Custom grouping cutoffs can pool sites or compare sites within each band. Dense comparisons scroll within the chart. |
| Empty charts | Measurement controls remain available without axes. Unavailable new fields/groupings show their coverage and cannot be selected while alternatives have data. Existing choices remain editable; an entirely empty subset permits measurement changes. |
| Histogram | Density fills close at zero. Bins have synchronized slider and numeric input. Density values refresh when observations change without changing the group count. |
| Clock | Measurement and units are visible; measurement picker and season overlay are available. Mixed calendars shade each site's own sectors. Ring labels render above the bars. |
| Heatmap | Measurement picker; chronological year-months or Jan–Dec pooled across years; readable month labels and contained scrolling. |
| Composition | Wrapped season labels; separate season panels for donuts and other hierarchy encodings. Family/species legends share the parent/child presentation used for sites/seasons. |
| Legends | Site color and site toggle appear once beside the site's seasons. Hiding a parent crosses off its children. |
| Map | Long network provenance moved into Explain. Map site ordering remains fixed while toggling. |
| Filters sidebar | Ordering stays fixed while open; selected sites move to the top on reopening. |
| Data availability | Sites menu section: notebook-based network product matrix, monthly HIPS/public coverage, cadence/gap statistics, four-site sample-day and physical-filter overlaps, completeness thresholds, historical cross-network comparison, and the filter explorer with editable recency threshold (default two years). See [source and counting notes](gallery-data-availability.md). No recent exported records are not treated as proof of site closure. |
| PMF | Data marks clipped to the plot bounds, including the first/last daily bars. |
| Calibration | Specification rule text appears once as wrapping HTML. Slope/intercept uses the shared guardrail templates and editor. Shared query settings update mounted views immediately. Configuration controls and their diagnostics are grouped, and the CV curve names its configuration. |
| Experimental Addis filters | Shared crossplot with record selection, hand edits, live regression, clickable axes, explicit Custom state, model preset reset, and rectangular brush review/keep/exclude actions. |
| Frame controls | Focus icon is a sized SVG button. Controls wrap on narrow screens. |

## Verification

Chromium layout sweep: distribution, correlation, ranking, evolution, composition, PMF, map, availability, and calibration at **320, 360, 375, 390, 414, 480, 640, 768, 820, 1024, 1280, 1440, 1920 and 2560 px**. The expanded sweep includes the colour/scatter/every-field correlograms, all availability views, 80% completeness, and both network filter inventories. All 224 cases passed page-overflow, control-overflow, legend-position and invalid SVG-coordinate checks, with no browser exceptions. Dense chart regions intentionally retain horizontal scrolling.

Repeat this check with a running server:

```sh
cd gallery/app
python scripts/check-responsive.py --base-url http://localhost:5178
```

Interaction checks cover synchronized bins, density path closure, month pooling, missing-data recovery, grouping modes, parent legends, focus mode, map/sidebar ordering, and the shared guardrail editor. Seasonal donuts and the experimental page also receive the 14-width sweep. Availability interaction checks also cover product/year/search/presence controls, exact overlap and completeness states, copied URL restoration, both network inventories, the decoded IMPROVE monthly history (2,591 Acadia samples), historical comparison, and the provenance download. React review includes hooks, accessibility labels, keyboard handling, listener cleanup, and preserving chart/tile settings.

The production build, the prior 30 focused checks, and all 15 availability/native-loading checks pass. The broader suite reports **123 passed / 8 failed**. These eight checks depend on unavailable or incompatible scientific exports: one baseline-comparison prediction cohort, three specification exports lacking the saved-guardrail inputs, and four rank-70 migration checks requiring absent research output files. They do not indicate a successful scientific-data validation.

The checkout lacks the experimental analogs export. The SPARTAN filter/native export has now been restored from the existing published gallery: 4,350 content-addressed files were verified, including 4,270 spectrum assets. Experimental interaction checks use an explicitly mocked UI fixture; they do not validate model predictions. Available real focus-site data and adjacent static-gallery exports support the other browser checks. Data availability can fall back to the existing SPARTAN inventory/reference export. The availability export packages existing saved tables and recomputes study-day coverage from canonical inputs; no scientific inputs or notebook files were changed.

## Mobile filter popup follow-up

ETAD-0075 and ETAD-0090 now load their native raw, AIRSpec and VIBES traces.
Missing data URLs returning HTML are rejected as unavailable exports before
JSON/gzip parsing. Measurement tables, ratios, native HIPS readings, neighbouring
filters and plots have independent horizontal scroll regions; the drawer has
no horizontal overflow at the 14 tested widths.

Touch dropdowns and axis pickers no longer autofocus search. Text controls use
a minimum 16px font to avoid iOS focus zoom, with pinch zoom left enabled. Menu
placement follows the visual viewport when the keyboard opens. Chromium touch
checks cover deliberate search, keyboard-sized viewport bounds and preserved
desktop mouse/keyboard behavior; these are browser simulations, not a physical
iPhone run.

## Consistent fit summaries — 25 October 2026

Scatterplot and hexbin previously used separate mobile rules: the scatterplot
reserved SVG space above its axes, while hexbin placed a wrapping HTML summary
below them. The shared `useRegressionSummary` now reserves space above the axes
at every width. It shows n / R² before one equation per line, wraps long group
names and confidence intervals at 11px, and keeps the summary in SVG exports.
Grouped fits stack on phones and use up to two columns on wide panels. Shared
research crossplots, station comparisons and the filter drawer use the same
format. Summary boxes reflow immediately with their text during resizing.

Single scatterplot and hexbin panels now fit the phone card instead of retaining
a minimum width larger than the card. Dense multi-panel rows can still scroll.
The displayed default fit remains n = 544, R² = 0.832 and Deming y = 1.258x − 0.590.
Regression calculations and observations are unchanged. The 23 regression,
confidence-interval, grouping and axis checks pass.

Chromium geometry checks cover pooled, per-site and per-season fits plus per-site
and per-season facets at all 14 widths listed above. All three estimators with
confidence intervals pass at 320, 390, 768 and 1280px. Checks also cover hidden
fits, the focused mobile chart, the filter drawer, six station panels, and SVG
exports retaining the unchanged default coefficients. No page exceptions were
observed in the completed interaction checks. The production build passes.

## Mobile Ref menu and connected scatter

The Ref menu is portalled to the document body and positioned against the visual
viewport, so a scrolling chart card cannot clip it. It supports arrow-key
navigation, Escape with focus restoration, outside clicks and scrolling when
vertical space is limited. Grouping picker actions (New, Edit and Build on this)
share the picker's font, line height and vertical padding.

Connected scatter starts with one path on phone widths (640px or less), with
Compare all available on the path selector. Its axes fit that path so low-loading
sites remain legible. Straight segments connect the existing monthly/yearly
medians in chronological order. Month/year selection, previous/next buttons and
tappable/keyboard-accessible points show the selected medians and paired-filter
count. The guidance is visible without enabling Explain. Path, step, labels and
selected point survive copied URLs. The paired-filter calculation and minimum
of four pairs per step are unchanged.

Verification covers Ref menus on scatter, hexbin and connected scatter at all
14 widths (42 menu cases), keyboard navigation/focus restoration, a 280px-high
landscape viewport and outside-tap dismissal. Mobile inspection checks cover
previous/next, month/year selection, point taps/keyboard activation, low-loading
JPL axes, paired-data medians, copied URL restoration and SVG export. New and
Build on this match the picker at three text sizes. The production build and
nine axis/grouping checks pass. Browser checks use Chromium touch simulation.

## Stationary bottom legends and mobile field pickers

Bottom HTML legends use the visible chart width and stay horizontally pinned
while wide plots pan. Colour keys wrap within that width and their SVG ramps
scale without clipping. Dashboard legends placed on the right retain their
column layout. A real touch swipe on the correlogram leaves the Pearson-r key
in place; palette selection and SVG export still work after panning.

The Fields multi-select now shares the floating-menu positioner with the other
pickers. Touch opens it without focusing search; deliberate search remains
available with a 16px minimum input font. It has a Done button, bounded scrolling,
and correct field-button labels. Axis and palette popovers also follow their
live anchors and visual viewport. Keyboard/page scroll no longer dismisses an
active field chooser. Popup height is capped to the visible viewport even when
its anchor moves behind an open keyboard.

Interaction checks cover multi-field presets, individual checkboxes, search,
clear/recovery, copied URL restoration, resized keyboard viewports, axis fields,
sidebar x/y fields, scatter size/colour, distribution/evolution measurements,
focused cards and filter-drawer axes. The Fields and axis popups fit all 14
tested widths. All ten field-view/axis unit checks and the production build pass.
These remain Chromium touch simulations rather than a physical iPhone run.

## Integration with main, 26 October 2026

The gallery changes, availability exports and VIBES plan were integrated on top
of upstream theme/materials commit `1befb62`. The production build passes, as do
all 224 layout checks at the 14 widths from 320 to 2560px. A mobile dark-theme
check also passes for regression summaries, the Fields picker, connected scatter
and seasonal labels, with no browser exceptions. The native-export Python tests
pass (5/5). The full JavaScript suite reports 123 passes and the same eight known
failures involving missing or incompatible scientific export fixtures.

## Public gallery publication, 26 October 2026

The public gallery at https://projects.ahmadjalil.com/aethmodular-gallery/ now
serves the updated application and all 4,750 data assets. The application build
identifies source commit `366a571` in `build-info.json`; later source commits
change publication helpers and documentation only. Gallery repository commit
`2f4b735` deployed successfully in
[Actions run 40391159632](https://github.com/ahzs645/aethmodular-gallery/actions/runs/40391159632).

Release uploads returned HTTP 401 and changing Pages settings returned HTTP
403 for the connected integration. Publication therefore uses the prebuilt
Actions workflow while preserving the existing Pages workflow mode. The source
workflow remains available. The publisher includes linked data directories,
records build provenance, and accepts generic build-service logins in browser
identifiers while retaining personal-path checks. Its five publication tests
pass. The unused draft from the failed release attempt was removed.

All 224 layout cases were repeated against the public URL at the 14 widths
from 320 to 2560px: no layout failures or browser exceptions. Live touch checks
also pass for Fields, the floating Ref menu, connected-scatter inspection,
ETAD-0090 spectrum loading and local drawer scrolling. HTTP checks verify the
availability export, corrected site labels, native index and content hashes of
the ETAD-0075 and ETAD-0090 spectrum assets. These browser checks use Chromium
touch simulation; physical iPhone Safari remains a separate device check.
