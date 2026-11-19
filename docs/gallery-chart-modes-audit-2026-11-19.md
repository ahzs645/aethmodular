# Gallery chart modes: network, site, axes, grouping (19 Nov 2026)

The question (18 Nov): when a reader changes a mode (the network, the active site, the x and y
quantities, a grouping such as "Iron · median split"), does every chart follow it the same way? Before
this change most charts kept their own copy of each mode, so one page could show Addis in one chart
and Delhi in the next, and a chart's Network switch did not move the gallery's network.

Paths are relative to `gallery/app/src`.

## The rule

1. **Site.** A chart that draws one site reads the gallery's page site (`psite`, `lib/siteFocus`
   `usePageSite` + `SitePicker`). Its default is the first site picked in Samples
   (`lib/gallerySamples` `useFirstSampleSite()`), then the chart's own choice. The chart's old key goes
   in `legacy`, so old links still open the same site. A choice that means something else (an
   instrument, the availability explorer's IMPROVE-or-SPARTAN inventory) keeps its own `key`, as
   `usePageSite` already allows.
2. **Network.** A chart with a network split reads the gallery's network (`net`), through
   `lib/gallerySamples` `useGalleryNetwork()` on pages without the Samples panel. A page whose story
   needs another starting point gets a Samples preset (`MEETING_PRESETS`, or a section page's
   `samples`), so a change there stays with that page and does not move the Correlation tab.
3. **Axes.** An x-y chart with sensible alternative quantities declares them with `AxisChoices` and
   `pick`, and offers `swap`. Layers that only make sense in one view (a fitted curve, a
   construction) are drawn only there, and the chart says so.
4. **Grouping.** A chart with per-filter rows offers the reader's groupings (`grp` in a link,
   `components/GroupingDialog` `useChartGrouping`) to colour its points. Rows from a bespoke export
   join a grouping's field by filter id (for SPARTAN, from `spartan_reference.json`).

## The pilot (HIPS meeting pages)

| Chart | Before | Now |
|---|---|---|
| The bend (`#hl-bend`) | Own Network (`hl.bend.network`), own SPARTAN site (`hl.bend.site`), fixed axes, no grouping | Network is the gallery's `net` (the HIPS pages open on IMPROVE + Addis through `MEETING_PRESETS`); site is `psite` with an "All sites" choice; x = EC / PM2.5 / iron on the filter (µg), y = τ / Fabs / Fabs ÷ EC, with swap; grouping colours the SPARTAN filters. Curves, construction and loading groups only in EC against τ; IMPROVE groups show wherever they have the quantity (EC with τ or Fabs ÷ EC). |
| Stretch the HIPS axis (`#hl-against-hips`) | Addis only, fixed axes | Any SPARTAN site with ≥ 20 filters with FTIR EC, HIPS Fabs and volume (`psite`). Addis keeps the closure workflow's 189 filters and λ = 2.96. Other sites are rebuilt from `spartan_reference.json` (HIPS BC = Fabs ÷ MAC 10, τ from Fabs × volume) and use Deming λ = 1, as the subtitle says; their exclusions follow Filters → excluded. y = FTIR EC or PM2.5 (the 1:1 line and shared square range only for FTIR EC); swap; grouping. |
| One site's filters on the curve (`#hw-site-curve`), Less air (`#hw-volume`) | `hw.site` (a site name) | `psite`; `hw.site` read as legacy. |

Old links: `?hl.bend.network=Both` is moved onto `net=both` on load; `hl.bend.site` and `hw.site` are
read as legacy keys of `psite`.

TOR is not in `linearise.json` or `spartan_reference.json`, so "Stretch the HIPS axis" cannot offer TOR
on y yet. An Addis TOR export joined by filter id would add it.

## What else in the gallery does not follow the rule yet

Found by search (`grep`) on 19 Nov, then checked file by file for the site and network rows. The axes
and grouping rows are candidates: a chart with a fixed pair of quantities on purpose is fine as it is.

### Site: one-site charts on their own key

| Chart | Key now | Change |
|---|---|---|
| Meeting figure specs with `choose: [{ column: 'site' … }]`: `c1110` (closure, l.164), `hs-site-seasons` and `hs-levels` (hips-seasons l.75, l.108), `hp-traced` and `hp-points` (hips-procedure l.66, l.109), `h1111-…` (hips-network l.65, l.81, l.135) | each figure's own option key | **Highest value.** One change in `components/Figure.tsx`: a `choose` column named `site` reads `psite` (matching by name). Several of these are on the HIPS parts page, so until then the page's borrowed figures keep their own site. |
| `charts/correlation/Correlogram.tsx:75`, `charts/correlation/SpeciesGraph.tsx:91` | chart option `site` (default ETAD, with All) | `psite` with `all` |
| `charts/ranking/CorrelationRanking.tsx:125` | chart option `site` by name (default Addis Ababa) | `psite`; names as legacy values |
| `sections/instruments/sources/CrossSectionSweep.tsx:24` | `aes.sigmaSite` | `psite` |
| `sections/research/pages/baselineLab/data.tsx:53` | `blab.site` (with All) | `psite` with `all` |
| `sections/research/pages/network/RawVsBaselined.tsx:64` | chart option `site` | `psite` |
| `sections/ml/pages/decision/FlagOnSpartan.tsx:170` | chart option `site` | `psite` |
| `pages/experimental/VibesTool.tsx:158` | `vt-site` | `psite` |

Kept on their own key on purpose: `sections/instruments/quality/FlowFixCompare.tsx` and `SmoothRaw.tsx`
(`aq.site`, an aethalometer instrument), `pages/DataAvailabilityPage.tsx` (`availability.site`, either
network's inventory). Already on `psite` or `useSiteMode`: Similarity, Weather, AERONET, FlagStudy.

### Network: charts with their own network choice

| Chart | Now | Change |
|---|---|---|
| `charts/ranking/NetworkRankPlacement.tsx:55` | option `networks` (comparison background, default both) | Default it from the gallery network; keep the option, since the background can differ from what Samples draws |
| `charts/map/SiteBubbleMap.tsx:136` | `Sites`: Selected / All SPARTAN / All IMPROVE | Leave: it is a map scope over the Samples pick, not a second network |
| `pages/DataAvailabilityPage.tsx` | its own Network | Leave: an inventory browser of either network |

### Axes: x-y charts with no axis picker

`ScatterPlot` with no `pick`: `pages/HipsCaveats.tsx`, `pages/HipsDecimals.tsx` (2),
`pages/BaselineExternalPilot.tsx` (3), `pages/BaselineComparisonPage.tsx`,
`sections/research/pages/baselineLab/Ethiopia.tsx`, `…/baselineLab/Experiments.tsx`, and on the HIPS
pages `CorrectionCurve`, `NetworkExponents` (`pages/HipsLinearisePage.tsx`) and `VolumeReplay`
(`pages/HipsPartsPage.tsx`). Most draw one derived quantity against another (a correction against τ,
an exponent against median τ), where an axis picker has nothing to offer; `swap` is the cheap win
there.

`Crossplot` with no `pick` (Crossplot is the 1:1 comparison of two measurements of one thing, so
`pick` applies only where several measurements of that thing exist): `charts/aeronet/AeronetNonAbsorbing.tsx`,
`charts/pmf/ResidualVsBiomass.tsx`, `sections/research/pages/OffsetShift.tsx`, `…/TwoProblem.tsx` (5),
`…/filterOnly/Predictions.tsx`, `…/charcoal/Scores.tsx`, `…/charcoal/Controls.tsx`,
`sections/instruments/filters/MethodAgreement.tsx`, `…/ConstraintCheck.tsx`,
`sections/instruments/diurnal/SurfaceCharts.tsx`, `sections/instruments/sources/PmfAgreement.tsx`.
`MethodAgreement` (aethalometer against filter BC/EC) is the clearest case for a picker: the
aethalometer channels are fields now that the section is public.

### Grouping: per-filter charts that cannot colour by a grouping

Only these offer the shared grouping: Scatterplot, ConnectedScatter, DensityHexbin, Barplot,
DistributionPanel, Timeseries (Correlation, Distribution, Ranking, Evolution tabs), and the two HIPS
charts above. Every `Crossplot` and `ScatterPlot` listed above, and the meeting `Figure` scatters, draw
per-filter points without it.

Recommended: give `Crossplot`, `ScatterPlot` and `Figure` a `grouping` prop (chart id + rows + a
filter-id join) that does what `useChartGrouping` does here: picker in the controls, band colours on
the points, the "No data" note, the band key. Then each chart opts in with one line instead of the
per-chart wiring the pilot needed.

## Also changed on 19 Nov

- `sections.json`: Instruments is `"public": true`. A public build lists the Aethalometer tab, and its
  per-filter fields (UV…IR BCc, smoothed IR, AAE, Red absorption; CHTS, ETAD, INDH, USPA) join every
  tab's pickers, including the Correlation scatterplot. Each field's description now says that the
  9 AM daily matching is unverified and that Addis ran at 55:45.
- `gallery/data/rebuild.py` `check-public`: checks every published section (listed or unlisted) ships
  its data, not only unlisted ones.

## Known

`npm test`: 620 of 621 pass. The one failure is `investigationHistoryCatalog.test.mjs` (the checked-in
investigation history is behind the 16 Nov VIBES rank/tau commit; `npm run index:investigations`
regenerates it). It fails the same way without these changes.
