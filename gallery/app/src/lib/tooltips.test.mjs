import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

/**
 * docs/gallery-chart-standards.md "Tooltip": a mark shows the gallery tooltip (hooks/useTooltip, bound with
 * `tip.bind(body)` or ScatterLayer's `tipOf`) and not the browser's native `<title>` as well; with both, two
 * cards open on one hover, the native one late and unstyled. A file that has the gallery tooltip and a
 * `<title>` fails here, unless it is in ALLOWED with the reason, which is usually that the `<title>` only
 * restores the full text of a label clipped to fit (a row name, an axis name), and that label has no tooltip.
 */
const src = new URL('..', import.meta.url).pathname
const files = []
const walk = (dir) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p)
    else if (/\.tsx$/.test(e.name)) files.push(p)
  }
}
walk(src)
const code = (p) => fs.readFileSync(p, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1')
const rel = (p) => path.relative(src, p)
// the gallery tooltip: the hook itself, or a tip handed down from it (tip.show / tip.bind / showTip)
const usesTip = (s) => /\buseTooltip\(|\btip\.(show|bind)\(|\bshowTip\(/.test(s)

test('a mark with the gallery tooltip has no native <title> as well', () => {
  const ALLOWED = {
    'hooks/useTooltip.tsx': 'defines it',
    'charts/ranking/CorrelationRanking.tsx': 'the full name of a field or site label clipped to the row gutter or column; the labels have no tooltip',
    'charts/ranking/ParallelCoordinates.tsx': 'the full name of an axis title clipped to its column; the title has no tooltip',
    'charts/correlation/Correlogram.tsx': 'the full field name of the wrapped row, column and diagonal headers; the headers have no tooltip',
    'charts/calibration/SpecCurvePlots.tsx': 'the full panel and selection names clipped to fit, and the help text on the choice-strip rows; none of those labels has a tooltip',
    'charts/partOfWhole/CompositionTreemap.tsx': 'the full season name behind the shortened strip label, and the "of PM2.5" text a narrow bar drops; neither mark has a tooltip',
    'components/IntervalStrip.tsx': 'the full row and block names behind labels clipped to a narrow card\'s gutter; a row that has the gallery tooltip gets no <title>',
    'charts/weather/SiteWeatherSummary.tsx': 'the full variable name behind an abbreviated column header; the header has no tooltip',
    'pages/SatoshiFollowupPage.tsx': 'excluded × marks drawn beside the points, not through ScatterLayer: the native title is their only readout (no per-mark handlers on filter marks)',
    'pages/experimental/FlagStudy.tsx': 'excluded × marks drawn beside the points, not through ScatterLayer: the native title is their only readout (no per-mark handlers on filter marks)',
    'pages/experimental/TorStudy.tsx': 'excluded × marks drawn beside the points, not through ScatterLayer: the native title is their only readout (no per-mark handlers on filter marks)',
  }
  const offenders = files.filter((p) => !(rel(p) in ALLOWED)).filter((p) => { const s = code(p); return usesTip(s) && /<title[\s>]/.test(s) }).map(rel)
  assert.deepEqual(offenders, [],
    'drop the native <title> from a mark that has the gallery tooltip (put its text in tip.bind(...) instead); a <title> that only shows clipped label text goes in ALLOWED with the reason')
})
