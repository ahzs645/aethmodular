/**
 * The per-filter charts of char_05 and char_06, recomputed from the scores joined to the rows in view: raw feedstock
 * against char for every filter (char_05 [14] left, [18]), the dry-season collapse over time and by month
 * (char_06 [8]), and the two populations the mixture finds (char_06 [11]).
 */
import { useMemo } from 'react'
import type { ScaleTime } from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Key } from '@/components/Key'
import { timeSeasonBands } from '@/components/SeasonBands'
import { SweepPlot, histogramSeries, type SweepSeries } from '@/components/SweepPlot'
import { crossPath, markTip } from '@/lib/exclusionMode'
import { fmt, pct } from '@/lib/format'
import { ExcludedNote, FixedSetNote, useMarkReason } from '@/lib/markedRows'
import { INK } from '@/lib/theme'
import type { FilterRow } from '@/lib/types'
import { num } from '@/lib/table'
import type { ScoreSet, Scored } from '../../common/charcoal'
import { byGroup, monthlyRate } from '../../common/charcoal'
import { Bars, DataTable, Grid, line, shortSeason } from '../../common/ui'
import { provenance, rows as rowsOfTable, type SpectralExport } from '../spectralCommon'
import { NB, SET_LABEL, SET_TITLE, type Charcoal } from './data'

const rChar = (s: Scored<FilterRow>) => s.rChar
const MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export const R_CHAR = 'best r vs McCall char (r_char)', R_RAW = 'best r vs McCall raw feedstock (r_raw)', DELTA = 'delta = r_raw − r_char'

/** the page's filter-set picker (one URL key, ch.set, for every chart) */
export function SetPick({ value, onChange }: { value: ScoreSet; onChange: (s: ScoreSet) => void }) {
  return <Segmented label="Filters" value={value} options={['view', 'plus', 'notebook'] as const} onChange={onChange} optionLabel={(o) => SET_LABEL[o]} title={SET_TITLE} />
}

/** a scored filter as a crossplot point: a gallery row opens its record, a filter the gallery lacks only tips */
export function scorePoint(s: Scored<FilterRow>, x: number, y: number, color: string, xLabel: string, yLabel: string, hollow = false, extra: string[] = []): CrossPoint {
  return { x, y, color, hollow, id: s.row ? s.id : undefined, season: s.season, seasonColor: color,
    tip: [s.filterId, `${s.date} · ${shortSeason(s.season)}${s.lot ? ` · lot ${s.lot}` : ''}${s.row ? '' : ' · not a gallery row'}`, `${xLabel} ${fmt(x, 3)} · ${yLabel} ${fmt(y, 3)}`, ...extra],
    rec: s.row ? { id: s.id, source: 'Research · charcoal similarity', site: s.row.site, date: s.date, fields: [[xLabel, x], [yLabel, y], ['component', s.low ? 'low-similarity' : 'normal']] } : undefined }
}

/**
 * The one exclusion note a chart of the scores carries (lib/markedRows): under Mark, how many excluded Addis rows in
 * view it leaves out (`drawn`: its crossplots draw them as ×; `of` narrows them to what the chart reads); for the
 * notebooks' own set, which the exclusions do not touch, FixedSetNote under Include or Mark.
 */
export function SetNote({ c, drawn = false, of }: { c: Charcoal; drawn?: boolean; of?: (s: Scored<FilterRow>) => boolean }) {
  if (c.set === 'notebook') return <FixedSetNote what={`every filter the notebooks scored (${SET_LABEL.notebook}), with their seasons, Fabs and EC`} />
  return <ExcludedNote count={(of ? c.marked.filter(of) : c.marked).length} drawn={drawn} />
}

/**
 * Filters → excluded: Mark on a SweepPlot, which draws no × of its own: a series that only carries the marked
 * filters' hover (no dot, no line) and the × over the plot, for the marks inside the counted points' range, so the
 * axes stay the counted ones. `children` goes to the SweepPlot.
 */
function useSweepMarks(marked: Scored<FilterRow>[], counted: { x: Date; y: number }[], y: (s: Scored<FilterRow>) => number, yLabel: string) {
  const reasonOf = useMarkReason()
  return useMemo(() => {
    if (!marked.length || !counted.length) return { series: [] as SweepSeries[], children: undefined }
    const xs = counted.map((p) => +p.x), ys = counted.map((p) => p.y)
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
    const inside = marked.filter((s) => +new Date(s.date) >= x0 && +new Date(s.date) <= x1 && y(s) >= y0 && y(s) <= y1)
    if (!inside.length) return { series: [] as SweepSeries[], children: undefined }
    const series: SweepSeries[] = [{ id: 'excluded', label: 'excluded', color: INK.muted, width: 0, r: 0, marks: true,
      points: inside.map((s) => ({ x: new Date(s.date), y: y(s), tip: markTip({ id: s.filterId, site: s.row?.site, date: `${s.date} · ${shortSeason(s.season)}` }, reasonOf(s.id) ?? 'excluded', [`${yLabel} ${fmt(y(s), 3)}`]) })) }]
    const children = ({ x, y: sy }: { x: unknown; y: (v: number) => number }) => <path className="scatter-marks" fill="none" stroke={INK.muted} strokeWidth={1.3} strokeOpacity={0.85} strokeLinecap="round" pointerEvents="none"
      d={inside.map((s) => crossPath((x as ScaleTime<number, number>)(new Date(s.date)), sy(y(s)))).join('')} />
    return { series, children }
  }, [marked, counted, y, yLabel, reasonOf])
}

/** the notebook's value beside the page's, when the page shows the notebooks' set */
const vs = (mine: number, theirs: number | null | undefined, digits = 3) => theirs == null ? fmt(mine, digits) : `${fmt(mine, digits)} (nb ${fmt(theirs, digits)})`

export function RawVsChar({ c, data, setPick }: { c: Charcoal; data: SpectralExport; setPick: React.ReactNode }) {
  const groups = useMemo(() => [...byGroup(c.scored, () => 'All filters', ['All filters']), ...byGroup(c.scored, (s) => s.season, c.seasons.map((s) => s.name))].filter((g) => g.n), [c])
  const saved = rowsOfTable(data, 'by_season_05')
  const nbOf = (season: string) => (c.set === 'notebook' ? saved.find((r) => r.season === season) : undefined)
  const pts = c.scored.map((s) => scorePoint(s, s.rChar, s.rRaw, c.colorOf(s.season), 'r_char', 'r_raw', false, [`${DELTA}: ${fmt(s.delta, 3)}`]))
  const marked = c.marked.map((s) => scorePoint(s, s.rChar, s.rRaw, c.colorOf(s.season), 'r_char', 'r_raw', false, [`${DELTA}: ${fmt(s.delta, 3)}`]))
  const all = groups[0], dry = groups.find((g) => g.group !== 'All filters' && c.isDry({ season: g.group, month: 0 }))
  return (
    <ChartFrame id="ch-raw-vs-char" controls={setPick}
      title={all ? `Within McCall, raw feedstock matches ${pct(all.rawWins)} of these ${all.n} filters better than char${dry ? `; ${pct(dry.rawWins)} in the ${shortSeason(dry.group)} season` : ''}` : 'Raw feedstock against char, per filter'}
      subtitle={`Each Addis filter's best Pearson r against McCall's charred feedstock (x) and against its raw feedstock (y), on AIRSpec-baselined, detrended, SNV spectra over 1430–3500 cm⁻¹ (char_05). Above the 1:1 line raw biomass is the closer analogue. Coloured by season; the table recomputes char_05's by-season summary for the filters shown${c.set === 'notebook' ? ', with the notebook\'s own numbers in brackets' : ''}.`}
      provenance={<>{NB.c05} [14] left, [18]. {provenance(data, ['per_filter', 'by_season_05'])}</>}>
      {pts.length < 3 ? <Empty>No scored Addis filters in view. Choose “{SET_LABEL.notebook}” or bring Addis into view.</Empty> : <Grid>
        <Crossplot points={pts} marked={marked} title="r_raw vs r_char" xLabel={R_CHAR} yLabel={R_RAW} fit={false} extraLines={line({ slope: 1, intercept: 0 }, INK.identity, '1:1', '5 4')} maxSide={420} />
        <DataTable rows={groups} rowKey={(g) => g.group} columns={[
          { key: 'g', label: 'Filters', value: (g) => g.group === 'All filters' ? g.group : <span style={{ color: c.colorOf(g.group) }}>{shortSeason(g.group)}</span> },
          { key: 'n', label: 'n', num: true, value: (g) => vs(g.n, num(nbOf(g.group)?.filters ?? null), 0) },
          { key: 'raw', label: 'Median r_raw', num: true, value: (g) => vs(g.rawMedian, num(nbOf(g.group)?.['best r vs raw feedstock'] ?? null)) },
          { key: 'char', label: 'Median r_char', num: true, value: (g) => vs(g.charMedian, num(nbOf(g.group)?.['best r vs char'] ?? null)) },
          { key: 'w', label: 'Share raw > char', num: true, value: (g) => vs(g.rawWins, num(nbOf(g.group)?.['share better matched by raw'] ?? null)) },
          { key: 'd', label: 'Median delta', num: true, value: (g) => fmt(g.deltaMedian, 3) },
        ]} />
      </Grid>}
      <Key items={[...c.seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: '1:1', color: INK.identity, kind: 'dash' }]} />
      <SetNote c={c} drawn />
      <Note>Best r is the maximum over a class's reference spectra, so it rises with the number of references (McCall: 24 raw, 216 char spectra); the notebooks compare feedstocks on the mean r for that reason. The full charcoal library matches Addis better than either McCall class (median best r 0.78, char_02), so this is a statement about McCall, not evidence that Addis is unburned.</Note>
    </ChartFrame>
  )
}

export function Recurrence({ c, data, setPick }: { c: Charcoal; data: SpectralExport; setPick: React.ReactNode }) {
  const counted = useMemo(() => c.scored.map((s) => ({ x: new Date(s.date), y: s.rChar })), [c.scored])
  const marks = useSweepMarks(c.marked, counted, rChar, 'r_char')
  const series: SweepSeries[] = c.seasons.map((se) => ({ id: se.name, label: shortSeason(se.name), color: se.color, marks: true, width: 0, r: 3,
    points: c.scored.filter((s) => s.season === se.name).sort((a, b) => a.date.localeCompare(b.date))
      .map((s) => ({ x: new Date(s.date), y: s.rChar, id: s.row ? s.id : undefined, tip: [s.filterId, `${s.date} · ${shortSeason(s.season)}`, `r_char ${fmt(s.rChar, 3)} · delta ${fmt(s.delta, 3)}`] })) }))
  const monthly = monthlyRate(c.scored)
  const saved = c.set === 'notebook' ? rowsOfTable(data, 'monthly_06') : []
  const dryMonths = new Set(c.drySeasons.flatMap((s) => s.months))
  const inDry = monthly.filter((m) => dryMonths.has(m.month) && m.n), outDry = monthly.filter((m) => !dryMonths.has(m.month) && m.n)
  const share = (ms: typeof monthly) => { const n = ms.reduce((a, m) => a + m.n, 0); return n ? ms.reduce((a, m) => a + m.rate * m.n, 0) / n : NaN }
  return (
    <ChartFrame id="ch-recurrence" controls={setPick}
      title={`Similarity to char collapses every ${c.dryName} season: ${pct(share(inDry))} of ${c.dryName}-season filters are anomalous, ${pct(share(outDry))} of the rest`}
      subtitle={`Top: each filter's best r against McCall char over time, the ${c.dryName} season shaded. Bottom: the share of filters per calendar month (all years pooled) with delta = r_raw − r_char above 0.05, char_06's anomaly rule. An instrument or filter-lot artefact would not switch on and off with the calendar for four years.`}
      provenance={<>{NB.c06} [8]. {provenance(data, ['per_filter', 'monthly_06'])}</>}>
      {!c.scored.length ? <Empty>No scored Addis filters in view.</Empty> : <>
        <SweepPlot series={[...series, ...marks.series]} xScale="time" yLabel="r_char" height={260} children={marks.children}
          background={({ x, iw, ih }) => timeSeasonBands(x as ScaleTime<number, number>, c.drySeasons, iw).map((b, i) => <rect key={i} x={b.x} y={0} width={b.w} height={ih} fill={b.color} opacity={0.1} />)} />
        <Key items={c.seasons.map((s) => ({ label: shortSeason(s.name), color: s.color }))} />
        <Bars label={`share of filters with delta > 0.05${c.set === 'notebook' ? ' (nb = char_06\'s saved rate)' : ''}`} max={1} format={(v) => pct(v)}
          bars={monthly.map((m) => {
            const nbRate = num(saved.find((r) => r.month === m.month)?.rate ?? null)
            return { key: String(m.month), label: MONTH[m.month - 1], value: m.n ? m.rate : null, signed: dryMonths.has(m.month) ? -1 : 1,
              note: `n=${m.n}${nbRate != null ? ` · nb ${pct(nbRate)}` : ''}`, tip: [MONTH[m.month - 1], `${pct(m.rate)} of ${m.n} filters`] }
          })} />
        <Note>Red bars: months of the {c.dryName} season under the calendar in use; blue: the rest.</Note>
      </>}
      <SetNote c={c} drawn />
    </ChartFrame>
  )
}

export function Populations({ c, data, setPick }: { c: Charcoal; data: SpectralExport; setPick: React.ReactNode }) {
  const edges = Array.from({ length: 31 }, (_, i) => 0.15 + (i * 0.75) / 30)
  const hist = c.seasons.map((se) => {
    const v = c.scored.filter((s) => s.season === se.name).map((s) => s.rChar)
    const counts = edges.slice(0, -1).map((e, i) => v.filter((x) => x >= e && (i === edges.length - 2 ? x <= edges[i + 1] : x < edges[i + 1])).length)
    return histogramSeries({ id: se.name, label: shortSeason(se.name), color: se.color, edges, counts, fill: 0.25, tip: (i, n) => [shortSeason(se.name), `r_char ${fmt(edges[i], 2)}–${fmt(edges[i + 1], 2)}`, `${n} filters`] })
  })
  const pts = c.scored.map((s) => scorePoint(s, s.rChar, s.delta, c.colorOf(s.season), 'r_char', 'delta', !s.low))
  const marked = c.marked.map((s) => scorePoint(s, s.rChar, s.delta, c.colorOf(s.season), 'r_char', 'delta', !s.low))
  const saved = rowsOfTable(data, 'mixture_vs_season')
  const counts = c.seasons.map((se) => {
    const rs = c.scored.filter((s) => s.season === se.name), nb = c.set === 'notebook' ? saved.find((r) => r.season === se.name) : undefined
    return { season: se.name, n: rs.length, low: rs.filter((s) => s.low).length, nbLow: num(nb?.['low-similarity'] ?? null) }
  })
  const low = c.scored.filter((s) => s.low)
  return (
    <ChartFrame id="ch-populations" controls={setPick}
      title={`Two populations, not one shifted one: ${low.length} of ${c.scored.length} filters sit in the low-similarity component, ${counts.filter((r) => c.isDry({ season: r.season, month: 0 })).reduce((a, r) => a + r.low, 0)} of them in the ${c.dryName} season`}
      subtitle="Left: r_char by season. Right: r_char against delta; filled points are in char_06's low-similarity component, hollow ones in the normal one. The component comes from a two-component Gaussian mixture on r_char fitted on all 239 filters without the season (BIC −214.9 against −202.3 for one component; means 0.393 and 0.649), so it is the notebook's label: it does not refit on the filters shown."
      provenance={<>{NB.c06} [11]. {provenance(data, ['per_filter', 'mixture_vs_season'])}</>}>
      {!c.scored.length ? <Empty>No scored Addis filters in view.</Empty> : <Grid>
        <div><SweepPlot series={hist} xLabel="r_char" yLabel="filters" height={260} include={[0]} /></div>
        <Crossplot points={pts} marked={marked} title="delta vs r_char" xLabel={R_CHAR} yLabel={DELTA} fit={false} sameScale={false} maxSide={380}
          extraLines={line({ slope: 0, intercept: 0 }, INK.axis, 'delta = 0', '4 3')} />
      </Grid>}
      <Key items={[...c.seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: 'low-similarity component', color: INK.text, kind: 'dot' }, { label: 'normal component', color: INK.text, kind: 'hollow' }]} />
      <DataTable rows={counts} rowKey={(r) => r.season} columns={[
        { key: 's', label: 'Season', value: (r) => <span style={{ color: c.colorOf(r.season) }}>{shortSeason(r.season)}</span> },
        { key: 'n', label: 'Filters', num: true, value: (r) => r.n },
        { key: 'l', label: 'Low-similarity', num: true, value: (r) => vs(r.low, r.nbLow, 0) },
        { key: 'p', label: 'Share', num: true, value: (r) => pct(r.n ? r.low / r.n : NaN) },
      ]} />
      <SetNote c={c} drawn />
    </ChartFrame>
  )
}
