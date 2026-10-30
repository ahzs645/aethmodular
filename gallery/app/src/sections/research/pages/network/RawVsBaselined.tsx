/**
 * ftir_52 §7: what baselining does to who looks like whom, for the network and for one site. The notebook's
 * raw_vs_baselined_sites.png (cell [22]: every IMPROVE site pair's dissimilarity 1 − r in raw against
 * AIRSpec-baselined space, and both histograms), and a per-site view it does not draw: one site's (or one SPARTAN
 * target's) dissimilarity to every other site in both spaces, and how its nearest neighbours reorder.
 *
 * Both are computed here from the two site × site matrices the page ships: raw_sites + raw_pairs (median raw
 * spectra over 1425–3999 cm⁻¹, recomputed from the Davis Data spectra by the exporter) and site_correlation
 * (the notebook's own baselined matrix, four decimals). No HIPS, no Fabs: the MAC does not enter.
 */
import { memo, useMemo, useRef } from 'react'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { IdentityLine } from '@/components/RefLines'
import { SweepPlot, histogramSeries, type SweepSeries } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useChartOptions } from '@/lib/chartOptions'
import { fmt, pct } from '@/lib/format'
import { BASELINE_COLOR, FONT, INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { logHistogram, members, neighbours, networkSummary, pairs, siteLabel, siteSummary, spaces, type Cells, type Neighbour, type Pair, type Spaces } from '../../common/rawBaselined'
import { Grid } from '../../common/ui'
import { provenance, type SpectralExport } from '../spectralCommon'

const RAW = INK.muted
const BASE = BASELINE_COLOR.AIRSpec
/** the notebook marks the SPARTAN sites in red (site_dendrogram.png); pairs with one are drawn in it here */
const SPARTAN = '#B23327'
const CODE: Record<string, string> = { '*Addis': 'ETAD', '*Bishoftu': 'ETBI', '*Delhi': 'INDH', '*Beijing': 'CHTS', '*JPL': 'USPA', '*Pasadena': 'USPA' }
/** ftir_52 cell [21], printed: the figure's numbers on the notebook's 151 IMPROVE sites */
const NB = { rho: 0.587, medianRaw: 0.999, medianBaselined: 0.9681, changed: 139 }
const BINS = 60
const NB_CELL = 'ftir_52 §7, cells [21]–[23]'

/** r from a dissimilarity, with enough digits to tell raw pairs apart */
const rOf = (d: number, digits = 4) => fmt(1 - d, digits)
/** a log axis labels its powers of ten only (SweepPlot's x axis does the same over more than three decades) */
const decadeLabel = (v: number) => (Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9 ? String(+v.toPrecision(1)) : '')
const dOf = (d: number) => (d < 1e-3 ? d.toExponential(2) : d.toPrecision(3))

function useSpaces(data: SpectralExport) {
  return useMemo(() => spaces(data.tables.raw_sites as Cells | undefined, data.tables.raw_pairs as Cells | undefined, data.tables.site_correlation as Cells | undefined), [data])
}

/**
 * The network panel and the per-site panel, sharing the site in focus (`network-raw-site.site` in the URL): a click
 * on a pair in the network chart puts one of its sites in focus below.
 */
export function RawVsBaselined({ data, meta }: { data: SpectralExport; meta: MetaFile }) {
  const s = useSpaces(data)
  const choices = useMemo(() => {
    if (!s) return []
    const order = s.names.map((n, i) => ({ n, i })).sort((a, b) => Number(s.spartan[b.i]) - Number(s.spartan[a.i]) || (s.spartan[a.i] ? a.i - b.i : a.n.localeCompare(b.n)))
    return order.map(({ n, i }) => ({ value: n, label: s.spartan[i] ? `${siteLabel(n)} (SPARTAN)` : n }))
  }, [s])
  const colourOf = useMemo(() => (name: string) => {
    const code = CODE[name]
    return code ? meta.sites.find((x) => x.code === code)?.color ?? (code === 'ETBI' ? '#8e44ad' : SPARTAN) : BASE
  }, [meta.sites])
  const site = useChartOptions('network-raw-site', [
    { key: 'site', label: 'Site', style: 'select', default: '*Addis', choices },
    { key: 'partners', label: 'Compared with', default: 'all', choices: [{ value: 'all', label: 'Every site' }, { value: 'improve', label: 'IMPROVE sites (as ftir_52)' }] },
    { key: 'k', label: 'Nearest', default: '10', choices: ['5', '10', '15'] },
  ])
  const prov = provenance(data, ['raw_sites', 'raw_pairs', 'site_correlation'])
  if (!s) {
    return <ChartFrame id="network-raw-pairs" title="What baselining does to the network" provenance={prov}>
      <Empty>The raw-space site matrix (raw_sites, raw_pairs) or the baselined one (site_correlation) is not in this export.</Empty>
    </ChartFrame>
  }
  const focus = Math.max(0, s.names.indexOf(String(site.values.site)))
  return <>
    <NetworkPanel s={s} focus={focus} onFocus={(i) => site.set('site', s.names[i])} prov={prov} />
    <SitePanel s={s} focus={focus} onFocus={(i) => site.set('site', s.names[i])} partners={String(site.values.partners)} k={Number(site.values.k)} colourOf={colourOf} controls={site.controls()} prov={prov} />
  </>
}

/** ftir_52 [22]: every site pair in both spaces, and the two histograms */
function NetworkPanel({ s, focus, onFocus, prov }: { s: Spaces; focus: number; onFocus: (i: number) => void; prov: React.ReactNode }) {
  const o = useChartOptions('network-raw-pairs', [
    { key: 'pairs', label: 'Pairs', default: 'improve', choices: [{ value: 'improve', label: 'IMPROVE (as ftir_52)' }, { value: 'all', label: 'With the SPARTAN sites' }] },
  ])
  const withTargets = o.values.pairs === 'all'
  const view = useMemo(() => {
    const sites = members(s, withTargets)
    const ps = pairs(s, sites)
    const all = ps.flatMap((p) => [p.raw, p.baselined])
    const lims: [number, number] = [Math.min(...all) * 0.8, Math.max(...all) * 1.2]
    const h = logHistogram([ps.map((p) => p.raw), ps.map((p) => p.baselined)], lims[0], lims[1], BINS)
    // the site in focus is drawn against the sites in view even when it is a SPARTAN target left out of them
    const mine = pairs(s, [focus, ...sites.filter((j) => j !== focus)]).filter((p) => p.a === focus)
    return { sites, ps, mine, lims, h, sum: networkSummary(s, sites) }
  }, [s, withTargets, focus])
  const { ps, mine, lims, h, sum } = view
  const name = (i: number) => siteLabel(s.names[i])
  const tipOf = (p: Pair) => [`${name(p.a)} – ${name(p.b)}`, `raw r ${rOf(p.raw, 5)} (1 − r ${dOf(p.raw)})`, `baselined r ${rOf(p.baselined)} (1 − r ${dOf(p.baselined)})`,
    p.baselined > p.raw ? `baselining separates them ${fmt(p.baselined / p.raw, 0)}×` : 'closer after baselining', 'click to put a site of this pair in focus below']
  const series: SweepSeries[] = [
    // the cloud is drawn by <Cloud> underneath; this series only carries the hover and the click
    { id: 'pairs', label: 'site pairs', color: BASE, width: 0, marks: false, interval: 'none', points: ps.map((p) => ({ x: p.raw, y: p.baselined, id: `${p.a}-${p.b}`, tip: tipOf(p) })) },
    { id: 'focus', label: `pairs with ${name(focus)}`, color: INK.accent, width: 0, marks: true, r: 2.6, interval: 'none',
      points: mine.map((p) => ({ x: p.raw, y: p.baselined, id: `${p.a}-${p.b}`, tip: tipOf(p) })) },
  ]
  const hist = [
    histogramSeries({ id: 'raw', label: 'raw', color: RAW, edges: h.edges, counts: h.counts[0], fill: 0.4, tip: (i, c) => ['raw', `1 − r ${dOf(h.edges[i])}–${dOf(h.edges[i + 1])}`, `${c} site pairs`] }),
    histogramSeries({ id: 'baselined', label: 'AIRSpec-baselined', color: BASE, edges: h.edges, counts: h.counts[1], fill: 0.4, tip: (i, c) => ['AIRSpec-baselined', `1 − r ${dOf(h.edges[i])}–${dOf(h.edges[i + 1])}`, `${c} site pairs`] }),
  ]
  const nb = !withTargets
  return (
    <ChartFrame id="network-raw-pairs" controls={o.controls()}
      title={`What baselining does to the network — raw, every site looks the same (median pair r ${fmt(sum.medianRaw, 4)}); baselined, structure appears (${fmt(sum.medianBaselined, 4)})`}
      subtitle={`Every pair of ${sum.sites} ${withTargets ? 'sites (IMPROVE and the five SPARTAN targets)' : 'IMPROVE sites'} (${sum.pairs.toLocaleString()} pairs). x: dissimilarity 1 − r between the two sites' median RAW spectra, on the corrected grid's 1425–3999 cm⁻¹ window; y: the same pair after the AIRSpec baseline. Log axes, as in the notebook: raw pairs all sit at r ≈ 0.999, which a linear r axis collapses onto one tick. Spearman ρ between the two spaces ${fmt(sum.rho, 3)}${nb ? ` (ftir_52: ${NB.rho})` : ''}; ${sum.separated.toLocaleString()} of ${sum.pairs.toLocaleString()} pairs (${pct(sum.separated / sum.pairs)}) sit above the 1:1 line, i.e. baselining moves them apart; ${sum.changed} of ${sum.sites} sites${nb ? ` (ftir_52: ${NB.changed} of 151)` : ''} change which site is their nearest. Removing the shared PTFE and background leaves the differences between sites as a much larger share of what is left. Click a pair to put one of its sites in focus in the next chart.`}
      provenance={<>{NB_CELL}: raw_vs_baselined_sites.png. {prov} Pairs, ρ, medians and nearest sites computed here from the two matrices; the baselined r is rounded to four decimals in the export, so its lowest 1 − r values step by 0.0001.</>}>
      <Grid min={320}>
        <SweepPlot series={series} xScale="log" yScale="log" xDomain={lims} yDomain={lims} height={360}
          xLabel="site-pair dissimilarity 1 − r, RAW" yLabel="1 − r, AIRSpec-baselined" yFormat={decadeLabel}
          onPoint={(_, p) => { const [a, b] = String(p.id).split('-').map(Number); onFocus(a === focus ? b : a) }}
          background={({ x, y }) => <Cloud ps={ps} x={x as Scale} y={y as Scale} spartan={s.spartan} />}>
          {({ x, y }) => {
            const sx = x as Scale, sy = y as Scale
            return <>
              <IdentityLine x1={sx(lims[0])} y1={sy(lims[0])} x2={sx(lims[1])} y2={sy(lims[1])} />
              <text x={8} y={14} fontSize={10.5} fill={INK.muted} fontFamily={FONT.family}>above the line: baselining separates the pair</text>
            </>
          }}
        </SweepPlot>
        <SweepPlot series={hist} xScale="log" xDomain={lims} include={[0]} height={360} xLabel="site-pair dissimilarity 1 − r" yLabel="site pairs"
          xMarkers={[{ at: 1 - sum.medianRaw, label: `raw median r ${fmt(sum.medianRaw, 4)}`, color: RAW }, { at: 1 - sum.medianBaselined, label: `baselined ${fmt(sum.medianBaselined, 4)}`, color: BASE }]} />
      </Grid>
      <Key items={[{ label: 'IMPROVE–IMPROVE pair', color: BASE }, ...(withTargets ? [{ label: 'pair with a SPARTAN site', color: SPARTAN }] : []),
        { label: `pairs with ${name(focus)} (in focus below)`, color: INK.accent }, { label: '1:1', color: INK.identity, kind: 'dash' as const },
        { label: 'raw (histogram)', color: RAW, kind: 'bar' as const }, { label: 'AIRSpec-baselined (histogram)', color: BASE, kind: 'bar' as const }]} />
    </ChartFrame>
  )
}

type Scale = (v: number) => number

/** the 11–12 thousand pair dots, drawn once per layout rather than on every hover of the chart above them */
const Cloud = memo(function Cloud({ ps, x, y, spartan }: { ps: Pair[]; x: Scale; y: Scale; spartan: boolean[] }) {
  return <g pointerEvents="none">
    {ps.map((p) => {
      const t = spartan[p.a] || spartan[p.b]
      return <circle key={`${p.a}-${p.b}`} cx={x(p.raw)} cy={y(p.baselined)} r={t ? 2 : 1.5} fill={t ? SPARTAN : BASE} fillOpacity={t ? 0.6 : 0.25} />
    })}
  </g>
}, (a, b) => a.ps === b.ps && a.spartan === b.spartan && sameScale(a.x, b.x) && sameScale(a.y, b.y))

/** two d3 scales place points alike when their domains and ranges agree */
function sameScale(a: Scale, b: Scale) {
  const da = (a as unknown as { domain(): number[]; range(): number[] }), db = (b as unknown as { domain(): number[]; range(): number[] })
  return da.domain().join() === db.domain().join() && da.range().join() === db.range().join()
}

/** one site against every other: both dissimilarities, and how its nearest neighbours reorder */
function SitePanel({ s, focus, onFocus, partners, k, colourOf, controls, prov }: {
  s: Spaces; focus: number; onFocus: (i: number) => void; partners: string; k: number; colourOf: (name: string) => string; controls: React.ReactNode; prov: React.ReactNode
}) {
  const view = useMemo(() => {
    const sites = members(s, partners === 'all' || s.spartan[focus])
    const scope = partners === 'all' ? sites : sites.filter((j) => j === focus || !s.spartan[j])
    const list = neighbours(s, focus, scope)
    const sum = siteSummary(list, k)
    const all = list.flatMap((n) => [n.raw, n.baselined])
    const lims: [number, number] = [Math.min(...all) * 0.8, Math.max(...all) * 1.2]
    return { list, sum, lims }
  }, [s, focus, partners, k])
  const { list, sum, lims } = view
  const me = siteLabel(s.names[focus])
  if (list.length < 3 || !sum.nearestRaw || !sum.nearestBaselined) {
    return <ChartFrame id="network-raw-site" title={`${me} against every other site`} controls={controls} provenance={prov}><Empty>Too few sites to compare.</Empty></ChartFrame>
  }
  const nr = sum.nearestRaw, nb = sum.nearestBaselined
  const name = (j: number) => siteLabel(s.names[j])
  const tipOf = (n: Neighbour) => [`${me} – ${name(n.j)}${s.spartan[n.j] ? ' (SPARTAN)' : ''}`, `raw r ${rOf(n.raw, 5)} · rank ${n.rankRaw}`, `baselined r ${rOf(n.baselined)} · rank ${n.rankBaselined}`, 'click to put it in focus']
  const pointOf = (n: Neighbour) => ({ x: n.raw, y: n.baselined, id: String(n.j), tip: tipOf(n) })
  const targets = list.filter((n) => s.spartan[n.j])
  const series: SweepSeries[] = [
    { id: 'improve', label: 'IMPROVE sites', color: BASE, width: 0, marks: true, r: 2.6, interval: 'none', points: list.filter((n) => !s.spartan[n.j]).map(pointOf) },
    ...targets.map((n): SweepSeries => ({ id: s.names[n.j], label: name(n.j), color: colourOf(s.names[n.j]), width: 0, marks: true, r: 5, interval: 'none', points: [pointOf(n)] })),
  ]
  const top = (by: 'rankRaw' | 'rankBaselined', n: number) => list.filter((x) => x[by] <= n)
  const labelled = new Map<number, Neighbour>([...top('rankRaw', 3), ...top('rankBaselined', 3)].map((n) => [n.j, n]))
  const same = nr.j === nb.j
  const scope = partners === 'all' || s.spartan[focus] ? 'every other site' : 'the IMPROVE sites'
  return (
    <ChartFrame id="network-raw-site" controls={controls}
      title={same ? `${me} keeps ${name(nr.j)} as its nearest site through baselining; ${sum.keptOfTop} of its ${k} nearest stay among the ${k} nearest`
        : `${me}: nearest site ${name(nr.j)} in raw space, ${name(nb.j)} after baselining — ${name(nr.j)} falls to rank ${nr.rankBaselined}`}
      subtitle={`${me}${s.spartan[focus] ? ` (SPARTAN, ${s.n[focus]} filters)` : ` (${s.n[focus]} library spectra)`} against ${scope} (${sum.partners}): left, its dissimilarity 1 − r to each in raw space (x) and after AIRSpec (y), log axes; right, its ${k} nearest sites in each space, joined where they are the same site. Raw: ${name(nr.j)} at r ${rOf(nr.raw, 5)}; baselined: ${name(nb.j)} at r ${rOf(nb.baselined)}. Its raw nearest is rank ${nr.rankBaselined} after baselining (ftir_52's rank_of_raw_nn_after); ${sum.keptOfTop} of its raw ${k} nearest are among the baselined ${k} nearest; Spearman ρ between its two rows ${fmt(sum.rho, 2)}; baselining separates it from ${sum.separated} of ${sum.partners} sites. SPARTAN sites are drawn large in their site colours. Raw-space neighbours are background neighbours: who shares the filter's PTFE and baseline shape, not its aerosol.`}
      provenance={<>{NB_CELL} (the per-site table [21] and the SPARTAN targets' raw nearest sites [23], which the notebook could give for Addis only). {prov}</>}>
      <Grid min={320}>
        <SweepPlot series={series} xScale="log" yScale="log" xDomain={lims} yDomain={lims} height={360}
          xLabel={`1 − r with ${me}, RAW`} yLabel={`1 − r with ${me}, baselined`} yFormat={decadeLabel}
          onPoint={(_, p) => onFocus(Number(p.id))}>
          {({ x, y }) => {
            const sx = x as Scale, sy = y as Scale
            return <>
              <IdentityLine x1={sx(lims[0])} y1={sy(lims[0])} x2={sx(lims[1])} y2={sy(lims[1])} />
              {placeLabels([...labelled.values()].map((n) => ({ j: n.j, x: sx(n.raw), y: sy(n.baselined), text: name(n.j) }))).map((l) =>
                <text key={l.j} x={l.left ? l.x - 6 : l.x + 6} y={l.y + l.dy + 3.5} textAnchor={l.left ? 'end' : 'start'} fontSize={10} fontFamily={FONT.family} fill={INK.text} pointerEvents="none">{l.text}</text>)}
            </>
          }}
        </SweepPlot>
        <RankShift list={list} k={k} n={sum.partners} name={name} spartan={(j) => s.spartan[j]} colourOf={(j) => colourOf(s.names[j])} />
      </Grid>
      <Key items={[{ label: 'IMPROVE site', color: BASE }, ...targets.map((n) => ({ label: `${name(n.j)} (SPARTAN)`, color: colourOf(s.names[n.j]) })), { label: '1:1', color: INK.identity, kind: 'dash' as const },
        { label: `in both ${k}-nearest lists`, color: INK.text, kind: 'line' as const }, { label: 'drops out after baselining', color: RAW, kind: 'dash' as const }, { label: 'comes in after baselining', color: BASE, kind: 'line' as const }]} />
      <Note>Ranks are 1 for the nearest site; below rank {k} the axis is logarithmic down to rank {sum.partners}. The baselined r is the notebook's matrix rounded to four decimals, so two partners can tie.</Note>
    </ChartFrame>
  )
}

const ROW = 22, HEAD = 22, TAIL = 96, LABEL_W = 118

/** point labels to the right of their points, else to the left, above or below, wherever they do not overprint one already placed */
function placeLabels<T extends { x: number; y: number; text: string }>(items: T[]): (T & { left: boolean; dy: number })[] {
  const out: (T & { left: boolean; dy: number; box: [number, number, number] })[] = []
  for (const it of items.slice().sort((a, b) => a.x - b.x)) {
    const wide = it.text.length * 6
    const box = (left: boolean, dy: number): [number, number, number] => [left ? it.x - 6 - wide : it.x + 6, left ? it.x - 6 : it.x + 6 + wide, it.y + dy]
    // clear of the labels placed so far and of the other labelled points
    const free = ([x0, x1, y]: [number, number, number]) => out.every((o) => Math.abs(o.box[2] - y) >= 11 || x0 >= o.box[1] + 3 || x1 <= o.box[0] - 3)
      && items.every((o) => o === it || Math.abs(o.y - y) >= 8 || o.x + 4 <= x0 || o.x - 4 >= x1)
    const tries: [boolean, number][] = [[false, 0], [true, 0], [false, -10], [false, 10], [true, -10], [true, 10]]
    const [left, dy] = tries.find(([l, d]) => free(box(l, d))) ?? tries[0]
    out.push({ ...it, left, dy, box: box(left, dy) })
  }
  return out
}

/**
 * A slope chart of neighbour ranks: raw rank on the left, baselined rank on the right, one line per site that is
 * among the k nearest in either space. Ranks 1–k get a row each; a rank beyond k sits in a log-scaled tail below.
 */
function RankShift({ list, k, n, name, spartan, colourOf }: {
  list: Neighbour[]; k: number; n: number; name: (j: number) => string; spartan: (j: number) => boolean; colourOf: (j: number) => string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const w = Math.max(300, width)
  const x0 = LABEL_W, x1 = w - LABEL_W
  const tailTop = HEAD + k * ROW + 16, tailBottom = tailTop + TAIL - 16
  const lo = Math.log(k + 1), hi = Math.log(Math.max(n, k + 2))
  const yOf = (rank: number) => (rank <= k ? HEAD + (rank - 0.5) * ROW : tailTop + ((Math.log(rank) - lo) / (hi - lo)) * (tailBottom - tailTop))
  // one row per site: tied partners (the baselined matrix has four decimals) take consecutive rows, raw order first;
  // the text keeps each site's own rank (ties share the better one, as ftir_52's rank_of_raw_nn_after counts)
  const pos = (by: 'raw' | 'baselined') => new Map(list.slice().sort((a, b) => a[by] - b[by] || a.rankRaw - b.rankRaw).map((m, i) => [m.j, i + 1]))
  const rawPos = pos('raw'), basPos = pos('baselined')
  const shown = list.map((m) => ({ ...m, pr: rawPos.get(m.j)!, pb: basPos.get(m.j)! })).filter((m) => m.pr <= k || m.pb <= k)
  type Shown = (typeof shown)[number]
  const height = tailBottom + 14
  // tail rank numbers, thinned so they do not overprint; the site that was (or becomes) the nearest is placed first
  const tailLabels = (side: 'raw' | 'baselined') => {
    const out: { j: number; rank: number; y: number }[] = []
    const cands = shown.filter((m) => (side === 'raw' ? m.pr : m.pb) > k)
      .map((m) => ({ j: m.j, rank: side === 'raw' ? m.rankRaw : m.rankBaselined, y: yOf(side === 'raw' ? m.pr : m.pb), first: m.pr === 1 || m.pb === 1 }))
      .sort((a, b) => Number(b.first) - Number(a.first) || a.rank - b.rank)
    for (const c of cands) if (out.every((o) => Math.abs(o.y - c.y) >= 10)) out.push(c)
    return out
  }
  const style = (m: Shown) => (m.pr <= k && m.pb <= k ? { stroke: INK.text, dash: undefined, opacity: 0.75 }
    : m.pr <= k ? { stroke: RAW, dash: '4 3', opacity: 0.9 } : { stroke: BASE, dash: undefined, opacity: 0.9 })
  const label = (j: number, r: string, anchor: 'start' | 'end', x: number, y: number) =>
    <text x={x} y={y + 3.5} textAnchor={anchor} fontSize={10.5} fontFamily={FONT.family} fill={spartan(j) ? colourOf(j) : INK.text} fontWeight={spartan(j) ? 600 : 400}>
      {name(j)} {r}
    </text>
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={w} height={height} fontFamily={FONT.family}>
        <text x={x0} y={12} textAnchor="end" fontSize={11} fontWeight={600} fill={INK.text}>raw rank</text>
        <text x={x1} y={12} textAnchor="start" fontSize={11} fontWeight={600} fill={INK.text}>baselined rank</text>
        {Array.from({ length: k }, (_, i) => <g key={i}>
          <text x={x0 + 6} y={yOf(i + 1) + 3.5} fontSize={9.5} fill={INK.muted}>{i + 1}</text>
          <text x={x1 - 6} y={yOf(i + 1) + 3.5} textAnchor="end" fontSize={9.5} fill={INK.muted}>{i + 1}</text>
        </g>)}
        <line x1={0} x2={w} y1={tailTop - 9} y2={tailTop - 9} stroke={INK.axis} strokeDasharray="1 3" />
        {shown.map((m) => {
          const st = style(m), ya = yOf(m.pr), yb = yOf(m.pb)
          const tipText = [`${name(m.j)}${spartan(m.j) ? ' (SPARTAN)' : ''}`, `raw rank ${m.rankRaw} (r ${rOf(m.raw, 5)})`, `baselined rank ${m.rankBaselined} (r ${rOf(m.baselined)})`]
          return <g key={m.j} onMouseMove={(e) => tip.show(e, tipText)} onMouseLeave={tip.hide}>
            <line x1={x0 + 18} y1={ya} x2={x1 - 18} y2={yb} stroke="transparent" strokeWidth={8} />
            <line x1={x0 + 18} y1={ya} x2={x1 - 18} y2={yb} stroke={spartan(m.j) ? colourOf(m.j) : st.stroke} strokeWidth={spartan(m.j) ? 2 : 1.4} strokeDasharray={st.dash} strokeOpacity={st.opacity} />
            <circle cx={x0 + 18} cy={ya} r={2.4} fill={st.stroke} />
            <circle cx={x1 - 18} cy={yb} r={2.4} fill={st.stroke} />
          </g>
        })}
        <text x={2} y={tailTop - 12} fontSize={9.5} fill={INK.muted} stroke={INK.panel} strokeWidth={3} paintOrder="stroke" pointerEvents="none">below rank {k} (log scale, to {n})</text>
        {shown.filter((m) => m.pr <= k).map((m) => <g key={`l${m.j}`}>{label(m.j, rOf(m.raw, 5), 'end', x0 - 4, yOf(m.pr))}</g>)}
        {shown.filter((m) => m.pb <= k).map((m) => <g key={`r${m.j}`}>{label(m.j, rOf(m.baselined), 'start', x1 + 4, yOf(m.pb))}</g>)}
        {tailLabels('raw').map((t) => <text key={`tl${t.j}`} x={x0 - 4} y={t.y + 3.5} textAnchor="end" fontSize={9.5} fill={INK.muted}>{name(t.j)} #{t.rank}</text>)}
        {tailLabels('baselined').map((t) => <text key={`tr${t.j}`} x={x1 + 4} y={t.y + 3.5} fontSize={9.5} fill={INK.muted}>#{t.rank} {name(t.j)}</text>)}
      </svg>
      {tip.node}
    </div>
  )
}
