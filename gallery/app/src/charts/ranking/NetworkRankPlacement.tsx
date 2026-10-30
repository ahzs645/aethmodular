import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { HitLayer } from '@/components/HitLayer'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useChartOptions } from '@/lib/chartOptions'
import { useFitHeight } from '@/lib/chartFit'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { groupQuantiles, placeAmong, rankedReference, referenceQuantity, type Population, type RankedSite, type ReferenceFile } from '@/lib/referenceRank'
import { FONT, IMPROVE_GREY, INK, MARGIN, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { ExcludedNote } from '@/lib/markedRows'

const FILE = 'improve_site_quantiles.json'
const REF = 'IMPROVE sites'
/** a site in view needs this many filters to be placed (run_ftir_51 §11 used 10 for SPARTAN) */
const MIN_VIEW_N = 10
const POPULATIONS: { value: Population; label: string }[] = [
  { value: 'paired', label: 'OC, EC and Fabs > 0' },
  { value: 'own', label: 'Every filter with it' },
]

/** the notebook's digits: two for OC/EC, three below 1 (OC/fAbs) */
const val = (v: number) => fmt(v, Math.abs(v) >= 10 ? 1 : Math.abs(v) >= 1 ? 2 : 3)

type Mark = { kind: 'ref'; rank: number; s: RankedSite } | { kind: 'view'; rank: number; site: string; color: string; code: string; n: number; p25: number; p50: number; p75: number }

/**
 * Where each site sits in a network's ranking: every IMPROVE site ranked by its median of the field (or of a
 * per-filter ratio of two fields, from the Ratio dialog), drawn as median and interquartile range in grey, and each
 * site in view placed as a star at the rank its own median would take in that order. Generalises run_ftir_51's
 * site_rank_figure (ftir_51 11a OC/EC, 11b OC/fAbs), which put the four focus sites' FTIR OC/EC among the IMPROVE
 * sites' thermal-optical OC/EC.
 *
 * The sites in view are computed here from the rows, so they follow the filters, seasons, exclusions and MAC; the
 * IMPROVE side is per-site quantiles (improve_site_quantiles.json) of per-filter values, scaled to the reader's MAC
 * where Fabs / MAC enters (lib/referenceRank).
 */
export function NetworkRankPlacement({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const fit = useFitHeight()
  const m = useMac()
  const lg = useLegend()
  const o = useChartOptions('netrank', [
    { key: 'pop', label: 'Filters', default: 'paired', choices: POPULATIONS,
      title: 'Which filters count, on both sides. "OC, EC and Fabs > 0" is ftir_51 §11 (figures 11a/11b); "Every filter with it" takes a field wherever it is reported and a ratio wherever both parts are > 0' },
    { key: 'minN', label: 'IMPROVE sites with ≥', default: '50', choices: ['10', '30', '50', '100'].map((v) => ({ value: v, label: `${v} filters` })),
      title: 'An IMPROVE site is ranked only with at least this many filters (ftir_51: 50)' },
    { key: 'scale', label: 'Scale', default: 'Linear', choices: ['Linear', 'Log'] },
  ])
  const pop = o.values.pop as Population
  const minN = Number(o.values.minN)

  const { data: ref, error } = useExport<ReferenceFile>(FILE, (d) => (d.schema_version === 1 && d.quantities ? null : 'unexpected schema'))
  const q = useMemo(() => referenceQuantity(field, m.mac), [field, m.mac])
  const quantity = 'reason' in q ? null : q
  // µg per filter (the mass view) cannot be ranked against IMPROVE's µg/m³ unless the units cancel in a ratio
  const massClash = !!quantity && meta.basis === 'mass' && quantity.concPower !== 0
  const ranked = useMemo(() => (ref && quantity ? rankedReference(ref, pop, quantity, minN) : []), [ref, quantity, pop, minN])

  const siteInfo = useMemo(() => {
    const out = new Map<string, { code: string; color: string }>()
    for (const r of rows) if (!out.has(r.site)) out.set(r.site, { code: r.code, color: r.color })
    return out
  }, [rows])
  const placed = useMemo(() => {
    if (!quantity || massClash || !ranked.length) return []
    return groupQuantiles(rows, (r) => r.site, quantity, pop, MIN_VIEW_N).map((g) => {
      const info = siteInfo.get(g.key)!
      return { kind: 'view' as const, rank: placeAmong(ranked, g.p50) + 1, site: g.key, code: info.code, color: info.color, n: g.n, p25: g.p25, p50: g.p50, p75: g.p75 }
    })
  }, [rows, quantity, massClash, ranked, pop, siteInfo])
  const unplaced = [...siteInfo.keys()].filter((s) => !placed.some((p) => p.site === s))
  const shown = placed.filter((p) => lg.show(p.site))

  const N = ranked.length
  const height = fit ?? 440
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const lo = Math.min(d3.min(ranked, (s) => s.p25) ?? 0, d3.min(shown, (s) => s.p25) ?? Infinity)
  const hi = Math.max(d3.max(ranked, (s) => s.p75) ?? 1, d3.max(shown, (s) => s.p75) ?? -Infinity)
  const log = o.values.scale === 'Log' && lo > 0
  const x = useMemo(() => d3.scaleLinear().domain([0, N + 1]).range([0, innerW]), [N, innerW])
  const y = useMemo(() => (log ? d3.scaleLog().domain([lo / 1.15, hi * 1.15]).range([innerH, 0]).nice()
    : d3.scaleLinear().domain([Math.min(0, lo), hi * 1.05]).range([innerH, 0]).nice()), [log, lo, hi, innerH])

  const showRef = lg.show(REF)
  const marks = useMemo<Mark[]>(() => [...(showRef ? ranked.map((s, i) => ({ kind: 'ref' as const, rank: i + 1, s })) : []), ...shown], [ranked, shown, showRef])
  const xy = useMemo(() => marks.map((d) => [x(d.rank), y(d.kind === 'ref' ? d.s.p50 : d.p50)] as const), [marks, x, y])

  const refLabel = quantity?.label ?? field
  const popInfo = ref?.populations[pop]
  // past the highest IMPROVE median the rank is N + 1, which reads better said
  const rankText = (rank: number) => (rank > N ? `above all ${N}` : `rank ${rank} of ${N}`)
  const star = d3.symbol(d3.symbolStar, 210)() ?? ''
  // the stars' names, lifted clear of each other where neighbouring ranks crowd them (left to right, ~6 px a character)
  const labels = new Map<string, { x: number; y: number; end: boolean }>()
  const boxes: { x0: number; x1: number; y: number }[] = []
  for (const p of [...shown].sort((a, b) => a.rank - b.rank)) {
    const w = p.site.length * 6 + 4, end = x(p.rank) + 9 + w > innerW
    const x0 = end ? x(p.rank) - 9 - w : x(p.rank) + 9, x1 = x0 + w
    let ly = y(p.p50) - 7
    while (boxes.some((b) => x0 < b.x1 && b.x0 < x1 && Math.abs(b.y - ly) < 12)) ly -= 12
    boxes.push({ x0, x1, y: ly })
    labels.set(p.site, { x: end ? x1 : x0, y: ly, end })
  }

  return (
    <ChartFrame
      id="netrank"
      title={`Network rank — where each site's median ${field} sits among the IMPROVE sites`}
      subtitle={`Every IMPROVE site with at least ${minN} filters, ranked by its median ${refLabel} (grey: median and interquartile range). Each site in view (≥ ${MIN_VIEW_N} filters) is a star at the rank its own median would take in that order: rank = 1 + the IMPROVE sites with a lower median. Ratios are per filter on both sides, then summarised per site.`}
      provenance={<>ftir_51 11a/11b (research/ftir_ec_phase3/scripts/run_ftir_51.py site_rank_figure), generalised to any field or ratio · IMPROVE: <code>{FILE}</code> (gallery/data/export_improve_site_quantiles.py, the FED export ftir_51 read) · sites in view: the rows in view</>}
      tip={<>
        {quantity?.methods.length ? <p>Not the same method on both sides: {quantity.methods.join('; ')}. The placement compares methods as much as aerosols.</p> : null}
        {popInfo && <p>Filters: {popInfo.rule}. IMPROVE: {popInfo.n_filters.toLocaleString()} filters at {popInfo.n_sites} sites{ref ? `; ${ref.note}` : ''}</p>}
        <p>The IMPROVE side does not follow the gallery's site, season, date or exclusion filters; the stars do.{quantity && quantity.scale !== 1 ? ` HIPS BC is Fabs / ${m.mac} m²/g on both sides (the reader's MAC).` : ''}</p>
        {ref && <p>{ref.data_use}</p>}
      </>}
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap">
        {!quantity ? <Empty>{field} cannot be ranked against IMPROVE: {(q as { reason: string }).reason}. Pick EC, OC, HIPS Fabs or BC, PM2.5 mass, iron or soil, or a ratio of two of them.</Empty>
          : massClash ? <Empty>The IMPROVE reference is in µg/m³; with Units set to mass per filter, {field} is µg per filter. Switch Units to concentration, or rank a ratio whose units cancel.</Empty>
          : error ? <Empty>The IMPROVE site quantiles are not here ({error}). Rebuild with gallery/data/export_improve_site_quantiles.py.</Empty>
          : !ref ? <Empty>Loading the IMPROVE site quantiles…</Empty>
          : !N ? <Empty>No IMPROVE site has {minN} or more filters with {refLabel} under “{popInfo?.label}”.</Empty>
          : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`} fontFamily={FONT.family}>
              <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} pick="field" format={log ? (v: number) => d3.format('~g')(v) : undefined} />
              <XAxis scale={x} y={innerH} label={`IMPROVE sites, ranked by median ${refLabel}`} format={(v: number) => (Number.isInteger(v) && v >= 1 && v <= N ? String(v) : '')} />
              {lg.show(REF) && <g opacity={lg.hover && lg.hover !== REF ? 0.45 : 1}>
                {ranked.map((s, i) => (
                  <line key={s.code} x1={x(i + 1)} x2={x(i + 1)} y1={y(Math.max(y.domain()[0], s.p25))} y2={y(s.p75)} stroke={IMPROVE_GREY} strokeOpacity={0.35} strokeWidth={1} />
                ))}
                {ranked.map((s, i) => <circle key={s.code} cx={x(i + 1)} cy={y(s.p50)} r={2.3} fill={IMPROVE_GREY} />)}
              </g>}
              {shown.map((p) => (
                <g key={p.site} opacity={lg.dim(p.site)} data-export-color={p.color}>
                  <line x1={x(p.rank)} x2={x(p.rank)} y1={y(Math.max(y.domain()[0], p.p25))} y2={y(p.p75)} stroke={p.color} strokeWidth={1.6} />
                  <path d={star} transform={`translate(${x(p.rank)},${y(p.p50)})`} fill={p.color} stroke={INK.panel} strokeWidth={0.8} />
                  {shown.length <= 8 && (
                    <text x={labels.get(p.site)!.x} y={labels.get(p.site)!.y} textAnchor={labels.get(p.site)!.end ? 'end' : 'start'} fontSize={10.5} fill={INK.text} paintOrder="stroke" stroke={INK.panel} strokeWidth={3}>{p.site}</text>
                  )}
                </g>
              ))}
              <HitLayer items={marks} xy={xy} width={innerW} height={innerH} hit={12}
                onHover={(e, d) => tip.show(e, d.kind === 'ref'
                  ? [`${d.s.name} (${d.s.code}) · IMPROVE`, `median ${val(d.s.p50)} · IQR ${val(d.s.p25)}–${val(d.s.p75)}`, `n = ${d.s.n} · ${rankText(d.rank)}`]
                  : [`${d.site} (${d.code})`, `median ${val(d.p50)} · IQR ${val(d.p25)}–${val(d.p75)}`, `n = ${d.n} · ${d.rank > N ? '' : 'would be '}${rankText(d.rank)}`])}
                onLeave={tip.hide} />
            </g>
          </svg>
        )}
        {quantity && !massClash && N > 0 && (
          <Legend
            items={[
              { label: REF, color: IMPROVE_GREY, shape: 'dot' as const, detail: `${N} sites, median ± IQR` },
              ...placed.map((p) => ({ label: p.site, color: p.color, shape: 'diamond' as const, detail: `${val(p.p50)} · n=${p.n} · ${rankText(p.rank)}` })),
            ]}
            {...lg.props}
            note={unplaced.length ? `not placed (fewer than ${MIN_VIEW_N} filters with ${field}${pop === 'paired' ? ' and OC, EC, Fabs > 0' : ''}): ${unplaced.join(', ')}` : undefined}
          />
        )}
        {tip.node}
      </div>
      <ExcludedNote fields={[field]} />
    </ChartFrame>
  )
}
