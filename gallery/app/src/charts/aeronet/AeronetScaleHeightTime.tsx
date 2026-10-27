import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusLast, focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps } from '@/lib/filterEdits'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { seasonOnly } from '@/lib/seasons'
import { AAOD675, HIPS_LOT_PERIODS, lotPeriodOf, scaleHeight, surfaceFabs } from '@/lib/aeronet'
import type { FilterRow, MetaFile } from '@/lib/types'

const BY = ['Lot period', 'Year', 'Season'] as const
type By = (typeof BY)[number]
const LOT_COLOURS = ['#1b9e77', '#bdbdbd', '#d95f02', '#7570b3', '#e7298a']
const M = { top: 10, right: 18, bottom: 40, left: 62 }

/**
 * The absorption scale height H = AAOD 675 / (surface Fabs × 10⁻⁶) filter by filter through time, on a log
 * axis, with each group's median as a step. One site: coloured by HIPS filter lot where the repo documents the
 * lot periods (Addis), else by year or season; several sites: by site, a median per site and year. A step that
 * lines up with a lot or calibration change, not with the seasons, would point at the measurement, not the air.
 */
export function AeronetScaleHeightTime({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const pts = useMemo(() => rows.flatMap((r) => {
    const h = scaleHeight(r)
    return h !== null && h > 0 ? [{ row: r, t: new Date(r.date), h }] : []
  }), [rows])
  const codes = [...new Set(pts.map((p) => p.row.code))]
  const one = codes.length === 1
  const hasLots = one && !!HIPS_LOT_PERIODS[codes[0]]
  const [byParam, setBy] = useSearchParam<By>('aeronet.h.by', 'Lot period', BY)
  const options = BY.filter((b) => b !== 'Lot period' || hasLots)
  const by: By | 'Site' = !one ? 'Site' : options.includes(byParam) ? byParam : 'Year'

  const groupOf = (r: FilterRow): string => by === 'Site' ? r.site : by === 'Year' ? r.date.slice(0, 4)
    : by === 'Season' ? String(r.season ?? '') : lotPeriodOf(r.code, r.date) ?? 'No documented lot'
  const groups = useMemo(() => {
    const keys = [...new Set(pts.map((p) => groupOf(p.row)))]
    if (by === 'Site') return keys.sort((a, b) => meta.sites.findIndex((s) => s.name === a) - meta.sites.findIndex((s) => s.name === b))
    if (by === 'Season') return keys.sort((a, b) => meta.seasons.findIndex((s) => s.name === a) - meta.seasons.findIndex((s) => s.name === b))
    if (by === 'Lot period') return [...(HIPS_LOT_PERIODS[codes[0]] ?? []).map((p) => p.label), 'No documented lot'].filter((k) => keys.includes(k))
    return keys.sort()
  }, [pts, by]) // eslint-disable-line react-hooks/exhaustive-deps
  const yearRamp = d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(1, groups.length - 1)])
  const colorOf = (g: string) => by === 'Site' ? meta.sites.find((s) => s.name === g)?.color ?? INK.muted
    : by === 'Season' ? meta.seasons.find((s) => s.name === g)?.color ?? INK.muted
      : by === 'Lot period' ? (g === 'No documented lot' ? INK.neutral : LOT_COLOURS[groups.indexOf(g) % LOT_COLOURS.length])
        : yearRamp(groups.indexOf(g))
  const label = (g: string) => (by === 'Season' ? seasonOnly(g) : g)

  // medians as steps: per group (lot, year), or per group and year (site, season)
  const steps = useMemo(() => {
    const perYear = by === 'Site' || by === 'Season'
    return d3.groups(pts, (p) => `${groupOf(p.row)}${perYear ? `|${p.row.date.slice(0, 4)}` : ''}`).flatMap(([k, ps]) => ps.length < 3 ? [] : [{
      g: k.split('|')[0], a: d3.min(ps, (p) => p.t)!, b: d3.max(ps, (p) => p.t)!, med: d3.median(ps, (p) => p.h)!, n: ps.length,
    }])
  }, [pts, by]) // eslint-disable-line react-hooks/exhaustive-deps

  const iw = Math.max(260, width - M.left - M.right), ih = 260
  const x = d3.scaleTime().domain(d3.extent(pts, (p) => p.t) as [Date, Date]).range([0, iw]).nice()
  const y = d3.scaleLog().domain([Math.max(10, (d3.min(pts, (p) => p.h) ?? 100) * 0.8), (d3.max(pts, (p) => p.h) ?? 1000) * 1.25]).range([ih, 0])
  const yTicks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].filter((v) => v >= y.domain()[0] && v <= y.domain()[1])

  return (
    <ChartFrame id="aeronet-scale-height-time" title={`Absorption scale height through time — by ${by.toLowerCase()}`}
      subtitle={`H = AAOD 675 (window-mean inversion) ÷ surface Fabs (HIPS at 633 nm; public BC × 10 where HIPS Fabs is absent), one dot per filter, log axis; lines are each ${by === 'Site' || by === 'Season' ? `${by.toLowerCase()}'s median per calendar year` : `${by.toLowerCase()}'s median`} (3 filters or more). ${hasLots ? 'Lot periods are the sampling-date ranges in research/ftir_ec_phase3/SPARTAN_LOT_INVENTORY_2026-08-23.md (to the month; the gallery has no per-filter lot). ' : one ? 'No HIPS lot periods are documented for this site, so the colours are years or seasons. ' : ''}A step at a lot change and not at a season change points at the measurement, not the air.`}
      provenance="research/ftir_ec_phase3/AERONET_SPARTAN_2026-08-22.md (scale height); research/ftir_ec_phase3/SPARTAN_LOT_INVENTORY_2026-08-23.md (HIPS lots)"
      controls={one ? <Segmented label="Colour" value={by as By} options={options} onChange={setBy} optionLabel={(o) => o} /> : undefined}>
      <div ref={wrapRef} className="chart-wrap">
        {pts.length < 3 ? <Empty>Fewer than 3 filters in view have both an AERONET AAOD 675 and a surface Fabs.</Empty> : (
          <svg width={iw + M.left + M.right} height={ih + M.top + M.bottom} fontFamily={FONT.family}>
            <g transform={`translate(${M.left},${M.top})`}>
              {yTicks.map((t) => <g key={t}>
                <line x1={0} x2={iw} y1={y(t)} y2={y(t)} stroke={INK.grid} />
                <text x={-6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={INK.axis}>{t >= 1000 ? `${t / 1000}k` : t}</text>
              </g>)}
              <text transform={`translate(-48,${ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill={INK.text}>Scale height H (m, log)</text>
              <XAxis scale={x} y={ih} tickCount={8} format={d3.utcFormat('%b %Y')} />
              {focusLast(pts, (p) => p.row.id, hl.focusId).map(([p]) => {
                const st = focusStyle(p.row.id, hl.focusId, { r: 3, opacity: 0.7 }, hl.selected)
                const c = colorOf(groupOf(p.row))
                return <circle key={p.row.id} cx={x(p.t)} cy={y(p.h)} r={st.r} fill={c} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth}
                  style={{ cursor: 'pointer' }} {...fidProps(p.row.id)}
                  onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.site} · ${p.row.date}`, `H ${fmt(p.h, 0)} m`,
                    `AAOD 675 ${fmt(p.row[AAOD675] as number, 4)} · Fabs ${fmt(surfaceFabs(p.row), 1)} Mm⁻¹`, label(groupOf(p.row)), 'click for details']) }}
                  onMouseLeave={() => { hl.setHover(null); tip.hide() }} onClick={() => hl.openSample(p.row.id)} />
              })}
              {steps.map((s, i) => <line key={i} x1={x(s.a)} x2={Math.max(x(s.b), x(s.a) + 4)} y1={y(s.med)} y2={y(s.med)} stroke={colorOf(s.g)} strokeWidth={3} strokeLinecap="round"
                onMouseEnter={(e) => tip.show(e, [label(s.g), `median H ${fmt(s.med, 0)} m`, `${s.n} filters, ${d3.utcFormat('%b %Y')(s.a)} – ${d3.utcFormat('%b %Y')(s.b)}`])} onMouseLeave={tip.hide} />)}
            </g>
          </svg>
        )}
        <div className="legend">
          {groups.map((g) => <span key={g} className="legend-item"><span className="swatch" style={{ background: colorOf(g) }} />{label(g)}</span>)}
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
