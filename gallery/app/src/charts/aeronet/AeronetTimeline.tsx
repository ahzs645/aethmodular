import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusLast, focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps } from '@/lib/filterEdits'
import { fmt } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { loadDailyAeronet, L2_AOD440, type AeronetLevel, type DailyAeronet } from '@/lib/aeronet'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { useFitHeight } from '@/lib/chartFit'

const M = { top: 8, right: 20, bottom: 30, left: 68 }
const PANEL = 92, GAP = 14
type Panel = 'aod' | 'fmf' | 'aaod' | 'ssa' | 'field'
const TITLES: Record<Exclude<Panel, 'field'>, string> = { aod: 'AOD 500', fmf: 'Fine fraction', aaod: 'AAOD 675', ssa: 'SSA 440' }

/**
 * The column over one site's filters: AERONET's daily AOD at 500 nm and its fine-mode fraction,
 * the inversion's absorption AOD at 675 nm and single-scattering albedo at 440 nm (open circles
 * where the inversion's AOD 440 is under 0.4, AERONET's Level 2 threshold for absorption), above
 * the chosen measurement for each filter on the same axis, with the site's seasons shaded.
 */
export function AeronetTimeline({ rows, meta, field, site, station, level = '15' }: {
  rows: FilterRow[]; meta: MetaFile; field: string; site: { name: string; code: string } | null; station?: string; level?: AeronetLevel
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const [daily, setDaily] = useState<DailyAeronet | null>(null)
  const [error, setError] = useState('')
  const [cursor, setCursor] = useState<number | null>(null)

  useEffect(() => {
    if (!site) return
    let live = true
    setDaily(null); setError('')
    loadDailyAeronet(site.code, level).then((d) => { if (live) setDaily(d) }).catch(() => { if (live) setError(`No daily AERONET exported for ${site.name}.`) })
    return () => { live = false }
  }, [site, level])

  const pts = useMemo(() => rows.filter((r) => r.site === site?.name && typeof r[field] === 'number')
    .map((r) => ({ row: r, t: new Date(r.date), v: r[field] as number })), [rows, site, field])
  const domain = useMemo<[Date, Date] | null>(() => {
    const ds = rows.filter((r) => r.site === site?.name).map((r) => new Date(r.date).getTime())
    if (!ds.length) return null
    return [new Date(Math.min(...ds) - 14 * 864e5), new Date(Math.max(...ds) + 14 * 864e5)]
  }, [rows, site])
  const days = useMemo(() => {
    if (!daily || !domain) return []
    return daily.date.map((d, i) => ({ t: new Date(d), i })).filter((x) => x.t >= domain[0] && x.t <= domain[1])
  }, [daily, domain])

  const panels: Panel[] = ['aod', 'fmf', 'aaod', 'ssa', 'field']
  const innerW = Math.max(240, width - M.left - M.right)
  const fit = useFitHeight()
  const panelH = fit === null ? PANEL : Math.max(40, (fit - M.top - M.bottom - (panels.length - 1) * GAP) / panels.length)
  const height = M.top + M.bottom + panels.length * panelH + (panels.length - 1) * GAP
  const x = d3.scaleTime().domain(domain ?? [new Date(), new Date()]).range([0, innerW])
  const top = (k: Panel) => M.top + panels.indexOf(k) * (panelH + GAP)
  const val = (arr: (number | null)[] | undefined, i: number) => arr?.[i] ?? null
  const ext = (arr: (number | null)[] | undefined, lo?: number, hi?: number) => {
    const vs = days.map(({ i }) => val(arr, i)).filter((v): v is number => v !== null)
    const a = lo ?? d3.min(vs) ?? 0, b = hi ?? d3.max(vs) ?? 1
    return d3.scaleLinear().domain([a, b === a ? a + 1 : b]).range([panelH, 0]).nice()
  }
  const y: Record<Exclude<Panel, 'field'>, d3.ScaleLinear<number, number>> = {
    aod: ext(daily?.aod500, 0), fmf: d3.scaleLinear().domain([0, 1]).range([panelH, 0]), aaod: ext(daily?.aaod675, 0), ssa: ext(daily?.ssa440, undefined, 1),
  }
  const yF = d3.scaleLinear().domain([Math.min(0, d3.min(pts, (p) => p.v) ?? 0), (d3.max(pts, (p) => p.v) ?? 1) * 1.05]).range([panelH, 0]).nice()
  const line = (arr: (number | null)[] | undefined, s: d3.ScaleLinear<number, number>) =>
    d3.line<{ t: Date; i: number }>().defined(({ i }) => val(arr, i) !== null).x(({ t }) => x(t)).y(({ i }) => s(val(arr, i)!))(days) ?? ''
  // the inversion's own AOD 440 is not in the daily series; the daily AOD 440 stands in for the threshold
  const weak = (i: number) => (val(daily?.aod440, i) ?? 0) < L2_AOD440

  const seasons = site ? seasonsForSite(meta, site.name) : []
  const runs = useMemo(() => {
    const out: { a: Date; b: Date; color: string; name: string }[] = []
    for (const { t } of days) {
      const se = seasons.find((s) => s.months.includes(t.getUTCMonth() + 1))
      const last = out[out.length - 1]
      if (se && last && last.name === se.name) last.b = t
      else if (se) out.push({ a: t, b: t, color: se.color, name: se.name })
    }
    return out
  }, [days, seasons])
  const at = cursor === null ? null : days[d3.bisector((d: { t: Date }) => d.t).center(days, x.invert(cursor))]

  return (
    <ChartFrame id="aeronet-timeline" title={`${site ? `${site.name} · ` : ''}AERONET timeline`}
      subtitle={`AERONET's ${level === '20' ? 'Level 2.0' : 'Level 1.5'} daily means${station ? ` at ${station}` : ''}: AOD at 500 nm and its fine-mode fraction (direct sun and SDA), absorption AOD at 675 nm and single-scattering albedo at 440 nm (almucantar inversions; open circles on days with AOD 440 under ${L2_AOD440}, where absorption is uncertain), above the chosen measurement for each filter. Seasons are the site's local calendar.`}
      provenance="addis_04_aeronet and aeronet_addis_diagnostics generalised to every SPARTAN site · NASA AERONET Version 3">
      <div ref={wrapRef} className="chart-wrap">
        {error ? <Empty>{error}</Empty> : !site || !domain ? <Empty>No filters at this site in the current subset.</Empty> : !daily ? <Empty>Loading the daily AERONET…</Empty> : (
          <svg width={width} height={height} onMouseLeave={() => { setCursor(null); tip.hide() }}>
            <g transform={`translate(${M.left},0)`}>
              {panels.map((k) => (
                <g key={k} transform={`translate(0,${top(k)})`}>
                  {runs.map((r, i) => <rect key={i} x={x(r.a)} width={Math.max(1, x(r.b) - x(r.a) + innerW / Math.max(1, days.length))} height={panelH} fill={r.color} opacity={0.1} />)}
                  <YAxis scale={k === 'field' ? yF : y[k]} x={0} gridWidth={innerW} tickCount={3}
                    label={k === 'field' ? withUnit(field, meta.field_units) : TITLES[k]} labelX={-50} pick={k === 'field' ? 'field' : undefined} />
                  {k === 'aod' && <path d={line(daily.aod500, y.aod)} fill="none" stroke="#2c7fb8" strokeWidth={1.1} />}
                  {k === 'fmf' && <path d={line(daily.fmf, y.fmf)} fill="none" stroke="#41ab5d" strokeWidth={1.1} />}
                  {(k === 'aaod' || k === 'ssa') && days.map(({ t, i }) => {
                    const v = val(k === 'aaod' ? daily.aaod675 : daily.ssa440, i)
                    if (v === null) return null
                    const c = k === 'aaod' ? '#8c2d04' : '#6a51a3'
                    return <circle key={i} cx={x(t)} cy={y[k](v)} r={2.2} fill={weak(i) ? 'none' : c} stroke={c} strokeWidth={0.8} opacity={0.8} />
                  })}
                  {k === 'field' && focusLast(pts, (p) => p.row.id, hl.focusId).map(([p]) => {
                    const st = focusStyle(p.row.id, hl.focusId, { r: 3, opacity: 0.8 }, hl.selected)
                    return <circle key={p.row.id} cx={x(p.t)} cy={yF(p.v)} r={st.r} fill={meta.seasons.find((s) => s.name === p.row.season)?.color ?? INK.text}
                      fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} style={{ cursor: 'pointer' }} {...fidProps(p.row.id)}
                      onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.date} · ${seasonOnly(String(p.row.season ?? ''))}`, `${field}: ${fmt(p.v)}`, 'click for details · right-click to exclude']) }}
                      onMouseLeave={() => { hl.setHover(null); tip.hide() }} onClick={() => hl.openSample(p.row.id)} />
                  })}
                  {k !== 'field' && <rect width={innerW} height={panelH} fill="transparent"
                    onMouseMove={(e) => {
                      const mx = d3.pointer(e)[0]
                      setCursor(mx)
                      const d = days[d3.bisector((q: { t: Date }) => q.t).center(days, x.invert(mx))]
                      if (d) tip.show(e, [d3.utcFormat('%d %b %Y')(d.t), `AOD 500 ${fmt(daily.aod500[d.i], 3)} · AE ${fmt(daily.ae[d.i], 2)} · fine fraction ${fmt(daily.fmf[d.i], 2)}`,
                        `AAOD 675 ${fmt(daily.aaod675[d.i], 4)} · AAE ${fmt(daily.aae[d.i], 2)} · SSA 440 ${fmt(daily.ssa440[d.i], 3)}`, `precipitable water ${fmt(daily.pw[d.i], 2)} cm`])
                    }} />}
                </g>
              ))}
              {at && <line x1={x(at.t)} x2={x(at.t)} y1={M.top} y2={height - M.bottom} stroke={INK.text} strokeOpacity={0.35} strokeDasharray="3 3" pointerEvents="none" />}
              <XAxis scale={x} y={height - M.bottom} tickCount={8} format={d3.utcFormat('%b %Y')} />
            </g>
          </svg>
        )}
        <div className="legend">
          {seasons.map((s) => <span key={s.name} className="legend-item"><span className="swatch band" style={{ background: s.color }} />{s.name.replace(`${site?.name} · `, '')}</span>)}
          <span className="legend-item"><span className="swatch" style={{ background: '#8c2d04' }} />inversion, AOD 440 ≥ {L2_AOD440}</span>
          <span className="legend-item"><span className="swatch ring" style={{ color: '#8c2d04' }} />AOD 440 &lt; {L2_AOD440}</span>
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
