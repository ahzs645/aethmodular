import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { Panels } from '@/components/Panels'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlightActions } from '@/lib/highlight'
import { ScatterLayer } from '@/components/ScatterLayer'
import { fidProps } from '@/lib/filterEdits'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { AAE, AAE_440_675, EAE_INV, INV_AOD440, L2_AOD440, SAE_440_675 } from '@/lib/aeronet'
import { CAZORLA_CLASSES, CAZORLA_REGIONS, cazorlaClass, type CazorlaClass } from '@/lib/aeronetMath'
import type { FilterRow, MetaFile } from '@/lib/types'
import { colouring, num } from './shared'
import { toMarks, useMarkReason, useMarkedRows } from '@/lib/markedRows'

const PAIRS = ['Cazorla 440–675', 'Russell / Giles 440–870'] as const
type Pair = (typeof PAIRS)[number]
const M = { top: 12, right: 18, bottom: 46, left: 56 }
// the matrix's classes, shaded from the size axis's coarse end (warm, dust) to its fine end (grey, carbon)
const REGION_FILL: Record<CazorlaClass, string> = {
  'Dust dominated': '#e6a05a', 'OC/Dust mix': '#d9b46c', 'OC dominated': '#a6c36f', 'Dust/EC mix': '#c9a27e',
  'Mix': '#b8b8b8', 'EC/OC mixture': '#8fa3b8', 'Coated large particles': '#c2b4d6', 'EC dominated': '#6b7b8c',
}

/**
 * Aerosol typing from the inversion's spectral shapes: each filter window's mean absorption Ångström
 * exponent against its size exponent, coloured by site (by season with one site). Two literature frames:
 * Cazorla et al. (2013)'s Ångström matrix on 440–675 nm (AAE against the scattering exponent, classes
 * shaded), and the 440–870 nm plane Russell et al. (2010) and Giles et al. (2012) read (AAE near 1 for
 * black carbon, larger for dust; extinction AE 0.8 between mainly large and mainly small particles).
 */
export function AeronetTyping({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const tip = useTooltip(wrapRef)
  const [pair, setPair] = useSearchParam<Pair>('aeronet.typing', PAIRS[0], PAIRS)
  const cazorla = pair === PAIRS[0]
  const xField = cazorla ? SAE_440_675 : EAE_INV, yField = cazorla ? AAE_440_675 : AAE
  const col = useMemo(() => colouring(rows, meta), [rows, meta])
  const pts = useMemo(() => rows.filter((r) => num(r[xField]) && num(r[yField]))
    .map((r) => ({ row: r, x: r[xField] as number, y: r[yField] as number, weak: !(num(r[INV_AOD440]) && (r[INV_AOD440] as number) >= L2_AOD440) })), [rows, xField, yField])

  // the class shares per site (or season), on the Cazorla matrix
  const shares = useMemo(() => {
    if (!cazorla) return []
    return col.order(pts.map((p) => p.row)).map((k) => {
      const mine = pts.filter((p) => col.key(p.row) === k)
      const counts = d3.rollup(mine, (v) => v.length, (p) => cazorlaClass(p.x, p.y))
      return { k, n: mine.length, counts }
    })
  }, [pts, col, cazorla])

  const xDom: [number, number] = [Math.min(-0.2, d3.min(pts, (p) => p.x) ?? 0), Math.max(2.6, d3.max(pts, (p) => p.x) ?? 2.5)]
  const yDom: [number, number] = [Math.min(0, d3.min(pts, (p) => p.y) ?? 0), Math.max(3, d3.max(pts, (p) => p.y) ?? 3)]

  const legendKeys = col.order(pts.map((p) => p.row))
  return (
    <ChartFrame id="aeronet-typing" title="Aerosol typing — absorption against size Ångström exponents"
      subtitle={cazorla
        ? `Each filter window's mean AAE 440–675 against its scattering Ångström exponent (SAE, from AOD × SSA at 440 and 675 nm), per inversion then averaged. Shaded: the Ångström matrix of Cazorla et al. (2013, Fig. 2), cut at 1.0 and 1.5 on both axes; the middle cell is "Mix" (their diagonal through it is drawn, not used). Hollow: window inversion AOD 440 under ${L2_AOD440}, where absorption is uncertain.`
        : `Each filter window's mean AAE 440–870 against the inversion's extinction Ångström exponent 440–870. AAE 1 is black carbon's λ⁻¹ (urban-industrial clusters near it, Russell et al. 2010); extinction AE 0.8 splits mainly large from mainly small particles, and the shaded box is Giles et al. (2012)'s dust cluster (AAE 1.5–2.3, coarse). Hollow: window inversion AOD 440 under ${L2_AOD440}.`}
      provenance="Cazorla et al. (2013) ACP 13, 9337; Russell et al. (2010) ACP 10, 1155; Giles et al. (2012) JGR 117, D17203 · gallery/data/export_aeronet.py derive()"
      controls={<Segmented label="Frame" value={pair} options={PAIRS} onChange={setPair} optionLabel={(o) => o} />}>
      <div ref={wrapRef} className="chart-wrap aeronet-typing">
        {pts.length === 0 ? <Empty>No filter in view has an inversion with both exponents in its window.</Empty> : (
          <Panels label="Aerosol typing" panels={[
            { key: 'plane', title: cazorla ? 'Ångström matrix, 440–675 nm' : 'AAE against extinction AE, 440–870 nm', min: 320, basis: 594, max: 594, grow: 0, render: ({ width, height, expanded }) => {
              // square: as wide as the panel allows, and in a tile no taller than its share
              const side = Math.max(240, Math.min(expanded ? 760 : 520, width - M.left - M.right, height === null ? Infinity : height - M.top - M.bottom))
              const x = d3.scaleLinear().domain(xDom).range([0, side]).nice()
              const y = d3.scaleLinear().domain(yDom).range([side, 0]).nice()
              const clampX = (v: number) => x(Math.max(x.domain()[0], Math.min(x.domain()[1], v)))
              const clampY = (v: number) => y(Math.max(y.domain()[0], Math.min(y.domain()[1], v)))
              const clip = `aeronet-typing-clip-${expanded ? 'big' : 'page'}`
              return (
                <svg width={side + M.left + M.right} height={side + M.top + M.bottom} fontFamily={FONT.family}>
                  <defs><clipPath id={clip}><rect width={side} height={side} /></clipPath></defs>
                  <g transform={`translate(${M.left},${M.top})`}>
                    <g clipPath={`url(#${clip})`}>
                      {cazorla && CAZORLA_CLASSES.map((c) => {
                        const [x0, x1, y0, y1] = CAZORLA_REGIONS[c]
                        const rx = clampX(x0), rw = clampX(x1) - rx, ry = clampY(y1), rh = clampY(y0) - ry
                        return <g key={c}>
                          <rect x={rx} y={ry} width={rw} height={rh} fill={REGION_FILL[c]} fillOpacity={0.16} />
                          <text x={rx + rw / 2} y={ry + rh / 2 + 10} textAnchor="middle" dy="0.32em" fontSize={side < 400 ? 8.5 : 10.5} fill={INK.muted} fontWeight={600} pointerEvents="none">{c}</text>
                        </g>
                      })}
                      {cazorla && <line x1={x(1)} y1={y(1)} x2={x(1.5)} y2={y(1.5)} stroke={INK.muted} strokeDasharray="3 3" />}
                      {!cazorla && <>
                        <rect x={clampX(-10)} width={clampX(0.8) - clampX(-10)} y={y(2.3)} height={y(1.5) - y(2.3)} fill="#e6a05a" fillOpacity={0.18} />
                        <text x={clampX(-10) + 6} y={y(2.3) + 14} fontSize={10.5} fill={INK.muted} fontWeight={600}>Giles dust cluster</text>
                        <line x1={0} x2={side} y1={y(1)} y2={y(1)} stroke={INK.muted} strokeDasharray="4 3" />
                        <text x={side - 4} y={y(1) - 5} textAnchor="end" fontSize={10.5} fill={INK.muted}>AAE 1: black carbon (λ⁻¹)</text>
                        <line x1={x(0.8)} x2={x(0.8)} y1={0} y2={side} stroke={INK.muted} strokeDasharray="4 3" />
                        <text x={x(0.8) + 4} y={side - 6} fontSize={10.5} fill={INK.muted}>EAE 0.8: large | small</text>
                      </>}
                    </g>
                    <YAxis scale={y} x={0} gridWidth={side} tickCount={6} label={cazorla ? 'AAE 440–675' : 'AAE 440–870'} labelX={-40} />
                    <XAxis scale={x} y={side} tickCount={6} label={cazorla ? 'Scattering Ångström exponent 440–675' : 'Extinction Ångström exponent 440–870 (inversion)'} labelDy={38} />
                    {cazorla && [1, 1.5].map((v) => <g key={v}>
                      <line x1={0} x2={side} y1={y(v)} y2={y(v)} stroke={INK.muted} strokeDasharray="3 3" />
                      {/* SAE 1.0 bounds only the cells above AAE 1 (coated large particles span SAE < 1.5) */}
                      <line x1={x(v)} x2={x(v)} y1={0} y2={v === 1 ? y(1) : side} stroke={INK.muted} strokeDasharray="3 3" />
                    </g>)}
                    <TypingPoints pts={pts} x={x} y={y} side={side} col={col} cazorla={cazorla} tip={tip} fields={[xField, yField]} sites={rows} />
                  </g>
                </svg>
              )
            } },
            ...(cazorla && shares.length > 0 ? [{ key: 'shares', title: `Class shares by ${col.by === 'site' ? 'site' : 'season'}`, min: 300, basis: 420, expandable: false, render: () => (
                  <div style={{ overflowX: 'auto', maxWidth: '100%' }}><table className="aeronet-share-table" style={{ width: 'max-content', fontSize: 11.5, borderCollapse: 'collapse', fontFamily: FONT.family, color: INK.text }}>
                    <thead><tr><th style={{ textAlign: 'left', padding: '2px 6px' }}>{col.by === 'site' ? 'Site' : 'Season'}</th><th style={{ padding: '2px 6px' }}>n</th>
                      {CAZORLA_CLASSES.map((c) => <th key={c} style={{ padding: '2px 4px', fontWeight: 500, color: INK.muted, maxWidth: 64 }}>{c}</th>)}</tr></thead>
                    <tbody>{shares.map((s) => (
                      <tr key={s.k}>
                        <td style={{ padding: '2px 6px', whiteSpace: 'nowrap' }}><span className="swatch" style={{ background: col.color(s.k), display: 'inline-block', width: 9, height: 9, borderRadius: 5, marginRight: 5 }} />{col.label(s.k)}</td>
                        <td style={{ padding: '2px 6px', fontFamily: FONT.mono, textAlign: 'right' }}>{s.n}</td>
                        {CAZORLA_CLASSES.map((c) => {
                          const share = (s.counts.get(c) ?? 0) / s.n
                          return <td key={c} style={{ padding: '2px 4px', textAlign: 'right', fontFamily: FONT.mono, background: share ? `rgba(33,113,181,${0.08 + share * 0.6})` : undefined }}>{share ? `${Math.round(share * 100)}%` : '·'}</td>
                        })}
                      </tr>
                    ))}</tbody>
                  </table></div>
            ) }] : []),
          ]} />
        )}
        <div className="legend">
          {legendKeys.map((k) => <span key={k} className="legend-item"><span className="swatch" style={{ background: col.color(k) }} />{col.label(k)}</span>)}
          <span className="legend-item"><span className="swatch ring" style={{ color: INK.muted }} />inversion AOD 440 &lt; {L2_AOD440}</span>
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}

type TypingPoint = { row: FilterRow; x: number; y: number; weak: boolean }

/**
 * The plane's filters (components/ScatterLayer): drawn once per frame and range, so a hover only draws the focused
 * filter over them; a point outside the frame sits on its edge.
 */
function TypingPoints({ pts, x, y, side, col, cazorla, tip, fields: [xf, yf], sites }: {
  pts: TypingPoint[]
  /** the plane's two fields, and the rows whose sites it draws: for the excluded filters' marks (Filters → excluded: Mark) */
  fields: [string, string]; sites: FilterRow[]
  x: d3.ScaleLinear<number, number>; y: d3.ScaleLinear<number, number>; side: number
  col: ReturnType<typeof colouring>; cazorla: boolean; tip: ReturnType<typeof useTooltip>
}) {
  const hl = useHighlightActions()
  const xy = useMemo(() => {
    const [x0, x1] = x.domain(), [y0, y1] = y.domain()
    return pts.map((p) => [x(Math.max(x0, Math.min(x1, p.x))), y(Math.max(y0, Math.min(y1, p.y)))] as const)
  }, [pts, x.domain().join(), y.domain().join(), side]) // eslint-disable-line react-hooks/exhaustive-deps
  const dots = useMemo(() => pts.map((p, i) => {
    const st = focusStyle(p.row.id, null, { r: 3.4 }, hl.selected)
    const c = col.color(col.key(p.row))
    return <circle key={p.row.id} cx={xy[i][0]} cy={xy[i][1]} r={st.r} fill={p.weak ? 'none' : c} fillOpacity={st.opacity}
      stroke={p.weak || !st.stroke ? c : st.stroke} strokeWidth={st.strokeWidth ?? 1} data-dot={p.weak ? undefined : st.dot} {...fidProps(p.row.id)} />
  }), [pts, xy, col, hl.selected])
  const focusR = focusStyle('', '', { r: 3.4 }).r
  const markedAll = useMarkedRows()
  const markReason = useMarkReason()
  const marks = useMemo(() => {
    const here = new Set(sites.map((r) => r.site)), [x0, x1] = x.domain(), [y0, y1] = y.domain()
    return toMarks(markedAll.filter((r) => here.has(r.site)), (r) => {
      const a = r[xf], b = r[yf]
      return typeof a === 'number' && typeof b === 'number' && a >= x0 && a <= x1 && b >= y0 && b <= y1 ? [x(a), y(b)] : null
    }, markReason, (r) => [`${xf} ${fmt(r[xf] as number, 2)} · ${yf} ${fmt(r[yf] as number, 2)}`])
  }, [markedAll, sites, xf, yf, x.domain().join(), y.domain().join(), side, markReason]) // eslint-disable-line react-hooks/exhaustive-deps
  return <ScatterLayer items={pts} xy={xy} idOf={(p) => p.row.id} dots={dots} width={side} height={side} tip={tip} focusR={focusR} marks={marks}
    focus={(p, [cx, cy]) => {
      const st = focusStyle(p.row.id, p.row.id, { r: 3.4 })
      const c = col.color(col.key(p.row))
      return <circle cx={cx} cy={cy} r={st.r} fill={p.weak ? 'none' : c} fillOpacity={st.opacity} stroke={p.weak ? c : st.stroke} strokeWidth={st.strokeWidth} />
    }}
    tipOf={(p) => [p.row.id, `${p.row.site} · ${p.row.date}`, `${cazorla ? 'SAE 440–675' : 'EAE 440–870'} ${fmt(p.x, 2)} · ${cazorla ? 'AAE 440–675' : 'AAE 440–870'} ${fmt(p.y, 2)}`,
      ...(cazorla ? [`Ångström matrix: ${cazorlaClass(p.x, p.y)}`] : []), `inversion AOD 440 ${fmt(p.row[INV_AOD440] as number | null, 2)}`, 'click for details']} />
}
