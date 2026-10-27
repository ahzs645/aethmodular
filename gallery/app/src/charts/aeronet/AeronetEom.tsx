import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { AE, AOD500, DAOD, DSSA, FINE_AOD, FMF } from '@/lib/aeronet'
import { exclusionCurve, fineSoil } from '@/lib/aeronetMath'
import type { FilterRow, MetaFile } from '@/lib/types'
import { BC_CHOICES } from './AeronetNonAbsorbing'
import { colouring, num, recordOf } from './shared'

const COLUMN = [
  { key: AOD500, label: 'AOD 500', color: '#2c7fb8' },
  { key: AE, label: 'Ångström exponent 440–870', color: '#d95f02' },
  { key: FMF, label: 'Fine-mode fraction', color: '#41ab5d' },
  { key: FINE_AOD, label: 'Fine-mode AOD 500', color: '#7570b3' },
]
const STEPS = [0, 20, 40, 60, 80]
const M = { top: 14, right: 18, bottom: 46, left: 56 }

/**
 * The surface–column tests of Eom et al. (2025), as notebooks/analysis/surface_columnar_linkage.ipynb runs
 * them on the filters: the spectral AOD difference and the SSA difference (440 − 870 nm) against BC minus fine
 * soil (dust-dominated days to the left, BC-dominated to the right), and the sequential exclusion of the
 * lowest BC/AOD filters, after which surface BC should track the column better if those were the days with
 * aerosol aloft.
 */
export function AeronetEom({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const present = BC_CHOICES.filter((f) => rows.some((r) => num(r[f])))
  const [bcParam, setBc] = useSearchParam<string>('aeronet.eom.bc', BC_CHOICES[0], BC_CHOICES)
  const bc = present.includes(bcParam as (typeof BC_CHOICES)[number]) ? bcParam : present[0] ?? BC_CHOICES[0]
  const col = useMemo(() => colouring(rows, meta), [rows, meta])

  const diff = (yField: string, label: string): CrossPoint[] => rows.flatMap((r) => {
    const fs = fineSoil(r), b = r[bc], v = r[yField]
    if (fs === null || !num(b) || !num(v)) return []
    return [{ x: b - fs, y: v, color: col.color(col.key(r)), id: r.id,
      tip: [r.id, `${r.site} · ${r.date}`, `${bc} − fine soil ${fmt(b - fs, 2)} µg/m³ (fine soil ${fmt(fs, 2)})`, `${label} ${fmt(v, 3)}`],
      rec: recordOf(r, 'AERONET · Eom et al. tests', [[bc, b], ['Fine soil (Malm 1994)', +fs.toFixed(3)], [`${bc} − fine soil`, +(b - fs).toFixed(3)], [label, v]]) }]
  })
  const dAod = useMemo(() => diff(DAOD, 'dAOD 440 − 870'), [rows, bc, col]) // eslint-disable-line react-hooks/exhaustive-deps
  const dSsa = useMemo(() => diff(DSSA, 'dSSA 440 − 870'), [rows, bc, col]) // eslint-disable-line react-hooks/exhaustive-deps

  const steps = useMemo(() => exclusionCurve(rows.flatMap((r) => {
    const b = r[bc], aod = r[AOD500]
    if (!num(b) || !num(aod) || aod <= 0) return []
    return [{ ratio: b / aod, x: b, ys: Object.fromEntries(COLUMN.map((c) => [c.key, num(r[c.key]) ? (r[c.key] as number) : null])) }]
  }), COLUMN.map((c) => c.key), STEPS), [rows, bc])

  const iw = Math.max(240, Math.min(width - M.left - M.right, 520)), ih = 240
  const x = d3.scalePoint<number>().domain(STEPS).range([0, iw]).padding(0.3)
  const rs = steps.flatMap((s) => COLUMN.map((c) => s.r[c.key]?.r)).filter(num)
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(rs) ?? 0), Math.max(0.2, d3.max(rs) ?? 1)]).range([ih, 0]).nice()
  const xs = (v: number) => x(v) ?? 0

  return (
    <ChartFrame id="aeronet-eom" title="Surface against column — the Eom et al. (2025) tests"
      subtitle={`(a, b) dAOD = AOD 440 − AOD 870 and dSSA = SSA 440 − SSA 870 (per retrieval, then the window mean) against ${bc} minus fine soil (Malm 1994: 2.20 Al + 2.49 Si + 1.63 Ca + 2.42 Fe + 1.94 Ti, elemental Ca where measured). (c) Pearson r between ${bc} and each column quantity after dropping the filters whose ${bc} / AOD 500 is below the 0th, 20th … 80th percentile of the filters in view (pooled over the sites picked).`}
      provenance="notebooks/analysis/surface_columnar_linkage.ipynb sections 1, 3 and 8 (Eom et al. 2025: BC/AOD exclusion, dAOD and dSSA vs BC − fine soil); the notebook used aethalometer BC for the exclusion and FTIR EC for BC − FS"
      controls={<Select label="BC" value={bc} options={present.length ? [...present] : [...BC_CHOICES]} onChange={setBc} />}>
      <div ref={wrapRef} className="chart-wrap">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
          {dAod.length ? <Crossplot title={`(a) dAOD vs ${bc} − fine soil (n = ${dAod.length})`} points={dAod} xLabel={`${bc} − fine soil (µg/m³)`} yLabel="dAOD 440 − 870" sameScale={false} maxSide={320} />
            : <Empty>(a) No filter in view has fine soil, {bc} and a direct-sun AOD at 440 and 870 nm.</Empty>}
          {dSsa.length ? <Crossplot title={`(b) dSSA vs ${bc} − fine soil (n = ${dSsa.length})`} points={dSsa} xLabel={`${bc} − fine soil (µg/m³)`} yLabel="dSSA 440 − 870" sameScale={false} maxSide={320} />
            : <Empty>(b) No filter in view has fine soil, {bc} and an inversion SSA.</Empty>}
        </div>
        {steps.length === 0 ? <Empty>(c) Fewer than 5 filters in view have {bc} and AOD 500.</Empty> : (
          <svg width={iw + M.left + M.right} height={ih + M.top + M.bottom + 16} fontFamily={FONT.family} style={{ display: 'block', margin: '8px auto 0' }}>
            <text x={M.left} y={12} fontSize={12.5} fontWeight={600} fill={INK.text}>(c) Sequential exclusion of the lowest {bc} / AOD 500</text>
            <g transform={`translate(${M.left},${M.top + 16})`}>
              <YAxis scale={y} x={0} gridWidth={iw} tickCount={5} label={`Pearson r with ${bc}`} labelX={-42} />
              <XAxis scale={x as unknown as d3.ScaleBand<string>} y={ih} label="Bottom % of BC / AOD excluded" labelDy={36} format={(v) => `Ex${v}`} />
              {y.domain()[0] < 0 && <line x1={0} x2={iw} y1={y(0)} y2={y(0)} stroke={INK.muted} strokeDasharray="3 3" />}
              {COLUMN.map((c) => {
                const pts = steps.map((s) => ({ ex: s.excluded, r: s.r[c.key]?.r ?? null, n: s.r[c.key]?.n ?? 0 }))
                const line = d3.line<(typeof pts)[number]>().defined((p) => p.r !== null).x((p) => xs(p.ex)).y((p) => y(p.r!))
                return <g key={c.key}>
                  <path d={line(pts) ?? ''} fill="none" stroke={c.color} strokeWidth={2} />
                  {pts.map((p) => p.r === null ? null : <circle key={p.ex} cx={xs(p.ex)} cy={y(p.r)} r={4} fill={c.color}
                    onMouseEnter={(e) => tip.show(e, [`${c.label} · Ex${p.ex}`, `r = ${fmt(p.r, 3)}`, `${p.n} filters`])} onMouseLeave={tip.hide} />)}
                </g>
              })}
              {steps.map((s) => <text key={s.excluded} x={xs(s.excluded)} y={-4} textAnchor="middle" fontSize={10} fill={INK.muted}>n={s.n}</text>)}
            </g>
          </svg>
        )}
        <div className="legend">
          {COLUMN.map((c) => <span key={c.key} className="legend-item"><span className="swatch" style={{ background: c.color }} />{c.label}</span>)}
          <span className="legend-item u">points in (a, b) coloured by {col.by}</span>
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
