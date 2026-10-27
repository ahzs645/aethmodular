import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { fmt } from '@/lib/stats'
import { useSearchParam } from '@/lib/url'
import { SSA_BY_WAVELENGTH } from '@/lib/aeronet'
import { nonAbsorbingFraction } from '@/lib/aeronetMath'
import type { FilterRow, MetaFile } from '@/lib/types'
import { colouring, num, recordOf } from './shared'

/** the surface BC Eom's w can use, in the notebook's order of preference (it fell back from ChemSpec BC to FTIR EC to HIPS) */
export const BC_CHOICES = ['EC (FTIR)', 'HIPS BC', 'BC (public HIPS)'] as const

/**
 * Eom et al. (2025)'s non-absorbing fraction against the column's SSA: w = (AS + AN) / (AS + AN + BC) from the
 * filter (ammonium sulfate 1.375 × sulfate, ammonium nitrate 1.290 × nitrate), against the inversion's SSA in
 * the same window, one panel per wavelength. A scattering-dominated surface should sit under a bright column.
 */
export function AeronetNonAbsorbing({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const present = BC_CHOICES.filter((f) => rows.some((r) => num(r[f])))
  const [bcParam, setBc] = useSearchParam<string>('aeronet.w.bc', BC_CHOICES[0], BC_CHOICES)
  const bc = present.includes(bcParam as (typeof BC_CHOICES)[number]) ? bcParam : present[0] ?? BC_CHOICES[0]
  const col = useMemo(() => colouring(rows, meta), [rows, meta])
  const panels = useMemo(() => SSA_BY_WAVELENGTH.map(([w, f]) => {
    const points: CrossPoint[] = rows.flatMap((r) => {
      const wv = nonAbsorbingFraction(r, bc), ssa = r[f]
      if (wv === null || !num(ssa)) return []
      const tip = [r.id, `${r.site} · ${r.date}`, `w ${fmt(wv, 3)} (${bc})`, `SSA ${w} nm ${fmt(ssa, 3)}`]
      return [{ x: wv, y: ssa, color: col.color(col.key(r)), tip, id: r.id,
        rec: recordOf(r, 'AERONET · w vs SSA', [['w = (AS+AN)/(AS+AN+BC)', +wv.toFixed(3)], [bc, r[bc] as number], ['Sulfate', r['Sulfate'] as number], ['Nitrate', r['Nitrate'] as number], [`SSA ${w} nm`, ssa]]) }]
    })
    return { w, points }
  }), [rows, bc, col])
  const all = panels.flatMap((p) => p.points)
  const yLo = Math.max(0.5, (d3.min(all, (p) => p.y) ?? 0.8) - 0.02)
  const n = panels[0].points.length

  return (
    <ChartFrame id="aeronet-w-ssa" title="Non-absorbing fraction against SSA — by wavelength"
      subtitle={`w = (AS + AN) / (AS + AN + BC): ammonium sulfate (1.375 × sulfate) and ammonium nitrate (1.290 × nitrate) over themselves plus ${bc}, all from the filter, against the window-mean single-scattering albedo at each AERONET wavelength. Eom et al. (2025) found SSA rising with w across 14 sites; ${n} filters here have both. Fits follow the gallery's estimator.`}
      provenance="notebooks/analysis/surface_columnar_linkage.ipynb sections 2 and 8 (w parameter; w vs SSA, which the notebook could not draw without the inversion)"
      controls={<Select label="BC" value={bc} options={present.length ? [...present] : [...BC_CHOICES]} onChange={setBc} />}>
      {all.length === 0 ?<Empty>No filter in view has sulfate, nitrate, {bc} and an inversion SSA in its window.</Empty> : (
        <div className="aeronet-panels" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 8 }}>
          {panels.map((p) => (
            <Crossplot key={p.w} title={`SSA ${p.w} nm (n = ${p.points.length})`} points={p.points} xLabel="w (non-absorbing fraction)" yLabel={`SSA ${p.w} nm`}
              sameScale={false} domain={[0, 1]} yDomain={[yLo, 1]} maxSide={300} />
          ))}
        </div>
      )}
      <div className="legend">
        {col.order(rows.filter((r) => nonAbsorbingFraction(r, bc) !== null && num(r[SSA_BY_WAVELENGTH[0][1]]))).map((k) =>
          <span key={k} className="legend-item"><span className="swatch" style={{ background: col.color(k) }} />{col.label(k)}</span>)}
      </div>
    </ChartFrame>
  )
}
