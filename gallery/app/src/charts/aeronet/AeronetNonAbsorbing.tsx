import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { BC_CHOICES, bcOption, useChartOptions } from '@/lib/chartOptions'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { fmt } from '@/lib/stats'
import { SSA_BY_WAVELENGTH } from '@/lib/aeronet'
import { nonAbsorbingFraction } from '@/lib/aeronetMath'
import type { FilterRow, MetaFile } from '@/lib/types'
import { colouring, num, recordOf } from './shared'
import { useMarkedRows } from '@/lib/markedRows'
import { Legend } from '@/components/Legend'

/** the surface BC Eom's w can use, in the notebook's order of preference (it fell back from ChemSpec BC to FTIR EC to HIPS): lib/chartOptions BC_CHOICES */

/**
 * Eom et al. (2025)'s non-absorbing fraction against the column's SSA: w = (AS + AN) / (AS + AN + BC) from the
 * filter (ammonium sulfate 1.375 × sulfate, ammonium nitrate 1.290 × nitrate), against the inversion's SSA in
 * the same window, one panel per wavelength. A scattering-dominated surface should sit under a bright column.
 */
export function AeronetNonAbsorbing({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const present = BC_CHOICES.filter((f) => rows.some((r) => num(r[f])))
  // the filter BC (lib/chartOptions bcOption): kept in the link as aeronet.w.bc
  const o = useChartOptions('aeronet.w', [bcOption({ present })])
  const bc = o.values.bc as string
  const col = useMemo(() => colouring(rows, meta), [rows, meta])
  // Filters → excluded: Mark (lib/markedRows): these sites' excluded filters, built the same way and drawn as ×
  const marked = useMarkedRows()
  const panels = useMemo(() => SSA_BY_WAVELENGTH.map(([w, f]) => {
    const build = (src: FilterRow[]): CrossPoint[] => src.flatMap((r) => {
      const wv = nonAbsorbingFraction(r, bc), ssa = r[f]
      if (wv === null || !num(ssa)) return []
      const tip = [r.id, `${r.site} · ${r.date}`, `w ${fmt(wv, 3)} (${bc})`, `SSA ${w} nm ${fmt(ssa, 3)}`]
      return [{ x: wv, y: ssa, color: col.color(col.key(r)), tip, id: r.id,
        rec: recordOf(r, 'AERONET · w vs SSA', [['w = (AS+AN)/(AS+AN+BC)', +wv.toFixed(3)], [bc, r[bc] as number], ['Sulfate', r['Sulfate'] as number], ['Nitrate', r['Nitrate'] as number], [`SSA ${w} nm`, ssa]]) }]
    })
    const sites = new Set(rows.map((r) => r.site))
    return { w, points: build(rows), marked: build(marked.filter((r) => sites.has(r.site))) }
  }), [rows, bc, col, marked])
  const all = panels.flatMap((p) => p.points)
  const yLo = Math.max(0.5, (d3.min(all, (p) => p.y) ?? 0.8) - 0.02)
  const n = panels[0].points.length

  return (
    <ChartFrame id="aeronet-w-ssa" title="Non-absorbing fraction against SSA — by wavelength"
      subtitle={`w = (AS + AN) / (AS + AN + BC): ammonium sulfate (1.375 × sulfate) and ammonium nitrate (1.290 × nitrate) over themselves plus ${bc}, all from the filter, against the window-mean single-scattering albedo at each AERONET wavelength. Eom et al. (2025) found SSA rising with w across 14 sites; ${n} filters here have both. Fits follow the gallery's estimator.`}
      provenance="notebooks/analysis/surface_columnar_linkage.ipynb sections 2 and 8 (w parameter; w vs SSA, which the notebook could not draw without the inversion)"
      controls={o.controls()}>
      {all.length === 0 ?<Empty>No filter in view has sulfate, nitrate, {bc} and an inversion SSA in its window.</Empty> : (
        <div className="aeronet-panels" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 8 }}>
          {panels.map((p) => (
            <Crossplot key={p.w} title={`SSA ${p.w} nm (n = ${p.points.length})`} points={p.points} marked={p.marked} xLabel="w (non-absorbing fraction)" yLabel={`SSA ${p.w} nm`}
              sameScale={false} domain={[0, 1]} yDomain={[yLo, 1]} maxSide={300} />
          ))}
        </div>
      )}
      <Legend items={col.order(rows.filter((r) => nonAbsorbingFraction(r, bc) !== null && num(r[SSA_BY_WAVELENGTH[0][1]]))).map((k) =>
        ({ label: col.label(k), color: col.color(k) }))} />
    </ChartFrame>
  )
}
