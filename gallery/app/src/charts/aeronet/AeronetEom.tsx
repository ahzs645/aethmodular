import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Panels } from '@/components/Panels'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { AE, AOD500, DAOD, DSSA, FINE_AOD, FMF } from '@/lib/aeronet'
import { exclusionCurve, fineSoil } from '@/lib/aeronetMath'
import type { FilterRow, MetaFile } from '@/lib/types'
import { BC_CHOICES } from './AeronetNonAbsorbing'
import { colouring, num, recordOf } from './shared'
import { useMarkedPoints } from '@/lib/markedRows'

const COLUMN = [
  { key: AOD500, label: 'AOD 500', color: '#2c7fb8' },
  { key: AE, label: 'Ångström exponent 440–870', color: '#d95f02' },
  { key: FMF, label: 'Fine-mode fraction', color: '#41ab5d' },
  { key: FINE_AOD, label: 'Fine-mode AOD 500', color: '#7570b3' },
]
const STEPS = [0, 20, 40, 60, 80]

/**
 * The surface–column tests of Eom et al. (2025), as notebooks/analysis/surface_columnar_linkage.ipynb runs
 * them on the filters: the spectral AOD difference and the SSA difference (440 − 870 nm) against BC minus fine
 * soil (dust-dominated days to the left, BC-dominated to the right), and the sequential exclusion of the
 * lowest BC/AOD filters, after which surface BC should track the column better if those were the days with
 * aerosol aloft.
 */
export function AeronetEom({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const present = BC_CHOICES.filter((f) => rows.some((r) => num(r[f])))
  const [bcParam, setBc] = useSearchParam<string>('aeronet.eom.bc', BC_CHOICES[0], BC_CHOICES)
  const bc = present.includes(bcParam as (typeof BC_CHOICES)[number]) ? bcParam : present[0] ?? BC_CHOICES[0]
  const col = useMemo(() => colouring(rows, meta), [rows, meta])

  const diff = (yField: string, label: string, src: FilterRow[] = rows): CrossPoint[] => src.flatMap((r) => {
    const fs = fineSoil(r), b = r[bc], v = r[yField]
    if (fs === null || !num(b) || !num(v)) return []
    return [{ x: b - fs, y: v, color: col.color(col.key(r)), id: r.id,
      tip: [r.id, `${r.site} · ${r.date}`, `${bc} − fine soil ${fmt(b - fs, 2)} µg/m³ (fine soil ${fmt(fs, 2)})`, `${label} ${fmt(v, 3)}`],
      rec: recordOf(r, 'AERONET · Eom et al. tests', [[bc, b], ['Fine soil (Malm 1994)', +fs.toFixed(3)], [`${bc} − fine soil`, +(b - fs).toFixed(3)], [label, v]]) }]
  })
  const dAod = useMemo(() => diff(DAOD, 'dAOD 440 − 870'), [rows, bc, col]) // eslint-disable-line react-hooks/exhaustive-deps
  const dSsa = useMemo(() => diff(DSSA, 'dSSA 440 − 870'), [rows, bc, col]) // eslint-disable-line react-hooks/exhaustive-deps
  // Filters → excluded: Mark (lib/markedRows): these sites' excluded filters as ×, in no fit
  const markedAod = useMarkedPoints((rs) => diff(DAOD, 'dAOD 440 − 870', rs.filter((r) => rows.some((q) => q.site === r.site))), [rows, bc, col])
  const markedSsa = useMarkedPoints((rs) => diff(DSSA, 'dSSA 440 − 870', rs.filter((r) => rows.some((q) => q.site === r.site))), [rows, bc, col])

  const steps = useMemo(() => exclusionCurve(rows.flatMap((r) => {
    const b = r[bc], aod = r[AOD500]
    if (!num(b) || !num(aod) || aod <= 0) return []
    return [{ ratio: b / aod, x: b, ys: Object.fromEntries(COLUMN.map((c) => [c.key, num(r[c.key]) ? (r[c.key] as number) : null])) }]
  }), COLUMN.map((c) => c.key), STEPS), [rows, bc])

  const sweep = useMemo<SweepSeries[]>(() => COLUMN.map((c) => ({ id: c.key, label: c.label, color: c.color, width: 2, r: 4,
    points: steps.map((s) => ({ x: s.excluded, y: s.r[c.key]?.r ?? null, tip: [`${c.label} · Ex${s.excluded}`, `r = ${fmt(s.r[c.key]?.r ?? NaN, 3)}`, `${s.r[c.key]?.n ?? 0} filters`] })) })), [steps])

  return (
    <ChartFrame id="aeronet-eom" title="Surface against column — the Eom et al. (2025) tests"
      subtitle={`(a, b) dAOD = AOD 440 − AOD 870 and dSSA = SSA 440 − SSA 870 (per retrieval, then the window mean) against ${bc} minus fine soil (Malm 1994: 2.20 Al + 2.49 Si + 1.63 Ca + 2.42 Fe + 1.94 Ti, elemental Ca where measured). (c) Pearson r between ${bc} and each column quantity after dropping the filters whose ${bc} / AOD 500 is below the 0th, 20th … 80th percentile of the filters in view (pooled over the sites picked).`}
      provenance="notebooks/analysis/surface_columnar_linkage.ipynb sections 1, 3 and 8 (Eom et al. 2025: BC/AOD exclusion, dAOD and dSSA vs BC − fine soil); the notebook used aethalometer BC for the exclusion and FTIR EC for BC − FS"
      controls={<Select label="BC" value={bc} options={present.length ? [...present] : [...BC_CHOICES]} onChange={setBc} />}>
      <div className="chart-wrap">
        <Panels label="Eom et al. tests" wrap gap={8} panels={[
          { key: 'daod', title: `dAOD vs ${bc} − fine soil`, note: dAod.length ? `n = ${dAod.length}` : undefined, min: 280, label: 'dAOD vs BC − fine soil',
            render: () => dAod.length ? <Crossplot title="" points={dAod} marked={markedAod} xLabel={`${bc} − fine soil (µg/m³)`} yLabel="dAOD 440 − 870" sameScale={false} maxSide={320} />
              : <Empty>No filter in view has fine soil, {bc} and a direct-sun AOD at 440 and 870 nm.</Empty> },
          { key: 'dssa', title: `dSSA vs ${bc} − fine soil`, note: dSsa.length ? `n = ${dSsa.length}` : undefined, min: 280, label: 'dSSA vs BC − fine soil',
            render: () => dSsa.length ? <Crossplot title="" points={dSsa} marked={markedSsa} xLabel={`${bc} − fine soil (µg/m³)`} yLabel="dSSA 440 − 870" sameScale={false} maxSide={320} />
              : <Empty>No filter in view has fine soil, {bc} and an inversion SSA.</Empty> },
          { key: 'exclusion', title: `Sequential exclusion of the lowest ${bc} / AOD 500`, min: 300, max: 600, label: 'Sequential exclusion',
            render: ({ width, height }) => steps.length === 0 ? <Empty>Fewer than 5 filters in view have {bc} and AOD 500.</Empty>
              : <SweepPlot series={sweep} xScale="point" width={width} height={height ?? 316} xLabel="Bottom % of BC / AOD excluded" yLabel={`Pearson r with ${bc}`}
                  xFormat={(v) => `Ex${v}`} include={[0, 0.2]} reference={0}
                  children={({ x }) => steps.map((s) => <text key={s.excluded} x={(x as d3.ScalePoint<string>)(String(s.excluded)) ?? 0} y={-2} textAnchor="middle" fontSize={10} fill={INK.muted}>n={s.n}</text>)} /> },
        ]} legend={<div className="legend">
          {COLUMN.map((c) => <span key={c.key} className="legend-item"><span className="swatch" style={{ background: c.color }} />{c.label}</span>)}
          <span className="legend-item u">points in (a, b) coloured by {col.by}</span>
        </div>} />
      </div>
    </ChartFrame>
  )
}
