import { useMemo, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Legend } from '@/components/Legend'
import { Panels } from '@/components/Panels'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useHighlight } from '@/lib/highlight'
import { fmt, regression } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { ScrubField } from '@/components/ScrubField'

const DRY = '#d08c2c', HUMID = '#2c7fb8'
const SWEEP = d3.range(30, 85, 2.5)
const MIN_N = 8
const BEST_N = 15

/**
 * Does humidity explain the scatter between two methods? The pair is split at a relative
 * humidity threshold over each filter's window, and each half gets its own fit. Below, the
 * threshold sweep: both halves' slopes at every threshold from 30 to 82.5 %, with the
 * threshold that separates them most marked (meteorology_source_interaction §4,
 * meteorology_revisions R2 / R2C, which did this with the Addis airport station).
 * The pair is the gallery's shared x and y: click an axis title (or use the Correlation tab's
 * sidebar) to change it, and every crossplot follows. Both panels are shared charts in the
 * shared composite layout (components/Panels): the crossplot (components/Crossplot), each half a
 * fitted group under the page's estimators, and the sweep (components/SweepPlot); click a
 * threshold on the sweep to split there.
 */
export function HumidityAgreement({ rows, meta, xField, yField, rhField }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; rhField: string }) {
  const hl = useHighlight()
  const [thr, setThr] = useState(50)

  const pts = useMemo(() => rows.filter((r) => [xField, yField, rhField].every((f) => typeof r[f] === 'number' && Number.isFinite(r[f])))
    .map((r) => ({ row: r, x: r[xField] as number, y: r[yField] as number, rh: r[rhField] as number })), [rows, xField, yField, rhField])
  const fit = (ps: typeof pts) => regression(ps.map((p) => p.x), ps.map((p) => p.y), { errorsInVariables: true })
  const nDry = pts.filter((p) => p.rh < thr).length, nHumid = pts.length - nDry
  // each half a fit group of the shared crossplot; memoised, so a hover does not refit (or re-bootstrap) the halves
  const points = useMemo<CrossPoint[]>(() => pts.map((p) => {
    const isDry = p.rh < thr
    return { x: p.x, y: p.y, id: p.row.id, color: isDry ? DRY : HUMID, site: isDry ? `Dry (RH < ${thr} %)` : `Humid (RH ≥ ${thr} %)`,
      tip: [p.row.id, `${p.row.site} · ${p.row.date}`, `${xField} ${fmt(p.x)} · ${yField} ${fmt(p.y)}`, `RH ${fmt(p.rh, 0)} %`],
      open: () => hl.openSample(p.row.id) }
  }), [pts, thr, xField, yField]) // eslint-disable-line react-hooks/exhaustive-deps
  const sweep = useMemo(() => SWEEP.map((t) => {
    const a = pts.filter((p) => p.rh < t), b = pts.filter((p) => p.rh >= t)
    const sa = a.length >= MIN_N ? fit(a) : null, sb = b.length >= MIN_N ? fit(b) : null
    return { t, nd: a.length, nh: b.length, sd: sa?.slope ?? null, sh: sb?.slope ?? null }
  }), [pts]) // eslint-disable-line react-hooks/exhaustive-deps
  // a half with a handful of filters swings wildly; only call a gap where both halves have BEST_N
  const best = sweep.filter((s) => s.sd !== null && s.sh !== null && s.nd >= BEST_N && s.nh >= BEST_N).sort((a, b) => Math.abs(b.sh! - b.sd!) - Math.abs(a.sh! - a.sd!))[0]

  const nMax = d3.max(sweep, (s) => Math.max(s.nd, s.nh)) ?? 1
  const tipAt = (s: (typeof sweep)[number]) => [`RH < ${s.t} %: n=${s.nd}, slope ${fmt(s.sd, 3)}`, `RH ≥ ${s.t} %: n=${s.nh}, slope ${fmt(s.sh, 3)}`, 'click to use this threshold']
  const series: SweepSeries[] = [
    { id: 'dry', label: 'dry', color: DRY, width: 2, points: sweep.map((s) => ({ x: s.t, y: s.sd, tip: tipAt(s) })) },
    { id: 'humid', label: 'humid', color: HUMID, width: 2, points: sweep.map((s) => ({ x: s.t, y: s.sh, tip: tipAt(s) })) },
  ]

  return (
    <ChartFrame id="weather-humidity" title="Humidity and method agreement — dry vs humid filters"
      subtitle={`${yField} against ${xField}, split at ${thr} % relative humidity over each filter's window (${nDry} dry, ${nHumid} humid); a fit per half under the page's estimators. The sweep shows both slopes at every threshold, with the count of filters on each side (thin lines). With several sites picked the halves mix sites, so pick one for the notebook's question.`}
      provenance="meteorology_source_interaction §4 · meteorology_revisions R2, R2C (threshold sweep)"
      controls={<>
        <span className="control">RH threshold
          <ScrubField label="RH threshold" value={thr} onChange={setThr} min={30} max={82.5} step={2.5} unit="%" width={110} /></span>
      </>}>
      <div className="chart-wrap">
        {pts.length < MIN_N * 2 ? <Empty>Fewer than {MIN_N * 2} filters carry {xField}, {yField} and {rhField} in this subset.</Empty> : <Panels label="Humidity split and threshold sweep" panels={[
          { key: 'cross', title: `Split at ${thr} %`, note: `${nDry} dry · ${nHumid} humid`, label: 'Dry and humid crossplot', min: 300, basis: 480, max: 500, grow: 0,
            // the crossplot is square: in a tile its side also fits the height (less its fit readout and axes)
            render: ({ width, height }) => <Crossplot points={points} title="" xLabel={withUnit(xField, meta.field_units)} yLabel={withUnit(yField, meta.field_units)}
              maxSide={Math.min(width - 70, height === null ? Infinity : height - 150)} grouping="site" groupNoun="humidity group" pick /> },
          { key: 'sweep', title: 'Threshold sweep', note: best ? `widest gap ${best.t} %` : undefined, label: 'Threshold sweep of both slopes', min: 280, basis: 420,
            render: ({ width, height }) => <SweepPlot series={series} width={width} height={height ?? 300} xLabel="RH threshold (%)" yLabel="OLS slope" xDomain={[SWEEP[0], SWEEP[SWEEP.length - 1]]}
              include={[0]} chosen={[{ series: 'dry', x: thr }, { series: 'humid', x: thr }]} onPoint={(_, p) => setThr(p.x as number)}
              // the threshold in use is the ringed pair (and the panel title); the widest gap is labelled
              xMarkers={[{ at: thr, color: INK.text }, ...(best && best.t !== thr ? [{ at: best.t, label: 'widest gap', color: INK.fit }] : [])]}
              // filters on each side, on their own scale in the lower half of the plot
              background={({ x, iw, ih }) => {
                const sn = d3.scaleLinear().domain([0, nMax]).range([ih, ih * 0.55]), sx = x as d3.ScaleLinear<number, number>
                const count = (k: 'nd' | 'nh') => d3.line<(typeof sweep)[number]>().x((s) => sx(s.t)).y((s) => sn(s[k]))(sweep) ?? ''
                return <g pointerEvents="none">
                  <path d={count('nd')} fill="none" stroke={DRY} strokeOpacity={0.4} strokeDasharray="2 3" />
                  <path d={count('nh')} fill="none" stroke={HUMID} strokeOpacity={0.4} strokeDasharray="2 3" />
                  <text x={iw - 2} y={sn(nMax) - 4} textAnchor="end" fontSize={9.5} fill={INK.muted}>dotted: filters per side (max {nMax})</text>
                </g>
              }} /> },
        ]} />}
      </div>
      <Legend items={[{ label: 'dry (RH below the threshold)', color: DRY }, { label: 'humid', color: HUMID },
        { label: '1:1', color: INK.identity, shape: 'dashed' }]} note="sweep: dotted = filters per side" />
      {best && <Note>Where both halves have at least {BEST_N} filters, the slopes differ most at {best.t} % ({best.nd} dry, {best.nh} humid). A gap at one threshold is not a test.</Note>}
    </ChartFrame>
  )
}
