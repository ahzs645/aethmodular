import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Legend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useHighlight } from '@/lib/highlight'
import { fmt, regression } from '@/lib/stats'
import { FONT, INK, withUnit } from '@/lib/theme'
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
 * sidebar) to change it, and every crossplot follows. The crossplot is the shared one
 * (components/Crossplot), each half a fitted group under the page's estimators.
 */
export function HumidityAgreement({ rows, meta, xField, yField, rhField }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; rhField: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
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

  // scatter and sweep side by side when there is room, stacked on a narrow screen
  const sq = Math.min(420, Math.max(240, (width - 24) / 2 - 82))
  const sideW = Math.max(220, width - sq - 82 - 24 - 80)
  const sh = 200
  const sx = d3.scaleLinear().domain([SWEEP[0], SWEEP[SWEEP.length - 1]]).range([0, sideW])
  const slopes = sweep.flatMap((s) => [s.sd, s.sh]).filter((v): v is number => v !== null)
  const sy = d3.scaleLinear().domain([Math.min(0, d3.min(slopes) ?? 0), (d3.max(slopes) ?? 1) * 1.1]).range([sh, 0]).nice()
  const nMax = d3.max(sweep, (s) => Math.max(s.nd, s.nh)) ?? 1
  const sn = d3.scaleLinear().domain([0, nMax]).range([sh, sh * 0.55])
  const sline = (k: 'sd' | 'sh') => d3.line<(typeof sweep)[number]>().defined((s) => s[k] !== null).x((s) => sx(s.t)).y((s) => sy(s[k]!))(sweep) ?? ''

  return (
    <ChartFrame id="weather-humidity" title="Humidity and method agreement — dry vs humid filters"
      subtitle={`${yField} against ${xField}, split at ${thr} % relative humidity over each filter's window (${nDry} dry, ${nHumid} humid); a fit per half under the page's estimators. The sweep shows both slopes at every threshold, with the count of filters on each side (thin lines). With several sites picked the halves mix sites, so pick one for the notebook's question.`}
      provenance="meteorology_source_interaction §4 · meteorology_revisions R2, R2C (threshold sweep)"
      controls={<>
        <span className="control">RH threshold
          <ScrubField label="RH threshold" value={thr} onChange={setThr} min={30} max={82.5} step={2.5} unit="%" width={110} /></span>
      </>}>
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        {pts.length < MIN_N * 2 ? <Empty>Fewer than {MIN_N * 2} filters carry {xField}, {yField} and {rhField} in this subset.</Empty> : <>
          <div style={{ flex: `0 1 ${sq + 70}px`, minWidth: 240 }}>
            <Crossplot points={points} title="" xLabel={withUnit(xField, meta.field_units)} yLabel={withUnit(yField, meta.field_units)} maxSide={sq}
              grouping="site" groupNoun="humidity group" pick />
          </div>
          <svg width={sideW + 80} height={sh + 70} fontFamily={FONT.family}>
            <text x={52} y={12} fontSize={12} fontWeight={600} fill={INK.text}>Threshold sweep</text>
            <g transform={`translate(52,24)`}>
              <YAxis scale={sy} x={0} label="OLS slope" gridWidth={sideW} tickCount={4} labelX={-40} />
              <XAxis scale={sx} y={sh} label="RH threshold (%)" tickCount={6} />
              <path d={d3.line<(typeof sweep)[number]>().x((s) => sx(s.t)).y((s) => sn(s.nd))(sweep) ?? ''} fill="none" stroke={DRY} strokeOpacity={0.4} strokeDasharray="2 3" />
              <path d={d3.line<(typeof sweep)[number]>().x((s) => sx(s.t)).y((s) => sn(s.nh))(sweep) ?? ''} fill="none" stroke={HUMID} strokeOpacity={0.4} strokeDasharray="2 3" />
              <path d={sline('sd')} fill="none" stroke={DRY} strokeWidth={2} />
              <path d={sline('sh')} fill="none" stroke={HUMID} strokeWidth={2} />
              <line x1={sx(thr)} x2={sx(thr)} y1={0} y2={sh} stroke={INK.text} strokeOpacity={0.5} />
              {best && <g><line x1={sx(best.t)} x2={sx(best.t)} y1={0} y2={sh} stroke={INK.fit} strokeDasharray="3 3" />
                <text x={sx(best.t) + 4} y={10} fontSize={10} fill={INK.fit}>widest gap {best.t} %</text></g>}
              {sweep.map((s) => <rect key={s.t} x={sx(s.t) - sideW / SWEEP.length / 2} width={sideW / SWEEP.length} height={sh} fill="transparent"
                onMouseEnter={(e) => tip.show(e, [`RH < ${s.t} %: n=${s.nd}, slope ${fmt(s.sd, 3)}`, `RH ≥ ${s.t} %: n=${s.nh}, slope ${fmt(s.sh, 3)}`, 'click to use this threshold'])}
                onMouseLeave={tip.hide} onClick={() => setThr(s.t)} style={{ cursor: 'pointer' }} />)}
            </g>
          </svg>
        </>}
        {tip.node}
      </div>
      <Legend items={[{ label: 'dry (RH below the threshold)', color: DRY }, { label: 'humid', color: HUMID },
        { label: '1:1', color: INK.identity, shape: 'dashed' }]} note="sweep: dotted = filters per side" />
      {best && <Note>Where both halves have at least {BEST_N} filters, the slopes differ most at {best.t} % ({best.nd} dry, {best.nh} humid). A gap at one threshold is not a test.</Note>}
    </ChartFrame>
  )
}
