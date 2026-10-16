import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps } from '@/lib/filterEdits'
import { fmt, regression, type RegressionStats } from '@/lib/stats'
import { FONT, INK, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

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
 * sidebar) to change it, and every crossplot follows.
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
  const dry = pts.filter((p) => p.rh < thr), humid = pts.filter((p) => p.rh >= thr)
  const fd = dry.length >= MIN_N ? fit(dry) : null, fh = humid.length >= MIN_N ? fit(humid) : null
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
  const M = { top: 16, left: 62, bottom: 46 }
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
  const hi = Math.max(d3.quantile(xs.slice().sort(d3.ascending), 0.995) ?? 1, d3.quantile(ys.slice().sort(d3.ascending), 0.995) ?? 1) * 1.05
  const x = d3.scaleLinear().domain([0, hi]).range([0, sq]).nice(), y = d3.scaleLinear().domain([0, hi]).range([sq, 0]).nice()
  const sh = 200
  const sx = d3.scaleLinear().domain([SWEEP[0], SWEEP[SWEEP.length - 1]]).range([0, sideW])
  const slopes = sweep.flatMap((s) => [s.sd, s.sh]).filter((v): v is number => v !== null)
  const sy = d3.scaleLinear().domain([Math.min(0, d3.min(slopes) ?? 0), (d3.max(slopes) ?? 1) * 1.1]).range([sh, 0]).nice()
  const nMax = d3.max(sweep, (s) => Math.max(s.nd, s.nh)) ?? 1
  const sn = d3.scaleLinear().domain([0, nMax]).range([sh, sh * 0.55])
  const fitLine = (f: RegressionStats | null, c: string) => f && <line x1={x(0)} y1={y(f.intercept)} x2={x(hi)} y2={y(f.intercept + f.slope * hi)} stroke={c} strokeWidth={2} />
  const sline = (k: 'sd' | 'sh') => d3.line<(typeof sweep)[number]>().defined((s) => s[k] !== null).x((s) => sx(s.t)).y((s) => sy(s[k]!))(sweep) ?? ''

  return (
    <ChartFrame id="weather-humidity" title="Humidity and method agreement — dry vs humid filters"
      subtitle={`${yField} against ${xField}, split at ${thr} % relative humidity over each filter's window; OLS fit per half. The sweep shows both slopes at every threshold, with the count of filters on each side (thin lines). With several sites picked the halves mix sites, so pick one for the notebook's question.`}
      provenance="meteorology_source_interaction §4 · meteorology_revisions R2, R2C (threshold sweep)"
      controls={<>
        <label className="control">RH threshold
          <input type="range" min={30} max={82.5} step={2.5} value={thr} onChange={(e) => setThr(Number(e.target.value))} style={{ width: 110 }} />
          <span style={{ fontFamily: 'var(--mono)' }}>{thr} %</span></label>
      </>}>
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        {pts.length < MIN_N * 2 ? <Empty>Fewer than {MIN_N * 2} filters carry {xField}, {yField} and {rhField} in this subset.</Empty> : <>
          <svg width={sq + M.left + 20} height={sq + M.top + M.bottom} fontFamily={FONT.family}>
            <g transform={`translate(${M.left},${M.top})`}>
              <YAxis scale={y} x={0} label={withUnit(yField, meta.field_units)} gridWidth={sq} labelX={-46} pick="y" />
              <XAxis scale={x} y={sq} label={withUnit(xField, meta.field_units)} pick="x" />
              <line x1={0} y1={sq} x2={sq} y2={0} stroke={INK.muted} strokeDasharray="4 4" />
              {pts.map((p) => {
                const st = focusStyle(p.row.id, hl.focusId, { r: 3, opacity: 0.7 }, hl.selected)
                return <circle key={p.row.id} cx={x(Math.min(p.x, hi))} cy={y(Math.min(p.y, hi))} r={st.r} fill={p.rh < thr ? DRY : HUMID} fillOpacity={st.opacity}
                  stroke={st.stroke} strokeWidth={st.strokeWidth} style={{ cursor: 'pointer' }} {...fidProps(p.row.id)}
                  onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.site} · ${p.row.date}`, `${xField} ${fmt(p.x)} · ${yField} ${fmt(p.y)}`, `RH ${fmt(p.rh, 0)} %`]) }}
                  onMouseLeave={() => { hl.setHover(null); tip.hide() }} onClick={(e) => { e.stopPropagation(); hl.openSample(p.row.id) }} />
              })}
              {fitLine(fd, DRY)}{fitLine(fh, HUMID)}
              <g fontFamily={FONT.mono} fontSize={10.5}>
                <rect x={6} y={4} width={210} height={36} rx={4} fill="#fff" fillOpacity={0.85} stroke={INK.grid} />
                <text x={12} y={18} fill={DRY}>dry  n={dry.length}  slope {fmt(fd?.slope, 3)}  R² {fmt(fd?.r2, 2)}</text>
                <text x={12} y={33} fill={HUMID}>humid n={humid.length}  slope {fmt(fh?.slope, 3)}  R² {fmt(fh?.r2, 2)}</text>
              </g>
            </g>
          </svg>
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
      <div className="legend">
        <span className="legend-item"><span className="swatch" style={{ background: DRY }} />dry (RH below the threshold)</span>
        <span className="legend-item"><span className="swatch" style={{ background: HUMID }} />humid</span>
        <span className="legend-item"><span className="swatch dashed" style={{ color: INK.muted }} />1:1 · dotted = filters per side</span>
      </div>
      {best && <Note>Where both halves have at least {BEST_N} filters, the slopes differ most at {best.t} % ({best.nd} dry, {best.nh} humid). A gap at one threshold is not a test.</Note>}
    </ChartFrame>
  )
}
