import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { fmt } from '@/lib/stats'
import { INK, MARGIN } from '@/lib/theme'
import type { CalibRun, CalibRunsFile, MetaFile } from '@/lib/types'
import { groupColorFn } from './common'

/**
 * The explorer's Series tab: predicted EC by sampling date under one
 * configuration, with the 45-day rolling median from ftir_29, the deployed
 * SPARTAN EC on the same filters where it exists, and the plausibility card
 * (negative days, days above 8 µg/m³, group medians) that decides whether a
 * calibration produces a believable record, not just a good crossplot.
 */
export function RunSeries({ run, runs, meta, knownIds }: { run: CalibRun; runs: CalibRunsFile; meta: MetaFile; knownIds: Set<string> }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const height = 400

  const [showDeployed, setShowDeployed] = useState(true)
  const [showRoll, setShowRoll] = useState(true)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const color = useMemo(() => groupColorFn(meta, run.eval.group), [meta, run.eval.group])
  const target = runs.targets[run.target]

  const dated = useMemo(() => {
    const e = run.eval
    if (!e.date) return []
    return e.pred
      .map((v, i) => ({ i, d: e.date![i] ? new Date(e.date![i] + 'T00:00:00') : null, v, g: e.group[i], id: e.id[i], dep: e.deployed?.[i] ?? null }))
      .filter((p): p is typeof p & { d: Date } => p.d !== null && Number.isFinite(p.v))
      .sort((a, b) => +a.d - +b.d)
  }, [run.eval])
  const shown = dated.filter((p) => !hidden.has(p.g))
  const hasDeployed = dated.some((p) => p.dep !== null)

  const roll = useMemo(() =>
    shown.map((p) => {
      const win = shown.filter((q) => Math.abs(+q.d - +p.d) <= 45 * 864e5).map((q) => q.v).sort(d3.ascending)
      return { d: p.d, v: win[Math.floor(win.length / 2)] }
    }), [shown])

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const x = d3.scaleTime().domain(d3.extent(shown, (p) => p.d) as [Date, Date]).range([0, innerW])
  const allV = shown.flatMap((p) => [p.v, ...(showDeployed && p.dep !== null ? [p.dep] : [])])
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(allV) ?? 0), (d3.max(allV) ?? 1) * 1.08]).range([innerH, 0]).nice()
  const line = d3.line<{ d: Date; v: number }>().x((p) => x(p.d)).y((p) => y(p.v)).curve(d3.curveMonotoneX)

  const preds = shown.map((p) => p.v).sort(d3.ascending)
  const q = (p: number) => preds[Math.floor(p * (preds.length - 1))]
  const neg = preds.filter((v) => v < 0).length
  const high = preds.filter((v) => v > 8).length
  const gmed = [...new Set(shown.map((p) => p.g))].map((g) => ({ g, med: d3.median(shown.filter((p) => p.g === g), (p) => p.v) ?? 0, n: shown.filter((p) => p.g === g).length }))

  return (
    <ChartFrame
      id="run-series"
      title={`Dated series — predicted EC at ${target?.site ?? run.target} under this configuration`}
      subtitle="Every dated evaluation filter as a point, with the 45-day rolling median. Negative days and days above 8 µg/m³ are the plausibility checks from ftir_29; the deployed SPARTAN EC (hollow) is the record this calibration would replace."
      provenance="calibration_explorer Series tab · /api/run eval.date / eval.deployed"
      controls={
        <>
          <Toggle label="45-day rolling median" checked={showRoll} onChange={setShowRoll} />
          {hasDeployed && <Toggle label="deployed SPARTAN EC" checked={showDeployed} onChange={setShowDeployed} />}
          <span className="control">{shown.length} dated filters</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {shown.length < 2 ? (
          <Empty>No dated filters for this target.</Empty>
        ) : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label="predicted EC (µg/m³)" gridWidth={innerW} />
              <XAxis scale={x as any} y={innerH} format={(d: Date) => d3.timeFormat('%b %Y')(d)} tickCount={7} />
              <line x1={0} x2={innerW} y1={y(0)} y2={y(0)} stroke={INK.negative} strokeWidth={1} strokeOpacity={0.7} />
              {y.domain()[1] >= 8 && <line x1={0} x2={innerW} y1={y(8)} y2={y(8)} stroke={INK.muted} strokeDasharray="3 3" strokeOpacity={0.7} />}
              {showDeployed && shown.filter((p) => p.dep !== null).map((p) => (
                <circle key={`d${p.i}`} cx={x(p.d)} cy={y(p.dep as number)} r={3} fill="none" stroke={INK.axis} strokeWidth={1} />
              ))}
              {shown.map((p) => {
                const st = focusStyle(p.id ?? `#${p.i}`, hl.focusId, { r: 3.2, opacity: showRoll ? 0.55 : 0.8 })
                return (
                  <circle key={p.i} cx={x(p.d)} cy={y(p.v)} r={st.r} fill={color(p.g)} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth}
                    style={{ cursor: p.id && knownIds.has(p.id) ? 'pointer' : 'default' }}
                    onMouseEnter={(e) => { if (p.id) hl.setHover(p.id); tip.show(e, [p.id ?? `filter #${p.i + 1}`, `${run.eval.date?.[p.i]} · ${p.g}`, `predicted EC ${fmt(p.v, 3)} µg/m³`, ...(p.dep !== null ? [`deployed EC ${fmt(p.dep, 3)}`] : [])]) }}
                    onMouseLeave={() => { hl.setHover(null); tip.hide() }}
                    onClick={() => p.id && knownIds.has(p.id) && hl.openSample(p.id)} />
                )
              })}
              {showRoll && <path d={line(roll) ?? ''} fill="none" stroke={INK.text} strokeWidth={2} pointerEvents="none" />}
            </g>
          </svg>
        )}
        <Legend
          items={[
            ...[...new Set(run.eval.group)].map((g) => ({ label: g, color: color(g) })),
            ...(showRoll ? [{ label: '45-day rolling median', color: INK.text, shape: 'line' as const }] : []),
            ...(hasDeployed && showDeployed ? [{ label: 'deployed SPARTAN EC (hollow)', color: INK.axis }] : []),
          ]}
          hidden={hidden}
          onToggle={(l) => { if (run.eval.group.includes(l)) setHidden((h) => toggleIn(h, l)) }}
          note="red line = 0 · dotted = 8 µg/m³"
        />
        {preds.length > 0 && (
          <table className="placement" style={{ maxWidth: 640 }}>
            <tbody>
              <tr><td className="f">n dated</td><td className="v">{preds.length}</td><td className="f">median [IQR]</td><td className="v">{fmt(q(0.5), 2)} [{fmt(q(0.25), 2)} – {fmt(q(0.75), 2)}]</td></tr>
              <tr><td className="f">negative days</td><td className="v" style={{ color: neg ? INK.negative : INK.good }}>{neg} ({((100 * neg) / preds.length).toFixed(1)} %)</td><td className="f">days above 8 µg/m³</td><td className="v" style={{ color: high ? INK.negative : INK.good }}>{high}</td></tr>
              <tr><td className="f">group medians</td><td className="u" colSpan={3}>{gmed.map((g) => `${g.g}: ${fmt(g.med, 2)} (n=${g.n})`).join(' · ')}</td></tr>
            </tbody>
          </table>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
