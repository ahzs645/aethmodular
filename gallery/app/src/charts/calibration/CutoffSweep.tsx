import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibFile, CalibRow } from '@/lib/types'
import { METRICS, SPECTRA_DASH, cohortColor, fmtFit, isInterceptMetric, isSlopeMetric, metricOf, sameConfig, type Config, type Metric } from './common'

/**
 * Cutoff sweep — the "try somewhat more and somewhat less" experiment from
 * the July-17 meeting, as a line chart. One line per calibration spectra
 * space, x = how many IMPROVE filters the cohort keeps, y = the readout at the
 * rule k. The 2026-08-20 dense sweep found that the locked ocec-800 cutoff was
 * never the optimum: a basin at 440–490 × AIRSpec beats it on intercept *and*
 * on the held-out TOR test, which the coarse 600–1000 ladder could not see.
 */
export function CutoffSweep({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 400

  const rankedCohorts = Object.keys(calib.default_cutoff)
  const [cohort, setCohort] = useState(rankedCohorts.includes('ocec') ? 'ocec' : rankedCohorts[0])
  const [mode, setMode] = useState('site_heldout')
  const [target, setTarget] = useState('addis')
  const [metric, setMetric] = useState<Metric>('Deming intercept')
  const [allPairs, setAllPairs] = useState(false)
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const targets = useMemo(() => [...new Set(calib.grid.map((r) => r.tg))].sort(), [calib.grid])
  const modes = useMemo(() => Object.keys(calib.modes).filter((m) => calib.grid.some((r) => r.mo === m)), [calib])

  const series = useMemo(() => {
    const rows = calib.grid.filter((r) => r.co === cohort && r.mo === mode && r.tg === target && r.lot === 'all' && r.el === 'all' && r.cut !== null)
    const groups = d3.group(rows, (r) => `${r.sel ?? 'raw'}|${r.sp}`)
    return [...groups.entries()]
      .map(([key, rs]) => {
        const [sel, sp] = key.split('|')
        const pts = rs
          .map((r) => ({ r, x: r.cut as number, y: metricOf(r, metric, allPairs) }))
          .filter((p): p is { r: CalibRow; x: number; y: number } => p.y !== null && Number.isFinite(p.y))
          .sort((a, b) => a.x - b.x)
        return { key, sel, sp, label: `${sp}${sel !== 'raw' ? ` · selected on ${sel}` : ''}`, pts, color: sel === 'raw' ? cohortColor(cohort) : INK.deming, dash: SPECTRA_DASH[sp] }
      })
      .filter((s) => s.pts.length > 1)
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [calib.grid, cohort, mode, target, metric, allPairs])

  const shown = series.filter((s) => !hidden.has(s.label))
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const all = shown.flatMap((s) => s.pts)
  const x = d3.scaleLinear().domain(d3.extent(all, (p) => p.x) as [number, number]).range([0, innerW]).nice()
  // Tiny cohorts (cutoff ≈ 100) produce wild fits that would squash the
  // readable range into a few pixels; the axis follows the 2–98 % bulk and the
  // extremes are pinned to the edge, with a count below the chart.
  const sortedY = all.map((p) => p.y).sort(d3.ascending)
  const yExt: [number, number] = [d3.quantile(sortedY, 0.02) ?? 0, d3.quantile(sortedY, 0.98) ?? 1]
  const yLo = isInterceptMetric(metric) ? Math.min(0, yExt[0]) : isSlopeMetric(metric) ? Math.min(0.5, yExt[0]) : Math.min(0, yExt[0])
  const y = d3.scaleLinear().domain([yLo, Math.max(isSlopeMetric(metric) ? 1.2 : 0, yExt[1])]).range([innerH, 0]).nice()
  const clampY = (v: number) => Math.max(y.domain()[0], Math.min(y.domain()[1], v))
  const nClamped = all.filter((p) => p.y < y.domain()[0] || p.y > y.domain()[1]).length
  const isClamped = (v: number) => v < y.domain()[0] || v > y.domain()[1]
  // extremes are not joined into the line: a spike to the edge of the axis is
  // a lie about the neighbouring cutoffs; they are drawn as hollow edge markers
  const line = d3.line<{ x: number; y: number }>().defined((p) => !isClamped(p.y)).x((p) => x(p.x)).y((p) => y(p.y))
  const defaultCut = calib.default_cutoff[cohort]

  return (
    <ChartFrame
      id="cutoff-sweep"
      title="Cutoff sweep — how big should the IMPROVE cohort be?"
      subtitle="The readout at the explorer's rule k as the cohort cutoff grows, one line per calibration spectra space. The vertical line is the locked cutoff. Click a point to load that configuration into the k-sweep below."
      provenance={`calibration_explorer batch results · ${calib.n_source_rows.toLocaleString()} rows · DENSE_SWEEP_RESULTS_2026-08-20.md`}
      controls={
        <>
          <Select label="cohort" value={cohort} options={rankedCohorts} onChange={setCohort} />
          <Select label="protocol" value={mode} options={modes} onChange={setMode} title={Object.entries(calib.modes).map(([k, v]) => `${k}: ${v}`).join('\n')} />
          <Select label="evaluate on" value={target} options={targets} onChange={setTarget} />
          <Select label="readout" value={metric} options={[...METRICS]} onChange={(v) => setMetric(v as Metric)} />
          <Toggle label="all pairs (not the fixed set)" checked={allPairs} onChange={setAllPairs} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {all.length < 2 ? (
          <Empty>No rows for {cohort} · {mode} · {target}.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {isSlopeMetric(metric) && (
                <rect x={0} y={Math.max(0, y(calib.slope_box[1]))} width={innerW} height={Math.max(0, Math.min(innerH, y(calib.slope_box[0])) - Math.max(0, y(calib.slope_box[1])))} fill={INK.accent} fillOpacity={0.06} />
              )}
              {metric === 'held-out TOR R²' && <line x1={0} x2={innerW} y1={y(calib.heldout_floor)} y2={y(calib.heldout_floor)} stroke={INK.axis} strokeDasharray="4 3" />}
              <YAxis scale={y} x={0} label={metric} gridWidth={innerW} />
              <XAxis scale={x} y={innerH} label="cohort cutoff (IMPROVE filters kept)" />
              {(isInterceptMetric(metric) || isSlopeMetric(metric)) && (
                <line x1={0} x2={innerW} y1={y(isSlopeMetric(metric) ? 1 : 0)} y2={y(isSlopeMetric(metric) ? 1 : 0)} stroke={INK.axis} strokeWidth={1} strokeOpacity={0.7} />
              )}
              {defaultCut !== undefined && x(defaultCut) >= 0 && x(defaultCut) <= innerW && (
                <g>
                  <line x1={x(defaultCut)} x2={x(defaultCut)} y1={0} y2={innerH} stroke={INK.muted} strokeDasharray="3 3" />
                  <text x={x(defaultCut) + 4} y={12} fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>locked {defaultCut}</text>
                </g>
              )}
              {shown.map((s) => (
                <g key={s.key}>
                  <path d={line(s.pts) ?? ''} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dash} strokeOpacity={0.85} pointerEvents="none" />
                  {s.pts.map((p) => {
                    const isSel = sameConfig({ co: p.r.co, cut: p.r.cut, sel: p.r.sel, sp: p.r.sp }, selected)
                    return (
                      <circle
                        key={p.x}
                        cx={x(p.x)} cy={y(clampY(p.y))} r={isSel ? 6 : isClamped(p.y) ? 2.5 : 3}
                        fill={isSel || isClamped(p.y) ? '#fff' : s.color} stroke={isSel ? INK.text : s.color} strokeWidth={isSel ? 2 : 1}
                        style={{ cursor: 'pointer' }}
                        onMouseEnter={(e) =>
                          tip.show(e, [
                            `${cohort}-${p.x} × ${s.sp}${s.sel !== 'raw' ? ` (sel ${s.sel})` : ''}`,
                            `k = ${p.r.k} (rule)`,
                            `Deming ${fmtFit(allPairs ? p.r.adm : p.r.dm, allPairs ? p.r.adb : p.r.db)} · OLS ${fmtFit(allPairs ? p.r.aom : p.r.om, allPairs ? p.r.aob : p.r.ob)}`,
                            `R² ${(allPairs ? p.r.ar2 : p.r.r2)?.toFixed(3) ?? '—'} · held-out TOR R² ${p.r.ho?.toFixed(3) ?? '—'}`,
                            'click to load into the k-sweep',
                          ])
                        }
                        onMouseLeave={tip.hide}
                        onClick={() => onSelect({ co: p.r.co, cut: p.r.cut, sel: p.r.sel, sp: p.r.sp })}
                      />
                    )
                  })}
                </g>
              ))}
            </g>
          </svg>
        )}
        <Legend
          items={series.map((s) => ({ label: s.label, color: s.color, shape: s.dash ? ('dashed' as const) : ('line' as const) }))}
          hidden={hidden}
          onToggle={(l) => setHidden((h) => toggleIn(h, l))}
          note={isSlopeMetric(metric) ? `shaded = slope box ${calib.slope_box[0]}–${calib.slope_box[1]}` : metric === 'held-out TOR R²' ? `dashed = floor ${calib.heldout_floor}` : undefined}
        />
        {nClamped > 0 && <p className="chart-note">{nClamped} extreme readouts (tiny cohorts) are drawn hollow at the edge of the axis and left out of the line.</p>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
