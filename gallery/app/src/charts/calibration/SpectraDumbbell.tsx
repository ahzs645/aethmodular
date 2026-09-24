import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibFile, CalibRow } from '@/lib/types'
import { METRICS, SPECTRA_COLOR, cohortColor, fmtFit, isInterceptMetric, isSlopeMetric, metricLabel, metricOf, sameConfig, type Config, type Metric } from './common'
import { COHORT, CV_PROTOCOL, METRIC, PREPROCESSING, SELECTION_SPACE, label as nameOf } from '@/lib/labels'

const ROWSETS = ['Locked cutoffs', 'One cohort, every cutoff'] as const
const spectraColor = (sp: string) => SPECTRA_COLOR[sp] ?? INK.muted

interface DRow { key: string; cfg: Omit<Config, 'sp'>; label: string; pts: { sp: string; v: number; r: CalibRow }[] }

/**
 * Spectra-space dumbbell — what baseline correction does to one readout,
 * per configuration. Each row is a cohort × cutoff × selection space; the
 * dots are the same configuration calibrated on raw, AIRSpec-corrected and
 * second-derivative spectra, joined so the *change* is the mark. This is the
 * question every "raw vs AIRSpec" pair in the write-ups answers one table at
 * a time: AIRSpec shrinks every intercept, but does it keep the slope?
 *
 * react-graph-gallery's dumbbell (lollipop page) with a third dot.
 */
export function SpectraDumbbell({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const lg = useLegend()

  const cohorts = Object.keys(calib.default_cutoff)
  const [rowset, setRowset] = useState<(typeof ROWSETS)[number]>('Locked cutoffs')
  const [cohort, setCohort] = useState(cohorts.includes('ocec') ? 'ocec' : cohorts[0])
  const [target, setTarget] = useState('addis')
  const [metric, setMetric] = useState<Metric>('Deming intercept')
  const targets = useMemo(() => [...new Set(calib.grid.map((r) => r.tg))].sort(), [calib.grid])

  const rows = useMemo<DRow[]>(() => {
    const base = calib.grid.filter((r) => r.mo === 'site_heldout' && r.tg === target && r.lot === 'all' && r.el === 'all')
    const keep = (r: CalibRow) =>
      rowset === 'Locked cutoffs'
        ? r.cut === null || r.cut === calib.default_cutoff[r.co]
        : r.co === cohort && r.cut !== null
    const groups = d3.group(base.filter(keep), (r) => `${r.co}|${r.cut ?? ''}|${r.sel ?? ''}`)
    const out: DRow[] = []
    for (const [key, rs] of groups) {
      const r0 = rs[0]
      const pts = rs
        .map((r) => ({ sp: r.sp, v: metricOf(r, metric, false), r }))
        .filter((p): p is { sp: string; v: number; r: CalibRow } => p.v !== null && Number.isFinite(p.v))
        .sort((a, b) => Object.keys(SPECTRA_COLOR).indexOf(a.sp) - Object.keys(SPECTRA_COLOR).indexOf(b.sp))
      if (pts.length < 2) continue
      const label = `${r0.co}${r0.cut !== null ? `-${r0.cut}` : ''}${r0.sel && r0.sel !== 'raw' ? ` · ${nameOf(SELECTION_SPACE, r0.sel)}` : ''}`
      out.push({ key, cfg: { co: r0.co, cut: r0.cut, sel: r0.sel }, label, pts })
    }
    const order = ['pool', 'smoke', 'eth_shaped', 'analogs', 'ocec']
    return out.sort((a, b) =>
      rowset === 'Locked cutoffs'
        ? order.indexOf(a.cfg.co) - order.indexOf(b.cfg.co) || (a.cfg.cut ?? 0) - (b.cfg.cut ?? 0) || (a.cfg.sel ?? '').localeCompare(b.cfg.sel ?? '')
        : (a.cfg.cut ?? 0) - (b.cfg.cut ?? 0) || (a.cfg.sel ?? '').localeCompare(b.cfg.sel ?? '')
    )
  }, [calib, rowset, cohort, target, metric])

  const spLabel = (sp: string) => PREPROCESSING[sp] ?? calib.spectra[sp] ?? sp
  const coLabel = (c: string) => COHORT[c] ?? calib.cohorts[c] ?? c
  const siteName = calib.targets[target]?.site ?? target
  const axisLabel = target === 'addis' ? metricLabel(metric) : metricLabel(metric).replace('Addis', siteName)
  const vis = rows
    .filter((r) => rowset !== 'Locked cutoffs' || lg.show(coLabel(r.cfg.co)))
    .map((r) => ({ ...r, pts: r.pts.filter((p) => lg.show(spLabel(p.sp))) }))
    .filter((r) => r.pts.length > 0)
  const hoverSp = Object.keys(SPECTRA_COLOR).some((sp) => spLabel(sp) === lg.hover)
  const rowH = rowset === 'Locked cutoffs' ? 26 : 16
  const labelW = rowset === 'Locked cutoffs' ? 236 : 130
  const innerW = Math.max(240, width - labelW - MARGIN.right - 60)
  const innerH = vis.length * rowH
  const all = vis.flatMap((r) => r.pts.map((p) => p.v)).sort(d3.ascending)
  // tiny cohorts (cutoff ≈ 100) fit wildly; the axis follows the 2–98 % bulk
  // and extremes are pinned hollow at the edge, as in the cutoff sweep
  const ext: [number, number] = all.length ? [d3.quantile(all, 0.02) ?? 0, d3.quantile(all, 0.98) ?? 1] : [0, 1]
  const lo = isInterceptMetric(metric) ? Math.min(0, ext[0]) : isSlopeMetric(metric) ? Math.min(0.5, ext[0]) : Math.min(0, ext[0])
  const hi = isInterceptMetric(metric) ? Math.max(0, ext[1]) : isSlopeMetric(metric) ? Math.max(1.2, ext[1]) : Math.max(1, ext[1])
  const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]).nice()
  const [x0, x1] = x.domain() as [number, number]
  const clamp = (v: number) => Math.max(x0, Math.min(x1, v))
  const isClamped = (v: number) => v < x0 || v > x1
  const nClamped = all.filter(isClamped).length
  const ref = isSlopeMetric(metric) ? 1 : isInterceptMetric(metric) ? 0 : null

  return (
    <ChartFrame
      id="spectra-dumbbell"
      title="Preprocessing dumbbell — what baseline correction does to each configuration"
      subtitle={`One row per cohort × calibration cohort size × selection space; the dots are the same configuration calibrated on raw spectra, spline-baselined spectra and second-derivative spectra, joined so the change is the mark. ${CV_PROTOCOL.site_heldout}, fixed test set, rule number of PLS factors. Click a dot to load that configuration above.`}
      provenance="calibration_explorer batch results · react-graph-gallery.com/lollipop (dumbbell)"
      controls={
        <>
          <Segmented label="rows" value={rowset} options={ROWSETS} onChange={setRowset} />
          {rowset !== 'Locked cutoffs' && <Select label="cohort" value={cohort} options={cohorts} onChange={setCohort} optionLabel={(c) => nameOf(COHORT, c)} />}
          <Select label="test set" value={target} options={targets} onChange={setTarget} optionLabel={(t) => calib.targets[t]?.label ?? t} />
          <Select label="readout" value={metric} options={[...METRICS]} onChange={(v) => setMetric(v as Metric)} optionLabel={metricLabel} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length === 0 ? (
          <Empty>No configuration has two preprocessings at {siteName}.</Empty>
        ) : (
          <svg width={width} height={innerH + MARGIN.top + MARGIN.bottom} className="animated">
            <g transform={`translate(${labelW},${MARGIN.top})`}>
              {isSlopeMetric(metric) && (
                <rect x={Math.max(0, x(calib.slope_box[0]))} y={0} width={Math.max(0, Math.min(innerW, x(calib.slope_box[1])) - Math.max(0, x(calib.slope_box[0])))} height={innerH} fill={INK.accent} fillOpacity={0.06} />
              )}
              {metric === 'held-out TOR R²' && <line x1={x(calib.heldout_floor)} x2={x(calib.heldout_floor)} y1={0} y2={innerH} stroke={INK.axis} strokeDasharray="4 3" />}
              {x.ticks(6).map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={innerH} stroke={INK.grid} />
              ))}
              {ref !== null && ref >= x0 && ref <= x1 && <line x1={x(ref)} x2={x(ref)} y1={0} y2={innerH} stroke={INK.axis} strokeWidth={1} strokeOpacity={0.7} />}
              {vis.map((row, i) => {
                const cy = i * rowH + rowH / 2
                const vs = row.pts.map((p) => clamp(p.v))
                const isSelRow = row.pts.some((p) => sameConfig({ ...row.cfg, sp: p.sp }, selected))
                return (
                  <g key={row.key} opacity={lg.hover && !hoverSp ? lg.dim(coLabel(row.cfg.co)) : 1}>
                    <text x={-10} y={cy} dy="0.32em" textAnchor="end" fontSize={rowset === 'Locked cutoffs' ? 11.5 : 10} fill={isSelRow ? INK.text : INK.muted} fontWeight={isSelRow ? 600 : 400} fontFamily={FONT.mono}>
                      {rowset === 'Locked cutoffs' ? row.label : `${row.cfg.cut}${row.cfg.sel && row.cfg.sel !== 'raw' ? ` · ${nameOf(SELECTION_SPACE, row.cfg.sel)}` : ''}`}
                    </text>
                    <line x1={x(d3.min(vs)!)} x2={x(d3.max(vs)!)} y1={cy} y2={cy} stroke={cohortColor(row.cfg.co)} strokeWidth={rowset === 'Locked cutoffs' ? 3 : 2} strokeOpacity={0.45} strokeLinecap="round" />
                    {row.pts.map((p) => {
                      const cfg: Config = { ...row.cfg, sp: p.sp }
                      const isSel = sameConfig(cfg, selected)
                      const pinned = isClamped(p.v)
                      return (
                        <circle
                          key={p.sp}
                          cx={x(clamp(p.v))} cy={cy} r={isSel ? 7 : rowset === 'Locked cutoffs' ? 5.5 : 4}
                          fill={pinned ? '#fff' : spectraColor(p.sp)} stroke={isSel ? INK.text : spectraColor(p.sp)} strokeWidth={isSel ? 2.2 : 1.2}
                          opacity={hoverSp ? lg.dim(spLabel(p.sp)) : 1}
                          style={{ cursor: 'pointer' }}
                          onMouseEnter={(e) =>
                            tip.show(e, [
                              `${row.label} × ${spLabel(p.sp)}`,
                              `${axisLabel} = ${p.v.toFixed(3)}${pinned ? ' (off the axis)' : ''}`,
                              `${METRIC.k}: ${p.r.k} · test set Deming ${fmtFit(p.r.dm, p.r.db)} · R² ${p.r.r2?.toFixed(3) ?? '—'}`,
                              `${METRIC.cvR2}: ${p.r.ho?.toFixed(3) ?? '—'}`,
                              'click to load this configuration',
                            ])
                          }
                          onMouseLeave={tip.hide}
                          onClick={() => onSelect(cfg)}
                        />
                      )
                    })}
                  </g>
                )
              })}
              <XAxis scale={x} y={innerH} label={axisLabel} />
            </g>
          </svg>
        )}
        <Legend
          items={[
            ...Object.keys(SPECTRA_COLOR).map((sp) => ({ label: spLabel(sp), color: spectraColor(sp) })),
            ...(rowset === 'Locked cutoffs' ? cohorts.filter((c) => rows.some((r) => r.cfg.co === c)).map((c) => ({ label: coLabel(c), color: cohortColor(c), shape: 'line' as const })) : []),
          ]}
          {...lg.props}
          note={isSlopeMetric(metric) ? `shaded = slope box ${calib.slope_box[0]}–${calib.slope_box[1]}` : metric === 'held-out TOR R²' ? `dashed = floor ${calib.heldout_floor}` : 'the joining line is the cohort colour; the dots are the preprocessing'}
        />
        {nClamped > 0 && <p className="chart-note">{nClamped} extreme readouts (tiny cohorts) are drawn hollow at the edge of the axis.</p>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
