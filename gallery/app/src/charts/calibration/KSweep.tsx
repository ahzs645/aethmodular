import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, FONT } from '@/lib/theme'
import type { CalibFile } from '@/lib/types'
import { configLabel, sameConfig, type Config } from './common'

const PANELS: { title: string; keys: { key: 'db' | 'ob' | 'dm' | 'om' | 'r2' | 'ho'; label: string; color: string; dash?: string }[]; ref?: number }[] = [
  { title: 'intercept (µg/m³)', keys: [{ key: 'db', label: 'Deming', color: INK.deming }, { key: 'ob', label: 'OLS', color: INK.fit }], ref: 0 },
  { title: 'slope', keys: [{ key: 'dm', label: 'Deming', color: INK.deming }, { key: 'om', label: 'OLS', color: INK.fit }], ref: 1 },
  { title: 'R²', keys: [{ key: 'r2', label: 'Addis crossplot R²', color: INK.accent }, { key: 'ho', label: 'held-out TOR R²', color: INK.good, dash: '5 3' }] },
]

/**
 * k sweep — the component-count "optimization" for one configuration, read
 * out on Addis under protocol A. The explorer's rule k is marked. The trap
 * this chart exists to show: intercept usually keeps shrinking with k while
 * slope and the held-out TOR test deteriorate, so the k with the smallest
 * intercept is rarely the k to report (CONVERSATION_OPTIMIZATION_AUDIT).
 */
export function KSweep({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const cohorts = useMemo(() => [...new Set(calib.sweeps.map((s) => s.co))], [calib.sweeps])
  const cutoffs = useMemo(
    () => [...new Set(calib.sweeps.filter((s) => s.co === selected.co).map((s) => s.cut))].filter((c): c is number => c !== null).sort((a, b) => a - b),
    [calib.sweeps, selected.co]
  )
  const sels = useMemo(() => [...new Set(calib.sweeps.filter((s) => s.co === selected.co && s.cut === selected.cut).map((s) => s.sel ?? 'raw'))], [calib.sweeps, selected])
  const sps = useMemo(() => [...new Set(calib.sweeps.filter((s) => s.co === selected.co && s.cut === selected.cut && (s.sel ?? 'raw') === (selected.sel ?? 'raw')).map((s) => s.sp))], [calib.sweeps, selected])

  const rows = useMemo(
    () => calib.sweeps.filter((s) => sameConfig({ co: s.co, cut: s.cut, sel: s.sel, sp: s.sp }, selected)).sort((a, b) => a.k - b.k),
    [calib.sweeps, selected]
  )
  const autoK = rows[0]?.ak ?? null

  const cols = width > 760 ? 3 : 1
  const panelW = Math.floor((width - (cols - 1) * 12) / cols)
  const m = { top: 24, right: 14, bottom: 42, left: 52 }
  const iw = Math.max(160, panelW - m.left - m.right)
  const ih = 200
  const x = d3.scaleLinear().domain([1, d3.max(rows, (r) => r.k) ?? 30]).range([0, iw])

  return (
    <ChartFrame
      id="k-sweep"
      title="k sweep — the component-count optimisation for one configuration"
      subtitle="Every PLS component count the batch tried for the configuration chosen here (or clicked in the sweep above), read out on Addis under protocol A. The dashed vertical line is the rule k the explorer would choose on its own."
      provenance="calibration_explorer batch results · Addis · site-held-out · MAC 10 · fixed set"
      controls={
        <>
          <Select label="cohort" value={selected.co} options={cohorts} onChange={(co) => {
            const first = calib.sweeps.find((s) => s.co === co)
            onSelect({ co, cut: first?.cut ?? null, sel: first?.sel ?? null, sp: first?.sp ?? 'raw' })
          }} />
          {cutoffs.length > 0 && (
            <Select label="cutoff" value={String(selected.cut ?? '')} options={cutoffs.map(String)} onChange={(v) => onSelect({ ...selected, cut: Number(v) })} />
          )}
          {sels.length > 1 && <Select label="selected on" value={selected.sel ?? 'raw'} options={sels} onChange={(v) => onSelect({ ...selected, sel: v })} />}
          <Select label="calibration spectra" value={selected.sp} options={sps.length ? sps : [selected.sp]} onChange={(v) => onSelect({ ...selected, sp: v })} />
          <span className="control">{rows.length} k values{autoK !== null ? ` · rule k = ${autoK}` : ''}</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length < 2 ? (
          <Empty>No k sweep saved for {configLabel(selected, calib)} on Addis.</Empty>
        ) : (
          <div className="facet-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            {PANELS.map((panel) => {
              const vals = panel.keys.flatMap((k) => rows.map((r) => r[k.key]).filter((v): v is number => v !== null))
              const ext = d3.extent(vals) as [number, number]
              const lo = panel.ref !== undefined ? Math.min(panel.ref, ext[0]) : Math.min(0, ext[0])
              const hi = panel.ref !== undefined ? Math.max(panel.ref, ext[1]) : Math.max(1, ext[1])
              const y = d3.scaleLinear().domain([lo, hi]).range([ih, 0]).nice()
              return (
                <div key={panel.title}>
                  <p className="facet-title">{panel.title}</p>
                  <svg width={panelW} height={ih + m.top + m.bottom} className="animated">
                    <g transform={`translate(${m.left},${m.top})`}>
                      <YAxis scale={y} x={0} gridWidth={iw} tickCount={5} />
                      <XAxis scale={x} y={ih} label="components k" tickCount={6} />
                      {panel.ref !== undefined && <line x1={0} x2={iw} y1={y(panel.ref)} y2={y(panel.ref)} stroke={INK.axis} strokeOpacity={0.7} />}
                      {panel.title === 'slope' && (
                        <rect x={0} y={Math.max(0, y(calib.slope_box[1]))} width={iw} height={Math.max(0, Math.min(ih, y(calib.slope_box[0])) - Math.max(0, y(calib.slope_box[1])))} fill={INK.accent} fillOpacity={0.06} />
                      )}
                      {panel.title === 'R²' && <line x1={0} x2={iw} y1={y(calib.heldout_floor)} y2={y(calib.heldout_floor)} stroke={INK.good} strokeDasharray="2 3" strokeOpacity={0.6} />}
                      {autoK !== null && (
                        <g>
                          <line x1={x(autoK)} x2={x(autoK)} y1={0} y2={ih} stroke={INK.muted} strokeDasharray="3 3" />
                          <text x={x(autoK) + 3} y={-6} fontSize={9.5} fill={INK.muted} fontFamily={FONT.mono}>rule k={autoK}</text>
                        </g>
                      )}
                      {panel.keys.map((k) => {
                        const pts = rows.map((r) => ({ k: r.k, v: r[k.key] })).filter((p): p is { k: number; v: number } => p.v !== null)
                        const line = d3.line<{ k: number; v: number }>().x((p) => x(p.k)).y((p) => y(p.v))
                        return (
                          <g key={k.key}>
                            <path d={line(pts) ?? ''} fill="none" stroke={k.color} strokeWidth={1.8} strokeDasharray={k.dash} pointerEvents="none" />
                            {pts.map((p) => (
                              <circle
                                key={p.k} cx={x(p.k)} cy={y(p.v)} r={p.k === autoK ? 4.5 : 2.8} fill={p.k === autoK ? '#fff' : k.color} stroke={k.color} strokeWidth={p.k === autoK ? 2 : 1}
                                onMouseEnter={(e) => {
                                  const r = rows.find((q) => q.k === p.k)!
                                  tip.show(e, [
                                    `k = ${p.k}${p.k === autoK ? ' (rule)' : ''}`,
                                    `Deming ${r.dm?.toFixed(3) ?? '—'}x ${(r.db ?? 0) < 0 ? '−' : '+'} ${Math.abs(r.db ?? 0).toFixed(3)}`,
                                    `OLS ${r.om?.toFixed(3) ?? '—'}x ${(r.ob ?? 0) < 0 ? '−' : '+'} ${Math.abs(r.ob ?? 0).toFixed(3)}`,
                                    `R² ${r.r2?.toFixed(3) ?? '—'} · held-out TOR R² ${r.ho?.toFixed(3) ?? '—'}`,
                                  ])
                                }}
                                onMouseLeave={tip.hide}
                              />
                            ))}
                          </g>
                        )
                      })}
                    </g>
                  </svg>
                  <Legend items={panel.keys.map((k) => ({ label: k.label, color: k.color, shape: k.dash ? ('dashed' as const) : ('line' as const) }))} />
                </div>
              )
            })}
          </div>
        )}
        <p className="chart-note">{configLabel(selected, calib)}</p>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
