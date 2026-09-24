import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, FONT, RAMP_DIVERGING, RAMP_SEQUENTIAL } from '@/lib/theme'
import type { CalibFile, CalibRow } from '@/lib/types'
import { configKey, fmtFit, passes, sameConfig, score, shortConfig, type Config } from './common'
import { METRIC } from '@/lib/labels'

const CELLS = ['Deming intercept', 'Deming slope', 'R²'] as const
type Cell = (typeof CELLS)[number]

/**
 * Cross-site heatmap — is the answer Addis-specific? Rows are configurations
 * (the top N at Addis by the explorer's score inside the guardrails, plus
 * whatever is selected above), columns are the five SPARTAN evaluation sites.
 * CROSS_SITE_EVALUATION_2026-08-22 turned "the offset is Addis-specific" into
 * an ordering: large negative intercepts at Addis and Delhi, near zero at
 * Beijing, JPL and Bishoftu — a compositional signature, not one city's
 * quirk. Quote OFFSET_ADJUDICATION_2026-08-23 for the weighted numbers.
 */
export function CrossSiteHeatmap({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const [cell, setCell] = useState<Cell>('Deming intercept')
  const [topN, setTopN] = useState('15')
  const [mode] = useState('site_heldout')

  const targets = useMemo(() => {
    const present = [...new Set(calib.grid.filter((r) => r.mo === mode).map((r) => r.tg))]
    const order = ['addis', 'etbi', 'indh', 'chts', 'uspa']
    return present.sort((a, b) => (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 99 : order.indexOf(b)))
  }, [calib.grid, mode])

  const { rows, byKey } = useMemo(() => {
    const base = calib.grid.filter((r) => r.mo === mode && r.lot === 'all' && r.el === 'all')
    const byKey = new Map<string, CalibRow>()
    for (const r of base) byKey.set(`${configKey({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })}|${r.tg}`, r)
    const addis = base.filter((r) => r.tg === 'addis' && passes(r, calib)).sort((a, b) => (score(a) ?? 1e9) - (score(b) ?? 1e9))
    const cfgs: Config[] = []
    const seen = new Set<string>()
    const push = (c: Config) => {
      const k = configKey(c)
      if (!seen.has(k)) { seen.add(k); cfgs.push(c) }
    }
    push(selected)
    for (const r of addis.slice(0, Number(topN))) push({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })
    return { rows: cfgs, byKey }
  }, [calib, mode, topN, selected])

  const valueOf = (r: CalibRow | undefined): number | null => {
    if (!r) return null
    if (cell === 'Deming intercept') return r.db
    if (cell === 'Deming slope') return r.dm
    return r.r2
  }
  const color = useMemo(() => {
    if (cell === 'Deming intercept') return d3.scaleLinear<string>().domain([-6, -4, -2, 0, 2, 4, 6]).range(RAMP_DIVERGING).clamp(true)
    if (cell === 'Deming slope') return d3.scaleLinear<string>().domain([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]).range(RAMP_DIVERGING).clamp(true)
    return d3.scaleQuantize<string>().domain([0, 1]).range(RAMP_SEQUENTIAL)
  }, [cell])

  const labelW = 250
  const cellW = Math.max(70, Math.min(140, (width - labelW - 20) / Math.max(1, targets.length)))
  const rowH = 26
  const innerH = rows.length * rowH

  return (
    <ChartFrame
      id="cross-site"
      title="Cross-site heatmap — does the calibration travel?"
      subtitle="The best Addis configurations (by the explorer's score, inside the slope box and above the IMPROVE cross-validation R² floor) read out at every SPARTAN site with FTIR spectra and HIPS Fabs; each site is a test set the model never saw. A row that is blue at Addis and Delhi but white at Beijing and JPL is the compositional-offset signature. The first row is whatever is selected above."
      provenance="CROSS_SITE_EVALUATION_2026-08-22.md · FIVE_SITE_GRID_2026-08-23.md · site-grouped CV · MAC 10 · fixed set"
      controls={
        <>
          <Segmented label="cell" value={cell} options={CELLS} onChange={setCell} />
          <Select label="top N at Addis" value={topN} options={['5', '10', '15', '25', '40']} onChange={setTopN} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap centered">
        {rows.length === 0 || targets.length === 0 ? (
          <Empty>No cross-site rows in the export.</Empty>
        ) : (
          <svg width={width} height={innerH + 40}>
            <g transform={`translate(${labelW},30)`}>
              {targets.map((t, j) => (
                <text key={t} x={j * cellW + cellW / 2} y={-10} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={calib.targets[t]?.color ?? INK.text} fontFamily={FONT.family}>
                  {calib.targets[t]?.site ?? t}
                </text>
              ))}
              {rows.map((cfg, i) => {
                const isSel = sameConfig(cfg, selected)
                return (
                  <g key={configKey(cfg)} transform={`translate(0,${i * rowH})`} style={{ cursor: 'pointer' }} onClick={() => onSelect(cfg)}>
                    <text x={-8} y={rowH / 2} dy="0.32em" textAnchor="end" fontSize={11} fill={isSel ? INK.text : INK.muted} fontWeight={isSel ? 600 : 400} fontFamily={FONT.mono}>
                      {shortConfig(cfg)}
                    </text>
                    {targets.map((t, j) => {
                      const r = byKey.get(`${configKey(cfg)}|${t}`)
                      const v = valueOf(r)
                      return (
                        <g key={t}>
                          <rect x={j * cellW} y={1} width={cellW - 2} height={rowH - 2} rx={2} fill={v === null ? INK.empty : color(v)} stroke={isSel ? INK.text : '#fff'} strokeWidth={isSel ? 1.2 : 1}
                            onMouseEnter={(e) => r && tip.show(e, [
                              `${shortConfig(cfg)} @ ${calib.targets[t]?.site ?? t}`,
                              `${METRIC.k}: ${r.k}`,
                              `Test set Deming ${fmtFit(r.dm, r.db)} · R² ${r.r2?.toFixed(3) ?? '—'}`,
                              `${METRIC.cvR2}: ${r.ho?.toFixed(3) ?? '—'}`,
                            ])}
                            onMouseLeave={tip.hide}
                          />
                          {v !== null && (
                            <text x={j * cellW + cellW / 2} y={rowH / 2} dy="0.32em" textAnchor="middle" fontSize={10.5} fontFamily={FONT.mono} fill={cell === 'R²' ? (v > 0.6 ? '#fff' : INK.text) : Math.abs(cell === 'Deming slope' ? v - 1 : v) > (cell === 'Deming slope' ? 0.45 : 3.5) ? '#fff' : INK.text} pointerEvents="none">
                              {cell === 'R²' ? v.toFixed(2) : v.toFixed(2)}
                            </text>
                          )}
                        </g>
                      )
                    })}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <div className="legend">
          <ColorLegend
            scale={color}
            label={cell === 'Deming intercept' ? 'Test set Deming intercept (µg/m³)' : `Test set ${cell}`}
            width={200}
            ticks={cell === 'R²' ? 5 : 6}
            format={(v) => (cell === 'Deming intercept' ? (v > 0 ? `+${v}` : String(v)) : String(v))}
            note={cell === 'Deming intercept' ? 'white = 0 · clamped at ±6' : cell === 'Deming slope' ? 'white = slope 1' : undefined}
          />
          <span className="legend-note">grey cell = not evaluated at that site</span>
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
