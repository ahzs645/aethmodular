import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibFile } from '@/lib/types'
import { cohortColor, fmtFit, passes, sameConfig, score, shortConfig, type Config } from './common'

/**
 * Slope vs intercept — every configuration the batches ever fitted, as one
 * point, at the rule k. This is the "full-scale slope trap" (ftir_17,
 * FIVE_SITE_GRID): ranked by |intercept| alone the winners are slope-0.4
 * lines with intercepts near zero, because a flat enough line always has a
 * small intercept. The shaded column is the slope box the explorer applies
 * before it ranks anything; dimmed points fail it or the held-out TOR floor.
 */
export function SlopeInterceptTrap({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 460

  const [target, setTarget] = useState('addis')
  const [mode, setMode] = useState('site_heldout')
  const [onlyPassing, setOnlyPassing] = useState(false)
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const targets = useMemo(() => [...new Set(calib.grid.map((r) => r.tg))].sort(), [calib.grid])
  const modes = useMemo(() => Object.keys(calib.modes).filter((m) => calib.grid.some((r) => r.mo === m)), [calib])

  const rows = useMemo(
    () =>
      calib.grid.filter(
        (r) => r.tg === target && r.mo === mode && r.lot === 'all' && r.el === 'all' && r.dm !== null && r.db !== null && !hidden.has(r.co) && (!onlyPassing || passes(r, calib))
      ),
    [calib, target, mode, hidden, onlyPassing]
  )
  const cohorts = useMemo(() => [...new Set(calib.grid.filter((r) => r.tg === target && r.mo === mode).map((r) => r.co))], [calib.grid, target, mode])

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  // clip the wild tails so the box is readable; count what is off-canvas
  const xs = rows.map((r) => r.dm as number)
  const ys = rows.map((r) => r.db as number)
  const xHi = Math.min(d3.quantile(xs.slice().sort(d3.ascending), 0.98) ?? 3, 4)
  const yLo = Math.max(d3.quantile(ys.slice().sort(d3.ascending), 0.02) ?? -12, -15)
  const x = d3.scaleLinear().domain([0, Math.max(1.3, xHi)]).range([0, innerW]).nice()
  const yHi = Math.min(Math.max(1, d3.quantile(ys.slice().sort(d3.ascending), 0.98) ?? 1), 8)
  const y = d3.scaleLinear().domain([Math.min(-1, yLo), yHi]).range([innerH, 0]).nice()
  const clampX = (v: number) => Math.max(x.domain()[0], Math.min(x.domain()[1], v))
  const clampYv = (v: number) => Math.max(y.domain()[0], Math.min(y.domain()[1], v))
  const off = rows.filter((r) => clampX(r.dm as number) !== r.dm || clampYv(r.db as number) !== r.db).length
  const best = useMemo(() => rows.filter((r) => passes(r, calib)).sort((a, b) => (score(a) ?? 1e9) - (score(b) ?? 1e9)).slice(0, 5), [rows, calib])

  return (
    <ChartFrame
      id="slope-trap"
      title="Slope vs intercept — the full-scale slope trap"
      subtitle="Every fitted configuration at its rule k, Deming at MAC 10 on the fixed set. A flat line always has a small intercept, so ranking by intercept alone rewards slope-0.4 fits; the shaded column is the slope box the explorer enforces first, and dimmed points fail it or the held-out TOR floor. Click a point to load it into the k-sweep."
      provenance="FIVE_SITE_GRID_2026-08-23.md “scoring lesson first” · ftir_17"
      controls={
        <>
          <Select label="evaluate on" value={target} options={targets} onChange={setTarget} />
          <Select label="protocol" value={mode} options={modes} onChange={setMode} />
          <Toggle label="only configurations that pass the guardrails" checked={onlyPassing} onChange={setOnlyPassing} />
          <span className="control">{rows.length} configurations</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length === 0 ? (
          <Empty>No rows for {target} · {mode}.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <rect x={x(calib.slope_box[0])} y={0} width={x(calib.slope_box[1]) - x(calib.slope_box[0])} height={innerH} fill={INK.accent} fillOpacity={0.07} />
              <YAxis scale={y} x={0} label="Deming intercept (µg/m³)" gridWidth={innerW} />
              <XAxis scale={x} y={innerH} label="Deming slope" />
              <line x1={0} x2={innerW} y1={y(0)} y2={y(0)} stroke={INK.axis} />
              <line x1={x(1)} x2={x(1)} y1={0} y2={innerH} stroke={INK.axis} strokeDasharray="4 3" />
              {rows.map((r, i) => {
                const ok = passes(r, calib)
                const cfg: Config = { co: r.co, cut: r.cut, sel: r.sel, sp: r.sp }
                const isSel = sameConfig(cfg, selected)
                return (
                  <circle
                    key={i}
                    cx={x(clampX(r.dm as number))}
                    cy={y(clampYv(r.db as number))}
                    r={isSel ? 7 : ok ? 4 : 2.6}
                    fill={isSel ? '#fff' : cohortColor(r.co)}
                    fillOpacity={ok ? 0.85 : 0.25}
                    stroke={isSel ? INK.text : r.sp === 'raw' ? 'none' : cohortColor(r.co)}
                    strokeWidth={isSel ? 2 : r.sp === 'airspec' ? 1.5 : 0.8}
                    strokeDasharray={r.sp === 'deriv2' ? '1 1' : undefined}
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={(e) =>
                      tip.show(e, [
                        shortConfig(cfg),
                        `k = ${r.k} · Deming ${fmtFit(r.dm, r.db)}`,
                        `OLS ${fmtFit(r.om, r.ob)} · R² ${r.r2?.toFixed(3) ?? '—'}`,
                        `held-out TOR R² ${r.ho?.toFixed(3) ?? '—'} · score ${score(r)?.toFixed(2) ?? '—'}`,
                        ok ? 'passes slope box + held-out floor' : 'fails the guardrails',
                      ])
                    }
                    onMouseLeave={tip.hide}
                    onClick={() => onSelect(cfg)}
                  />
                )
              })}
              {best.map((r, i) => (
                <text key={i} x={x(clampX(r.dm as number)) + 8} y={y(clampYv(r.db as number)) + 3} fontSize={9.5} fill={INK.text} fontFamily={FONT.mono} pointerEvents="none">
                  {i + 1}
                </text>
              ))}
            </g>
          </svg>
        )}
        <Legend
          items={cohorts.map((c) => ({ label: c, color: cohortColor(c), detail: calib.cohorts[c] }))}
          hidden={hidden}
          onToggle={(l) => setHidden((h) => toggleIn(h, l))}
          note="filled = raw spectra · ring = AIRSpec · dotted ring = 2nd derivative · numbers = top 5 by score among passing"
        />
        {off > 0 && <p className="chart-note">{off} extreme configurations are pinned to the edge of the canvas.</p>}
        {best.length > 0 && (
          <table className="placement">
            <thead><tr><th>#</th><th>configuration</th><th>k</th><th>Deming</th><th>R²</th><th>held-out</th><th>score</th></tr></thead>
            <tbody>
              {best.map((r, i) => (
                <tr key={i} className="link" onClick={() => onSelect({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })}>
                  <td className="f">{i + 1}</td>
                  <td className="f">{shortConfig({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })}</td>
                  <td className="v">{r.k}</td>
                  <td className="v">{fmtFit(r.dm, r.db)}</td>
                  <td className="v">{r.r2?.toFixed(3)}</td>
                  <td className="v">{r.ho?.toFixed(3)}</td>
                  <td className="v">{score(r)?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
