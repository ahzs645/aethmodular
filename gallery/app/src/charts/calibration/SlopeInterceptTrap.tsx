import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { hexbin as d3Hexbin } from 'd3-hexbin'
import { ChartFrame, Empty, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, MARGIN, FONT, RAMP_SEQUENTIAL } from '@/lib/theme'
import type { CalibFile, CalibRow } from '@/lib/types'
import { cohortColor, fmtFit, passes, sameConfig, score, shortConfig, type Config } from './common'
import { COHORT, CV_PROTOCOL, METRIC, label } from '@/lib/labels'

/**
 * Slope vs intercept — every configuration the batches ever fitted, as one
 * point, at the rule k. This is the "full-scale slope trap" (ftir_17,
 * FIVE_SITE_GRID): ranked by |intercept| alone the winners are slope-0.4
 * lines with intercepts near zero, because a flat enough line always has a
 * small intercept. The shaded column is the slope box the explorer applies
 * before it ranks anything; dimmed points fail it or the IMPROVE
 * cross-validation R² floor.
 */
/** Which evaluation set supplies the Deming intercept (and its matching slope). Deming only: agreed with Ann, 23 Sep 2026. */
const INTERCEPTS = {
  'Deming · fixed set': { m: (r: CalibRow) => r.dm, b: (r: CalibRow) => r.db },
  'Deming · all pairs': { m: (r: CalibRow) => r.adm, b: (r: CalibRow) => r.adb },
} as const
type InterceptKey = keyof typeof INTERCEPTS
const VIEWS = ['Points', 'Density hex'] as const
const MAX_SQUARE = 560

export function SlopeInterceptTrap({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [intercept, setIntercept] = useState<InterceptKey>('Deming · fixed set')
  const [view, setView] = useState<(typeof VIEWS)[number]>('Points')
  const fit = INTERCEPTS[intercept]

  const [target, setTarget] = useState('addis')
  const [mode, setMode] = useState('site_heldout')
  const [onlyPassing, setOnlyPassing] = useState(false)
  const { hidden, dim, props: legendProps } = useLegend()

  const targets = useMemo(() => [...new Set(calib.grid.map((r) => r.tg))].sort(), [calib.grid])
  const modes = useMemo(() => Object.keys(calib.modes).filter((m) => calib.grid.some((r) => r.mo === m)), [calib])

  const rows = useMemo(
    () =>
      calib.grid.filter(
        (r) => r.tg === target && r.mo === mode && r.lot === 'all' && r.el === 'all' && fit.m(r) !== null && fit.b(r) !== null && !hidden.has(r.co) && (!onlyPassing || passes(r, calib))
      ),
    [calib, target, mode, hidden, onlyPassing, fit]
  )
  const cohorts = useMemo(() => [...new Set(calib.grid.filter((r) => r.tg === target && r.mo === mode).map((r) => r.co))], [calib.grid, target, mode])

  // a crossplot is square: slope and intercept read against each other, not against the page width
  const innerW = Math.min(MAX_SQUARE, Math.max(240, width - MARGIN.left - MARGIN.right))
  const innerH = innerW
  const height = innerH + MARGIN.top + MARGIN.bottom
  const mOf = (r: CalibRow) => fit.m(r) as number
  const bOf = (r: CalibRow) => fit.b(r) as number
  // clip the wild tails so the box is readable; count what is off-canvas
  const xs = rows.map(mOf)
  const ys = rows.map(bOf)
  const xHi = Math.min(d3.quantile(xs.slice().sort(d3.ascending), 0.98) ?? 3, 4)
  const yLo = Math.max(d3.quantile(ys.slice().sort(d3.ascending), 0.02) ?? -12, -15)
  const x = d3.scaleLinear().domain([0, Math.max(1.3, xHi)]).range([0, innerW]).nice()
  const yHi = Math.min(Math.max(1, d3.quantile(ys.slice().sort(d3.ascending), 0.98) ?? 1), 8)
  const y = d3.scaleLinear().domain([Math.min(-1, yLo), yHi]).range([innerH, 0]).nice()
  const clampX = (v: number) => Math.max(x.domain()[0], Math.min(x.domain()[1], v))
  const clampYv = (v: number) => Math.max(y.domain()[0], Math.min(y.domain()[1], v))
  const off = rows.filter((r) => clampX(mOf(r)) !== mOf(r) || clampYv(bOf(r)) !== bOf(r)).length

  // density view: bin in pixel space, so hexagons stay regular on the square panel
  const hexes = useMemo(() => {
    if (view !== 'Density hex') return []
    const hb = d3Hexbin<CalibRow>().x((r) => x(clampX(mOf(r)))).y((r) => y(clampYv(bOf(r)))).radius(12).extent([[0, 0], [innerW, innerH]])
    return hb(rows).map((bin) => ({ bin, path: hb.hexagon(), pass: bin.filter((r) => passes(r, calib)).length }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, rows, innerW, innerH, fit, x.domain().join(), y.domain().join()])
  const hexMax = d3.max(hexes, (h) => h.bin.length) ?? 1
  const hexColor = d3.scaleQuantize<string>().domain([0, hexMax]).range(RAMP_SEQUENTIAL.slice(1))
  const best = useMemo(() => rows.filter((r) => passes(r, calib)).sort((a, b) => (score(a) ?? 1e9) - (score(b) ?? 1e9)).slice(0, 5), [rows, calib])

  return (
    <ChartFrame
      id="slope-trap"
      title="Slope vs intercept — the full-scale slope trap"
      subtitle={`Every fitted configuration at its rule number of PLS factors, test-set ${intercept} at MAC 10. A flat line always has a small intercept, so ranking by intercept alone rewards slope-0.4 fits; the shaded column is the slope box the explorer enforces first, and dimmed points fail it or the IMPROVE cross-validation R² floor. Click a point to load it into the PLS-factor sweep. Density hex bins the same points to show where the grid is crowded. The guardrails (slope box, cross-validation R² floor) stay the explorer rule on the fixed-set Deming fit, whichever intercept is plotted.`}
      provenance="FIVE_SITE_GRID_2026-08-23.md “scoring lesson first” · ftir_17"
      controls={
        <>
          <Select label="test set" value={target} options={targets} onChange={setTarget} optionLabel={(t) => calib.targets[t]?.label ?? t} />
          <Select label="cross-validation protocol" value={mode} options={modes} onChange={setMode} optionLabel={(m) => label(CV_PROTOCOL, m)} />
          <Select label="intercept" value={intercept} options={Object.keys(INTERCEPTS)} onChange={(v) => setIntercept(v as InterceptKey)} title="Which fit's intercept goes on y, with its matching slope on x" />
          <Segmented value={view} options={VIEWS} onChange={setView} />
          <Toggle label="only configurations that pass the guardrails" checked={onlyPassing} onChange={setOnlyPassing} />
          <span className="control">{rows.length} configurations</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length === 0 ? (
          <Empty>No rows for {calib.targets[target]?.label ?? target} · {label(CV_PROTOCOL, mode)}.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <rect x={x(calib.slope_box[0])} y={0} width={x(calib.slope_box[1]) - x(calib.slope_box[0])} height={innerH} fill={INK.accent} fillOpacity={0.07} />
              <YAxis scale={y} x={0} label={`Test set Deming intercept (${calib.targets[target]?.site ?? target}, µg/m³)`} gridWidth={innerW} />
              <XAxis scale={x} y={innerH} label={`Test set Deming slope (${calib.targets[target]?.site ?? target})`} />
              <line x1={0} x2={innerW} y1={y(0)} y2={y(0)} stroke={INK.axis} />
              <line x1={x(1)} x2={x(1)} y1={0} y2={innerH} stroke={INK.axis} strokeDasharray="4 3" />
              {view === 'Density hex' && hexes.map((h, i) => (
                <path
                  key={i} d={h.path} transform={`translate(${h.bin.x},${h.bin.y})`}
                  fill={hexColor(h.bin.length)} stroke="#fff" strokeWidth={0.6}
                  onMouseEnter={(e) => tip.show(e, [`${h.bin.length} configuration${h.bin.length === 1 ? '' : 's'} in this hex`, `${h.pass} pass the guardrails`, `slope ≈ ${x.invert(h.bin.x).toFixed(2)} · intercept ≈ ${y.invert(h.bin.y).toFixed(2)}`])}
                  onMouseLeave={tip.hide}
                />
              ))}
              {rows.map((r, i) => {
                const ok = passes(r, calib)
                // in the density view only the selected configuration keeps its own mark
                if (view === 'Density hex' && !sameConfig({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp }, selected)) return null
                const cfg: Config = { co: r.co, cut: r.cut, sel: r.sel, sp: r.sp }
                const isSel = sameConfig(cfg, selected)
                return (
                  <circle
                    key={i}
                    cx={x(clampX(mOf(r)))}
                    cy={y(clampYv(bOf(r)))}
                    r={isSel ? 7 : ok ? 4 : 2.6}
                    fill={isSel ? '#fff' : cohortColor(r.co)}
                    fillOpacity={(ok ? 0.85 : 0.25) * dim(r.co)}
                    strokeOpacity={dim(r.co)}
                    stroke={isSel ? INK.text : r.sp === 'raw' ? 'none' : cohortColor(r.co)}
                    strokeWidth={isSel ? 2 : r.sp === 'airspec' ? 1.5 : 0.8}
                    strokeDasharray={r.sp === 'deriv2' ? '1 1' : undefined}
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={(e) =>
                      tip.show(e, [
                        shortConfig(cfg),
                        `${METRIC.k}: ${r.k}`,
                        `Test set Deming ${fmtFit(r.dm, r.db)} (fixed) · ${fmtFit(r.adm, r.adb)} (all pairs) · R² ${r.r2?.toFixed(3) ?? '—'}`,
                        `${METRIC.cvR2}: ${r.ho?.toFixed(3) ?? '—'} · score ${score(r)?.toFixed(2) ?? '—'}`,
                        ok ? 'passes slope box + cross-validation R² floor' : 'fails the guardrails',
                      ])
                    }
                    onMouseLeave={tip.hide}
                    onClick={() => onSelect(cfg)}
                  />
                )
              })}
              {best.map((r, i) => (
                <text key={i} x={x(clampX(mOf(r))) + 8} y={y(clampYv(bOf(r))) + 3} fontSize={9.5} fill={INK.text} fillOpacity={dim(r.co)} fontFamily={FONT.mono} pointerEvents="none">
                  {i + 1}
                </text>
              ))}
            </g>
          </svg>
        )}
        <Legend
          items={cohorts.map((c) => ({ label: c, color: cohortColor(c), detail: COHORT[c] ?? calib.cohorts[c] }))}
          {...legendProps}
          note="filled = Raw spectra · ring = Spline baseline · dotted ring = Second derivative · numbers = top 5 by score among passing"
        />
        {off > 0 && <p className="chart-note">{off} extreme configurations are pinned to the edge of the canvas.</p>}
        {best.length > 0 && (
          <table className="placement">
            <thead><tr><th>#</th><th>configuration</th><th title={METRIC.k}>PLS factors</th><th>Test set Deming</th><th>Test set R²</th><th title={METRIC.cvR2}>IMPROVE CV R²</th><th>score</th></tr></thead>
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
