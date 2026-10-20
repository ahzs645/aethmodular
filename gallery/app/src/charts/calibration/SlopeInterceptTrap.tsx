import { useMemo, useRef, useState } from 'react'
import { AxisChoices } from '@/components/AxisChoices'
import { useSearchParam } from '@/lib/url'
import { robustDomain } from '@/lib/axes'
import { RefLine } from '@/components/RefLines'
import { EstimatorName } from '@/lib/regressionPrefs'
import * as d3 from 'd3'
import { hexbin as d3Hexbin } from 'd3-hexbin'
import { ChartFrame, Empty, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, MARGIN, FONT } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'
import { ColorLegend } from '@/components/ColorLegend'
import type { CalibFile, CalibRow } from '@/lib/types'
import { fmt } from '@/lib/stats'
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
/**
 * What each axis can show (click its title, components/AxisChoices): the test-set Deming slope and
 * intercept on the fixed set or on all pairs (Deming only: agreed with Ann, 23 Sep 2026), the fit
 * quality, the number of PLS factors, the cohort size and the explorer's score. The trap itself is
 * slope (x) against intercept (y); the slope box and the reference lines follow slope and intercept
 * wherever they sit.
 */
type Kind = 'slope' | 'intercept' | 'other'
const METRICS: Record<string, { get: (r: CalibRow) => number | null; kind: Kind; unit?: string }> = {
  'Slope · fixed set': { get: (r) => r.dm, kind: 'slope' },
  'Intercept · fixed set': { get: (r) => r.db, kind: 'intercept', unit: 'µg/m³' },
  'Slope · all pairs': { get: (r) => r.adm, kind: 'slope' },
  'Intercept · all pairs': { get: (r) => r.adb, kind: 'intercept', unit: 'µg/m³' },
  'Test set R²': { get: (r) => r.r2, kind: 'other' },
  'IMPROVE cross-validation R²': { get: (r) => r.ho, kind: 'other' },
  'PLS factors': { get: (r) => r.k, kind: 'other' },
  'Cohort size': { get: (r) => r.cut ?? r.n_cohort ?? null, kind: 'other' },
  'Score': { get: (r) => score(r) ?? null, kind: 'other' },
}
const METRIC_KEYS = Object.keys(METRICS)
const VIEWS = ['Points', 'Density hex'] as const
const MAX_SQUARE = 560

export function SlopeInterceptTrap({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const hexColours = useColourScheme('slope-trap-hex', 'sequential')
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const ax = useSearchParam<string>('trapx', 'Slope · fixed set', METRIC_KEYS)
  const ay = useSearchParam<string>('trapy', 'Intercept · fixed set', METRIC_KEYS)
  const [view, setView] = useState<(typeof VIEWS)[number]>('Points')
  const mx = METRICS[ax[0]], my = METRICS[ay[0]]

  const [target, setTarget] = useState('addis')
  const [mode, setMode] = useState('site_heldout')
  const [onlyPassing, setOnlyPassing] = useState(false)
  const { hidden, dim, props: legendProps } = useLegend()
  // the legend names cohorts in words; hiding and dimming go by the same name
  const cohortName = (c: string) => COHORT[c] ?? calib.cohorts[c] ?? c

  const targets = useMemo(() => [...new Set(calib.grid.map((r) => r.tg))].sort(), [calib.grid])
  const modes = useMemo(() => Object.keys(calib.modes).filter((m) => calib.grid.some((r) => r.mo === m)), [calib])

  const rows = useMemo(
    () =>
      calib.grid.filter(
        (r) => r.tg === target && r.mo === mode && r.lot === 'all' && r.el === 'all' && Number.isFinite(mx.get(r) as number) && Number.isFinite(my.get(r) as number) && !hidden.has(cohortName(r.co)) && (!onlyPassing || passes(r, calib))
      ),
    [calib, target, mode, hidden, onlyPassing, mx, my]
  )
  const cohorts = useMemo(() => [...new Set(calib.grid.filter((r) => r.tg === target && r.mo === mode).map((r) => r.co))], [calib.grid, target, mode])

  // a crossplot is square: slope and intercept read against each other, not against the page width
  const innerW = Math.min(MAX_SQUARE, Math.max(240, width - MARGIN.left - MARGIN.right))
  const innerH = innerW
  // centred in the frame: the square leaves the rest of a wide frame as even margins either side
  const left = MARGIN.left + Math.max(0, (width - MARGIN.left - MARGIN.right - innerW) / 2)
  const height = innerH + MARGIN.top + MARGIN.bottom
  const mOf = (r: CalibRow) => mx.get(r) as number
  const bOf = (r: CalibRow) => my.get(r) as number
  // clip the wild tails so the box is readable; count what is off-canvas
  const xs = rows.map(mOf)
  const ys = rows.map(bOf)
  // each kind keeps its own sensible range: slopes from 0 past 1, intercepts around 0, the rest robust
  const rangeOf = (kind: Kind, v: number[]): [number, number] => {
    if (kind === 'slope') return [0, Math.max(1.3, robustDomain(v, 0, 0.98, { limit: [-Infinity, 4], fallback: [0, 3] }).domain[1])]
    if (kind === 'intercept') { const d = robustDomain(v, 0.02, 0.98, { include: [1], limit: [-15, 8], fallback: [-12, 1] }).domain; return [Math.min(-1, d[0]), d[1]] }
    const d = robustDomain(v, 0.01, 0.99, { fallback: [0, 1] }).domain; const pad = (d[1] - d[0]) * 0.05 || 0.5; return [d[0] - pad, d[1] + pad]
  }
  const x = d3.scaleLinear().domain(rangeOf(mx.kind, xs)).range([0, innerW]).nice()
  const y = d3.scaleLinear().domain(rangeOf(my.kind, ys)).range([innerH, 0]).nice()
  const clampX = (v: number) => Math.max(x.domain()[0], Math.min(x.domain()[1], v))
  const clampYv = (v: number) => Math.max(y.domain()[0], Math.min(y.domain()[1], v))
  const off = rows.filter((r) => clampX(mOf(r)) !== mOf(r) || clampYv(bOf(r)) !== bOf(r)).length

  // density view: bin in pixel space, so hexagons stay regular on the square panel
  const hexes = useMemo(() => {
    if (view !== 'Density hex') return []
    const hb = d3Hexbin<CalibRow>().x((r) => x(clampX(mOf(r)))).y((r) => y(clampYv(bOf(r)))).radius(12).extent([[0, 0], [innerW, innerH]])
    return hb(rows).map((bin) => ({ bin, path: hb.hexagon(), pass: bin.filter((r) => passes(r, calib)).length }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, rows, innerW, innerH, mx, my, x.domain().join(), y.domain().join()])
  const hexMax = d3.max(hexes, (h) => h.bin.length) ?? 1
  // the palette is picked from the density ramp (lib/colourSchemes), as on the other density charts
  const hexColor = d3.scaleQuantize<string>().domain([0, hexMax]).range(hexColours.stops.slice(1))
  const best = useMemo(() => rows.filter((r) => passes(r, calib)).sort((a, b) => (score(a) ?? 1e9) - (score(b) ?? 1e9)).slice(0, 5), [rows, calib])

  return (
    <ChartFrame
      id="slope-trap"
      title="Slope vs intercept — the full-scale slope trap"
      subtitle={`Every fitted configuration at its rule number of PLS factors, test-set Deming at MAC 10; by default slope against intercept (click an axis title for another metric). A flat line always has a small intercept, so ranking by intercept alone rewards slope-0.4 fits; the shaded column is the slope box the explorer enforces first, and dimmed points fail it or the IMPROVE cross-validation R² floor. Click a point to load it into the PLS-factor sweep. Density hex bins the same points to show where the grid is crowded. The guardrails (slope box, cross-validation R² floor) stay the explorer rule on the fixed-set Deming fit, whichever intercept is plotted.`}
      provenance="FIVE_SITE_GRID_2026-08-23.md “scoring lesson first” · ftir_17"
      controls={
        <>
          <Select label="Test set" value={target} options={targets} onChange={setTarget} optionLabel={(t) => calib.targets[t]?.label ?? t} />
          <Select label="Cross-validation protocol" value={mode} options={modes} onChange={setMode} optionLabel={(m) => label(CV_PROTOCOL, m)} />
          <Segmented value={view} options={VIEWS} onChange={setView} />
          <Toggle label="Only configurations that pass the guardrails" checked={onlyPassing} onChange={setOnlyPassing} />
          <span className="control">{rows.length} configurations</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length === 0 ? (
          <Empty>No rows for {calib.targets[target]?.label ?? target} · {label(CV_PROTOCOL, mode)}.</Empty>
        ) : (
          <AxisChoices noun="metrics" x={ax} y={ay} options={METRIC_KEYS.map((k) => ({ key: k, group: METRICS[k].kind === 'other' ? 'Quality and size' : 'Test set Deming fit', unit: METRICS[k].unit }))}>
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${left},${MARGIN.top})`}>
              {mx.kind === 'slope' && <rect x={x(calib.slope_box[0])} y={0} width={x(calib.slope_box[1]) - x(calib.slope_box[0])} height={innerH} fill={INK.accent} fillOpacity={0.07} />}
              {my.kind === 'slope' && <rect x={0} y={y(calib.slope_box[1])} width={innerW} height={y(calib.slope_box[0]) - y(calib.slope_box[1])} fill={INK.accent} fillOpacity={0.07} />}
              <YAxis scale={y} x={0} label={`${my.kind !== 'other' ? 'Test set ' : ''}${ay[0]}${my.kind !== 'other' ? ` (${calib.targets[target]?.site ?? target}${my.unit ? `, ${my.unit}` : ''})` : ''}`} gridWidth={innerW} pick="y" />
              <XAxis scale={x} y={innerH} label={`${mx.kind !== 'other' ? 'Test set ' : ''}${ax[0]}${mx.kind !== 'other' ? ` (${calib.targets[target]?.site ?? target}${mx.unit ? `, ${mx.unit}` : ''})` : ''}`} pick="x" />
              {my.kind === 'intercept' && <line x1={0} x2={innerW} y1={y(0)} y2={y(0)} stroke={INK.axis} />}
              {mx.kind === 'intercept' && <line x1={x(0)} x2={x(0)} y1={0} y2={innerH} stroke={INK.axis} />}
              {mx.kind === 'slope' && <RefLine orient="v" at={x(1)} from={0} to={innerH} />}
              {my.kind === 'slope' && <RefLine orient="h" at={y(1)} from={0} to={innerW} />}
              {view === 'Density hex' && hexes.map((h, i) => (
                <path
                  key={i} d={h.path} transform={`translate(${h.bin.x},${h.bin.y})`}
                  fill={hexColor(h.bin.length)} stroke="#fff" strokeWidth={0.6}
                  onMouseEnter={(e) => tip.show(e, [`${h.bin.length} configuration${h.bin.length === 1 ? '' : 's'} in this hex`, `${h.pass} pass the guardrails`, `${ax[0]} ≈ ${x.invert(h.bin.x).toFixed(2)} · ${ay[0]} ≈ ${y.invert(h.bin.y).toFixed(2)}`])}
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
                    fillOpacity={(ok ? 0.85 : 0.25) * dim(cohortName(r.co))}
                    strokeOpacity={dim(cohortName(r.co))}
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
                <text key={i} x={x(clampX(mOf(r))) + 8} y={y(clampYv(bOf(r))) + 3} fontSize={9.5} fill={INK.text} fillOpacity={dim(cohortName(r.co))} fontFamily={FONT.mono} pointerEvents="none">
                  {i + 1}
                </text>
              ))}
            </g>
          </svg>
          </AxisChoices>
        )}
        <Legend
          items={cohorts.map((c) => ({ label: cohortName(c), color: cohortColor(c) }))}
          {...legendProps}
          note="filled = Raw spectra · ring = Spline baseline · dotted ring = Second derivative · numbers = top 5 by score among passing"
        />
        {view === 'Density hex' && hexes.length > 0 && <div className="legend"><ColorLegend scale={hexColor} scheme={hexColours.scheme} label="configurations per hex" width={160} format={(v) => String(Math.round(v))} /></div>}
        {off > 0 && <p className="chart-note">{off} extreme configurations are pinned to the edge of the canvas.</p>}
        {best.length > 0 && (
          <table className="placement">
            <thead><tr><th>#</th><th>configuration</th><th title={METRIC.k}>PLS factors</th><th>Test set <EstimatorName /></th><th>Test set R²</th><th title={METRIC.cvR2}>IMPROVE cross-validation R²</th><th>score</th></tr></thead>
            <tbody>
              {best.map((r, i) => (
                <tr key={i} className="link" onClick={() => onSelect({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })}>
                  <td className="f">{i + 1}</td>
                  <td className="f">{shortConfig({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })}</td>
                  <td className="v">{r.k}</td>
                  <td className="v">{fmtFit(r.dm, r.db)}</td>
                  <td className="v">{fmt(r.r2, 3)}</td>
                  <td className="v">{fmt(r.ho, 3)}</td>
                  <td className="v">{fmt(score(r), 2)}</td>
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
