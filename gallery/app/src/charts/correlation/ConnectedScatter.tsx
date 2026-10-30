import { useId, useMemo, useRef } from 'react'
import { StepButton } from '@/components/StepButton'
import { IdentityLine } from '@/components/RefLines'
import * as d3 from 'd3'
import { seasonsForSite } from '@/siteSeasons'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { axesForUnits, pairDomain, type AxesOpts } from '@/lib/axes'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { useChartGrouping } from '@/components/GroupingDialog'
import { useSearchParam } from '@/lib/url'
import { Combobox } from '@/components/Combobox'
import { useNarrowScreen } from '@/hooks/useMediaQuery'
import { ChartAside } from '@/lib/singleView'
import { ExcludedNote } from '@/lib/markedRows'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const STEPS = ['Month of year', 'Year'] as const
const PATHS_BY = ['Site', 'Group'] as const
/** an SVG id from a site or band name (band labels carry '<', spaces and dots) */
const markerId = (name: string) => `arrow-${[...name].map((c) => (/[A-Za-z0-9_-]/.test(c) ? c : `_${c.charCodeAt(0)}`)).join('')}`

/**
 * Connected scatterplot — the census classified 31 figures as this and none
 * were built. The pair from the bar is aggregated to a median per step
 * (calendar month or year) and the steps are joined in order, so the path
 * traces how the *relationship* moves through the seasons rather than how
 * each variable moves on its own.
 *
 * Follows the bar's axes policy: with the 1:1 line on, the panel is square
 * and both axes share a domain, so a path that runs along the diagonal is
 * a site where the two methods agree all year.
 */
export function ConnectedScatter({ rows, meta, xField, yField, axes }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; axes: AxesOpts }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const [step, setStep] = useSearchParam<(typeof STEPS)[number]>('connected.step', 'Month of year', STEPS)
  const [labelSetting, setLabels] = useSearchParam<string>('connected.labels', '1', ['0', '1'])
  const labels = labelSetting === '1'
  const [pathChoice, setPathChoice] = useSearchParam<string>('connected.path', 'auto')
  const [pointChoice, setPointChoice] = useSearchParam<string>('connected.point', '')
  const narrow = useNarrowScreen()
  const { hidden, setHidden, dim, hover, setHover } = useLegend()
  const minN = 4
  const opts: AxesOpts = { ...axesForUnits(axes, meta.field_units?.[xField], meta.field_units?.[yField]), log: false, mode: axes.mode === 'Show intercept' ? 'Data' : axes.mode }

  // one path per site, or per band of the reader's grouping (components/GroupingDialog), every site pooled
  const paired = useMemo(() => rows.filter((r) => typeof r[xField] === 'number' && typeof r[yField] === 'number'), [rows, xField, yField])
  const { grp, picker, missingNote } = useChartGrouping('connected', paired, () => { setPathsBy('Group'); setHidden(new Set()) })
  const [pathsChoice, setPathsBy] = useSearchParam<'Site' | 'Group'>('connected.paths', 'Site', PATHS_BY)
  // Group is offered only once a grouping is picked
  const pathsBy = grp ? pathsChoice : 'Site'
  const byGroup = pathsBy === 'Group'
  const sources = useMemo(() => byGroup
    ? grp!.labels.map((l, i) => ({ name: l, color: grp!.colors[i], keep: (r: FilterRow) => grp!.labelOf(r) === l }))
    : meta.sites.map((s) => ({ name: s.name, color: s.color, keep: (r: FilterRow) => r.site === s.name })), [byGroup, grp, meta.sites])
  const series = useMemo(() => {
    const keyOf = (r: FilterRow) => (step === 'Month of year' ? r.month ?? 0 : r.year ?? 0)
    return sources
      .filter((s) => !hidden.has(s.name))
      .map((s) => {
        const byKey = new Map<number, { xs: number[]; ys: number[] }>()
        for (const r of rows) {
          if (!s.keep(r)) continue
          const x = r[xField]
          const y = r[yField]
          if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) continue
          const k = keyOf(r)
          if (!k) continue
          const g = byKey.get(k) ?? { xs: [], ys: [] }
          g.xs.push(x)
          g.ys.push(y)
          byKey.set(k, g)
        }
        const pts = [...byKey.entries()]
          .filter(([, g]) => g.xs.length >= minN)
          .sort((a, b) => a[0] - b[0])
          .map(([k, g]) => ({ key: k, x: d3.median(g.xs) ?? 0, y: d3.median(g.ys) ?? 0, n: g.xs.length }))
        return { name: s.name, color: s.color, pts }
      })
      .filter((s) => s.pts.length >= 2)
  }, [rows, sources, xField, yField, step, hidden])

  // On phones start with one readable path and its own axes. An explicit
  // selection is shared in the URL; Compare all preserves the combined view.
  const compareAll = pathChoice === 'all' || (pathChoice === 'auto' && !narrow)
  const selectedPath = series.find(s => s.name === pathChoice) ?? series[0]
  const shown = compareAll ? series : selectedPath ? [selectedPath] : []
  const selectedIndex = Math.max(0, selectedPath?.pts.findIndex(p => String(p.key) === pointChoice) ?? 0)
  const selectedPoint = selectedPath?.pts[selectedIndex]
  const all = shown.flatMap((s) => s.pts)
  // a median outside the axes (negative under "From 0,0") is clipped, not drawn into the margin
  const clipId = useId()
  const MAX_SQUARE = 560
  const availW = Math.max(120, width - MARGIN.left - MARGIN.right)
  const fit = useFitHeight()
  const innerW = Math.min(availW, fit === null ? MAX_SQUARE : Math.max(100, fit - MARGIN.top - MARGIN.bottom))
  const innerH = innerW
  const height = innerH + MARGIN.top + MARGIN.bottom

  const domain = pairDomain(all.map((p) => p.x), all.map((p) => p.y), opts)
  const x = d3.scaleLinear().domain(domain.x).range([0, innerW]).nice()
  const y = d3.scaleLinear().domain(domain.y).range([innerH, 0]).nice()
  const [d0, d1] = x.domain() as [number, number]
  const line = d3.line<{ x: number; y: number }>().x((p) => x(p.x)).y((p) => y(p.y))
  const label = (k: number) => (step === 'Month of year' ? MONTHS[k - 1] : String(k))
  const choosePoint = (name: string, key: number) => { tip.hide(); setPathChoice(name); setPointChoice(String(key)) }

  // Thin labels across overlaid paths while retaining their first/last steps.
  // The single-path view labels only the inspected step.
  const placed: [number, number][] = []
  const labelOk = (px: number, py: number, force: boolean) => {
    if (!force && placed.some(([qx, qy]) => Math.abs(qx - px) < 34 && Math.abs(qy - py) < 14)) return false
    placed.push([px, py])
    return true
  }
  const seasonOf = (m: number, site: string) => seasonsForSite(meta, site).find((s) => s.months.includes(m))?.name

  return (
    <ChartFrame
      id="connected"
      title="Connected scatter — the pair's path through the year"
      subtitle="Each point is the median of both measurements for a calendar month (pooling the selected years), or for a year. Straight segments join the available steps in time order. Inspect one path with the month/year controls, or compare all paths on shared axes. Steps with fewer than 4 paired filters are skipped."
      provenance="31 figures classified as connected scatter, none built until now · react-graph-gallery.com/connected-scatter-plot"
      // how to read it, behind the ⓘ rather than a line above the plot
      tip={<>{step === 'Month of year' ? 'Monthly medians · selected years pooled · Jan → Dec.' : 'Yearly medians · earliest → latest.'}
        {' '}{compareAll ? 'Tap a point to inspect its path on its own.' : 'Tap a point or use the step controls below.'}
        {opts.identity && ' Grey dashed line = equal measurements.'}</>}
      controls={
        <>
          {grp && <Segmented label="Paths" value={pathsBy} options={PATHS_BY} onChange={(v) => { setPathsBy(v); setHidden(new Set()) }}
            title="One path per site, or per band of a grouping (every site pooled in each band)" />}
          {picker}
          <Combobox label="Show" value={compareAll ? 'all' : selectedPath?.name ?? 'all'}
            options={[{ key: 'all', label: 'Compare all' }, ...series.map(s => ({ key: s.name, label: s.name }))]}
            onChange={setPathChoice} />
          <Segmented label="Step" value={step} options={STEPS} onChange={v => { setStep(v); setPointChoice('') }} />
          <Toggle label="Labels" checked={labels} onChange={v => setLabels(v ? '1' : '0')} />
        </>
      }
    >
      <div ref={wrapRef} className={`chart-wrap centered${fit === null ? '' : ' fit-square'}`} style={fit === null ? undefined : { minHeight: fit }}>
        {all.length < 2 ? (
          <Empty>Not enough steps with ≥{minN} filters carrying both fields.</Empty>
        ) : (
          <svg width={innerW + MARGIN.left + MARGIN.right} height={height}>
            <defs>
              {shown.map((s) => (
                <marker key={s.name} data-export-site={s.name} data-export-color={s.color} id={`${clipId}-${markerId(s.name)}`} viewBox="0 0 10 10" refX={8} refY={5} markerWidth={5} markerHeight={5} orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={s.color} />
                </marker>
              ))}
              <clipPath id={clipId}>
                <rect width={innerW} height={innerH} />
              </clipPath>
            </defs>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={withUnit(yField, meta.field_units)} gridWidth={innerW} pick="y" />
              <XAxis scale={x} y={innerH} label={withUnit(xField, meta.field_units)} pick="x" />
              <g clipPath={`url(#${clipId})`}>
              {opts.identity && (
                <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />
              )}
              {shown.map((s) => (
                <g key={s.name} opacity={dim(s.name)} data-export-site={s.name} data-export-color={s.color}>
                  <path
                    d={line(s.pts) ?? ''}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={2}
                    strokeOpacity={0.75}
                    markerEnd={`url(#${clipId}-${markerId(s.name)})`}
                  />
                  {s.pts.map((p, i) => (
                    <g key={p.key} role="button" tabIndex={0}
                      aria-label={`${s.name}, ${label(p.key)}, ${p.n} filters; inspect medians`}
                      aria-pressed={!compareAll && selectedPoint?.key === p.key}
                      onClick={() => choosePoint(s.name, p.key)}
                      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choosePoint(s.name, p.key) } }}>
                      <circle cx={x(p.x)} cy={y(p.y)} r={13} fill="transparent" className="connected-hit" />
                      {!compareAll && selectedPoint?.key === p.key && <circle cx={x(p.x)} cy={y(p.y)} r={9} fill="none" stroke={s.color} strokeWidth={2} pointerEvents="none" />}
                      <circle
                        cx={x(p.x)}
                        cy={y(p.y)}
                        r={i === 0 ? 5.5 : 4}
                        fill={i === 0 ? INK.panel : s.color}
                        stroke={s.color}
                        strokeWidth={i === 0 ? 2.2 : 1}
                        onMouseEnter={(e) =>
                          tip.show(e, [
                            `${s.name} · ${label(p.key)}`,
                            `${xField} median = ${fmt(p.x)}`,
                            `${yField} median = ${fmt(p.y)}`,
                            `ratio y/x = ${fmt(p.x !== 0 ? p.y / p.x : null, 3)}`,
                            `n = ${p.n}`,
                            ...(step === 'Month of year' && !byGroup ? [seasonOf(p.key, s.name) ?? ''] : []),
                          ])
                        }
                        onMouseLeave={tip.hide}
                      />
                      {labels && (compareAll ? labelOk(x(p.x) + 7, y(p.y) - 6, i === 0 || i === s.pts.length - 1) : selectedPoint?.key === p.key) && (
                        <text x={Math.max(4, Math.min(innerW - 25, x(p.x) + 10))} y={Math.max(12, y(p.y) - 9)} fontSize={11} fill={s.color} fontFamily={FONT.mono} pointerEvents="none" stroke={INK.panel} strokeWidth={3} paintOrder="stroke">
                          {label(p.key)}
                        </text>
                      )}
                    </g>
                  ))}
                </g>
              ))}
              </g>
            </g>
          </svg>
        )}
        {/* the step's readout and its stepper: under the plot, or beside the options in single-chart view */}
        {!compareAll && selectedPoint && selectedPath && <ChartAside><div className="connected-inspector">
          <div className="connected-step-controls">
            <StepButton dir="prev" label={step === 'Month of year' ? 'Previous month' : 'Previous year'} disabled={selectedIndex === 0}
              onClick={() => setPointChoice(String(selectedPath.pts[selectedIndex - 1].key))} />
            <Combobox label={step === 'Month of year' ? 'Month' : 'Year'} value={String(selectedPoint.key)}
              options={selectedPath.pts.map(p => ({ key: String(p.key), label: label(p.key), hint: `${p.n} filters` }))} onChange={setPointChoice} />
            <StepButton dir="next" label={step === 'Month of year' ? 'Next month' : 'Next year'} disabled={selectedIndex === selectedPath.pts.length - 1}
              onClick={() => setPointChoice(String(selectedPath.pts[selectedIndex + 1].key))} />
          </div>
          <div className="connected-readout" role="region" aria-label="Selected step medians" aria-live="polite" data-export="text">
            <strong><span className="swatch" style={{ background: selectedPath.color }} />{selectedPath.name} · {label(selectedPoint.key)}</strong>
            <span>{selectedPoint.n} paired filters</span>
            <dl><div><dt>{withUnit(xField, meta.field_units)}</dt><dd>{fmt(selectedPoint.x, 3)}</dd></div>
              <div><dt>{withUnit(yField, meta.field_units)}</dt><dd>{fmt(selectedPoint.y, 3)}</dd></div></dl>
          </div>
          <p className="chart-note connected-guide" data-export="text">Hollow point = {label(selectedPath.pts[0].key)} (start); arrow = {label(selectedPath.pts[selectedPath.pts.length - 1].key)} (end). Axes fit this path; choose Compare all for shared axes.</p>
        </div></ChartAside>}
        {compareAll && <Legend
          items={sources.map((s) => ({ label: s.name, color: s.color, shape: 'line' as const }))}
          hidden={hidden}
          onToggle={(l) => setHidden((h) => toggleIn(h, l))}
          onHover={setHover}
          highlighted={hover}
          note="hollow marker = first step · arrow = last · choose a path for its step-by-step medians"
        />}
        {byGroup && missingNote('left out of the paths')}
        <ExcludedNote fields={[xField, yField]} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
