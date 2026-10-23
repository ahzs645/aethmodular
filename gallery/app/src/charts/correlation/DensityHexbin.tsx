import { fitUncertainty } from '@/lib/fitUncertainty'
import { IdentityLine } from '@/components/RefLines'
import { RegressionLines, RegressionReadout, RegressionReadoutText, RegressionGroupLines, RegressionGroupReadouts, regressionGroupReadoutLayout } from '@/components/RegressionOverlay'
import { fitsFor, useRegressions } from '@/lib/regressionPrefs'
import { fitGroupingOption, useChartOptions } from '@/lib/chartOptions'
import { useChartGrouping } from '@/components/GroupingDialog'
import { groupedRegressions, type FitGrouping } from '@/lib/siteRegression'
import { useId, useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { hexbin as d3Hexbin } from 'd3-hexbin'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { inDomain, pairDomain, type AxesOpts } from '@/lib/axes'
import { regression, fmt } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'
import type { FilterRow, MetaFile } from '@/lib/types'
import { fidsProps, useFilterEdits } from '@/lib/filterEdits'
import { useFitHeight } from '@/lib/chartFit'
import { useNarrowScreen } from '@/hooks/useMediaQuery'

type Pt = [number, number, string]

/**
 * 2D density (hexbin) — 16 figures in the estate, all of them on the large
 * IMPROVE pool where a plain scatter overplots into a solid blob. Same idea
 * here: at 900+ filters the dense core of a crossplot is unreadable as points.
 * Reads the same x/y pair and the same axes policy as the scatterplot above,
 * so the two frames line up. (Log axes are ignored here: hex bins on a log
 * scale bin the pixels, not the data.)
 */
export function DensityHexbin({ rows, meta, xField, yField, axes }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; axes: AxesOpts }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const edits = useFilterEdits()
  const clipId = useId()
  const { methods, grouping: chosenGrouping, setGrouping, showCI } = useRegressions()
  const MAX_SQUARE = 520

  const opts: AxesOpts = { ...axes, log: false }
  const pairedRows = useMemo(() => rows.filter(r => typeof r[xField] === 'number' && typeof r[yField] === 'number' && Number.isFinite(r[xField]) && Number.isFinite(r[yField])), [rows, xField, yField])
  // one site in view: a fit per site is the pooled fit, so it is not offered
  const oneSite = useMemo(() => new Set(pairedRows.map((r) => r.site)).size <= 1, [pairedRows])
  // one panel: every grouping applies, except a fit per site with a single site in view (it is the pooled fit)
  // the bins' colours (lib/colourSchemes): chosen by clicking the ramp under the plot
  const colours = useColourScheme('hexbin', 'sequential')
  // fits per band of the reader's grouping (components/GroupingDialog) instead of the gallery's fit grouping
  const { grp, picker, missingNote } = useChartGrouping('hexbin', pairedRows, () => o.set('byGroup', true))
  const o = useChartOptions('hexbin', [
    { key: 'radius', label: 'Hex radius', choices: ['8', '11', '14', '18', '24'], default: '14', style: 'select' },
    { key: 'groupPicker', node: () => picker },
    { key: 'byGroup', label: 'Fit per group', toggle: true, default: false, when: (c) => (c.estimators as number) > 0 && !!c.hasGroup,
      title: 'One fit per band of a grouping (high vs low iron, rain classes …), drawn over the same hexes' },
    fitGroupingOption({ bind: [chosenGrouping, setGrouping as (v: never) => void], when: (c) => (c.estimators as number) > 0 && !c.byGroup, offerSite: (c) => !c.oneSite,
      hiddenNote: (c) => ((c.estimators as number) > 0 && c.byGroup ? 'one fit per band' : null) }),
  ], { estimators: methods.length, oneSite, hasGroup: !!grp })
  const radius = o.values.radius as string
  const grouping = o.values.grouping as FitGrouping
  const perBand = !!o.values.byGroup && !!grp
  // several fits drawn: per site or season (the gallery's grouping), or per band of the grouping
  const split = perBand || grouping !== 'pooled'
  const fitGroups = useMemo(() => perBand ? groupedRegressions(pairedRows.filter((r) => grp!.labelOf(r) !== null), 'site', {
    x: r => r[xField] as number, y: r => r[yField] as number, site: r => grp!.labelOf(r)!, color: r => grp!.colorOf(grp!.labelOf(r)!),
    options: members => fitUncertainty(members, xField, yField, meta),
  }).sort((a, b) => grp!.labels.indexOf(a.name) - grp!.labels.indexOf(b.name)) : groupedRegressions(pairedRows, grouping, {
    x: r => r[xField] as number, y: r => r[yField] as number, site: r => r.site,
    season: r => r.season, seasonColor: r => meta.seasons.find(se => se.name === r.season)?.color ?? INK.muted,
    color: r => meta.sites.find(s => s.name === r.site)?.color ?? INK.muted,
    options: members => fitUncertainty(members, xField, yField, meta),
  }), [pairedRows, grouping, xField, yField, meta, perBand, grp])

  // [x, y, filter id]: the id rides along so a hex can list and exclude its filters
  const points = useMemo(
    () => pairedRows.map((r) => [r[xField] as number, r[yField] as number, r.id] as Pt),
    [pairedRows, xField, yField]
  )

  const fit = useFitHeight()
  const availW = Math.max(200, width - MARGIN.left - MARGIN.right)
  const innerW = Math.min(availW, fit === null ? MAX_SQUARE : Math.max(100, fit - MARGIN.top - MARGIN.bottom))
  const innerH = innerW
  const svgWidth = innerW + MARGIN.left + MARGIN.right
  const siteFits = split && methods.length > 0
  // on a phone the fit readout goes under the plot (in it, the box covers hexes and runs off the panel)
  const narrow = useNarrowScreen()
  const svgHeight = innerH + MARGIN.top + MARGIN.bottom + (siteFits ? regressionGroupReadoutLayout(fitGroups.length, svgWidth, methods, showCI).height : 0)

  const { x, y, stats, domain } = useMemo(() => {
    const xs = points.map((p) => p[0])
    const ys = points.map((p) => p[1])
    // Fit the same paired observations as the scatterplot.
    const stats = regression(xs, ys, fitUncertainty(rows.filter(r => typeof r[xField] === 'number' && typeof r[yField] === 'number' && Number.isFinite(r[xField]) && Number.isFinite(r[yField])), xField, yField, meta))
    const intercepts = fitGroups.flatMap(group => fitsFor(group.stats, methods).map(fit => fit.intercept))
    const domain = pairDomain(xs, ys, opts, intercepts)
    const scale = (d: [number, number], range: [number, number]) => {
      const s = d3.scaleLinear().domain(d).range(range)
      return opts.mode === 'Show intercept' ? s : s.nice()
    }
    return {
      x: scale(domain.x, [0, innerW]),
      y: scale(domain.y, [innerH, 0]),
      stats,
      domain,
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, rows, xField, yField, meta, methods, fitGroups, innerW, innerH, opts.identity, opts.mode])

  const bins = useMemo(() => {
    const hb = d3Hexbin<Pt>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .radius(Number(radius))
      .extent([[0, 0], [innerW, innerH]])
    return hb(points.filter((p) => inDomain(domain, p[0], p[1]))).map((b) => Object.assign(b, { path: hb.hexagon() }))
  }, [points, x, y, radius, innerW, innerH, domain])

  const maxCount = d3.max(bins, (b) => b.length) ?? 1
  const binTitle = (b: { x: number; y: number }) => `Hex at ${xField} ≈ ${fmt(x.invert(b.x), 2)}, ${yField} ≈ ${fmt(y.invert(b.y), 2)}`
  // the hex whose filters are in the side panel stays outlined until the panel closes
  const openTitle = edits.group?.title ?? null
  const color = d3.scaleQuantize<string>().domain([0, maxCount]).range(colours.stops)
  const [d0, d1] = x.domain() as [number, number]
  const originAxes = opts.mode === 'Show intercept' && d0 < 0 && (y.domain()[0] as number) < 0

  return (
    <ChartFrame
      id="hexbin"
      title="2D density (hexbin) — where the crossplot is actually dense"
      subtitle="The same pair and the same axes as the scatterplot, binned. Overplotting hides that most filters sit in a small low-loading core; the hexbin shows the mass of the data rather than its outline."
      provenance="stands in for 16 hexbin+plot+scatter figures · react-graph-gallery.com/2d-density-plot"
      controls={o.controls()}
    >
      <div ref={wrapRef} className={`chart-wrap centered${fit === null ? '' : ' fit-square'}`} style={fit === null ? undefined : { minHeight: fit }}>
        {points.length < 3 ? (
          <Empty>Fewer than 3 filters carry both fields in this subset.</Empty>
        ) : (
          <svg width={svgWidth} height={svgHeight} className="animated">
            <defs>
              <clipPath id={clipId}>
                <rect x={0} y={0} width={innerW} height={innerH} />
              </clipPath>
            </defs>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {originAxes ? (
                <>
                  <YAxis scale={y} x={x(0)} label={withUnit(yField, meta.field_units)} gridWidth={innerW} gridFrom={0} labelX={-50} pick="y" />
                  <XAxis scale={x} y={y(0)} label={withUnit(xField, meta.field_units)} labelDy={innerH - y(0) + 40} pick="x" />
                </>
              ) : (
                <>
                  <YAxis scale={y} x={0} label={withUnit(yField, meta.field_units)} gridWidth={innerW} pick="y" />
                  <XAxis scale={x} y={innerH} label={withUnit(xField, meta.field_units)} pick="x" />
                </>
              )}

              {/* hexes centred near an edge would spill over the axes and their tick labels */}
              <g clipPath={`url(#${clipId})`}>
              {bins.map((b, i) => {
                const title = binTitle(b)
                const ids = b.map((p) => p[2])
                return (
                  <path
                    key={i}
                    d={b.path}
                    transform={`translate(${b.x},${b.y})`}
                    fill={color(b.length)}
                    stroke={openTitle === title ? INK.fit : INK.panel}
                    strokeWidth={openTitle === title ? 2 : 0.4}
                    style={{ cursor: 'pointer' }}
                    {...fidsProps(ids, title)}
                    onMouseEnter={(e) =>
                      tip.show(e, [`${b.length} filters`, `${xField} ≈ ${fmt(x.invert(b.x), 2)}`, `${yField} ≈ ${fmt(y.invert(b.y), 2)}`, 'click to list them · right-click to exclude'])
                    }
                    onMouseLeave={tip.hide}
                    onClick={(e) => { e.stopPropagation(); tip.hide(); edits.openGroup({ title, ids, fields: [xField, yField] }) }}
                  />
                )
              })}
              {/* the open hex on top, so its outline is not covered by its neighbours */}
              {bins.filter((b) => binTitle(b) === openTitle).map((b) => (
                <path key="open" d={b.path} transform={`translate(${b.x},${b.y})`} fill="none" stroke={INK.fit} strokeWidth={2} pointerEvents="none" />
              ))}
              </g>

              <g clipPath={`url(#${clipId})`} pointerEvents="none">
                {opts.identity && <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />}
                {stats && (
                  <>
                    {split ? <RegressionGroupLines groups={fitGroups} x={x} y={y} width={innerW} height={innerH} />
                      : <RegressionLines stats={stats} x={x} y={y} domain={x.domain()} width={innerW} height={innerH} />}
                  </>
                )}
              </g>
              {stats && opts.mode === 'Show intercept' && (
                <g pointerEvents="none" fontFamily={FONT.mono} fontSize={10.5}>
                  {fitGroups.flatMap(group => fitsFor(group.stats, methods).map(fit => <g key={group.name + fit.method}>
                    <circle cx={x(0)} cy={y(fit.intercept)} r={4.5} fill={INK.panel} stroke={group.color ?? fit.color} strokeWidth={2} />
                    <title>{fit.label} b = {fmt(fit.intercept, 3)}</title>
                  </g>))}
                </g>
              )}

              {!split && methods.length > 0 && !narrow && <RegressionReadout stats={stats} />}
            </g>
            {siteFits && <RegressionGroupReadouts groups={fitGroups} width={svgWidth} y={innerH + MARGIN.top + MARGIN.bottom} />}
          </svg>
        )}
        {!split && methods.length > 0 && narrow && <RegressionReadoutText stats={stats} />}
        <div className="legend">
          <ColorLegend scale={color} scheme={colours.scheme} label="filters per hex" format={(v) => String(Math.round(v))} />
        </div>
        {domain.dropped > 0 && <Note>{domain.dropped} points below zero are outside the axes and not binned; they are still in the fit.</Note>}
        {perBand && missingNote('in the hexes but left out of the per-band fits')}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
