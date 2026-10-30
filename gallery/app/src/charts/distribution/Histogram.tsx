import { useMemo, useRef, useState } from 'react'
import { robustDomain } from '@/lib/axes'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Select } from '@/components/ChartFrame'
import { Legend, toggleIn, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { kde, fmt, twoSample } from '@/lib/stats'
import { useChartOptions } from '@/lib/chartOptions'
import { fidsProps, useFilterEdits } from '@/lib/filterEdits'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useChartLabel, useFitHeight } from '@/lib/chartFit'
import { MeasurementControl } from '@/components/AxisFieldPopover'
import { useTileSetting } from '@/lib/url'
import { ChartAside } from '@/lib/singleView'
import { ScrubField } from '@/components/ScrubField'
import { ecdfAt, ecdfSteps } from '@/lib/ecdf'
import { parseCuts } from '@/lib/splitFit'

const LAYOUTS = ['Overlay', 'Density', 'Small multiples', 'Mirror', 'ECDF'] as const
type Layout = (typeof LAYOUTS)[number]

/** `ids[i]` is the filter behind `values[i]`, so a bar can name its filters */
interface Series { name: string; color: string; values: number[]; ids: string[] }

/**
 * Histogram + density — 36 histogram figures in the estate. Three layouts,
 * following react-graph-gallery's histogram variations:
 *
 *  Overlay          sites on one axis, which is the point: the Delhi and
 *                   Addis distributions overlap far less than separate axes
 *                   suggest.
 *  Density          the same, curves only (the gallery's density plot with
 *                   several groups) — when four sets of translucent bars
 *                   hide each other, the smoothed shapes do not.
 *  Small multiples  one panel per site with shared x and y, for when four
 *                   translucent overlays become mud.
 *  Mirror           two groups back to back (any site or season against any
 *                   other), the honest way to compare two distributions of
 *                   different n: dry vs wet at one site, Addis vs Beijing.
 *  ECDF             each site's cumulative share of filters at or below a
 *                   value (primary_tasks_notebook c11's Fe/EC CDF, the
 *                   vs_improve and mass-closure CDFs): no bins to choose, and
 *                   "what fraction lies under the cutoff" reads straight off
 *                   the curve. Typed marks draw the cutoffs being weighed.
 *
 * The bin count is a slider, because the bin size changes what a histogram
 * appears to say and the reader should be able to see that happen.
 */
export function Histogram({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const chartLabel = useChartLabel()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const edits = useFilterEdits()

  const [binsParam, setBinsParam] = useTileSetting<string>('histogram.bins', '30')
  const bins = Math.min(80, Math.max(8, Number(binsParam) || 30))
  const setBins = (n: number) => setBinsParam(String(n))
  const { hidden, setHidden, dim, hover, setHover } = useLegend()
  const mirror = useLegend()
  const [mirrorA, setMirrorA] = useState<string>('')
  const [mirrorB, setMirrorB] = useState<string>('')
  // values marked on the axis (the cutoffs a notebook weighed: Fe/EC 50, 100, 200), as typed
  const [marksParam, setMarksParam] = useTileSetting<string>('histogram.marks', '')
  // the marks being typed (null: show the ones in use)
  const [marksDraft, setMarksDraft] = useState<string | null>(null)
  const marks = useMemo(() => parseCuts(marksParam), [marksParam])
  // the ECDF's pointer readout: the value under the pointer (null: none)
  const [probe, setProbe] = useState<number | null>(null)

  const seasonColor = useMemo(() => new Map(meta.seasons.map((s) => [s.name, s.color])), [meta.seasons])
  const valuesOf = (pred: (r: FilterRow) => boolean) => {
    const kept = rows.filter((r) => pred(r) && typeof r[field] === 'number' && Number.isFinite(r[field]))
    return { values: kept.map((r) => r[field] as number), ids: kept.map((r) => r.id) }
  }

  const allSeries = useMemo<Series[]>(
    () => meta.sites.map((s) => ({ name: s.name, color: s.color, ...valuesOf((r) => r.site === s.name) })).filter((s) => s.values.length >= 5),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, field, meta.sites]
  )
  /** Every group a mirror half can be: sites and seasons with enough filters. */
  const mirrorGroups = useMemo<Series[]>(
    () => [
      ...allSeries,
      ...meta.seasons.map((s) => ({ name: s.name, color: s.color, ...valuesOf((r) => r.season === s.name) })).filter((s) => s.values.length >= 5),
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [allSeries, rows, field, meta.seasons, seasonColor]
  )
  const groupNames = mirrorGroups.map((g) => g.name)
  // the options and when each applies (lib/chartOptions): small multiples need more than one
  // series, a mirror two groups; the density layout draws no bars, so no bins and no extra curve
  const o = useChartOptions('histogram', [
    { key: 'layout', label: 'Layout', default: 'Overlay', choices: [
      { value: 'Overlay' }, { value: 'Density' },
      { value: 'Small multiples', when: (c) => (c.nSeries as number) > 1 },
      { value: 'Mirror', when: (c) => (c.nGroups as number) >= 2 },
      { value: 'ECDF' },
    ], title: 'ECDF: the cumulative share of each site\'s filters at or below a value, with no bins' },
    { key: 'mirror', when: (c) => c.layout === 'Mirror', node: () => <>
      <Select label="Top" value={aName} options={groupNames} onChange={setMirrorA} />
      <Select label="Bottom" value={bName} options={groupNames.filter((n) => n !== aName)} onChange={setMirrorB} />
    </> },
    { key: 'bins', when: (c) => c.layout !== 'Density' && c.layout !== 'ECDF', node: () => <span className="control" title="Number of bins across the displayed range">
      Bins
      <ScrubField label="Number of bins" value={bins} onChange={setBins} min={8} max={80} step={1} width={90} />
    </span> },
    { key: 'normalise', label: 'Normalise to fraction', toggle: true, default: true, hidden: true, when: (c) => c.layout !== 'ECDF' },
    { key: 'density', label: 'Density curve', toggle: true, default: true, hidden: true, when: (c) => c.layout !== 'Density' && c.layout !== 'ECDF' },
    { key: 'marks', when: (c) => c.layout !== 'Mirror', node: () => <label className="control" title="Values to mark on the axis, e.g. the cutoffs a notebook tried (Fe/EC 0.05, 0.1, 0.2): commas or spaces between them">
      Mark at
      <input className="select" style={{ width: 120 }} value={marksDraft ?? marksParam} placeholder="none" aria-label="Values to mark"
        onChange={(e) => setMarksDraft(e.target.value)} onBlur={() => { if (marksDraft !== null) setMarksParam(parseCuts(marksDraft).join(', ')); setMarksDraft(null) }}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} />
    </label> },
  ], { nSeries: allSeries.length, nGroups: groupNames.length })
  const layout = o.values.layout as Layout
  const normalise = o.values.normalise as boolean
  const overlayDensity = o.values.density as boolean
  const aName = groupNames.includes(mirrorA) ? mirrorA : groupNames[0] ?? ''
  const bName = groupNames.includes(mirrorB) && mirrorB !== aName ? mirrorB : groupNames.find((n) => n !== aName) ?? ''

  const mirrorPair = mirrorGroups.filter((g) => g.name === aName || g.name === bName).sort((g) => (g.name === aName ? -1 : 1))
  const series: Series[] = layout === 'Mirror' ? mirrorPair.filter((g) => mirror.show(g.name)) : allSeries.filter((s) => !hidden.has(s.name))
  const dimOf = layout === 'Mirror' ? mirror.dim : dim

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const fit = useFitHeight()
  const panelH = fit === null ? (layout === 'Small multiples' ? 190 : 420) : Math.max(140, fit / (layout === 'Small multiples' ? Math.max(1, series.length) : 1))
  const innerH = panelH - MARGIN.top - MARGIN.bottom

  const all = series.flatMap((s) => s.values).sort(d3.ascending)
  // Clip both tails at 1/99 %: these distributions have a long right tail and
  // a handful of genuinely negative values near the detection limit, either of
  // which otherwise squeezes the bulk of the data into a few pixels. The count
  // outside is reported below rather than silently dropped.
  const [lo, hi] = robustDomain(all, 0.01, 0.99, { include: [0] }).domain
  const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]).nice()
  const [dLo, dHi] = x.domain() as [number, number]
  const nOutside = all.filter((v) => v < dLo || v > dHi).length
  const ecdfMode = layout === 'ECDF'
  const shownMarks = layout === 'Mirror' ? [] : marks.filter((m) => m >= dLo && m <= dHi)

  const densityOnly = layout === 'Density'
  const showBars = !densityOnly
  const showDensity = overlayDensity || densityOnly
  // bin the indices, not the values, so each bar still knows which filters it holds
  const binned = series.map((s) => ({ ...s, bins: d3.bin<number, number>().value((i) => s.values[i]).domain([dLo, dHi]).thresholds(bins)(d3.range(s.values.length)) }))

  const grid = useMemo(() => d3.range(80).map((i) => dLo + ((dHi - dLo) * i) / 79), [dLo, dHi])

  /** Density scaled onto the histogram's own y units (fraction or count per bin). */
  const scaledDensities = useMemo(
    () =>
      binned.map((s) => {
        // real bin width, not domain/bins — d3.bin rounds the threshold count
        const b0 = s.bins[0]
        const binWidth = b0 && b0.x1 !== undefined && b0.x0 !== undefined ? b0.x1 - b0.x0 : (dHi - dLo) / bins
        const factor = binWidth * (normalise ? 1 : s.values.length)
        return kde(s.values, grid).map((d) => d * factor)
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, field, hidden, mirror.hidden, aName, bName, layout, grid, normalise, dLo, dHi, bins]
  )

  // The y axis must cover the density curve too — a tightly peaked group (JPL)
  // produces a curve taller than any bar, and it used to run off the top.
  const val = (s: { values: number[] }, b: d3.Bin<number, number>) => (normalise ? b.length / s.values.length : b.length)
  const barMax = d3.max(binned, (s) => d3.max(s.bins, (b) => val(s, b)) ?? 0) ?? 0
  const densMax = showDensity ? (d3.max(scaledDensities.flat()) ?? 0) : 0
  const yMax = ((densityOnly ? densMax : Math.max(barMax, densMax)) || 1) * 1.08
  const y = layout === 'Mirror'
    ? d3.scaleLinear().domain([-yMax, yMax]).range([innerH, 0]).nice()
    : d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice()
  const y0 = y(0)

  const drawSeries = (s: (typeof binned)[number], si: number, sign: 1 | -1) => {
    const dens = scaledDensities[si] ?? []
    const line = d3.line<number>().x((_, i) => x(grid[i])).y((_, i) => y(sign * (dens[i] ?? 0))).curve(d3.curveBasis)
    const area = d3.area<number>().x((_, i) => x(grid[i])).y0(y0).y1((_, i) => y(sign * (dens[i] ?? 0))).curve(d3.curveBasis)
    return (
      <g key={s.name} opacity={dimOf(s.name)} data-export-site={s.name} data-export-series={s.name} data-export-color={s.color}>
        {showBars && s.bins.map((b, i) => {
          const v = val(s, b)
          if (!v) return null
          const x0 = x(b.x0 ?? 0)
          const x1 = x(b.x1 ?? 0)
          const top = Math.min(y(sign * v), y0)
          return (
            <rect
              key={i}
              x={x0} y={top} width={Math.max(0, x1 - x0 - 0.7)} height={Math.abs(y(sign * v) - y0)}
              fill={s.color} fillOpacity={0.4} stroke={s.color} strokeOpacity={0.6}
              style={{ cursor: 'pointer' }}
              {...fidsProps(b.map((k) => s.ids[k]), `${s.name} · ${field} ${fmt(b.x0, 2)} – ${fmt(b.x1, 2)}`)}
              onMouseEnter={(e) =>
                tip.show(e, [s.name, `${fmt(b.x0, 2)} – ${fmt(b.x1, 2)}`, `${b.length} filters`, `${((b.length / s.values.length) * 100).toFixed(1)} % of ${s.name}`, 'click to list them · right-click to exclude'])
              }
              onMouseLeave={tip.hide}
              onClick={(e) => { e.stopPropagation(); tip.hide(); edits.openGroup({ title: `${s.name} · ${field} ${fmt(b.x0, 2)} – ${fmt(b.x1, 2)}`, ids: b.map((k) => s.ids[k]), fields: [field] }) }}
            />
          )
        })}
        {showDensity && (
          <path d={(densityOnly ? area(grid) : line(grid)) ?? ''} fill={densityOnly ? s.color : 'none'} fillOpacity={densityOnly ? 0.18 : 0} stroke={s.color} strokeWidth={2.2} pointerEvents={densityOnly ? 'visiblePainted' : 'none'}
            onMouseEnter={densityOnly ? (e) => tip.show(e, [s.name, `n = ${s.values.length}`, `median ${fmt(d3.median(s.values), 2)}`]) : undefined} onMouseLeave={densityOnly ? tip.hide : undefined} />
        )}
      </g>
    )
  }

  // the gallery's "stop chasing the p-value" playground: the mirror is the
  // picture, the statistics underneath are the caveat
  const mirrorStats = layout === 'Mirror' && series.length === 2 ? twoSample(series[0].values, series[1].values) : null
  const yLabel = ecdfMode ? 'cumulative % of filters' : normalise ? 'fraction of samples' : 'filters'
  const xLabel = withUnit(field, meta.field_units)
  const panels: { key: string; title: string | null; color: string; items: (typeof binned)[number][]; }[] =
    layout === 'Small multiples' ? binned.map((s) => ({ key: s.name, title: s.name, color: s.color, items: [s] })) : [{ key: 'all', title: null, color: INK.text, items: binned }]

  // ECDF: each series sorted once; a step path over the drawn range, in % of the series' filters
  const sortedAll = useMemo(() => new Map(allSeries.map((s) => [s.name, [...s.values].sort(d3.ascending)])), [allSeries])
  const sortedSeries = ecdfMode ? series.map((s) => ({ ...s, sorted: sortedAll.get(s.name) ?? [] })) : []
  const yPct = d3.scaleLinear().domain([0, 100]).range([innerH, 0])
  const marksLayer = (yTop: number) => shownMarks.map((m) => (
    <g key={m} pointerEvents="none">
      <line x1={x(m)} x2={x(m)} y1={yTop} y2={innerH} stroke={INK.text} strokeWidth={1.2} strokeDasharray="3 3" opacity={0.7} />
      <text x={x(m) + 3} y={yTop + 10} fontSize={10} fontFamily="var(--mono)" fill={INK.text}>{fmt(m, 3)}</text>
    </g>
  ))
  const ecdfPlot = (
    <svg width={width} height={panelH} className="animated">
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        <YAxis scale={yPct} x={0} label={yLabel} gridWidth={innerW} tickCount={6} />
        <XAxis scale={x} y={innerH} label={xLabel} />
        {sortedSeries.map((s) => {
          const path = d3.line<[number, number]>().x((d) => x(d[0])).y((d) => yPct(100 * d[1]))(ecdfSteps(s.sorted, dLo, dHi)) ?? ''
          return <path key={s.name} d={path} fill="none" stroke={s.color} strokeWidth={2.2} opacity={dim(s.name)}
            data-export-site={s.name} data-export-series={s.name} data-export-color={s.color} />
        })}
        {marksLayer(0)}
        {probe !== null && <line x1={x(probe)} x2={x(probe)} y1={0} y2={innerH} stroke={INK.muted} strokeWidth={1} pointerEvents="none" />}
        {probe !== null && sortedSeries.map((s) => <circle key={s.name} cx={x(probe)} cy={yPct(100 * ecdfAt(s.sorted, probe))} r={3.5} fill={s.color} stroke={INK.panel} pointerEvents="none" />)}
        {/* one pointer layer: the value under the pointer and each site's share at or below it */}
        <rect width={innerW} height={innerH} fill="transparent"
          onMouseMove={(e) => {
            const v = x.invert(d3.pointer(e.nativeEvent, e.currentTarget)[0])
            setProbe(v)
            tip.show(e, [`${field} ≤ ${fmt(v, 3)}`, ...sortedSeries.map((s) => { const f = ecdfAt(s.sorted, v); return `${s.name}: ${(100 * f).toFixed(1)} % (${Math.round(f * s.sorted.length)} of ${s.sorted.length})` })])
          }}
          onMouseLeave={() => { setProbe(null); tip.hide() }} />
      </g>
    </svg>
  )

  return (
    <ChartFrame
      id="histogram"
      title="Histogram + density — overlaid, panelled, or mirrored"
      subtitle="Counts per bin with an optional kernel density curve. Normalising to a fraction lets Delhi (n≈97) and Beijing (n≈375) be compared directly instead of the larger site simply looking taller. Mirror puts any two groups (a site or a season) back to back. Drag the bin slider to see how much the shape depends on it. ECDF draws each site's cumulative share of filters at or below a value, with no bins at all; Mark at draws the cutoffs being weighed."
      provenance="stands in for 36 histogram figures · react-graph-gallery.com/histogram · /histogram-mirror · /histogram-small-multiple · ECDF: research/ftir_hips_chem/primary_tasks_notebook.ipynb c11 (Fe/EC CDF), vs_improve fig02, mass M2"
      controls={<><MeasurementControl />{o.controls()}</>}
    >
      <div ref={wrapRef} className="chart-wrap">
        {(layout === 'Mirror' ? mirrorPair.length < 2 : series.length === 0) ? (
          <Empty>{layout === 'Mirror' ? `Fewer than two groups have 5 or more filters with ${field} in this subset.` : `No site has 5 or more filters with ${field} in this subset.`}</Empty>
        ) : ecdfMode ? ecdfPlot : (
          panels.map((panel) => (
            <div key={panel.key}>
              {panel.title && chartLabel(panel.title) && (
                <p className="facet-title">
                  <span className="swatch" style={{ background: panel.color }} /> {chartLabel(panel.title!)}
                  <span style={{ color: 'var(--ink-muted)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 11 }}>n={panel.items[0]?.values.length ?? 0}</span>
                </p>
              )}
              <svg width={width} height={panelH} className="animated">
                <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                  <YAxis scale={y} x={0} label={yLabel} gridWidth={innerW} tickCount={layout === 'Small multiples' ? 3 : 6} format={layout === 'Mirror' ? (v: number) => String(Math.abs(v)) : undefined} />
                  <XAxis scale={x} y={innerH} label={xLabel} />
                  {layout === 'Mirror' && <line x1={0} x2={innerW} y1={y0} y2={y0} stroke={INK.axis} strokeWidth={1} />}
                  {panel.items.map((s) => drawSeries(s, binned.indexOf(s), layout === 'Mirror' && s.name === bName ? -1 : 1))}
                  {marksLayer(0)}
                  {layout === 'Mirror' && (
                    <g fontFamily={FONT.family} fontSize={11.5} fontWeight={600} pointerEvents="none">
                      {mirror.show(aName) && <text x={innerW - 6} y={14} textAnchor="end" fill={mirrorPair.find((g) => g.name === aName)?.color} opacity={mirror.dim(aName)}>{aName} ▲</text>}
                      {mirror.show(bName) && <text x={innerW - 6} y={innerH - 8} textAnchor="end" fill={mirrorPair.find((g) => g.name === bName)?.color} opacity={mirror.dim(bName)}>{bName} ▼</text>}
                    </g>
                  )}
                </g>
              </svg>
            </div>
          ))
        )}
        {layout === 'Mirror' ? (
          <>
            <Legend items={mirrorPair.map((s) => ({ label: s.name, color: s.color, shape: 'square' as const, detail: `n=${s.values.length}` }))} {...mirror.props} />
            {mirrorStats && (
              <ChartAside><p className="chart-note" style={{ fontFamily: 'var(--mono)' }}>
                median {aName} {fmt(mirrorStats.medianA, 2)} vs {bName} {fmt(mirrorStats.medianB, 2)} (Δ {fmt(mirrorStats.medianDiff, 2)}) · Welch t = {fmt(mirrorStats.t, 2)}, p {mirrorStats.p < 0.001 ? '< 0.001' : `= ${fmt(mirrorStats.p, 3)}`} · Cohen's d = {fmt(mirrorStats.d, 2)} · Mann–Whitney AUC = {fmt(mirrorStats.auc, 2)}
                <span style={{ fontFamily: 'var(--font)' }}> — the effect sizes are the finding; p only says the n was large enough to see it.</span>
              </p></ChartAside>
            )}
          </>
        ) : (
          <Legend
            items={allSeries.map((s) => ({ label: s.name, color: s.color, shape: 'square' as const, detail: `n=${s.values.length}` }))}
            hidden={hidden}
            onToggle={(l) => setHidden((h) => toggleIn(h, l))}
            onHover={setHover}
            highlighted={hover}
          />
        )}
        {nOutside > 0 && <Note>{nOutside} of {all.length} values fall outside the displayed 1–99 % range and are not drawn.</Note>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
