/**
 * The population charts: held-out improvement over the median at every frozen denominator population (stability
 * 03), the three models across populations (proportionality 04), equal filters against equal quarters
 * (proportionality 05), and the stricter-denominator ratio sensitivity (filter_only_diagnostics 05).
 */
import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { FacetGrid } from '@/components/FacetGrid'
import { HeatCell } from '@/components/HeatCell'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt, signed } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { Grid } from '../../common/ui'
import { MODELS, POPULATIONS, denominatorSensitivity, evaluate, inPopulation, summarize, type FPoint, type Model, type Population, type Summary } from '../../common/filterOnly'
import { FoExcludedNote, MODEL_LABEL, NB_DIAG, NB_PROP, NB_STAB, SITES, SOURCE, popLabel, siteColor, type Settings, type SiteRun } from './data'

/** every site × population, leave-quarter-out: the summary and the population size */
function usePopulationGrid(points: FPoint[], s: Settings) {
  return useMemo(() => SITES.map((site) => {
    const diagnostic = points.filter((p) => p.site === site)
    return { site, cells: POPULATIONS.map(({ key }) => {
      const cohort = diagnostic.filter((p) => inPopulation(p, key))
      return { population: key as Population, n: cohort.length, sum: summarize(evaluate(diagnostic, cohort, 'leave_quarter_out', s.minTrain)) }
    }) }
  }), [points, s.minTrain])
}

/** stability 03: % MAE improvement of OLS over the training median, site × population */
export function PopulationHeat({ points, s }: { points: FPoint[]; s: Settings }) {
  const grid = usePopulationGrid(points, s)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const colours = useColourScheme('fo-heat', 'diverging')
  const vals = grid.flatMap((g) => g.cells.map((c) => c.sum.improvementPct)).filter(Number.isFinite)
  const bound = Math.max(10, ...vals.map(Math.abs))
  // blue = OLS better, red = worse, as the notebook's RdBu (and the gallery's r) draw it
  const color = d3.scaleQuantize<string>().domain([-bound, bound]).range([...colours.stops].reverse())
  const labelW = 96, top = 30, cw = Math.max(64, Math.min(120, (width - labelW - 8) / POPULATIONS.length)), ch = 44
  return (
    <ChartFrame id="fo-heat" title="Selection sensitivity — held-out improvement over the median in every frozen population"
      subtitle="Leave quarter out · filter-weighted · 100 × (median MAE − OLS MAE) / median MAE; positive = lower OLS error · n is the whole population"
      provenance={<>{NB_STAB} figure 03</>} source={`${SOURCE}/Populations.tsx › PopulationHeat`}
      tip={<>The memberships are the release's saved ones (EC ≥ k × its reported MDL among the 480 ratio filters); no threshold is chosen from these errors. At the release's settings (with Filters → excluded: Include) JPL keeps 84 ratio filters and 28 at 1.5 × MDL, where the improvement shrinks to 9.6 % and 0.5 %; its seven-filter 2 × population cannot supply ten training filters. Addis's 1.5 ×, 2 × and 3 × populations are the same 189 filters.</>}>
      <div ref={ref} style={{ position: 'relative', overflowX: 'auto' }}>
        <svg width={labelW + POPULATIONS.length * cw + 4} height={top + SITES.length * ch + 4} fontFamily={FONT.family}>
          {POPULATIONS.map((p, j) => <text key={p.key} x={labelW + j * cw + cw / 2} y={top - 10} textAnchor="middle" fontSize={11.5} fill={INK.text}>{p.label}</text>)}
          {grid.map((g, i) => <g key={g.site}>
            <text x={labelW - 8} y={top + i * ch + ch / 2} dy="0.35em" textAnchor="end" fontSize={12} fill={INK.text}>{g.site}</text>
            {g.cells.map((c, j) => { const v = c.sum.improvementPct, ok = Number.isFinite(v)
              return <HeatCell key={c.population} x={labelW + j * cw} y={top + i * ch} width={cw - 2} height={ch - 2} fill={ok ? color(v) : null} fontSize={11}
                label={ok ? `${signed(v, 1)}%  n=${c.n}` : `n/a  n=${c.n}`}
                onEnter={(e) => tip.show(e, [`${g.site} · ${popLabel(c.population)}`, ok ? `improvement ${signed(v, 1)} %` : 'no evaluated quarter', `OLS MAE ${fmt(c.sum.filters.mae.ols * s.k, 2)} · median MAE ${fmt(c.sum.filters.mae.median * s.k, 2)} ${s.u}`, `${c.sum.filters.n} evaluated of ${c.n} filters · ${c.sum.evaluatedBlocks} quarters`])} onLeave={tip.hide} /> })}
          </g>)}
        </svg>
        <ColorLegend scale={color} scheme={colours.scheme} label="MAE improvement (%)" format={(v) => fmt(v, 0)} />
        {tip.node}
      </div>
      <Note>An empty cell means no quarter had {s.minTrain} training filters: the population exists but cannot be evaluated.</Note>
      <FoExcludedNote s={s} any />
    </ChartFrame>
  )
}

const STYLE: Record<Model, { color?: string; dash?: string }> = { median: { color: '#b8b8b8', dash: '2 3' }, proportional: { color: '#8a8a8a', dash: '6 3' }, ols: {} }

/** proportionality 04: the three models' leave-quarter-out MAE in each population */
export function PopulationModels({ points, s, meta }: { points: FPoint[]; s: Settings; meta: MetaFile }) {
  const grid = usePopulationGrid(points, s)
  return (
    <ChartFrame id="fo-pop-models" title="The proportionality conclusion by population — three models across the frozen denominators"
      subtitle="Leave quarter out · equal-filter MAE · n is the population size, not the evaluated filters"
      provenance={<>{NB_PROP} figure 04</>} source={`${SOURCE}/Populations.tsx › PopulationModels`}
      tip={<>Every population is a saved membership. Changing the population changes the EC distribution and the baseline error together, so falling MAE at stricter denominators is not evidence that a method deteriorates at higher EC. No threshold was tuned.</>}>
      <Key items={MODELS.map((m) => ({ label: MODEL_LABEL[m], color: m === 'ols' ? INK.text : STYLE[m].color!, kind: m === 'ols' ? 'line' : 'dash' }))} />
      <FacetGrid items={grid} minCell={290} maxCols={2} title={(g) => g.site} render={(g, w) => {
        const color = siteColor(meta, g.site)
        const series: SweepSeries[] = MODELS.map((m) => ({ id: m, label: MODEL_LABEL[m], color: STYLE[m].color ?? color, dash: STYLE[m].dash, width: m === 'ols' ? 2 : 1.5,
          points: g.cells.map((c) => ({ x: POPULATIONS.find((p) => p.key === c.population)!.short, y: Number.isFinite(c.sum.filters.mae[m]) ? c.sum.filters.mae[m] * s.k : null,
            tip: [`${g.site} · ${popLabel(c.population)} (n ${c.n})`, ...MODELS.map((q) => `${MODEL_LABEL[q]} ${fmt(c.sum.filters.mae[q] * s.k, 2)} ${s.u}`), `${c.sum.filters.n} evaluated filters`] })) }))
        return <SweepPlot series={series} xScale="point" width={w} height={220} yLabel={`Held-out MAE (${s.u})`} include={[0]}
          xFormat={(v) => { const c = g.cells.find((x) => POPULATIONS.find((p) => p.key === x.population)!.short === v); return c ? `${v} ${c.n}` : String(v) }} />
      }} />
      <Note>Axis labels give each population and its size. A missing point is a population with no evaluated quarter.</Note>
      <FoExcludedNote s={s} any />
    </ChartFrame>
  )
}

/** proportionality 05: OLS mean signed error with filters weighted equally against quarters weighted equally */
export function Weighting({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  const sums = runs.map((r) => ({ site: r.site, lqo: summarize(r.lqo), later: summarize(r.later) }))
  const mark = (x: Summary, color: string) => [
    { mid: x.filters.mse.ols * s.k, color, shape: 'dot' as const, dy: -4 },
    { mid: x.quarters.mse.ols * s.k, color, shape: 'square' as const, dy: 4 },
  ]
  const rows: IntervalRow[] = sums.map((x) => {
    const color = siteColor(meta, x.site)
    return { key: x.site, label: x.site, labelColor: color, marks: [mark(x.lqo, color), mark(x.later, color)],
      tip: [x.site, `leave quarter out: ${signed(x.lqo.filters.mse.ols * s.k, 2)} (filters) · ${signed(x.lqo.quarters.mse.ols * s.k, 2)} (quarters) ${s.u}`,
        `later period: ${signed(x.later.filters.mse.ols * s.k, 2)} (filters) · ${signed(x.later.quarters.mse.ols * s.k, 2)} (quarters) ${s.u}`] }
  })
  return (
    <ChartFrame id="fo-weighting" title="Equal filters or equal quarters — averaging filters and averaging quarters answer different questions"
      subtitle={`${popLabel(s.population)} population · OLS with intercept · mean signed error over identical supported filters and folds within each scheme`}
      provenance={<>{NB_PROP} figure 05</>} source={`${SOURCE}/Populations.tsx › Weighting`}
      tip={<>At the release's settings (with Filters → excluded: Include) Delhi's leave-quarter-out intercept bias is −1.779 Mm⁻¹ over filters and +2.151 Mm⁻¹ over quarters; both are kept. Later-period bias stays negative under either weighting. The two marks are two estimands, not an uncertainty range.</>}>
      <Key items={[{ label: 'Each filter weighted equally', color: INK.text, kind: 'dot' }, { label: 'Each evaluated quarter weighted equally', color: INK.text, kind: 'square' }]} />
      <IntervalStrip labelWidth={110} rowHeight={30} blocks={[{ key: 'sites', rows }]}
        panels={[{ label: `Mean signed error (${s.u})`, heading: 'Withheld quarters', reference: 0, format: (v) => fmt(v, 1) },
          { label: `Mean signed error (${s.u})`, heading: 'Later-period prediction', reference: 0, format: (v) => fmt(v, 1) }]} />
      <Note>{sums.map((x) => `${x.site} ${signed(x.lqo.filters.mse.ols * s.k, 2)} / ${signed(x.lqo.quarters.mse.ols * s.k, 2)}`).join(' · ')} (leave quarter out, filters / quarters, {s.u}).</Note>
      <FoExcludedNote s={s} />
    </ChartFrame>
  )
}

/** filter_only_diagnostics 05: retained filters and the median HIPS / EC ratio at each stricter denominator rule */
export function DenominatorSensitivity({ points, s, meta }: { points: FPoint[]; s: Settings; meta: MetaFile }) {
  const rows = useMemo(() => SITES.map((site) => ({ site, color: siteColor(meta, site), d: denominatorSensitivity(points.filter((p) => p.site === site)) })), [points, meta])
  const nSeries: SweepSeries[] = rows.map((r) => ({ id: r.site, label: r.site, color: r.color, points: r.d.map((x) => ({ x: x.multiple, y: x.n, tip: [r.site, `EC ≥ ${x.multiple} × MDL`, `${x.n} filters`] })) }))
  const rSeries: SweepSeries[] = rows.map((r) => ({ id: r.site, label: r.site, color: r.color, fill: 0.06, points: r.d.map((x) => ({ x: x.multiple, y: x.n ? x.median : null, lo: x.n ? x.q25 : null, hi: x.n ? x.q75 : null,
    tip: [r.site, `EC ≥ ${x.multiple} × MDL · n ${x.n}`, x.n ? `median ${fmt(x.median, 2)} m²/g (IQR ${fmt(x.q25, 2)}–${fmt(x.q75, 2)})` : 'no filters left: no median'] })) }))
  return (
    <ChartFrame id="fo-denominator" title="Stricter EC denominators — the retained populations and their HIPS / FTIR-EC ratios"
      subtitle="Ratio population (EC > 0 and EC ≥ its reported MDL) at EC ≥ 1, 1.5, 2, 3 and 5 × the filter's own MDL · descriptive only"
      provenance={<>{NB_DIAG} figure 05</>} source={`${SOURCE}/Populations.tsx › DenominatorSensitivity`}
      tip={<>The 480-pair eligibility is unchanged; each stricter scenario reports its own count and median. A missing median means no retained filters, not a zero ratio. No threshold is optimised for agreement. The ratio is HIPS Fabs (Mm⁻¹) over FTIR EC (µg/m³), in m²/g: the MAC at which HIPS BC would equal FTIR EC on that filter.</>}>
      <Key items={rows.map((r) => ({ label: r.site, color: r.color, kind: 'line' }))} />
      <Grid min={320}>
        <SweepPlot series={nSeries} xScale="linear" xDomain={[0.8, 5.2]} xLabel="Minimum EC / reported MDL" yLabel="Retained filters" include={[0]} height={250} />
        <SweepPlot series={rSeries} xScale="linear" xDomain={[0.8, 5.2]} xLabel="Minimum EC / reported MDL" yLabel="HIPS / FTIR EC (m²/g)" include={[0]} height={250}
          yMarkers={[{ at: s.mac, label: `your MAC ${fmt(s.mac, 1)}`, color: INK.muted }]} />
      </Grid>
      <Note>Bands are the interquartile range of the ratio. The dashed marker is the reader's MAC (Filters → units): ratios above it mean HIPS BC exceeds FTIR EC at that MAC.</Note>
      <FoExcludedNote s={s} any />
    </ChartFrame>
  )
}
