import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { histogramSeries, SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { f2, fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { Grid } from '../../common/ui'
import { provenance, rows as tableRows, type SpectralExport } from '../spectralCommon'
import { residualDiagnostics } from './data'

const REMOVED = '#e45756', KEPT = '#bbbbbb'

/**
 * spartan_ec 08's residual diagnostics (fig08_2): measured − predicted on the full 906-filter fit at k = 20, as a
 * histogram with ±3σ, and |residual| ranked with the filters the iterative cleaning removes in red. Drawn from the
 * export's calibration_all (the Drive recomputation); with only the 61 tracked removals it says so instead.
 */
export function CleaningResiduals({ data }: { data: SpectralExport }) {
  const all = useMemo(() => tableRows(data, 'calibration_all'), [data])
  const d = useMemo(() => residualDiagnostics(all), [all])
  const full = all.length > tableRows(data, 'removed').length
  if (!d || !full) return null

  const lo = d3.min(d.points, (p) => p.residual)!, hi = d3.max(d.points, (p) => p.residual)!
  const bins = d3.bin().domain([lo, hi]).thresholds(d3.range(60).map((i) => lo + ((hi - lo) * i) / 60))(d.points.map((p) => p.residual))
  const edges = [...bins.map((b) => b.x0!), bins[bins.length - 1].x1!]
  const hist = histogramSeries({ id: 'filters', label: 'filters', color: '#54a24b', edges, counts: bins.map((b) => b.length), fill: 0.75, width: 1,
    tip: (i, n) => [`${f2(edges[i])} to ${f2(edges[i + 1])} µg`, `${n} filters`] })

  const ranked = [...d.points].sort((a, b) => Math.abs(b.residual) - Math.abs(a.residual)).map((p, i) => ({ ...p, rank: i + 1 }))
  const dots = (removed: boolean): SweepSeries => ({
    id: removed ? 'removed' : 'kept', label: removed ? 'removed by the cleaning' : 'kept', color: removed ? REMOVED : KEPT, width: 0, marks: true, r: removed ? 2.6 : 2, interval: 'none',
    points: ranked.filter((p) => p.removed === removed).map((p) => ({ x: p.rank, y: Math.abs(p.residual), id: p.id,
      tip: [p.id, `${p.site}`, `|residual| ${f2(Math.abs(p.residual))} µg`, removed ? `removed in round ${p.round}` : 'kept'] })),
  })
  const removed = d.points.filter((p) => p.removed).length
  const removedBelow = d.points.filter((p) => p.removed && Math.abs(p.residual) <= d.threshold).length
  const rounds = tableRows(data, 'cleaning_rounds')
  const later = rounds.slice(1).map((r) => String(r['3σ threshold (µg)']))

  return (
    <ChartFrame id="ethiopia-residuals" title={`spartan_ec 08's residual diagnostics — ${d.beyond} filters beyond the full fit's 3σ, ${removed} removed once the spread shrinks`}
      subtitle={`Residual = measured − predicted TOR EC on the full 906-filter fit at k = 20 (µg per filter). σ = ${f2(d.sigma)} µg (numpy's std), so the 3σ line is ${f2(d.threshold)} µg: ${d.beyond} filters lie beyond it${d.beyond === rounds[0]?.dropped ? `, exactly the ${d.beyond} the first cleaning round drops` : ''}. The rule then refits and re-measures σ on what is left (3σ of ${later.join(', ')} µg in rounds 2–${later.length + 1}), so ${removedBelow} of the ${removed} removed filters sit inside this first line — their residuals are judged against the cleaner fits of later rounds.`}
      provenance={provenance(data, ['calibration_all', 'cleaning_rounds'], 'research/spartan_ec_2026_06_16/08_outlier_removal_explainer.ipynb fig08_2.')}>
      <Grid>
        <SweepPlot series={[hist]} xScale="linear" xDomain={[lo, hi]} height={260} title="Residual distribution (full fit)" xLabel="Residual, measured − predicted (µg)" yLabel="Filters"
          include={[0]} yFormat={(v) => (Number.isInteger(v) ? fmt(v, 0) : '')}
          xMarkers={[{ at: -d.threshold, label: '−3σ', color: REMOVED }, { at: d.threshold, label: '+3σ', color: REMOVED }]} />
        <SweepPlot series={[dots(false), dots(true)]} xScale="linear" height={260} title="Filters ranked by |residual|" xLabel="Filters, sorted by |residual|" yLabel="|Residual| (µg)"
          include={[0]} yMarkers={[{ at: d.threshold, label: '3σ', color: REMOVED }]} />
      </Grid>
      <Key items={[{ label: `removed by the 3σ cleaning (${later.length + 1} rounds)`, color: REMOVED }, { label: 'kept', color: KEPT }, { label: 'full-fit ±3σ', color: REMOVED, kind: 'dash' as const }]} />
      <div className="legend-note" style={{ color: INK.muted }}>Hover a dot for its AnalysisId, site and cleaning round.</div>
    </ChartFrame>
  )
}
