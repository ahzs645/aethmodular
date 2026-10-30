import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { FieldSelect } from '@/components/FieldSelect'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { ExcludedNote, useMarkedRows } from '@/lib/markedRows'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { useAddisMet } from '../common/met'
import { bestLowThreshold, quantile, splitSweep, type SplitFit } from '../common/stats'
import { DataTable, Grid, HIGH, LOW, SitePick, num, ofSite, seasonsOf, shortSeason, useSite } from '../common/ui'

const PROVENANCE = 'notebooks/analysis/meteorology/rh_sweep_seasonal.ipynb · hips_ftir_intercept_rh_recursive_diagnostics.ipynb (RH threshold sweep)'

/** the notebooks' three method pairings, x then y (rh_sweep_seasonal PAIRS_FOR_RH) */
export const PAIRS = [
  { key: 'hips-ftir', x: 'EC (FTIR)', y: 'HIPS BC', label: 'HIPS BC vs FTIR EC' },
  { key: 'aeth-ftir', x: 'EC (FTIR)', y: 'IR BCc (aethalometer)', label: 'Aethalometer IR BCc vs FTIR EC' },
  { key: 'aeth-hips', x: 'HIPS BC', y: 'IR BCc (aethalometer)', label: 'Aethalometer IR BCc vs HIPS BC' },
] as const

/** the split variable's thresholds: the notebooks' 30 … 85 % for a humidity, else 12 steps across p10 – p90 */
export function thresholdsFor(values: number[], unit: string | undefined): number[] {
  if (unit === '%') return Array.from({ length: 12 }, (_, i) => 30 + 5 * i)
  const lo = quantile(values, 0.1), hi = quantile(values, 0.9)
  if (!Number.isFinite(lo) || !(hi > lo)) return []
  const step = (hi - lo) / 11, digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2
  return [...new Set(Array.from({ length: 12 }, (_, i) => +(lo + step * i).toFixed(digits)))]
}

export const defaultRh = (fields: string[], rows: Record<string, unknown>[]) =>
  ['RH (station, sample day)', 'Relative humidity (station)', 'Relative humidity (ERA5)'].find((f) => fields.includes(f) && rows.some((r) => num(r, f) !== null)) ?? fields[0] ?? ''

/**
 * The season-stratified RH threshold sweep: at each threshold, a separate OLS for the filters at or below it
 * and above it; slope, distance from the ideal slope of 1, intercept, and how many filters each side holds.
 */
export function RhSweep({ rows: viewRows, meta: viewMeta }: SectionPageProps) {
  const { rows, meta } = useAddisMet(viewRows, viewMeta)
  const [site, setSite, sites] = useSite(rows, 'rhs')
  const siteRows = useMemo(() => ofSite(rows, site), [rows, site])
  const pairs = PAIRS.filter((p) => meta.fields.includes(p.x) && meta.fields.includes(p.y))
  const [pairKey, setPairKey] = useSearchParam<string>('rhs.pair', 'hips-ftir', PAIRS.map((p) => p.key))
  const pair = pairs.find((p) => p.key === pairKey) ?? pairs[0]
  const [splitParam, setSplit] = useSearchParam<string>('rhs.by', '')
  const split = meta.fields.includes(splitParam) ? splitParam : defaultRh(meta.fields, siteRows)
  const [mode, setMode] = useSearchParam<'Pooled' | 'By season'>('rhs.mode', 'By season', ['Pooled', 'By season'])
  const [minParam, setMin] = useSearchParam<string>('rhs.min', '5', ['5', '10', '20'])
  const minN = Number(minParam)
  const seasons = useMemo(() => seasonsOf(siteRows, meta), [siteRows, meta])
  const unit = meta.field_units[split]
  const thresholds = useMemo(() => thresholdsFor(siteRows.map((r) => num(r, split)).filter((v): v is number => v !== null), unit), [siteRows, split, unit])

  const groups = useMemo(() => {
    if (!pair) return []
    const pts = (rs: typeof siteRows) => rs.flatMap((r) => { const x = num(r, pair.x), y = num(r, pair.y), s = num(r, split); return x !== null && y !== null && s !== null ? [{ x, y, s }] : [] })
    const all = [{ name: 'All filters', color: INK.text, points: pts(siteRows) }]
    const per = seasons.map((s) => ({ name: s.name, color: s.color, points: pts(siteRows.filter((r) => r.season === s.name)) }))
    return (mode === 'Pooled' ? all : per).map((g) => ({ ...g, sweep: splitSweep(g.points, thresholds, minN) }))
  }, [pair, siteRows, split, seasons, mode, thresholds, minN])

  const series = (metric: (f: SplitFit, side: 'low' | 'high') => number | null): SweepSeries[] => groups.flatMap((g) => (['low', 'high'] as const).map((side) => ({
    id: `${g.name}-${side}`, label: `${shortSeason(g.name)} · ${side === 'low' ? `${split} ≤ threshold` : 'above'}`,
    color: mode === 'Pooled' ? (side === 'low' ? LOW : HIGH) : g.color, dash: side === 'high' ? '5 4' : undefined,
    points: g.sweep.map((f) => ({ x: f.threshold, y: metric(f, side), tip: [`${shortSeason(g.name)}, ${side === 'low' ? '≤' : '>'} ${f.threshold}`, `n ${side === 'low' ? f.nLow : f.nHigh}`,
      `slope ${fmt(f[side]?.slope, 3)} · intercept ${fmt(f[side]?.intercept, 3)} · R² ${fmt(f[side]?.r2, 2)}`] })),
  })))
  const best = groups.map((g) => ({ g, b: bestLowThreshold(g.sweep, minN) }))
  const chosen = best.flatMap(({ g, b }) => (b ? [{ series: `${g.name}-low`, x: b.threshold }] : []))
  const xLabel = `${split} threshold${unit ? ` (${unit})` : ''}`
  const shade = (lo: number, hi: number) => ({ y, iw }: { y: (v: number) => number; iw: number }) => <rect x={0} width={iw} y={y(hi)} height={Math.max(0, y(lo) - y(hi))} fill={INK.good} fillOpacity={0.08} />

  // every pairing for the summary table (code 6), on the same site, split and seasons
  const summary = useMemo(() => pairs.flatMap((p) => (mode === 'Pooled' ? [{ name: 'All filters' }] : seasons).map((s) => {
    const rs = s.name === 'All filters' ? siteRows : siteRows.filter((r) => r.season === s.name)
    const pts = rs.flatMap((r) => { const x = num(r, p.x), y = num(r, p.y), v = num(r, split); return x !== null && y !== null && v !== null ? [{ x, y, s: v }] : [] })
    const b = bestLowThreshold(splitSweep(pts, thresholds, minN), minN)
    return { pair: p.label, season: s.name, n: pts.length, b }
  })), [pairs, mode, seasons, siteRows, split, thresholds, minN])

  // Filters → excluded: Mark (lib/markedRows): the site's excluded filters these fits leave out, counted in a note
  const markedMet = useAddisMet(useMarkedRows(), viewMeta).rows
  const nOut = useMemo(() => {
    const out = ofSite(markedMet, site).filter((r) => num(r, split) !== null)
    const has = (p: (typeof PAIRS)[number]) => (r: (typeof out)[number]) => num(r, p.x) !== null && num(r, p.y) !== null
    return { pair: pair ? out.filter(has(pair)).length : 0, any: out.filter((r) => pairs.some((p) => has(p)(r))).length }
  }, [markedMet, site, split, pair, pairs])

  const controls = <>
    <SitePick value={site} options={sites} onChange={setSite} />
    {pairs.length > 1 && <Segmented label="Pair" value={pair?.key ?? ''} options={pairs.map((p) => p.key)} onChange={setPairKey} optionLabel={(k) => PAIRS.find((p) => p.key === k)!.label.replace(' vs ', ' ~ ')} />}
    <FieldSelect label="Split by" value={split} meta={meta} onChange={setSplit} ratios={false} />
    <Segmented label="Seasons" value={mode} options={['Pooled', 'By season'] as const} onChange={setMode} />
    <Segmented label="Min filters per side" value={minParam} options={['5', '10', '20']} onChange={setMin} />
  </>
  const n = groups.reduce((a, g) => a + g.points.length, 0)
  return (
    <>
      <ChartFrame id="rhs-sweep" title={`Threshold sweep — ${pair?.label ?? 'no pair'} split by ${split}, ${site}`}
        subtitle={`At each threshold, an OLS fit (${pair?.y} on ${pair?.x}) for the filters with ${split} at or below it (solid) and above it (dashed), each only with at least ${minN} filters. Ringed: the threshold whose low side comes closest to slope 1. The green band is slope 0.9–1.1. ${split.includes('sample day') ? 'The split is the Addis station\'s day on the sample date, as the notebooks join it.' : 'The split is a mean over each filter\'s sampling window, not the notebooks\' sample-day station value.'}`}
        provenance={PROVENANCE} controls={controls}>
        {!pair ? <Empty>This view has none of the pairs (FTIR EC, HIPS BC, aethalometer IR BCc).</Empty>
          : n < minN * 2 || !thresholds.length ? <Empty>Fewer than {minN * 2} filters at {site} carry {pair.x}, {pair.y} and {split}.</Empty> : <>
            <Grid>
              <SweepPlot series={series((f, s) => f[s]?.slope ?? null)} xLabel={xLabel} yLabel="OLS slope" reference={1} chosen={chosen} background={shade(0.9, 1.1)} title="Slope" />
              <SweepPlot series={series((f, s) => (f[s] ? Math.abs(f[s]!.slope - 1) : null))} xLabel={xLabel} yLabel="|slope − 1|" reference={0} chosen={chosen} title="Distance from slope 1" />
              <SweepPlot series={series((f, s) => f[s]?.intercept ?? null)} xLabel={xLabel} yLabel={`Intercept (${meta.field_units[pair.y] ?? ''})`.replace(' ()', '')} reference={0} chosen={chosen} title="Intercept" />
              <SweepPlot series={series((f, s) => (s === 'low' ? f.nLow : f.nHigh))} xLabel={xLabel} yLabel="Filters" include={[0]} title="Filters on each side" />
            </Grid>
            <Key items={[...(mode === 'Pooled' ? [{ label: `${split} ≤ threshold`, color: LOW, kind: 'line' as const }, { label: 'above the threshold', color: HIGH, kind: 'dash' as const }]
              : [...groups.map((g) => ({ label: shortSeason(g.name), color: g.color, kind: 'line' as const })), { label: 'dashed: above the threshold', color: INK.muted, kind: 'dash' as const }])]} />
            <ExcludedNote count={nOut.pair} />
            {best.some(({ b }) => b) && <Note>Closest to slope 1 on the low side: {best.filter(({ b }) => b).map(({ g, b }) => `${shortSeason(g.name)} ${split} ≤ ${b!.threshold} (slope ${fmt(b!.low!.slope, 2)}, n ${b!.nLow})`).join('; ')}. A best threshold picked from a sweep is not a test; with a handful of filters on one side the slope swings widely.</Note>}
          </>}
      </ChartFrame>
      <ChartFrame id="rhs-best" title="Best threshold per season and pairing — where the low side comes closest to slope 1"
        subtitle={`The notebook's summary table (rh_sweep_seasonal, code 6) for every pairing this view has, at ${site}, split by ${split}, with at least ${minN} filters on the low side.`}
        provenance={PROVENANCE} exportable={false}>
        <DataTable rows={summary} rowKey={(r) => `${r.pair}-${r.season}`} columns={[
          { key: 'pair', label: 'Pairing', value: (r) => r.pair },
          { key: 'season', label: 'Season', value: (r) => shortSeason(r.season) },
          { key: 'n', label: 'Filters', num: true, value: (r) => r.n },
          { key: 't', label: `Best ${split}`, num: true, value: (r) => (r.b ? `≤ ${r.b.threshold}` : '—') },
          { key: 's', label: 'Slope (low)', num: true, value: (r) => fmt(r.b?.low?.slope, 3) },
          { key: 'd', label: '|slope − 1|', num: true, value: (r) => (r.b?.low ? fmt(Math.abs(r.b.low.slope - 1), 3) : '—') },
          { key: 'i', label: 'Intercept (low)', num: true, value: (r) => fmt(r.b?.low?.intercept, 3) },
          { key: 'r2', label: 'R² (low)', num: true, value: (r) => fmt(r.b?.low?.r2, 3) },
          { key: 'nl', label: 'n low / high', num: true, value: (r) => (r.b ? `${r.b.nLow} / ${r.b.nHigh}` : '—') },
        ]} />
        <ExcludedNote count={nOut.any} />
      </ChartFrame>
    </>
  )
}
