import { useMemo } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Crossplot, sharedCrossDomains, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { coef } from '@/lib/format'
import { regression } from '@/lib/stats'
import { useEstimator, type Estimator } from '@/lib/regressionPrefs'
import { withUnit } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { finite, instrumentColor, instrumentName, PERIOD_COLOR, PERIOD_LABEL, periodOf, type Period, type QualityIndex } from './data'

export const AETH_IR = 'IR BCc (aethalometer)'
/** the filter measurements the flow-fix notebook compared the aethalometer with */
export const REFERENCES = ['HIPS BC', 'EC (FTIR)'] as const
const REF_LABEL: Record<string, string> = { 'HIPS BC': 'HIPS', 'EC (FTIR)': 'FTIR EC' }
const ORDER: Period[] = ['before', 'gap', 'after', 'no_fix']

export const slopeFor = (s: ReturnType<typeof regression>, m: Estimator) =>
  !s ? null : m === 'ols' ? s.slope : m === 'deming' ? s.demingSlope : s.yorkSlope ?? s.demingSlope

export function crossPoint(r: FilterRow, x: string, y: string, color: string, source: string, meta: MetaFile, hollow = false): CrossPoint {
  const xv = r[x] as number, yv = r[y] as number
  return {
    x: xv, y: yv, color, hollow, id: r.id,
    tip: [r.id, r.date, `${x}: ${xv.toFixed(2)}`, `${y}: ${yv.toFixed(2)}`],
    rec: { id: r.id, source, site: r.site, date: r.date, fields: [[withUnit(x, meta.field_units), xv], [withUnit(y, meta.field_units), yv]] },
  }
}

/**
 * Before and after the flow fix against the filters: the aethalometer's IR BCc over each filter's sampling window
 * against HIPS absorption (as BC, over MAC 10) or FTIR EC, one crossplot per flow period at one site
 * (FlowFix_BeforeAfter_Analysis §6–7, plotting/comparisons.flow_periods), and every site's slope and r² by period
 * on one strip (§8–9, the summary bars). Filters are classified by their sample date with
 * flow_periods.classify_flow_period's rule; the dates, seasons and exclusions of the Filters panel apply.
 */
export function FlowFixCompare({ rows, meta, index }: { rows: FilterRow[]; meta: MetaFile; index: QualityIndex }) {
  const method = useEstimator()
  const sites = Object.keys(index.flow_periods).filter((c) => index.flow_periods[c].before_end || index.flow_periods[c].after_start)
  const [site, setSite] = useSearchParam<string>('aq.ffsite', 'USPA', sites)
  const [ref, setRef] = useSearchParam<string>('aq.ffref', 'HIPS BC', [...REFERENCES])
  const paired = useMemo(() => rows.filter((r) => finite(r[AETH_IR] as number) && finite(r[ref] as number)), [rows, ref])
  const byPeriod = useMemo(() => {
    const p = index.flow_periods[site]
    const out = new Map<Period, FilterRow[]>()
    for (const r of paired) if (r.code === site) { const k = periodOf(r.date, p); out.set(k, [...(out.get(k) ?? []), r]) }
    return ORDER.filter((k) => out.has(k)).map((k) => ({ k, rows: out.get(k)! }))
  }, [paired, site, index])
  const sets = byPeriod.map(({ k, rows: rs }) => rs.map((r) => crossPoint(r, AETH_IR, ref, PERIOD_COLOR[k], 'Aethalometer flow fix', meta)))
  const dom = sharedCrossDomains(sets)

  const blocks = useMemo<IntervalBlock[]>(() => Object.keys(index.sites).filter((c) => c in index.flow_periods).map((code) => {
    const p = index.flow_periods[code]
    const groups = new Map<Period, FilterRow[]>()
    for (const r of paired) if (r.code === code) { const k = periodOf(r.date, p); groups.set(k, [...(groups.get(k) ?? []), r]) }
    return {
      key: code, title: instrumentName(code, meta), color: instrumentColor(code, meta),
      rows: ORDER.filter((k) => groups.has(k)).map((k) => {
        const g = groups.get(k)!
        const s = regression(g.map((r) => r[AETH_IR] as number), g.map((r) => r[ref] as number), { errorsInVariables: true, lambda: 1 })
        const slope = slopeFor(s, method)
        return {
          key: k, label: PERIOD_LABEL[k], labelColor: PERIOD_COLOR[k], selected: code === site,
          marks: [[{ mid: slope, color: PERIOD_COLOR[k] }], [{ mid: s?.r2 ?? null, color: PERIOD_COLOR[k] }]],
          note: `n=${g.length}`,
          tip: [`${instrumentName(code, meta)} · ${PERIOD_LABEL[k]}`, `${g.length} filters`, `Deming slope ${coef(slope)} · r² ${coef(s?.r2)}`],
          onClick: () => setSite(code),
        }
      }),
    }
  }).filter((b) => b.rows.length), [paired, index, meta, method, site, setSite])

  const periods = index.flow_periods[site]
  const yLabel = withUnit(ref, meta.field_units), xLabel = withUnit(AETH_IR, meta.field_units)
  return (<>
    <ChartFrame id="aq-flowfix-cross" title={`${instrumentName(site, meta)} · before and after the flow fix — the aethalometer against ${REF_LABEL[ref]}`}
      subtitle={`Each filter's IR BCc (the overlap-weighted mean of the daily bins over its sampling window) against ${REF_LABEL[ref]}, one panel per flow period: before ≤ ${periods?.before_end ?? '—'}, after ≥ ${periods?.after_start ?? '—'}, the filters in between as the gap. ${periods?.description ?? ''}. Fits follow the sidebar's estimator. HIPS BC is HIPS Fabs over a MAC of ${meta.mac_value} m²/g; FTIR EC is a spectral prediction, not thermal EC.`}
      provenance="FlowFix_BeforeAfter_Analysis §6–7 · plotting/comparisons.flow_periods · flow_fix_explorer (test_flow_fix_dates)"
      controls={<>
        <Segmented label="Site" value={site} options={sites} onChange={setSite} optionLabel={(c) => instrumentName(c, meta)} />
        <Segmented label="Against" value={ref} options={[...REFERENCES]} onChange={setRef} optionLabel={(c) => REF_LABEL[c]} />
      </>}>
      {byPeriod.length ? <FacetGrid items={byPeriod} minCell={300} maxCols={3}
        render={({ k }, cell, i) => <Crossplot points={sets[i]} title={`${PERIOD_LABEL[k]} · ${sets[i].length} filters`} xLabel={xLabel} yLabel={yLabel}
          maxSide={Math.min(360, cell - 80)} domain={dom.domain} />} />
        : <Empty>No filter at {instrumentName(site, meta)} has both IR BCc and {REF_LABEL[ref]} in the current subset.</Empty>}
    </ChartFrame>
    <ChartFrame id="aq-flowfix-summary" title={`Slope and r² by flow period — every site, the aethalometer against ${REF_LABEL[ref]}`}
      subtitle={`The fit of ${REF_LABEL[ref]} on IR BCc per site and flow period (Deming slope with error ratio 1, r² of the pairs), the notebook's R² and slope bars on one strip. Click a row to draw that site above. A slope nearer 1 is not by itself agreement: the intercept, the MAC and the sampling overlap matter too.`}
      provenance="FlowFix_BeforeAfter_Analysis §8–9 (summary bars and table) · plotting/comparisons.summary_bars">
      {blocks.length ? <IntervalStrip blocks={blocks} labelWidth={150} noteWidth={48} minPanelWidth={220}
        panels={[{ label: 'Deming slope', reference: 1, format: (v) => v.toFixed(2) }, { label: 'r²', domain: [0, 1], format: (v) => v.toFixed(1) }]} />
        : <Empty>No paired filters in the current subset.</Empty>}
    </ChartFrame>
  </>)
}
