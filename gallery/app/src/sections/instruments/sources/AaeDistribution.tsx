import { useMemo } from 'react'
import { ChartFrame, Empty, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { f2, pct } from '@/lib/format'
import { densities } from '@/lib/table'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { AAE_PAIRS, CLASSES, MA350_NM, bccOffset, classShares, ma350Ratio, type AaePair, type AaeRegions } from './aae'
import { dailyAae, rowAae, type AaeFrom, type Basis } from './AaeValues'
import { NOTEBOOK, RegionBands, provenance } from './common'
import type { DailySite } from './data'

export interface SiteValues { code: string; name: string; color: string; values: number[] }

/** each site's AAE values on the chosen basis */
export function siteValues(basis: Basis, from: AaeFrom, pair: AaePair, rows: readonly FilterRow[], daily: readonly DailySite[], meta: MetaFile): SiteValues[] {
  const num = (v: number | null): v is number => v !== null && Number.isFinite(v)
  if (basis === 'Daily bins') return daily.map((d) => ({ code: d.code, name: d.name, color: d.color, values: dailyAae(d.daily, pair, from).filter(num) }))
  return meta.sites.map((s) => ({ code: s.code, name: s.name, color: s.color, values: rows.filter((r) => r.code === s.code).map((r) => rowAae(r, pair, from)).filter(num) }))
    .filter((s) => s.values.length)
}

/**
 * addis_01 Task 2: the AAE distribution with the fossil / mixed / biomass regions shaded, here for every
 * site in view, as densities on shared bins so sites of different n compare. The notebooks' BCc-based
 * AAE can be drawn beside it (dashed) to show the constant offset.
 */
export function AaeDistribution({ rows, meta, daily, pair, basis, regions }: {
  rows: FilterRow[]; meta: MetaFile; daily: DailySite[]; pair: AaePair; basis: Basis; regions: AaeRegions
}) {
  const [bccParam, setBcc] = useSearchParam<'0' | '1'>('aes.distBcc', '0', ['0', '1'])
  const showBcc = bccParam === '1'
  const lg = useLegend()
  const abs = useMemo(() => siteValues(basis, 'absorption', pair, rows, daily, meta), [basis, pair, rows, daily, meta])
  const bcc = useMemo(() => (showBcc ? siteValues(basis, 'BCc', pair, rows, daily, meta) : []), [showBcc, basis, pair, rows, daily, meta])
  // the notebook's axis (−1 to 3), a little wider; values beyond it are counted under the chart, not drawn
  const range: [number, number] = [showBcc ? -2 : -1, 3.5]
  const inRange = (v: number) => v >= range[0] && v <= range[1]
  const d = useMemo(() => densities([...abs.map((s) => ({ key: s.code, values: s.values.filter(inRange) })), ...bcc.map((s) => ({ key: `${s.code}·bcc`, values: s.values.filter(inRange) }))], { bins: 36 }),
    [abs, bcc, range[0]]) // eslint-disable-line react-hooks/exhaustive-deps
  // an AAE outside 0 to 3 is not ambient absorption (black carbon near 1, brown carbon up to about 3)
  const odd = abs.map((s) => ({ s, low: s.values.filter((v) => v < 0).length, high: s.values.filter((v) => v > 3).length })).filter((o) => (o.low + o.high) / o.s.values.length > 0.02)
  const outside = abs.reduce((a, s) => a + s.values.filter((v) => !inRange(v)).length, 0)
  const label = `AAE ${pair}`
  const series: SweepSeries[] = d.series.flatMap((s) => {
    const fromBcc = s.key.endsWith('·bcc'), code = s.key.replace('·bcc', '')
    const site = abs.find((a) => a.code === code)
    if (!site || !lg.show(site.name)) return []
    const name = fromBcc ? `${site.name} · from BCc` : site.name
    return [{ id: s.key, label: name, color: site.color, group: site.name, curve: 'monotone' as const, marks: false, interval: 'none' as const, dash: fromBcc ? '5 4' : undefined, width: fromBcc ? 1.4 : 2,
      points: s.centers.map((c, i) => ({ x: c, y: s.density[i], tip: [name, `${label}: ${f2(c)}`, `density ${f2(s.density[i])}`] })) }]
  })
  const unit = basis === 'Daily bins' ? 'daily bins' : 'filters'
  const p = AAE_PAIRS[pair], ratio = ma350Ratio(p.short, p.long), offset = bccOffset(ratio, MA350_NM[p.short], MA350_NM[p.long])
  return (
    <ChartFrame id="aeth-sources-aae-distribution"
      title={`AAE distribution by site — ${label} from absorption, with the fossil / mixed / biomass regions`}
      subtitle={`Each curve is one site's ${unit}, as a density so sites with different numbers compare. Regions from config.AAE_REGIONS (fossil at or below ${regions.fossil_max}, biomass at or above ${regions.biomass_min}, optics.classify_aae). With "BCc AAE" on, the dashed curves are the same bins with AAE taken from BCc, as the notebooks did: the whole distribution moves down by ${f2(-offset)}, the instrument's wavelength-dependent cross-section (σ ${p.short}/σ ${p.long} = ${f2(ratio)}), not the aerosol.`}
      provenance={provenance([NOTEBOOK.addis01, NOTEBOOK.addis01daily, NOTEBOOK.optics], `Reads aethalometer/daily/<CODE>.json (AAE per daily bin) or the filters' "${AAE_PAIRS[pair].field}".`)}
      controls={<Toggle label="BCc AAE (the notebooks')" checked={showBcc} onChange={(v) => setBcc(v ? '1' : '0')} title="Also draw each site's AAE taken from BCc, dashed: what addis_01 computed" />}>
      {!abs.length ? <Empty>No site in view has aethalometer AAE {basis === 'Daily bins' ? 'daily bins' : 'over its filters'}.</Empty> : <>
        <SweepPlot series={series} xLabel={label} yLabel="Density" include={[0]} height={280} dim={lg.dim} xDomain={range}
          background={({ x, iw, ih }) => <RegionBands scale={(v) => (x as (v: number) => number)(v)} length={iw} depth={ih} regions={regions} />} />
        <Legend {...lg.props} items={abs.map((s) => {
          const c = classShares(s.values, regions)
          return { label: s.name, color: s.color, shape: 'line' as const, detail: `n=${c.n} · median ${f2(c.median)} · ${CLASSES.map((k) => pct(c.shares[k])).join(' / ')}` }
        })} note="detail: fossil / mixed / biomass shares (absorption AAE); click a site to hide it" />
        {(outside > 0 || odd.length > 0) && <p className="chart-note">
          {outside > 0 && `${outside} value${outside > 1 ? 's' : ''} outside ${range[0]} to ${range[1]} ${outside > 1 ? 'are' : 'is'} not drawn but counted in the shares. `}
          {odd.map((o) => `${o.s.name}: ${pct(o.low / o.s.values.length)} of ${unit} below 0 and ${pct(o.high / o.s.values.length)} above 3`).join('; ')}
          {odd.length > 0 && '. An AAE outside 0 to 3 is not ambient absorption (black carbon sits near 1, brown carbon up to about 3); such bins are more likely noise in a channel than aerosol, and they swell the fossil and biomass shares at both ends. Read those sites’ classes as a spread, not a census.'}
        </p>}
      </>}
    </ChartFrame>
  )
}
