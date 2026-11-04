import { useMemo } from 'react'
import type { FilterRow } from '@/lib/types'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { FieldSelect } from '@/components/FieldSelect'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { ExcludedNote, useMarkedRows } from '@/lib/markedRows'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { useAddisMet } from '../common/met'
import { EC, MIN_FILTERS, defaultMoisture, macPoints, trimBySite, type MacPoint } from '../common/moistureMac'
import { ols, pearson, quantile, steiger } from '@/lib/stats'
import { BASIS_NOTE, DataTable, Grid, HIGH, LOW, MID, SitePick, initials, line, num, pText, ofSite, point, seasonsOf, shortSeason, useConcRows, useSite } from '../common/ui'
import { namedColor, siteColor } from '@/lib/siteColours'

const NB = 'notebooks/analysis/meteorology/pw_mac_calibration.ipynb'
const HIPS_BC = 'HIPS BC'
const PW = 'Precipitable water (AERONET)', AOD = 'AOD 500 nm (AERONET)'
// the opening choice: the first that serves every site in view (common/moistureMac defaultMoisture)
const MOISTURE = [PW, 'Precipitable water (ERA5)', 'RH (station, sample day)', 'Relative humidity (station)', 'Relative humidity (ERA5)', 'Dew point (ERA5)']
const ALL = 'All sites'
const shortSite = (site: string) => site.split(',')[0]

/**
 * Effective MAC against column and surface moisture (pw_mac_calibration): the "money plot" with its OLS and a
 * 95 % band on the slope, MAC and moisture by season, a moisture split asking whether moisture moves the
 * slope (MAC) or the intercept (a baseline), and the Steiger test of whether FTIR EC tracks AOD better
 * than HIPS does. It draws the sites picked in Samples together, each with its own fit, or one of them (Site).
 */
export function MoistureMac({ rows: viewRows, meta: viewMeta }: SectionPageProps) {
  // Fabs / EC per filter: EC in µg/m³ whatever the basis (common/ui useConcRows)
  const { rows, meta } = useAddisMet(useConcRows(viewRows, viewMeta), viewMeta)
  const [site, setSite, sites] = useSite(rows, 'mm', true)
  const siteRows = useMemo(() => ofSite(rows, site), [rows, site])
  const [mParam, setM] = useSearchParam<string>('mm.by', '')
  const offered = useMemo(() => MOISTURE.filter((f) => meta.fields.includes(f)), [meta.fields])
  const opening = useMemo(() => defaultMoisture(siteRows, offered), [siteRows, offered])
  const moisture = meta.fields.includes(mParam) ? mParam : opening ?? meta.fields[0] ?? PW
  const pooled = site === ALL
  // the sites drawn, in the Samples pick's order; pooled, they take the place of the seasons in the colours and strips
  const picked = useMemo(() => (pooled ? sites.filter((s) => s !== ALL) : [site]), [pooled, sites, site])
  const scope = pooled ? `${picked.length} sites` : site
  const [splitParam, setSplit] = useSearchParam<'median' | 'terciles'>('mm.split', 'median', ['median', 'terciles'])
  const seasons = useMemo(() => seasonsOf(siteRows, meta), [siteRows, meta])
  const colorOf = (r: FilterRow) => (pooled ? siteColor(meta, r.site) : namedColor(seasons, r.season))
  const strata = pooled ? picked.map((s) => ({ name: s, label: shortSite(s), color: siteColor(meta, s) })) : seasons.map((s) => ({ ...s, label: shortSeason(s.name) }))
  const stratumOf = (r: FilterRow) => (pooled ? r.site : r.season)
  const stratumName = pooled ? 'site' : 'season'
  const unit = meta.field_units[moisture] ?? ''
  const mLabel = `${moisture}${unit ? ` (${unit})` : ''}`

  const all = useMemo(() => macPoints(siteRows, moisture), [siteRows, moisture])
  // Filters → excluded: Mark (lib/markedRows): the site's excluded filters, joined like the rows; × on the crossplots only
  const markedMet = useAddisMet(useConcRows(useMarkedRows(), viewMeta), viewMeta).rows
  const markedSite = useMemo(() => ofSite(markedMet, site), [markedMet, site])
  const marked = useMemo(() => macPoints(markedSite, moisture), [markedSite, moisture])
  const trimmed = trimBySite(all)
  const money = ols(trimmed.map((p) => p.m), trimmed.map((p) => p.mac))
  const siteLines = pooled ? strata.flatMap((s) => {
    const g = trimmed.filter((p) => p.r.site === s.name)
    return line(ols(g.map((p) => p.m), g.map((p) => p.mac)), s.color, `OLS ${s.label}`)
  }) : [...line(money, INK.fit, 'OLS'), ...(money ? [...line({ slope: money.slope + 1.96 * money.seSlope, intercept: money.intercept }, INK.muted, '+1.96 SE', '4 3'),
    ...line({ slope: money.slope - 1.96 * money.seSlope, intercept: money.intercept }, INK.muted, '−1.96 SE', '4 3')] : [])]

  const byStratum = (value: (p: MacPoint<FilterRow>) => number): IntervalRow[] => strata.map((s) => {
    const v = trimmed.filter((p) => stratumOf(p.r) === s.name).map(value)
    return { key: s.name, label: s.label, note: `n=${v.length}`, labelColor: s.color,
      marks: [[{ mid: quantile(v, 0.5), lo: quantile(v, 0.25), hi: quantile(v, 0.75), color: s.color }]], tip: [s.label, `median ${fmt(quantile(v, 0.5), 2)}`, `IQR ${fmt(quantile(v, 0.25), 2)} – ${fmt(quantile(v, 0.75), 2)}`, `n ${v.length}`] }
  })

  // moisture groups: Fabs vs EC per group, slope = MAC, intercept = baseline (Mm⁻¹). Each site is split at its own
  // cuts and fitted alone: split across sites, Low and High would mostly be different sites. Pooled, a site keeps its
  // colour and the groups differ by line and mark.
  const groups = useMemo(() => picked.flatMap((s) => {
    const pts = pooled ? all.filter((p) => p.r.site === s) : all
    const ms = pts.map((p) => p.m)
    const cuts = splitParam === 'median' ? [quantile(ms, 0.5)] : [quantile(ms, 0.33), quantile(ms, 0.67)]
    const names = splitParam === 'median' ? ['Low', 'High'] : ['Low', 'Mid', 'High']
    const colors = splitParam === 'median' ? [LOW, HIGH] : [LOW, MID, HIGH]
    return names.map((name, i) => {
      const g = pts.filter((p) => (i === 0 || p.m > cuts[i - 1]) && (i === cuts.length || p.m <= cuts[i]))
      const fabsFit = ols(g.map((p) => p.ec), g.map((p) => p.fabs)), bcFit = ols(g.map((p) => p.ec), g.map((p) => num(p.r, HIPS_BC) ?? NaN))
      return { name: `${pooled ? `${shortSite(s)} · ` : ''}${name} ${initials(moisture)}`, color: pooled ? siteColor(meta, s) : colors[i], dash: pooled ? [undefined, '6 3', '2 3'][i] : undefined, hollow: pooled && i > 0,
        g, fabsFit, bcFit, range: [Math.min(...g.map((p) => p.m)), Math.max(...g.map((p) => p.m))], medianMac: quantile(g.map((p) => p.mac), 0.5) }
    })
  }).filter((g) => g.g.length > 0), [all, picked, pooled, meta, splitParam, moisture])

  // Steiger: does FTIR EC correlate with AOD better than HIPS does?
  const trioOf = (rs: FilterRow[]) => rs.flatMap((r) => { const e = num(r, EC), h = num(r, HIPS_BC), a = num(r, AOD); return e !== null && h !== null && a !== null && e > 0 ? [{ e, h, a }] : [] })
  const steigerOf = (rs: FilterRow[]) => {
    const trio = trioOf(rs)
    const rE = pearson(trio.map((t) => t.a), trio.map((t) => t.e)), rH = pearson(trio.map((t) => t.a), trio.map((t) => t.h)), rEH = pearson(trio.map((t) => t.e), trio.map((t) => t.h))
    return { n: trio.length, rE, rH, rEH, st: rE !== null && rH !== null && rEH !== null && trio.length > 10 ? steiger(rE, rH, rEH, trio.length) : null }
  }
  const nTrioOut = trioOf(markedSite).length
  // site by site: pooled, a correlation with AOD would mostly compare the sites' levels
  const siteTests = picked.map((s) => ({ site: s, ...steigerOf(ofSite(siteRows, s)) })).filter((t) => t.st)
  const { n: nTrio, rE, rH, rEH, st } = siteTests[0] ?? { n: 0, rE: null, rH: null, rEH: null, st: null }

  const controls = <>
    <SitePick value={site} options={sites} onChange={setSite} />
    <FieldSelect label="Moisture" value={moisture} meta={meta} onChange={setM} ratios={false} />
  </>
  if (trimmed.length < MIN_FILTERS || !money) {
    const enough = offered.find((f) => f !== moisture && macPoints(siteRows, f).length >= MIN_FILTERS)
    return <ChartFrame id="mm-empty" title="Effective MAC and moisture" provenance={NB} controls={controls}><Empty>Fewer than {MIN_FILTERS} filters at {scope} carry HIPS Fabs, FTIR EC and {moisture}.{enough ? ` ${enough} has enough: choose it under Moisture.` : ''}</Empty></ChartFrame>
  }
  return (
    <>
      <ChartFrame id="mm-money" title={`Effective MAC against ${moisture} — ${scope}`}
        subtitle={`Effective MAC = HIPS Fabs / FTIR EC per filter (m²/g), against ${moisture} over the same sampling window; trimmed to its 1st–99th percentile${pooled ? ' at each site' : ''} as the notebook does. ${pooled ? 'One OLS line per site' : 'OLS line; the dashed pair is the slope ± 1.96 SE through the same intercept'}; the grey line is the MAC set under Plot → units, ${meta.mac_value}.${meta.basis === 'mass' ? ` ${BASIS_NOTE}` : ''}`}
        provenance={`${NB} (§1 the money plot)`} controls={controls}>
        <Grid>
          <Crossplot points={trimmed.map((p) => point(p.r, p.m, p.mac, colorOf(p.r), moisture, 'effective MAC', 'Research · moisture and MAC'))} sameScale={false}
            marked={marked.map((p) => point(p.r, p.m, p.mac, colorOf(p.r), moisture, 'effective MAC', 'Research · moisture and MAC'))}
            title="Effective MAC vs moisture" xLabel={mLabel} yLabel="Effective MAC (m²/g)" fit={false}
            extraLines={[...siteLines, ...line({ slope: 0, intercept: meta.mac_value }, INK.identity, `MAC ${meta.mac_value}`, '2 3')]} />
          <div>
            <p className="subtitle" style={{ margin: '0 0 2px' }}>Effective MAC by {stratumName} (median, IQR)</p>
            <IntervalStrip blocks={[{ key: 'mac', rows: byStratum((p) => p.mac) }]} panels={[{ label: 'Effective MAC (m²/g)', reference: meta.mac_value }]} labelWidth={90} noteWidth={50} minPanelWidth={200} />
            <p className="subtitle" style={{ margin: '12px 0 2px' }}>{moisture} by {stratumName}</p>
            <IntervalStrip blocks={[{ key: 'm', rows: byStratum((p) => p.m) }]} panels={[{ label: mLabel }]} labelWidth={90} noteWidth={50} minPanelWidth={200} />
          </div>
        </Grid>
        <Key items={strata.map((s) => ({ label: s.label, color: s.color }))} />
        <ExcludedNote count={marked.length} drawn />
      </ChartFrame>

      <ChartFrame id="mm-split" title={`Does moisture move the slope (MAC) or the intercept (a baseline)? — ${scope}`}
        subtitle={`Filters split at the ${splitParam === 'median' ? 'median' : '33rd and 67th percentiles'} of ${moisture}${pooled ? ' at each site, each group fitted alone (the upper groups hollow and dashed)' : ''}; HIPS Fabs on FTIR EC per group. In Fabs the slope is the effective MAC (m²/g) and the intercept a baseline in Mm⁻¹; the grey dashed line is the MAC set under Plot → units, ${meta.mac_value}, through zero.`}
        provenance={`${NB} (§4 PW as predictor of slope vs intercept)`} controls={<Segmented label="Split" value={splitParam} options={['median', 'terciles'] as const} onChange={setSplit} />}>
        <Grid>
          <Crossplot points={groups.flatMap((g) => g.g.map((p) => point(p.r, p.ec, p.fabs, g.color, 'FTIR EC', 'Fabs', 'Research · moisture and MAC', [`${moisture} ${fmt(p.m, 2)}`], g.hollow)))} sameScale={false}
            marked={marked.map((p) => point(p.r, p.ec, p.fabs, INK.muted, 'FTIR EC', 'Fabs', 'Research · moisture and MAC', [`${moisture} ${fmt(p.m, 2)}`]))}
            title="HIPS Fabs vs FTIR EC" xLabel="FTIR EC (µg/m³)" yLabel="HIPS Fabs (Mm⁻¹)" fit={false}
            extraLines={[...groups.flatMap((g) => line(g.fabsFit, g.color, g.name, g.dash)), ...line({ slope: meta.mac_value, intercept: 0 }, INK.identity, `MAC ${meta.mac_value}`, '5 4')]} />
          <DataTable rows={groups} rowKey={(r) => r.name} columns={[
            { key: 'g', label: 'Group', value: (r) => <span style={{ color: r.color, fontWeight: 600 }}>{r.name}</span> },
            { key: 'r', label: `${moisture.split(' (')[0]} range`, num: true, value: (r) => `${fmt(r.range[0], 2)}–${fmt(r.range[1], 2)}` },
            { key: 'n', label: 'n', num: true, value: (r) => r.g.length },
            { key: 'm', label: 'Slope = MAC (m²/g)', num: true, value: (r) => fmt(r.fabsFit?.slope, 2) },
            { key: 'i', label: 'Intercept (Mm⁻¹)', num: true, value: (r) => fmt(r.fabsFit?.intercept, 2) },
            { key: 'b', label: 'BC slope / intercept', num: true, value: (r) => `${fmt(r.bcFit?.slope, 2)} / ${fmt(r.bcFit?.intercept, 2)}` },
            { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.fabsFit?.r2, 3) },
            { key: 'mm', label: 'Median MAC', num: true, value: (r) => fmt(r.medianMac, 2) },
          ]} />
        </Grid>
        <ExcludedNote count={marked.length} drawn />
      </ChartFrame>

      <ChartFrame id="mm-steiger" title={`Which tracks the column? — FTIR EC and HIPS BC against ${AOD}`}
        subtitle={`Pearson r of each with AERONET AOD at 500 nm over the same windows, and the Steiger (Williams) test for two correlations that share AOD. A significantly larger r for FTIR EC is the notebook's evidence that the thermal method follows the column loading more faithfully.${pooled ? " Tested at each site alone, since pooling would compare the sites' levels." : ''}`}
        provenance={`${NB} (§2 FTIR vs HIPS correlation gap with AOD)`} exportable={false}>
        {!st ? <Empty>Fewer than 10 filters at {scope} carry FTIR EC, HIPS BC and {AOD}.</Empty> : pooled ? <>
          <DataTable rows={siteTests} rowKey={(t) => t.site} columns={[
            { key: 's', label: 'Site', value: (t) => shortSite(t.site) },
            { key: 'e', label: 'FTIR EC vs AOD', num: true, value: (t) => fmt(t.rE, 3) },
            { key: 'h', label: 'HIPS BC vs AOD', num: true, value: (t) => fmt(t.rH, 3) },
            { key: 'd', label: 'Δr (FTIR − HIPS)', num: true, value: (t) => fmt((t.rE ?? NaN) - (t.rH ?? NaN), 3) },
            { key: 'z', label: 'Steiger z', num: true, value: (t) => fmt(t.st?.z, 3) },
            { key: 'p', label: 'two-sided p', num: true, value: (t) => (t.st ? pText(t.st.p) : '') },
            { key: 'n', label: 'n', num: true, value: (t) => t.n },
          ]} />
          <ExcludedNote count={nTrioOut} />
          {siteTests.length < picked.length && <Note>Not tested, fewer than 10 filters: {picked.filter((s) => !siteTests.some((t) => t.site === s)).map(shortSite).join(', ')}.</Note>}
        </> : <>
          <DataTable rows={[{ k: 'FTIR EC vs AOD', v: fmt(rE, 3) }, { k: 'HIPS BC vs AOD', v: fmt(rH, 3) }, { k: 'FTIR EC vs HIPS BC', v: fmt(rEH, 3) }, { k: 'Δr (FTIR − HIPS)', v: fmt((rE ?? NaN) - (rH ?? NaN), 3) },
            { k: 'Steiger z', v: fmt(st.z, 3) }, { k: 'two-sided p', v: pText(st.p) }, { k: 'n', v: String(nTrio) }]} rowKey={(r) => r.k}
            columns={[{ key: 'k', label: 'Quantity', value: (r) => r.k }, { key: 'v', label: 'Value', num: true, value: (r) => r.v }]} />
          <ExcludedNote count={nTrioOut} />
          <Note>{st.p < 0.05 ? `FTIR EC correlates significantly better with AOD than HIPS BC does (p ${pText(st.p)}).` : `The difference is not significant at p < 0.05 (p ${pText(st.p)}).`} The notebook's §3 (how much of a 24 h filter's mass lands in 6–10 am) needs minute aethalometer data, which the gallery does not hold.</Note>
        </>}
      </ChartFrame>
    </>
  )
}
