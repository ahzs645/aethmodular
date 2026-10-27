import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { FieldSelect } from '@/components/FieldSelect'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { correlationP, ols, pearsonR, quantile, steiger, type Fit } from '../common/stats'
import { DataTable, Grid, HIGH, LOW, MID, SitePick, initials, line, num, pText, ofSite, point, seasonsOf, shortSeason, useSite } from '../common/ui'

const NB = 'notebooks/analysis/meteorology/pw_mac_calibration.ipynb'
const EC = 'EC (FTIR)', FABS = 'HIPS Fabs', HIPS_BC = 'HIPS BC'
const PW = 'Precipitable water (AERONET)', AOD = 'AOD 500 nm (AERONET)'
const MOISTURE = [PW, 'Relative humidity (station)', 'Relative humidity (ERA5)', 'Dew point (ERA5)']

/**
 * Effective MAC against column and surface moisture (pw_mac_calibration): the "money plot" with its OLS and a
 * 95 % band on the slope, MAC and moisture by season, a moisture split asking whether moisture moves the
 * slope (MAC) or the intercept (a baseline), and the Steiger test of whether FTIR EC tracks AOD better
 * than HIPS does.
 */
export function MoistureMac({ rows, meta }: SectionPageProps) {
  const [site, setSite, sites] = useSite(rows, 'mm')
  const siteRows = useMemo(() => ofSite(rows, site), [rows, site])
  const [mParam, setM] = useSearchParam<string>('mm.by', PW)
  const moisture = meta.fields.includes(mParam) ? mParam : MOISTURE.find((f) => meta.fields.includes(f) && siteRows.some((r) => num(r, f) !== null)) ?? meta.fields[0]
  const [splitParam, setSplit] = useSearchParam<'median' | 'terciles'>('mm.split', 'median', ['median', 'terciles'])
  const seasons = useMemo(() => seasonsOf(siteRows, meta), [siteRows, meta])
  const colorOf = (season: string) => seasons.find((s) => s.name === season)?.color ?? INK.muted
  const unit = meta.field_units[moisture] ?? ''
  const mLabel = `${moisture}${unit ? ` (${unit})` : ''}`

  // effective MAC = HIPS Fabs / FTIR EC per filter (EC > 0), with the moisture over the same window
  const all = useMemo(() => siteRows.flatMap((r) => {
    const ec = num(r, EC), fabs = num(r, FABS), m = num(r, moisture)
    return ec !== null && fabs !== null && m !== null && ec > 0 ? [{ r, ec, fabs, m, mac: fabs / ec }] : []
  }), [siteRows, moisture])
  // the notebook trims effective MAC to its 1st–99th percentile before fitting
  const lo = quantile(all.map((p) => p.mac), 0.01), hi = quantile(all.map((p) => p.mac), 0.99)
  const trimmed = all.filter((p) => p.mac >= lo && p.mac <= hi)
  const money = ols(trimmed.map((p) => p.m), trimmed.map((p) => p.mac))
  const pMoney = money ? correlationP(money.r, money.n) : NaN

  const bySeason = (value: (p: (typeof trimmed)[number]) => number): IntervalRow[] => seasons.map((s) => {
    const v = trimmed.filter((p) => p.r.season === s.name).map(value)
    return { key: s.name, label: shortSeason(s.name), note: `n=${v.length}`, labelColor: s.color,
      marks: [[{ mid: quantile(v, 0.5), lo: quantile(v, 0.25), hi: quantile(v, 0.75), color: s.color }]], tip: [shortSeason(s.name), `median ${fmt(quantile(v, 0.5), 2)}`, `IQR ${fmt(quantile(v, 0.25), 2)} – ${fmt(quantile(v, 0.75), 2)}`, `n ${v.length}`] }
  })

  // moisture groups: Fabs vs EC per group, slope = MAC, intercept = baseline (Mm⁻¹)
  const groups = useMemo(() => {
    const ms = all.map((p) => p.m)
    const cuts = splitParam === 'median' ? [quantile(ms, 0.5)] : [quantile(ms, 0.33), quantile(ms, 0.67)]
    const names = splitParam === 'median' ? ['Low', 'High'] : ['Low', 'Mid', 'High']
    const colors = splitParam === 'median' ? [LOW, HIGH] : [LOW, MID, HIGH]
    return names.map((name, i) => {
      const g = all.filter((p) => (i === 0 || p.m > cuts[i - 1]) && (i === cuts.length || p.m <= cuts[i]))
      const fabsFit = ols(g.map((p) => p.ec), g.map((p) => p.fabs)), bcFit = ols(g.map((p) => p.ec), g.map((p) => num(p.r, HIPS_BC) ?? NaN))
      return { name: `${name} ${initials(moisture)}`, color: colors[i], g, fabsFit, bcFit, range: [Math.min(...g.map((p) => p.m)), Math.max(...g.map((p) => p.m))], medianMac: quantile(g.map((p) => p.mac), 0.5) }
    })
  }, [all, splitParam, moisture])

  // Steiger: does FTIR EC correlate with AOD better than HIPS does?
  const trio = siteRows.flatMap((r) => { const e = num(r, EC), h = num(r, HIPS_BC), a = num(r, AOD); return e !== null && h !== null && a !== null && e > 0 ? [{ e, h, a }] : [] })
  const rE = pearsonR(trio.map((t) => t.a), trio.map((t) => t.e)), rH = pearsonR(trio.map((t) => t.a), trio.map((t) => t.h)), rEH = pearsonR(trio.map((t) => t.e), trio.map((t) => t.h))
  const st = rE !== null && rH !== null && rEH !== null && trio.length > 10 ? steiger(rE, rH, rEH, trio.length) : null

  const controls = <>
    <SitePick value={site} options={sites} onChange={setSite} />
    <FieldSelect label="Moisture" value={moisture} meta={meta} onChange={setM} ratios={false} />
  </>
  if (trimmed.length < 10 || !money) return <ChartFrame id="mm-empty" title="Effective MAC and moisture" provenance={NB} controls={controls}><Empty>Fewer than 10 filters at {site} carry HIPS Fabs, FTIR EC and {moisture}.</Empty></ChartFrame>
  const macLine = (f: Fit | null) => (f ? `${fmt(f.slope, 2)} · ${moisture.split(' (')[0]} + ${fmt(f.intercept, 2)}` : '—')
  return (
    <>
      <ChartFrame id="mm-money" title={`Effective MAC against ${moisture} — ${site}`}
        subtitle={`Effective MAC = HIPS Fabs / FTIR EC per filter (m²/g), against ${moisture} over the same sampling window; trimmed to its 1st–99th percentile as the notebook does. OLS line; the dashed pair is the slope ± 1.96 SE through the same intercept; the grey line is the export's MAC of ${meta.mac_value}.`}
        provenance={`${NB} (§1 the money plot)`} controls={controls}>
        <Grid>
          <Crossplot points={trimmed.map((p) => point(p.r, p.m, p.mac, colorOf(p.r.season), moisture, 'effective MAC', 'Research · moisture and MAC'))} sameScale={false}
            title="Effective MAC vs moisture" xLabel={mLabel} yLabel="Effective MAC (m²/g)" fit={false}
            extraLines={[...line(money, INK.fit, 'OLS'), ...line({ slope: money.slope + 1.96 * money.seSlope, intercept: money.intercept }, INK.muted, '+1.96 SE', '4 3'),
              ...line({ slope: money.slope - 1.96 * money.seSlope, intercept: money.intercept }, INK.muted, '−1.96 SE', '4 3'), ...line({ slope: 0, intercept: meta.mac_value }, INK.identity, `MAC ${meta.mac_value}`, '2 3')]} />
          <div>
            <p className="subtitle" style={{ margin: '0 0 2px' }}>Effective MAC by season (median, IQR)</p>
            <IntervalStrip blocks={[{ key: 'mac', rows: bySeason((p) => p.mac) }]} panels={[{ label: 'Effective MAC (m²/g)', reference: meta.mac_value }]} labelWidth={90} noteWidth={50} minPanelWidth={200} />
            <p className="subtitle" style={{ margin: '12px 0 2px' }}>{moisture} by season</p>
            <IntervalStrip blocks={[{ key: 'm', rows: bySeason((p) => p.m) }]} panels={[{ label: mLabel }]} labelWidth={90} noteWidth={50} minPanelWidth={200} />
          </div>
        </Grid>
        <Key items={seasons.map((s) => ({ label: shortSeason(s.name), color: s.color }))} />
        <Note>MAC = {macLine(money)}; r {fmt(money.r, 3)} (p {pText(pMoney)}), R² {fmt(money.r2, 3)}, n {money.n}. The notebook used daily AERONET precipitable water on the filter's start day; here it is the mean of the retrievals inside each filter's window.</Note>
      </ChartFrame>

      <ChartFrame id="mm-split" title={`Does moisture move the slope (MAC) or the intercept (a baseline)? — ${site}`}
        subtitle={`Filters split at the ${splitParam === 'median' ? 'median' : '33rd and 67th percentiles'} of ${moisture}; HIPS Fabs on FTIR EC per group. In Fabs the slope is the effective MAC (m²/g) and the intercept a baseline in Mm⁻¹; the grey dashed line is the export's MAC of ${meta.mac_value} through zero.`}
        provenance={`${NB} (§4 PW as predictor of slope vs intercept)`} controls={<Segmented label="Split" value={splitParam} options={['median', 'terciles'] as const} onChange={setSplit} />}>
        <Grid>
          <Crossplot points={groups.flatMap((g) => g.g.map((p) => point(p.r, p.ec, p.fabs, g.color, 'FTIR EC', 'Fabs', 'Research · moisture and MAC', [`${moisture} ${fmt(p.m, 2)}`])))} sameScale={false}
            title="HIPS Fabs vs FTIR EC" xLabel="FTIR EC (µg/m³)" yLabel="HIPS Fabs (Mm⁻¹)" fit={false}
            extraLines={[...groups.flatMap((g) => line(g.fabsFit, g.color, g.name)), ...line({ slope: meta.mac_value, intercept: 0 }, INK.identity, `MAC ${meta.mac_value}`, '5 4')]} />
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
        {groups.length >= 2 && groups[0].fabsFit && groups[groups.length - 1].fabsFit && <Note>From the driest to the wettest group the slope changes by {fmt(groups[groups.length - 1].fabsFit!.slope - groups[0].fabsFit!.slope, 2)} m²/g ({fmt((100 * (groups[groups.length - 1].fabsFit!.slope - groups[0].fabsFit!.slope)) / groups[0].fabsFit!.slope, 0)} %) and the intercept by {fmt(groups[groups.length - 1].fabsFit!.intercept - groups[0].fabsFit!.intercept, 2)} Mm⁻¹. The notebook's reading: a slope change means moisture alters absorption per unit mass; an intercept change, a baseline artefact on the filter.</Note>}
      </ChartFrame>

      <ChartFrame id="mm-steiger" title={`Which tracks the column? — FTIR EC and HIPS BC against ${AOD}`}
        subtitle="Pearson r of each with AERONET AOD at 500 nm over the same windows, and the Steiger (Williams) test for two correlations that share AOD. A significantly larger r for FTIR EC is the notebook's evidence that the thermal method follows the column loading more faithfully."
        provenance={`${NB} (§2 FTIR vs HIPS correlation gap with AOD)`} exportable={false}>
        {!st ? <Empty>Fewer than 10 filters at {site} carry FTIR EC, HIPS BC and {AOD}.</Empty> : <>
          <DataTable rows={[{ k: 'FTIR EC vs AOD', v: fmt(rE, 3) }, { k: 'HIPS BC vs AOD', v: fmt(rH, 3) }, { k: 'FTIR EC vs HIPS BC', v: fmt(rEH, 3) }, { k: 'Δr (FTIR − HIPS)', v: fmt((rE ?? NaN) - (rH ?? NaN), 3) },
            { k: 'Steiger z', v: fmt(st.z, 3) }, { k: 'two-sided p', v: pText(st.p) }, { k: 'n', v: String(trio.length) }]} rowKey={(r) => r.k}
            columns={[{ key: 'k', label: 'Quantity', value: (r) => r.k }, { key: 'v', label: 'Value', num: true, value: (r) => r.v }]} />
          <Note>{st.p < 0.05 ? `FTIR EC correlates significantly better with AOD than HIPS BC does (p ${pText(st.p)}).` : `The difference is not significant at p < 0.05 (p ${pText(st.p)}).`} The notebook's §3 (how much of a 24 h filter's mass lands in 6–10 am) needs minute aethalometer data, which the gallery does not hold.</Note>
        </>}
      </ChartFrame>
    </>
  )
}
