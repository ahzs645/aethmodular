import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { Key } from '@/components/Key'
import { SweepPlot } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow } from '@/lib/types'
import type { SectionPageProps } from '../../types'
import { ols, pearsonR, quadratic, quantile, slidingWindow, median, type Fit } from '../lib/stats'
import { DataTable, Grid, SitePick, fitText, line, num, ofSite, point, sitesOf, useSite } from '../lib/ui'

const ADDIS_NB = 'notebooks/analysis/meteorology/addis_hips_two_problem_decomposition.ipynb'
const MULTI_NB = 'notebooks/analysis/meteorology/hips_two_problem_decomposition.ipynb'
const EC = 'EC (FTIR)', FABS = 'HIPS Fabs', MDL = 'HIPS MDL', AETH = 'IR BCc (aethalometer)'
const NEG = '#c0392b'

interface Pair { r: FilterRow; ec: number; fabs: number; bc: number }
/** the notebooks' pairs: both measured and positive (df[(hips_fabs > 0) & (ftir_ec > 0)]) */
const pairsOf = (rows: FilterRow[], mac: number): Pair[] => rows.flatMap((r) => {
  const ec = num(r, EC), fabs = num(r, FABS)
  return ec !== null && fabs !== null && ec > 0 && fabs > 0 ? [{ r, ec, fabs, bc: fabs / mac }] : []
})

/** the two-parameter correction: Fabs = MAC_eff · EC + baseline, so EC_corrected = (Fabs − baseline) / MAC_eff */
export function decompose(pairs: Pair[], mac: number) {
  const fabsFit = ols(pairs.map((p) => p.ec), pairs.map((p) => p.fabs))
  const bcFit = ols(pairs.map((p) => p.ec), pairs.map((p) => p.bc))
  if (!fabsFit || !bcFit) return null
  const corrected = pairs.map((p) => ({ ...p, corr: (p.fabs - fabsFit.intercept) / fabsFit.slope }))
  const nNeg = corrected.filter((p) => p.corr < 0).length
  return { fabsFit, bcFit, baseline: fabsFit.intercept, macEff: fabsFit.slope, sensitivity: fabsFit.slope / mac, corrected, nNeg,
    meanEc: d3.mean(pairs, (p) => p.ec)!, floor: d3.min(pairs, (p) => p.bc)! }
}

const fitRow = (name: string, xs: number[], ys: number[], extra: Record<string, unknown> = {}) => {
  const f = ols(xs, ys)
  return { name, n: xs.length, f, nNeg: ys.filter((v) => v < 0).length, ...extra }
}

/**
 * The HIPS–FTIR disagreement as two problems: an additive baseline (the intercept) and a compressed
 * sensitivity (the slope). Per site: both numbers, what subtracting the baseline does to the low filters,
 * how to handle the negatives it makes, whether the slope changes with loading, and whether the corrected
 * HIPS agrees better with the aethalometer.
 */
export function TwoProblem({ rows, meta }: SectionPageProps) {
  const mac = meta.mac_value || 10
  const [site, setSite, sites] = useSite(rows, 'tp')
  const siteRows = useMemo(() => ofSite(rows, site), [rows, site])
  const all = useMemo(() => sitesOf(rows).map((s) => ({ site: s, color: rows.find((r) => r.site === s)?.color ?? INK.text, d: decompose(pairsOf(ofSite(rows, s), mac), mac) })), [rows, mac])
  const pairs = useMemo(() => pairsOf(siteRows, mac), [siteRows, mac])
  const d = useMemo(() => decompose(pairs, mac), [pairs, mac])
  const [winParam, setWin] = useSearchParam<string>('tp.win', 'auto', ['auto', '20', '40', '60'])
  const size = winParam === 'auto' ? Math.max(20, Math.floor(pairs.length / 6)) : Number(winParam)
  const windows = useMemo(() => slidingWindow(pairs.map((p) => ({ x: p.ec, y: p.bc })), size, Math.max(3, Math.floor(size / 8))), [pairs, size])
  const quad = useMemo(() => quadratic(pairs.map((p) => p.ec), pairs.map((p) => p.bc)), [pairs])

  const strategies = useMemo(() => {
    if (!d) return null
    const xs = pairs.map((p) => p.ec)
    const raw = pairs.map((p) => p.bc - d.bcFit.intercept)
    const mdls = siteRows.map((r) => num(r, MDL)).filter((v): v is number => v !== null)
    const mdl = mdls.length ? median(mdls) / mac : Math.abs(d3.mean(raw.filter((v) => v < 0)) ?? 0.05) * 2
    const p10 = quantile(pairs.map((p) => p.bc), 0.1)
    const keep = raw.map((v) => v >= 0)
    const baseline = [
      fitRow('A · keep negatives', xs, raw),
      fitRow('B · censor to 0', xs, raw.map((v) => Math.max(0, v))),
      fitRow(`C · negatives to MDL / 2 (${fmt(mdl / 2, 3)})`, xs, raw.map((v) => (v < 0 ? mdl / 2 : v))),
      fitRow('D · drop negatives', xs.filter((_, i) => keep[i]), raw.filter((v) => v >= 0)),
      fitRow(`E · robust baseline (P10 = ${fmt(p10, 2)})`, xs, pairs.map((p) => p.bc - p10)),
    ]
    const corr = d.corrected.map((p) => p.corr)
    const two = [
      fitRow('A · keep negatives', xs, corr),
      fitRow('B · censor to 0', xs, corr.map((v) => Math.max(0, v))),
      fitRow('D · drop negatives', xs.filter((_, i) => corr[i] >= 0), corr.filter((v) => v >= 0)),
    ]
    return { baseline, two, mdl, mdlFromData: !!mdls.length }
  }, [d, pairs, siteRows, mac])

  const validation = useMemo(() => {
    if (!d) return null
    const v = d.corrected.flatMap((p) => { const a = num(p.r, AETH); return a !== null && a > 0 ? [{ ...p, a }] : [] })
    if (v.length < 10) return { n: v.length, rows: [], pts: v }
    const pos = v.filter((p) => p.corr >= 0)
    return { n: v.length, pts: v, rows: [
      fitRow('Original HIPS BC', v.map((p) => p.a), v.map((p) => p.bc)),
      fitRow('Corrected, keep negatives', v.map((p) => p.a), v.map((p) => p.corr)),
      fitRow('Corrected, drop negatives', pos.map((p) => p.a), pos.map((p) => p.corr)),
    ] }
  }, [d])

  const fitCols = <T extends { name: string; n: number; f: Fit | null; nNeg: number }>() => [
    { key: 'name', label: 'Strategy', value: (r: T) => r.name },
    { key: 'n', label: 'n', num: true, value: (r: T) => r.n },
    { key: 's', label: 'Slope', num: true, value: (r: T) => fmt(r.f?.slope, 3) },
    { key: 'i', label: 'Intercept', num: true, value: (r: T) => fmt(r.f?.intercept, 3) },
    { key: 'r2', label: 'R²', num: true, value: (r: T) => fmt(r.f?.r2, 3) },
    { key: 'e', label: 'RMSE', num: true, value: (r: T) => fmt(r.f?.rmse, 3) },
    { key: 'neg', label: 'Negative values', num: true, value: (r: T) => r.nNeg },
  ]
  const rWin = pearsonR(windows.map((w) => w.center), windows.map((w) => w.slope))
  const bcLabel = 'HIPS BC (µg/m³)', ecLabel = 'FTIR EC (µg/m³)'
  const color = rows.find((r) => r.site === site)?.color ?? INK.text

  return (
    <>
      <ChartFrame id="tp-sites" title="Two problems by site — baseline (intercept) and sensitivity (slope)"
        subtitle={`OLS of HIPS Fabs on FTIR EC per site: the intercept is the Fabs baseline, the slope the effective MAC. In BC units (Fabs / ${mac}) the slope is MAC_eff / ${mac}. Negatives: filters whose corrected EC, (Fabs − baseline) / MAC_eff, falls below zero. Filters with both values positive.`}
        provenance={`${MULTI_NB} (§1, §4 correction parameters)`} exportable={false}>
        <DataTable rows={all} rowKey={(r) => r.site} highlight={(r) => r.site === site} columns={[
          { key: 's', label: 'Site', value: (r) => <span style={{ color: r.color, fontWeight: 600 }}>{r.site}</span> },
          { key: 'n', label: 'n', num: true, value: (r) => r.d?.fabsFit.n ?? '—' },
          { key: 'b', label: 'Fabs baseline (Mm⁻¹)', num: true, value: (r) => fmt(r.d?.baseline, 1) },
          { key: 'm', label: 'MAC_eff (m²/g)', num: true, value: (r) => fmt(r.d?.macEff, 2) },
          { key: 'sl', label: `Slope, BC vs EC`, num: true, value: (r) => fmt(r.d?.bcFit.slope, 3) },
          { key: 'ic', label: 'Intercept (µg/m³)', num: true, value: (r) => fmt(r.d?.bcFit.intercept, 3) },
          { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.d?.bcFit.r2, 3) },
          { key: 'me', label: 'Mean EC', num: true, value: (r) => fmt(r.d?.meanEc, 2) },
          { key: 'fl', label: 'HIPS BC floor (min)', num: true, value: (r) => fmt(r.d?.floor, 2) },
          { key: 'neg', label: 'Negatives after correction', num: true, value: (r) => (r.d ? `${r.d.nNeg} (${fmt((100 * r.d.nNeg) / r.d.fabsFit.n, 0)} %)` : '—') },
        ]} />
        {all.filter((s) => s.d).length >= 3 && <Grid>
          <SweepPlot series={all.filter((s) => s.d).map((s) => ({ id: s.site, label: s.site, color: s.color, r: 6, points: [{ x: s.d!.meanEc, y: s.d!.bcFit.slope }] }))}
            xLabel="Mean FTIR EC (µg/m³)" yLabel="OLS slope (BC vs EC)" reference={1} include={[0]} title={`Slope vs loading — r = ${fmt(pearsonR(all.filter((s) => s.d).map((s) => s.d!.meanEc), all.filter((s) => s.d).map((s) => s.d!.bcFit.slope)), 2)}`} />
          <SweepPlot series={all.filter((s) => s.d).map((s) => ({ id: s.site, label: s.site, color: s.color, r: 6, points: [{ x: s.d!.floor, y: s.d!.bcFit.slope }] }))}
            xLabel="HIPS BC floor (µg/m³)" yLabel="OLS slope (BC vs EC)" reference={1} include={[0]} title={`Slope vs HIPS floor — r = ${fmt(pearsonR(all.filter((s) => s.d).map((s) => s.d!.floor), all.filter((s) => s.d).map((s) => s.d!.bcFit.slope)), 2)}`} />
        </Grid>}
      </ChartFrame>

      <ChartFrame id="tp-correction" title={`The correction at ${site} — original HIPS BC and (Fabs − baseline) / MAC_eff against FTIR EC`}
        subtitle="Left: HIPS BC with its OLS line (both problems). Right: the two-parameter corrected EC; red hollow points went negative (their Fabs is below the fitted baseline). The dashed line is 1:1."
        provenance={`${ADDIS_NB} (§1, §5) · ${MULTI_NB} (§2, §4)`} controls={<SitePick value={site} options={sites} onChange={setSite} />}>
        {!d ? <Empty>Fewer than 3 filters at {site} carry positive HIPS Fabs and FTIR EC.</Empty> : <>
          <Grid>
            <Crossplot points={pairs.map((p) => point(p.r, p.ec, p.bc, color, 'FTIR EC', 'HIPS BC', 'Research · two problems'))} title="Original" xLabel={ecLabel} yLabel={bcLabel} fit={false}
              extraLines={[...line(d.bcFit, INK.fit, 'OLS'), ...line({ slope: 0, intercept: d.bcFit.intercept }, INK.accent, 'baseline', '2 3')]} />
            <Crossplot points={d.corrected.map((p) => point(p.r, p.ec, p.corr, p.corr < 0 ? NEG : color, 'FTIR EC', 'corrected', 'Research · two problems', [`Fabs ${fmt(p.fabs, 1)} Mm⁻¹`], p.corr < 0))}
              title={`EC = (Fabs − ${fmt(d.baseline, 1)}) / ${fmt(d.macEff, 2)}`} xLabel={ecLabel} yLabel="Corrected HIPS EC (µg/m³)" fit={false} extraLines={line(ols(pairs.map((p) => p.ec), d.corrected.map((p) => p.corr)), INK.good, 'OLS')} />
          </Grid>
          <Key items={[{ label: site, color }, { label: 'negative after correction', color: NEG, kind: 'hollow' }, { label: 'OLS', color: INK.fit, kind: 'line' }, { label: 'baseline (intercept)', color: INK.accent, kind: 'dash' }]} />
          <Note>Baseline {fmt(d.baseline, 1)} Mm⁻¹ = {fmt(d.bcFit.intercept, 2)} µg/m³ at MAC {mac}; MAC_eff {fmt(d.macEff, 2)} m²/g, so HIPS follows {fmt(100 * d.sensitivity, 0)} % of the EC variability. {d.nNeg} of {d.corrected.length} filters ({fmt((100 * d.nNeg) / d.corrected.length, 1)} %) go negative after the correction: below detection, not negative carbon.</Note>
        </>}
      </ChartFrame>

      {strategies && <ChartFrame id="tp-negatives" title="Handling the negatives — five strategies, then after the two-parameter correction"
        subtitle={`Top: HIPS BC minus the regression intercept (${fmt(d!.bcFit.intercept, 3)} µg/m³), against FTIR EC. Bottom: the two-parameter corrected EC. ${strategies.mdlFromData ? 'MDL from the median HIPS MDL of these filters' : 'No HIPS MDL here: the notebook\'s fallback, twice the mean negative value'} (${fmt(strategies.mdl, 3)} µg/m³). The notebook's advice: keep negatives for regressions, censor or MDL/2 for reported concentrations, drop them for ratios and source work.`}
        provenance={`${ADDIS_NB} (§2–3, §5)`} exportable={false}>
        <p className="subtitle" style={{ margin: '0 0 4px' }}>Baseline subtracted (HIPS BC − intercept)</p>
        <DataTable rows={strategies.baseline} rowKey={(r) => r.name} columns={fitCols<(typeof strategies.baseline)[number]>()} />
        <p className="subtitle" style={{ margin: '14px 0 4px' }}>Two-parameter corrected, (Fabs − baseline) / MAC_eff</p>
        <DataTable rows={strategies.two} rowKey={(r) => r.name} columns={fitCols<(typeof strategies.two)[number]>()} />
      </ChartFrame>}

      {d && <ChartFrame id="tp-saturation" title={`Is the low slope saturation? — sliding-window slope and a quadratic at ${site}`}
        subtitle={`Left: OLS slope of HIPS BC on FTIR EC in windows of ${size} filters sorted by EC (step ${Math.max(3, Math.floor(size / 8))}), against the window's mean EC; a slope that falls as EC rises would be saturation. Right: linear (black) and quadratic (pink) fits.`}
        provenance={`${ADDIS_NB} (§4) · ${MULTI_NB} (§3)`} controls={<Segmented label="Window" value={winParam} options={['auto', '20', '40', '60']} onChange={setWin} optionLabel={(o) => (o === 'auto' ? `n / 6 (${Math.max(20, Math.floor(pairs.length / 6))})` : o)} />}>
        <Grid>
          {windows.length >= 2 ? <SweepPlot series={[{ id: 'w', label: 'window slope', color, points: windows.map((w) => ({ x: w.center, y: w.slope, tip: [`EC ${fmt(w.min, 2)}–${fmt(w.max, 2)} (mean ${fmt(w.center, 2)})`, `slope ${fmt(w.slope, 3)}, intercept ${fmt(w.intercept, 2)}, r ${fmt(w.r, 2)}`, `n ${w.n}`] })) }]}
            xLabel="Window mean FTIR EC (µg/m³)" yLabel="Local OLS slope" reference={1} yMarkers={[{ at: d.bcFit.slope, label: `overall ${fmt(d.bcFit.slope, 2)}`, color: INK.muted }]}
            background={({ y, iw }) => <rect x={0} width={iw} y={y(1.1)} height={Math.max(0, y(0.9) - y(1.1))} fill={INK.good} fillOpacity={0.08} />} />
            : <Empty>Too few filters for windows of {size}.</Empty>}
          <Crossplot points={pairs.map((p) => point(p.r, p.ec, p.bc, color, 'FTIR EC', 'HIPS BC', 'Research · two problems'))} title="Linear vs quadratic" xLabel={ecLabel} yLabel={bcLabel} fit={false}
            extraLines={line(d.bcFit, INK.fit, 'linear')}
            background={quad ? (x, y) => { const [x0, x1] = x.domain(); const xs = d3.range(0, 101).map((i) => x0 + ((x1 - x0) * i) / 100)
              return <path d={d3.line<number>().x((v) => x(v)).y((v) => y(quad.coef[0] + quad.coef[1] * v + quad.coef[2] * v * v))(xs) ?? ''} fill="none" stroke={INK.deming} strokeWidth={1.8} strokeDasharray="5 3" /> } : undefined} />
        </Grid>
        <Note>r(window EC, slope) = {fmt(rWin, 2)}{rWin !== null && rWin < -0.3 ? ': the slope falls with loading' : ': no clear saturation'}. Quadratic x² coefficient {fmt(quad?.coef[2], 4)}; R² linear {fmt(d.bcFit.r2, 3)}, quadratic {fmt(quad?.r2, 3)} (ΔR² {fmt((quad?.r2 ?? NaN) - d.bcFit.r2, 4)}).</Note>
      </ChartFrame>}

      {d && validation && <ChartFrame id="tp-validation" title={`Does the correction agree better with the aethalometer? — ${site}`}
        subtitle="HIPS BC and the corrected HIPS EC against the aethalometer's IR BCc over each filter's window (the Instruments section's field). If the correction is real and not fitted to FTIR alone, the corrected values should sit closer to 1:1."
        provenance={`${ADDIS_NB} (§6) · ${MULTI_NB} (§5)`}>
        {validation.n < 10 ? <Empty>Fewer than 10 filters at {site} carry the aethalometer's IR BCc.</Empty> : <>
          <Grid>
            <Crossplot points={validation.pts.map((p) => point(p.r, p.a, p.bc, color, 'IR BCc', 'HIPS BC', 'Research · two problems'))} title="Original HIPS BC" xLabel="Aethalometer IR BCc (µg/m³)" yLabel={bcLabel} fit={false} extraLines={line(validation.rows[0].f, INK.fit, 'OLS')} />
            <Crossplot points={validation.pts.map((p) => point(p.r, p.a, p.corr, p.corr < 0 ? NEG : color, 'IR BCc', 'corrected', 'Research · two problems', [], p.corr < 0))} title="Corrected HIPS EC" xLabel="Aethalometer IR BCc (µg/m³)" yLabel="Corrected HIPS EC (µg/m³)" fit={false} extraLines={line(validation.rows[1].f, INK.good, 'OLS')} />
          </Grid>
          <DataTable rows={validation.rows} rowKey={(r) => r.name} columns={fitCols<(typeof validation.rows)[number]>().slice(0, 5)} />
          <Note>Original: {fitText(validation.rows[0].f)}. Corrected: {fitText(validation.rows[1].f)}.</Note>
        </>}
      </ChartFrame>}
    </>
  )
}
