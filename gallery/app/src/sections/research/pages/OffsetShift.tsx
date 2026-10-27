import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { SweepPlot } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { deming, ols, quantile, spearman } from '../common/stats'
import { DataTable, Grid, SitePick, line, num, ofSite, point, seasonsOf, shortSeason, useSite } from '../common/ui'

const DEMING_NB = 'research/addis_fabs_ec_deming/addis_fabs_ec_deming.ipynb'
const OFFSET_NB = 'research/addis_fabs_ec_deming/addis_fabs_ec_offset_correction.ipynb'
const STORY_NB = 'research/spartan/presentation_offset_story.ipynb'
const EC = 'EC (FTIR)', FABS = 'HIPS Fabs'
const CHARCOAL = 'Charcoal share (PMF)'
const RATIOS = Array.from({ length: 80 }, (_, i) => 10 ** (-1 + (3.5 * i) / 79))   // σ_fAbs / σ_EC, 0.1 … ~316 (np.logspace(-1, 2.5, 80))
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/**
 * fAbs against FTIR EC and the shift in EC that would remove the intercept: how the fitted line moves with the
 * assumed error ratio (Deming), the additive Δ and the multiplier k, the three re-fits side by side, and whether
 * the "missing EC" this implies is constant or follows the seasons.
 */
export function OffsetShift({ rows, meta }: SectionPageProps) {
  const [site, setSite, sites] = useSite(rows, 'os')
  const siteRows = useMemo(() => ofSite(rows, site), [rows, site])
  const [sEcParam, setSEc] = useSearchParam<string>('os.sec', '0.2', ['0.1', '0.2', '0.5'])
  const [sFabsParam, setSFabs] = useSearchParam<string>('os.sfabs', '1', ['0.5', '1', '2'])
  const ratio = Number(sFabsParam) / Number(sEcParam)
  const pts = useMemo(() => siteRows.flatMap((r) => { const x = num(r, EC), y = num(r, FABS); return x !== null && y !== null ? [{ r, x, y }] : [] }), [siteRows])
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
  const fit = useMemo(() => ols(xs, ys), [pts]) // eslint-disable-line react-hooks/exhaustive-deps
  const dem = useMemo(() => deming(xs, ys, ratio * ratio), [pts, ratio]) // eslint-disable-line react-hooks/exhaustive-deps
  const sweep = useMemo(() => RATIOS.map((q) => ({ q, d: deming(xs, ys, q * q) })), [pts]) // eslint-disable-line react-hooks/exhaustive-deps
  const seasons = useMemo(() => seasonsOf(siteRows, meta), [siteRows, meta])
  const color = siteRows[0]?.color ?? INK.text

  const xbar = xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length), ybar = ys.reduce((a, b) => a + b, 0) / Math.max(1, ys.length)
  const shifts = [{ name: 'OLS', f: fit }, { name: `Deming (σ_fAbs / σ_EC = ${fmt(ratio, 1)})`, f: dem }].map(({ name, f }) => f
    ? { name, slope: f.slope, intercept: f.intercept, delta: f.intercept / f.slope, k: ybar / (f.slope * xbar) } : { name, slope: NaN, intercept: NaN, delta: NaN, k: NaN })
  const delta = shifts[0].delta, k = shifts[0].k
  const refits = [
    { name: 'Raw', transform: 'EC', f: ols(xs, ys), x: xs },
    { name: 'Additive', transform: `EC + ${fmt(delta, 2)}`, f: ols(xs.map((v) => v + delta), ys), x: xs.map((v) => v + delta) },
    { name: 'Multiplicative', transform: `EC × ${fmt(k, 2)}`, f: ols(xs.map((v) => v * k), ys), x: xs.map((v) => v * k) },
  ]

  // the implied missing EC per filter: fAbs / slope − EC (presentation_offset_story P3)
  const missing = fit ? pts.map((p) => ({ ...p, m: p.y / fit.slope - p.x })) : []
  const byMonth = MONTHS.map((label, i) => { const v = missing.filter((p) => p.r.month === i + 1).map((p) => p.m)
    return { x: label, y: v.length ? quantile(v, 0.5) : null, lo: v.length ? quantile(v, 0.25) : null, hi: v.length ? quantile(v, 0.75) : null, tip: [`${label}: n ${v.length}`, `median ${fmt(quantile(v, 0.5), 2)} µg/m³ (IQR ${fmt(quantile(v, 0.25), 2)} to ${fmt(quantile(v, 0.75), 2)})`] } })
  const seasonFits = seasons.map((s) => { const g = pts.filter((p) => p.r.season === s.name), f = ols(g.map((p) => p.x), g.map((p) => p.y)); return { ...s, f, missing: f ? f.intercept / f.slope : NaN } })
  const charcoal = missing.flatMap((p) => { const c = num(p.r, CHARCOAL); return c !== null ? [{ c, m: p.m }] : [] })
  const rhoCharcoal = charcoal.length >= 10 ? spearman(charcoal.map((c) => c.c), charcoal.map((c) => c.m)) : null

  const controls = <>
    <SitePick value={site} options={sites} onChange={setSite} />
    <Segmented label="σ EC (µg/m³)" value={sEcParam} options={['0.1', '0.2', '0.5']} onChange={setSEc} />
    <Segmented label="σ fAbs (Mm⁻¹)" value={sFabsParam} options={['0.5', '1', '2']} onChange={setSFabs} />
  </>
  if (pts.length < 10 || !fit) return <ChartFrame id="os-empty" title="The EC shift that zeros the intercept" provenance={DEMING_NB} controls={controls}><Empty>Fewer than 10 filters at {site} carry {FABS} and {EC}.</Empty></ChartFrame>
  const ecLabel = 'FTIR EC (µg/m³)', fabsLabel = 'HIPS Fabs (Mm⁻¹)'
  return (
    <>
      <ChartFrame id="os-lambda" title={`Errors in both variables — the Deming line against the assumed error ratio, ${site}`}
        subtitle={`Deming regression of fAbs on FTIR EC with λ = (σ_fAbs / σ_EC)². Large ratios approach OLS (all error in fAbs), small ones the inverse OLS (all error in EC); 1 is orthogonal regression. The marker is the assumed σ_fAbs / σ_EC = ${sFabsParam} / ${sEcParam}.`}
        provenance={`${DEMING_NB} (steps 1–2)`} controls={controls}>
        <Grid>
          <SweepPlot series={[{ id: 's', label: 'Deming slope', color: INK.good, marks: false, points: sweep.map((s) => ({ x: s.q, y: s.d?.slope })) }]} xScale="log" xLabel="σ_fAbs / σ_EC" yLabel="Slope (Mm⁻¹ per µg/m³)"
            xMarkers={[{ at: ratio, label: 'assumed', color: INK.negative }]} yMarkers={[{ at: fit.slope, label: `OLS ${fmt(fit.slope, 2)}`, color: INK.muted }]} title="Slope" />
          <SweepPlot series={[{ id: 'i', label: 'Deming intercept', color: INK.good, marks: false, points: sweep.map((s) => ({ x: s.q, y: s.d?.intercept })) }]} xScale="log" xLabel="σ_fAbs / σ_EC" yLabel="Intercept (Mm⁻¹)" reference={0}
            xMarkers={[{ at: ratio, label: 'assumed', color: INK.negative }]} yMarkers={[{ at: fit.intercept, label: `OLS ${fmt(fit.intercept, 1)}`, color: INK.muted }]} title="Intercept" />
        </Grid>
        <Note>OLS: slope {fmt(fit.slope, 3)}, intercept {fmt(fit.intercept, 2)} Mm⁻¹, R² {fmt(fit.r2, 3)}, n {fit.n}. Deming at the assumed ratio: slope {fmt(dem?.slope, 3)}, intercept {fmt(dem?.intercept, 2)}. Orthogonal (λ = 1): slope {fmt(deming(xs, ys, 1)?.slope, 3)}, intercept {fmt(deming(xs, ys, 1)?.intercept, 2)}. Inverse OLS slope {fmt(1 / (ols(ys, xs)?.slope ?? NaN), 3)}.</Note>
      </ChartFrame>

      <ChartFrame id="os-shift" title="How far must EC move to remove the intercept? — additive Δ against a multiplier k"
        subtitle={`Holding the slope fixed: an additive shift EC + Δ with Δ = intercept / slope puts the line through zero; k = mean fAbs / (slope × mean EC) only lands the mean point on a through-origin line. Re-fitting shows why: a constant shift keeps the slope and removes the intercept; a multiplier divides the slope by k and leaves the intercept where it was.`}
        provenance={`${DEMING_NB} (step 3) · ${OFFSET_NB} · ${STORY_NB} (P2)`}>
        <DataTable rows={shifts} rowKey={(r) => r.name} columns={[
          { key: 'n', label: 'Fit', value: (r) => r.name },
          { key: 's', label: 'Slope', num: true, value: (r) => fmt(r.slope, 3) },
          { key: 'i', label: 'Intercept (Mm⁻¹)', num: true, value: (r) => fmt(r.intercept, 2) },
          { key: 'd', label: 'Additive Δ (µg/m³)', num: true, value: (r) => fmt(r.delta, 2) },
          { key: 'dm', label: 'Δ / mean EC', num: true, value: (r) => fmt(r.delta / xbar, 2) },
          { key: 'k', label: 'Multiplier k', num: true, value: (r) => fmt(r.k, 2) },
        ]} />
        <Grid min={280}>
          {refits.map((r) => <Crossplot key={r.name} points={pts.map((p, i) => point(p.r, r.x[i], p.y, color, r.transform, 'fAbs', 'Research · EC shift'))} sameScale={false}
            title={`${r.name}: ${r.transform}`} xLabel={r.name === 'Raw' ? ecLabel : `${r.transform} (µg/m³)`} yLabel={fabsLabel} fit={false} maxSide={300} extraLines={line(r.f, INK.fit, 'OLS re-fit')} />)}
        </Grid>
        <DataTable rows={refits} rowKey={(r) => r.name} columns={[
          { key: 'n', label: 'Correction', value: (r) => r.name },
          { key: 't', label: 'EC becomes', value: (r) => r.transform },
          { key: 's', label: 'Re-fit slope', num: true, value: (r) => fmt(r.f?.slope, 3) },
          { key: 'i', label: 'Re-fit intercept (Mm⁻¹)', num: true, value: (r) => fmt(r.f?.intercept, 2) },
          { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.f?.r2, 3) },
        ]} />
        <Note>Removing the offset takes a constant {fmt(delta, 2)} µg/m³ on every filter, {fmt(delta / xbar, 2)} × the mean FTIR EC ({fmt(xbar, 2)}); the data cannot tell that from subtracting {fmt(fit.intercept, 1)} Mm⁻¹ from every fAbs (same line through zero, same R²).</Note>
      </ChartFrame>

      <ChartFrame id="os-missing" title="Is the implied missing EC constant, or does it follow the seasons?"
        subtitle={`Per filter, missing EC = fAbs / ${fmt(fit.slope, 2)} − FTIR EC (the all-data slope, i.e. a MAC of ${fmt(fit.slope, 1)}). Left: median and interquartile range by month. Right: independent fits by season and the missing EC each implies (intercept / slope). A constant points to something instrument-like; one that follows the charcoal season points to FTIR missing char.`}
        provenance={`${STORY_NB} (P3)`}>
        <Grid>
          <SweepPlot series={[{ id: 'm', label: 'missing EC', color, interval: 'whisker', points: byMonth }]} xScale="point" xLabel="Month" yLabel="Implied missing EC (µg/m³)" reference={0}
            yMarkers={[{ at: missing.reduce((a, p) => a + p.m, 0) / missing.length, label: 'all-data mean', color: INK.muted }]} />
          <DataTable rows={seasonFits} rowKey={(r) => r.name} columns={[
            { key: 's', label: 'Season', value: (r) => <span style={{ color: r.color, fontWeight: 600 }}>{shortSeason(r.name)}</span> },
            { key: 'n', label: 'n', num: true, value: (r) => r.f?.n ?? '—' },
            { key: 'sl', label: 'Slope', num: true, value: (r) => fmt(r.f?.slope, 2) },
            { key: 'i', label: 'Intercept', num: true, value: (r) => fmt(r.f?.intercept, 1) },
            { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.f?.r2, 2) },
            { key: 'm', label: 'Missing EC (µg/m³)', num: true, value: (r) => fmt(r.missing, 2) },
          ]} />
        </Grid>
        {rhoCharcoal !== null && <Note>Missing EC against the PMF charcoal share (n {charcoal.length} filters with a PMF day): Spearman ρ {fmt(rhoCharcoal, 2)}.</Note>}
      </ChartFrame>
    </>
  )
}
