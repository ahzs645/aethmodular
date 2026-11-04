import { useMemo } from 'react'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { FindingsList } from '@/components/Findings'
import { useSearchParam } from '@/lib/url'
import { baseFilterId } from '@/lib/highlight'
import { f2, signed } from '@/lib/format'
import { FixedSetNote } from '@/lib/markedRows'
import { Forest, rowsOf, Scatter, sourceOf, useCounted, useXp, type Cols, type XpBase } from './shared'

/**
 * Addis FTIR EC by season against the IMPROVE TOR yardstick: is the Bega < Belg < Kiremt pattern a season or a
 * composition effect? (research/ftir_hips_chem/season_composition_20261103.ipynb)
 */

interface SeasonData extends XpBase {
  medians: Cols; tertiles: Cols; effects: Cols; composition: Cols; improve_slope: Cols; improve_bins: Cols
  leave_out: Cols; bishoftu: Cols; windows: Cols; filters: Cols
}
interface Median { model: string; season: string; n: number; vs_hips: number; vs_tor: number; vs_tor_loading_adjusted: number }
interface Effect { model: string; covariates: string; n: number; r2: number; Belg_pct: number; Belg_lo: number; Belg_hi: number; Kiremt_pct: number; Kiremt_lo: number; Kiremt_hi: number; Kiremt_p: number; coef_lrm: number | null; coef_loc: number | null }
interface Win { model: string; window: string; kiremt_minus_bega: number; kiremt_adjusted: number }

const source = sourceOf('SeasonStudy')
const SEASONS = ['Bega', 'Belg', 'Kiremt'] as const
// config.ETHIOPIA_SEASONS hues (lib/seasons ANALYSIS_SEASON_COLOR), on the local calendar's names
const SEASON_COLOR: Record<string, string> = { Bega: '#E67E22', Belg: '#27AE60', Kiremt: '#3498DB' }
const MODELS = ['published', 'raw k18', 'anchored k18', 'vibes k18', 'spline k18', 'anchored_noCH k18', 'anchored_balanced k12']
const MODEL_LABEL: Record<string, string> = {
  published: 'SPARTAN published', 'raw k18': 'raw', 'anchored k18': 'PTFE-anchored', 'vibes k18': 'VIBES 87 r70',
  'spline k18': 'spline (AIRSpec)', 'anchored_noCH k18': 'anchored, C–H masked', 'anchored_balanced k12': 'anchored, OC/EC-balanced (k 12)',
}
const label = (m: string) => MODEL_LABEL[m] ?? m
const runLine = (d: SeasonData) => `season_composition_20261103.ipynb · ${d.source} · ${d.settings?.calendar ?? ''}`
const FIXED = 'the 2 Nov transfer-diagnosis fits (k 18 on 12,393 IMPROVE filters) and the 231 Addis filters with HIPS'

export function SeasonStudy() {
  const { data, state } = useXp<SeasonData>('season_composition', 'Season or composition?')
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <BySeason d={data} />
    <Effects d={data} />
    <AddisPoints d={data} />
    <Improve d={data} />
    <Windows d={data} />
    <Checks d={data} />
  </>
}

function Findings({ d }: { d: SeasonData }) {
  const med = rowsOf<Median>(d.medians)
  const eff = rowsOf<Effect>(d.effects)
  const adj = (m: string, s: string) => med.find((r) => r.model === m && r.season === s)?.vs_tor_loading_adjusted
  const k = (m: string, c: string) => eff.find((r) => r.model === m && r.covariates === c)?.Kiremt_pct
  return (
    <ExplainOnly>
      <ChartFrame id="xs-findings" title="Season or composition — short answers" exportable={false}>
        <FindingsList note={<>{runLine(d)}. The yardstick is what IMPROVE TOR would read at the filter's HIPS loading and OC/BC (2 Nov), not HIPS/10 at slope 1.</>}>
          <li><strong>The season effect is real.</strong> After the loading trend, published FTIR EC reads {f2(adj('published', 'Bega'))} of the yardstick in Bega, {f2(adj('published', 'Belg'))} in Belg and {f2(adj('published', 'Kiremt'))} in Kiremt, with the same order for every baseline, in every loading third and within lot 251.</li>
          <li><strong>About half is the organic stand-in.</strong> IMPROVE models over-read EC as organics rise (shown with independent TOR OC). A residual-mass organic proxy that doesn't use the spectrum takes the published Kiremt effect from {signed(k('published', '+ loading'), 0)} % to {signed(k('published', '+ loading + residual mass'), 0)} %.</li>
          <li><strong>FTIR OC removes all of it, but partly circularly.</strong> It needs a coefficient 2–5× IMPROVE's, and it comes from the same spectrum as FTIR EC.</li>
          <li><strong>The remaining Kiremt excess sits in organic bands</strong> (C–H, C=O), while the continuum window 2800–2000 cm⁻¹ reads Kiremt lower. So changing the baseline alone won't remove it, and a continuum-based EC estimate should show less of it.</li>
        </FindingsList>
      </ChartFrame>
    </ExplainOnly>
  )
}

type Measure = 'vs_tor_loading_adjusted' | 'vs_tor' | 'vs_hips'
const MEASURE_LABEL: Record<Measure, string> = { vs_tor_loading_adjusted: 'vs TOR, after loading', vs_tor: 'vs TOR yardstick', vs_hips: 'vs HIPS/10' }

function BySeason({ d }: { d: SeasonData }) {
  const [measure, setMeasure] = useSearchParam<Measure>('xs-season.m', 'vs_tor_loading_adjusted', Object.keys(MEASURE_LABEL))
  const med = rowsOf<Median>(d.medians)
  const rows = MODELS.filter((m) => med.some((r) => r.model === m)).map((m) => ({
    key: m, label: label(m), bold: m === 'published',
    values: Object.fromEntries(SEASONS.map((s) => { const r = med.find((x) => x.model === m && x.season === s); return [s, r ? { v: r[measure] } : undefined] })),
    tip: [label(m), ...SEASONS.map((s) => { const r = med.find((x) => x.model === m && x.season === s); return `${s} (n ${r?.n}): ${f2(r?.[measure])}` })],
  }))
  // one axis for the three seasons, so the step between them reads across the panels
  const vals = med.filter((r) => MODELS.includes(r.model)).map((r) => r[measure]).filter(Number.isFinite)
  const domain: [number, number] = [Math.min(1, ...vals) - 0.03, Math.max(1, ...vals) + 0.03]
  return (
    <ChartFrame id="xs-season" source={source('BySeason')} provenance={runLine(d)}
      title="Addis FTIR EC by season, every model"
      subtitle="Median ratio per season. 'After loading' removes the within-Addis log-log trend with HIPS loading (Kiremt filters are heavier: median 40 µg against 31 µg in Bega). 1 = reads like TOR would."
      controls={<Segmented label="Against" value={measure} options={Object.keys(MEASURE_LABEL) as Measure[]} optionLabel={(o) => MEASURE_LABEL[o]} onChange={setMeasure} />}>
      <Forest rows={rows} labelWidth={220} panels={SEASONS.map((s) => ({ key: s, label: `${s} (median ratio)`, ref: 1, fmt: f2, domain }))} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

function Effects({ d }: { d: SeasonData }) {
  const eff = rowsOf<Effect>(d.effects)
  const models = MODELS.filter((m) => eff.some((r) => r.model === m))
  const [model, setModel] = useSearchParam<string>('xs-eff.model', 'published', models)
  const COV_NOTE: Record<string, string> = {
    'season only': 'no covariates', '+ loading': 'log HIPS loading',
    '+ loading + residual mass': 'residual mass doesn\'t use the spectrum',
    '+ loading + residual mass + nsK': 'plus non-soil K (biomass marker)', '+ loading + residual mass + nsK + lot': 'plus the filter lot',
    '+ loading + FTIR OC': 'FTIR OC comes from the same spectrum as FTIR EC',
  }
  const rows = eff.filter((r) => r.model === model).map((r) => ({
    key: r.covariates, label: r.covariates, bold: r.covariates === '+ loading + residual mass', hollow: r.covariates === '+ loading + FTIR OC',
    values: { Belg: { v: r.Belg_pct, lo: r.Belg_lo, hi: r.Belg_hi }, Kiremt: { v: r.Kiremt_pct, lo: r.Kiremt_lo, hi: r.Kiremt_hi } },
    tip: [r.covariates, COV_NOTE[r.covariates] ?? '', `Belg ${signed(r.Belg_pct, 0)} % (${signed(r.Belg_lo, 0)} to ${signed(r.Belg_hi, 0)})`,
      `Kiremt ${signed(r.Kiremt_pct, 0)} % (${signed(r.Kiremt_lo, 0)} to ${signed(r.Kiremt_hi, 0)})`, `n ${r.n}, R² ${f2(r.r2)}`,
      ...(r.coef_lrm != null ? [`residual-mass coefficient ${f2(r.coef_lrm)}`] : []), ...(r.coef_loc != null ? [`FTIR OC coefficient ${f2(r.coef_loc)}`] : [])],
  }))
  return (
    <ChartFrame id="xs-effects" source={source('Effects')} provenance={runLine(d)}
      title="How much of the season effect the organic measures absorb"
      subtitle="OLS of log(FTIR EC / TOR yardstick) on season and covariates, relative to Bega, with 95 % intervals. Residual mass (PM2.5 − ammonium sulfate − soil − BC) doesn't use the spectrum. It roughly halves the Kiremt effect with a coefficient in IMPROVE's range. FTIR OC (hollow) removes all of it but needs a coefficient 2–5× IMPROVE's."
      controls={<Segmented label="Model" value={model} options={models} optionLabel={label} onChange={setModel} />}>
      <Forest rows={rows} labelWidth={280} panels={[{ key: 'Belg', label: 'Belg vs Bega (%)', ref: 0, fmt: (v) => signed(v, 0) }, { key: 'Kiremt', label: 'Kiremt vs Bega (%)', ref: 0, fmt: (v) => signed(v, 0) }]} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

interface FilterRow { filter_id: string; date: string; season: string; lot: string; hips_ug: number; oc_over_bc: number; rm_over_bc: number; [k: string]: string | number }

function AddisPoints({ d }: { d: SeasonData }) {
  const rows = useMemo(() => rowsOf<FilterRow>(d.filters), [d])
  const models = MODELS.filter((m) => `r|${m}` in d.filters)
  const [model, setModel] = useSearchParam<string>('xs-pts.model', 'raw k18', models)
  const [xv, setXv] = useSearchParam<'rm_over_bc' | 'oc_over_bc'>('xs-pts.x', 'rm_over_bc', ['rm_over_bc', 'oc_over_bc'])
  const lg = useLegend()
  const ok = rows.filter((r) => (r[xv] as number) > 0 && (r[`r|${model}`] as number) > 0)
  const { nOut } = useCounted(ok, (r) => r.filter_id)
  const xLabel = xv === 'rm_over_bc' ? 'residual mass / HIPS BC (doesn\'t use FTIR)' : 'FTIR OC / HIPS BC (same spectrum as FTIR EC)'
  const points = ok.filter((r) => lg.show(r.season)).map((r) => ({
    x: r[xv] as number, y: r[`r|${model}`] as number, color: SEASON_COLOR[r.season], id: r.filter_id, dim: lg.dim(r.season),
    tip: [`${r.filter_id} · ${r.season} · ${r.date} · lot ${r.lot}`, `${label(model)} / TOR yardstick ${f2(r[`r|${model}`])}`,
      `residual mass / BC ${f2(r.rm_over_bc)}, FTIR OC / BC ${f2(r.oc_over_bc)}`, `HIPS ${f2(r.hips_ug)} µg`],
    rec: { id: baseFilterId(r.filter_id), source: 'Season or composition · Addis', site: 'ETAD', date: r.date,
      fields: [['season (belg_feb)', r.season], ['lot', r.lot], [`${label(model)} / TOR yardstick`, f2(r[`r|${model}`])], ['residual mass / BC', f2(r.rm_over_bc)], ['FTIR OC / BC', f2(r.oc_over_bc)]] as [string, string][] },
  }))
  return (
    <ChartFrame id="xs-points" source={source('AddisPoints')} provenance={runLine(d)}
      title="Addis per filter: FTIR EC against organic matter per BC"
      subtitle={`Each dot is an Addis filter (log axes${nOut ? `; ${nOut} left out by Filters → excluded` : ''}). Kiremt sits to the right on both axes; within a season the ratio still rises with organics. The FTIR OC panel looks tighter partly because FTIR OC and FTIR EC are read from the same spectrum.`}
      controls={<><Segmented label="Model" value={model} options={models} optionLabel={label} onChange={setModel} />
        <Segmented label="Organics" value={xv} options={['rm_over_bc', 'oc_over_bc'] as const} optionLabel={(o) => o === 'rm_over_bc' ? 'residual mass' : 'FTIR OC'} onChange={setXv} /></>}>
      <Scatter points={points} xLabel={xLabel} yLabel={`${label(model)} FTIR EC / TOR yardstick`} xLog yLog maxSide={420} xDomain={[0.5, 8]} yDomain={[0.15, 2]} nice={false} />
      <Legend {...lg.props} items={SEASONS.map((s) => ({ label: s, color: SEASON_COLOR[s] }))} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

function Improve({ d }: { d: SeasonData }) {
  const s = rowsOf<{ model: string; k: number; covariates: string; n: number; coef_loc: number; se: number }>(d.improve_slope)
  const models = [...new Set(s.map((r) => r.model))]
  const COV: Record<string, string> = { 'loc + lt': 'loading as TOR EC', 'loc + lh': 'loading as HIPS', 'loc + lh + lt': 'both' }
  const rows = models.map((m) => ({
    key: m, label: `${m} (k ${s.find((r) => r.model === m)?.k})`, bold: m === 'raw',
    values: Object.fromEntries(Object.keys(COV).map((c) => { const r = s.find((x) => x.model === m && x.covariates === c); return [c, r ? { v: r.coef_loc, lo: r.coef_loc - 1.96 * r.se, hi: r.coef_loc + 1.96 * r.se } : undefined] })),
  }))
  return (
    <ChartFrame id="xs-improve" source={source('Improve')} provenance={runLine(d)}
      title="IMPROVE: FTIR over-reads EC as organics rise (independent TOR OC)"
      subtitle="Coefficient of log TOR OC in log(out-of-site FTIR EC / TOR EC) at fixed loading, IMPROVE filters with TOR EC 2–150 µg and HIPS > 5 µg (95 % intervals). Every baseline has it; training balanced by OC/EC halves it. Addis's residual-mass coefficient (0.26–0.32) is in this range."
    >
      <Forest rows={rows} labelWidth={200} panels={Object.entries(COV).map(([k, l]) => ({ key: k, label: `per log OC, ${l}`, ref: 0, fmt: (v) => v.toFixed(2) }))} />
      <FixedSetNote what="the 2 Nov out-of-site IMPROVE predictions" />
    </ChartFrame>
  )
}

function Windows({ d }: { d: SeasonData }) {
  const w = rowsOf<Win>(d.windows)
  const [what, setWhat] = useSearchParam<'kiremt_adjusted' | 'kiremt_minus_bega'>('xs-win.v', 'kiremt_adjusted', ['kiremt_adjusted', 'kiremt_minus_bega'])
  const models = [...new Set(w.map((r) => r.model))]
  const windows = [...new Set(w.map((r) => r.window))]
  const rows = windows.map((win) => ({
    key: win, label: win, bold: win === 'TOTAL',
    values: Object.fromEntries(models.map((m) => { const r = w.find((x) => x.model === m && x.window === win); return [m, r && r[what] != null ? { v: r[what] } : undefined] })),
  }))
  return (
    <ChartFrame id="xs-windows" source={source('Windows')} provenance={runLine(d)}
      title="Which spectral windows carry the Kiremt over-read"
      subtitle="The k 18 prediction split into window shares of FTIR EC / TOR yardstick: Kiremt minus Bega, raw or after loading and residual mass. Shares add up to TOTAL. For raw and anchored, the organic bands push Kiremt up and the continuum window (2800–2000) pulls it down. The O–H window adds little. Indicative only: neighbouring PLS coefficients offset each other."
      controls={<Segmented label="Read" value={what} options={['kiremt_adjusted', 'kiremt_minus_bega'] as const} optionLabel={(o) => o === 'kiremt_adjusted' ? 'after loading + residual mass' : 'raw difference'} onChange={setWhat} />}>
      <Forest rows={rows} labelWidth={250} panels={models.map((m) => ({ key: m, label: `${m} k18`, ref: 0, fmt: (v) => signed(v, 3) }))} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

function Checks({ d }: { d: SeasonData }) {
  const lo = rowsOf<{ model: string; proxy: string; held_out: string; n: number; observed: number; predicted: number }>(d.leave_out).filter((r) => ['published', 'raw k18', 'anchored k18'].includes(r.model))
  const bi = rowsOf<{ model: string; proxy: string; n: number; months: string; observed: number; predicted_from_addis: number; addis_bega_observed: number }>(d.bishoftu).filter((r) => r.proxy === 'lrm')
  const comp = rowsOf<{ season: string; n: number; hips_ug: number; oc_over_bc: number; rm_over_bc: number; pm25_ugm3: number; so4_ugm3: number; nsk_ugm3: number }>(d.composition)
  return (
    <ChartFrame id="xs-checks" source={source('Checks')} title="Composition by season, leave one season out, and Bishoftu" exportable={false}>
      <table className="census-table">
        <thead><tr><th>Season</th><th>Filters</th><th>HIPS (µg)</th><th>FTIR OC / BC</th><th>Residual mass / BC</th><th>PM2.5 (µg/m³)</th><th>SO4 (µg/m³)</th><th>nsK (µg/m³)</th></tr></thead>
        <tbody>{comp.map((r) => <tr key={r.season}><td>{r.season}</td><td>{r.n}</td><td>{f2(r.hips_ug)}</td><td>{f2(r.oc_over_bc)}</td><td>{f2(r.rm_over_bc)}</td><td>{f2(r.pm25_ugm3)}</td><td>{f2(r.so4_ugm3)}</td><td>{f2(r.nsk_ugm3)}</td></tr>)}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Model</th><th>Organic proxy</th><th>Held-out season</th><th>Observed</th><th>Predicted from the other two</th></tr></thead>
        <tbody>{lo.map((r) => <tr key={`${r.model}${r.proxy}${r.held_out}`}><td>{label(r.model)}</td><td>{r.proxy === 'lrm' ? 'residual mass' : 'FTIR OC'}</td><td>{r.held_out}</td><td>{f2(r.observed)}</td><td>{f2(r.predicted)}</td></tr>)}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Model</th><th>Bishoftu filters</th><th>Months</th><th>Observed</th><th>Predicted from Addis</th><th>Addis Bega observed</th></tr></thead>
        <tbody>{bi.map((r) => <tr key={r.model}><td>{label(r.model)}</td><td>{r.n}</td><td>{r.months}</td><td>{f2(r.observed)}</td><td>{f2(r.predicted_from_addis)}</td><td>{f2(r.addis_bega_observed)}</td></tr>)}</tbody>
      </table>
      <Note>Leave-one-season-out fits log ratio ~ log HIPS + organic proxy on two seasons and predicts the third. Bega comes out lower and Kiremt higher than predicted, matching the remaining season effect. Bishoftu has only Bega filters (Nov–Dec 2025), so the season question can't be tested there. The Addis relation over-predicts it for raw, anchored and published, so Bishoftu under-reads beyond what its composition implies.</Note>
    </ChartFrame>
  )
}
