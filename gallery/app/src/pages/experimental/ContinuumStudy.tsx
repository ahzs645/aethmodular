import { useMemo } from 'react'
import { ChartFrame, ExplainOnly, Note, Segmented, Select } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { FindingsList } from '@/components/Findings'
import { BarRows } from '@/components/BarRows'
import { closeoutReview } from '@/lib/closeoutReview'
import { saveSection } from '@/lib/sections'
import { epoSourceVerified, medianAbsoluteError, sourceScoreValue, sweepMedianAbsoluteError, type SourceScore } from '@/lib/sourceEvaluation'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { SpectrumPlot, type SpectrumSeries } from '@/components/SpectrumPlot'
import { Crossplot, CrossplotBrushControls, useCrossplotBrush, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { useSearchParam } from '@/lib/url'
import { INK } from '@/lib/theme'
import { coef, coefSigned, f2 } from '@/lib/format'
import { demingLambda, useMac } from '@/lib/mac'
import { baseFilterId } from '@/lib/highlight'
import { ExcludedNote, FixedSetNote } from '@/lib/markedRows'
import { Forest, num, rowsOf, sourceOf, useCounted, useXp, type Cols, type XpBase } from './shared'

/**
 * Why FTIR EC does not transfer to Addis, and what a continuum-preserving baseline changes: before and after
 * crossplots for Addis and for IMPROVE's heavy filters, the matched-loading curves, the broad continuum each
 * baseline keeps, every SPARTAN site against the TOR yardstick and the window split (2–3 Nov 2026,
 * research/ftir_hips_chem/transfer_diagnosis_20261102.ipynb; Colab VIBES rerun workflows/vibes_continuum_20261102.py).
 */

interface ContinuumData extends XpBase {
  addis_filters: Cols; addis_fits: Cols; improve_filters: Cols; mae: Cols; curves: Cols; continuum: Cols
  noise: Cols; injection: Cols; sites: Cols; windows: Cols; sweep: Cols; gain: Cols; aeth_sites: Cols
  hips_volume: Cols; hips_sat: Cols; hips_sites: Cols; hips_grid: Cols
  hips_stress: Cols; hips_by_site: Cols; tau_points: Cols; tau_holdout: Cols; aeth_bridge: Cols; true_over_hips: Cols; season_aeth: Cols
  aeth_flow: Cols; aeth_flow_coef: Cols; aeth_fix: Cols; aeth_spots: Cols; tau_curves: Cols; response_fit: Cols; response_addis: Cols; response_sites: Cols; adama: Cols; adama_summary: Cols
  rule_sim: Cols; scoreboard: Cols; spectra_story: Cols; lot_vs_aeth: Cols
}
interface Fit { model: string; x: XKey; n: number; slope: number; intercept: number; r2: number; ratio: number }

const source = sourceOf('ContinuumStudy')
const SEASONS = ['Bega', 'Belg', 'Kiremt'] as const
const SEASON_COLOR: Record<string, string> = { Bega: '#E67E22', Belg: '#27AE60', Kiremt: '#3498DB' }
const MODEL_LABEL: Record<string, string> = {
  published: 'SPARTAN published (raw, deployed)', 'raw k18': 'raw', 'spline k18': 'spline (AIRSpec)', 'vibes k18': 'VIBES 87 r70 (production)',
  'anchored k18': 'PTFE-anchored (linear)', 'vibes_anchorhp k18': 'VIBES on PTFE-anchored', 'vibes_deflate k18': 'VIBES, smooth shapes removed',
  'anchored_noCH k18': 'anchored, C–H masked', 'anchored_balanced k12': 'anchored, OC/EC-balanced (k 12)',
  'epo k18': 'anchored + EPO (OC, other PM, Fe, K removed)', 'dipls_g10 k18': 'anchored di-PLS, Addis target (γ 10)',
  'dipls_net k18': 'anchored di-PLS, network target (γ 3)', 'mask_org k18': 'anchored, O–H/C–H and 1800–1500 masked',
  'cont_only k10': 'anchored, continuum windows only (k 10)', 'iw k18': 'anchored, importance-weighted to Addis',
}
const SHORT: Record<string, string> = {
  published: 'published', 'raw k18': 'raw', 'spline k18': 'spline', 'vibes k18': 'VIBES (before)', 'anchored k18': 'anchored',
  'vibes_anchorhp k18': 'VIBES anchored (after)', 'vibes_deflate k18': 'VIBES deflated', 'anchored_noCH k18': 'anchored −C–H',
  'anchored_balanced k12': 'anchored balanced', 'epo k18': 'EPO', 'dipls_g10 k18': 'di-PLS (Addis)', 'dipls_net k18': 'di-PLS (network)',
  'mask_org k18': 'organic bands masked', 'cont_only k10': 'continuum only', 'iw k18': 'importance-weighted',
}
const MODEL_COLOR: Record<string, string> = {
  get published() { return INK.text }, 'raw k18': '#1f77b4', 'spline k18': '#9467bd', 'vibes k18': '#d62728', 'anchored k18': '#2ca02c',
  'vibes_anchorhp k18': '#8c564b', 'vibes_deflate k18': '#ff7f0e', 'anchored_noCH k18': '#17becf', 'anchored_balanced k12': '#bcbd22',
  'epo k18': '#e377c2', 'dipls_g10 k18': '#1f9e89', 'dipls_net k18': '#6a51a3', 'mask_org k18': '#17becf', 'cont_only k10': '#9467bd', 'iw k18': '#bcbd22',
}
const label = (m: string) => MODEL_LABEL[m] ?? m
const BEFORE = 'vibes k18', AFTER = 'vibes_anchorhp k18'
/** the curve/site tables name models without the k suffix */
const bare = (m: string) => m.replace(/ k\d+$/, '')
const runLine = (d: ContinuumData) => `transfer_diagnosis_20261102.ipynb · ${d.source} · IMPROVE out of site (5-fold by site), k 18 · Colab VIBES rerun ${d.settings?.colab?.fits ?? ''} fits, ${d.settings?.colab?.failed ?? '?'} failed`
const FIXED = 'the 233 Addis filters of the 22 Oct decision bundle, fitted once (k 18) on 12,393 IMPROVE filters'

export function ContinuumStudy() {
  const { data, state } = useXp<ContinuumData>('continuum_baseline', 'Continuum and the baseline', [1, 2])
  if (!data) return <>{state}</>
  return <>
    <CloseoutStart d={data} />
    <Coverage d={data} />
    <CandidatePerformance d={data} />
    <Findings d={data} />
    <WhereNext d={data} />
    <PassRuleProposal d={data} />
    <WhyAnchored d={data} />
    <AddisBeforeAfter d={data} />
    <AddisFits d={data} />
    <AethFlow d={data} />
    <AethCheck d={data} />
    <HipsNetwork d={data} />
    <TorTest d={data} />
    <SeasonAeth d={data} />
    <InterceptSweep d={data} />
    <WindowGain d={data} />
    <HipsCurve d={data} />
    <HipsSaturation d={data} />
    <ImproveBeforeAfter d={data} />
    <MatchedLoading d={data} />
    <Continuum d={data} />
    <Sites d={data} />
    <Windows d={data} />
  </>
}

function CloseoutStart({ d }: { d: ContinuumData }) {
  const verified = epoSourceVerified(d.settings?.source_evaluation)
  const openReview = () => {
    const section = saveSection(closeoutReview(d.generated, verified ? 'corrected retrospective evaluation' : 'EPO rerun pending'))
    window.location.hash = `tab=custom&sec=${section.id}`
  }
  return <ChartFrame id="xc-closeout" pageOnly exportable={false} title="Existing-data closeout review"
    evidence={verified ? 'Corrected source evaluation attached. Target absolute EC accuracy remains unresolved.' : 'EPO source evaluation pending: historical scores are withheld because preprocessing used held-out chemical labels.'}>
    <p data-export="text">The existing data support conclusions about baseline behavior, numerical convergence, seasonality and optical-response sensitivity. They do not yet select one independently validated Addis EC product.</p>
    <button type="button" className="btn" onClick={openReview}>Build six-figure review section</button>
    <Note>Uses the existing section editor, Copy link and PDF/Deck exports. The section records this export’s date; a scientific release also needs its frozen source/data receipt.</Note>
  </ChartFrame>
}

function Coverage({ d }: { d: ContinuumData }) {
  const all = useMemo(() => rowsOf(d.addis_filters), [d])
  const { counted, nOut } = useCounted(all, (r) => String(r.filter_id))
  const unique = (rows: typeof all) => new Set(rows.map((r) => baseFilterId(String(r.filter_id)))).size
  const total = unique(all), included = unique(counted)
  const rows = SEASONS.map((season) => ({ key: season, label: season, color: SEASON_COLOR[season], value: unique(counted.filter((r) => r.season === season)) }))
  const hips = unique(counted.filter((r) => Number.isFinite(num(r.hips_bc))))
  const aeth = unique(counted.filter((r) => Number.isFinite(num(r.aeth_ir))))
  return <ChartFrame id="xc-coverage" title="Physical filters in this comparison" source={source('Coverage')}
    evidence={`${included} counted physical Addis filters of ${total} in this export · ${all.length} rows · calendar belg_feb · generated ${d.generated}`}>
    <BarRows rows={rows} axis={{ label: 'Independent physical filters' }} format={(n) => String(n)} />
    <Note>{hips} counted filters have HIPS-derived optical BC (MAC 10); {aeth} have an aethalometer comparison. {nOut} exported rows are outside the current inclusion rules. Repeated scans do not add independent filters. The full scan/exclusion ledger belongs to the release bundle; this is the exported cohort, not the separate 232-filter seasonal study.</Note>
  </ChartFrame>
}

function CandidatePerformance({ d }: { d: ContinuumData }) {
  const evaluation = d.settings?.source_evaluation
  const scores = rowsOf<SourceScore>(d.scoreboard)
  const metrics = [
    ['medae', 'Median absolute error'], ['improve_mae', 'Mean absolute error'],
    ['improve_rmse', 'RMSE'], ['improve_mean_bias', 'Mean signed bias'], ['improve_p95_absolute_error', '95th-percentile absolute error'],
  ] as const
  const value = (r: SourceScore, key: string) => key === 'medae' ? medianAbsoluteError(r, evaluation) : sourceScoreValue(r, key, evaluation)
  const present = metrics.filter(([key]) => scores.some((r) => value(r, key) != null))
  const rows = scores.map((r) => ({ key: r.model, label: `${label(`${r.model} k18`)}${r.model === 'epo' && !epoSourceVerified(evaluation) ? ' · withheld' : ''}`,
    color: MODEL_COLOR[`${r.model} k18`], values: Object.fromEntries(present.map(([key]) => [key, { v: value(r, key) ?? NaN }])) }))
  return <ChartFrame id="xc-source" title="Source-reference performance on IMPROVE" source={source('CandidatePerformance')}
    evidence={`TOR EC reference · µg/filter · site-grouped cross-validation · fixed 18 components · finite TOR below 150 µg/filter · ${epoSourceVerified(evaluation) ? 'corrected retrospective comparison' : 'saved exploratory comparison; EPO withheld'}`}>
    <Forest rows={rows} panels={present.map(([key, title]) => ({ key, label: `${title} (µg/filter)`, fmt: f2 }))} labelWidth={210} />
    <div className="closeout-table-scroll"><table className="census-table" data-export="text">
      <thead><tr><th>Candidate</th><th>Eligible</th><th>Predicted</th><th>Common filters</th><th>Coverage</th><th>Evidence</th></tr></thead>
      <tbody>{scores.map((r) => <tr key={r.model}><td>{label(`${r.model} k18`)}</td><td>{num2(r.n_eligible, 0)}</td><td>{num2(r.n_available, 0)}</td><td>{num2(r.n_common, 0)}</td><td>{r.coverage == null ? 'Not recorded' : `${num2(num(r.coverage) * 100, 1)} %`}</td><td>{r.evaluation_status ?? (r.model === 'epo' ? 'Withheld: fold-safe rerun required' : 'Historical saved result')}</td></tr>)}</tbody>
    </table></div>
    <Note>Median absolute error = median |prediction − TOR|. Signed bias uses prediction − TOR. Missing metrics and counts are unavailable, not zero. A common-filter score must be read with coverage; failed predictions remain in the availability ledger. This source-domain comparison does not measure Addis accuracy.</Note>
  </ChartFrame>
}

function Findings({ d }: { d: ContinuumData }) {
  const inj = rowsOf<{ variant: string; shape: string; value: number }>(d.injection)
  const kept = inj.find((r) => r.variant === 'production (rank 70)' && r.shape === 'nu^1 (BC-like)')?.value
  return <ExplainOnly><ChartFrame id="xc-findings" title="Continuum and the baseline — short answers" exportable={false}>
    <FindingsList note={<>{runLine(d)}. Optical comparisons and transferred-response models do not provide independent Addis TOR.</>}>
      <li><strong>Baseline choice changes broad spectral preservation.</strong> The saved production-VIBES experiment recovers {kept != null ? `${(kept * 100).toFixed(1)} %` : 'under 1 %'} of its injected power-law shape. This is a synthetic-shape response, not a chemical identification.</li>
      <li><strong>PTFE anchoring remains a research candidate.</strong> It changes the Addis optical relationship and the saved source-reference results. The small Adama comparison and possible lot dependence prevent a general accuracy claim.</li>
      <li><strong>EPO requires a verified fold-safe source evaluation.</strong> The historical projection used all eligible source chemical labels before the held-out site split. Its original score cannot select a preferred method. Full-source target predictions are a separate exploratory product.</li>
      <li><strong>Seasonal structure remains important.</strong> Preserve Bega, Belg and Kiremt, the February convention and each comparator. Repetition against another optical instrument does not identify the mechanism or validate seasonal correction factors.</li>
      <li><strong>Numerical convergence and absolute EC accuracy are separate decisions.</strong> Preserve both baseline versions and their downstream scope. The existing-data stage can close with absolute Addis EC accuracy explicitly unresolved.</li>
    </FindingsList>
  </ChartFrame></ExplainOnly>
}

type XKey = 'hips_bc' | 'hips_lin' | 'tor_equiv' | 'aeth_ir'
const XKEYS = ['hips_bc', 'hips_lin', 'tor_equiv', 'aeth_ir'] as const
const X_SHORT: Record<XKey, string> = { hips_bc: 'HIPS BC', hips_lin: 'HIPS, transferred response', tor_equiv: 'inferred TOR comparator', aeth_ir: 'aethalometer' }
const X_TITLE: Record<XKey, string> = { hips_bc: 'HIPS BC', hips_lin: 'HIPS under the transferred IMPROVE response', tor_equiv: 'the inferred TOR comparator', aeth_ir: 'the aethalometer' }
/** HIPS on the MAC scale: the raw and the linearised (saturation-corrected, rescaled to the raw Addis mean) */
const onMac = (xk: XKey) => xk === 'hips_bc' || xk === 'hips_lin'
const AETH_LABEL = 'Aethalometer IR BCc (µg/m³, MA350, same 24 h)'
type View = 'pair' | 'all'

function AddisBeforeAfter({ d }: { d: ContinuumData }) {
  const mac = useMac()
  const models: string[] = d.settings?.models ?? []
  const [view, setView] = useSearchParam<View>('xc-addis.view', 'pair', ['pair', 'all'])
  const pickable = [...models, ...(d.settings?.fix_models ?? [])]
  const [before, setBefore] = useSearchParam<string>('xc-addis.before', BEFORE, pickable)
  const [after, setAfter] = useSearchParam<string>('xc-addis.after', AFTER, pickable)
  const [xk, setXk] = useSearchParam<XKey>('xc-addis.x', 'hips_bc', XKEYS)
  const [brush, setBrush] = useCrossplotBrush()
  const lg = useLegend()
  const all = useMemo(() => rowsOf(d.addis_filters).map((r): Record<string, any> => ({ ...r, hips_bc: mac.bc(num(r.hips_bc)), hips_lin: mac.bc(num(r.hips_lin)) })), [d, mac])
  const { counted, marked, nOut } = useCounted(all, (r) => String(r.filter_id))
  const xLabel = xk === 'hips_bc' ? `HIPS BC (µg/m³, ${mac.label})` : xk === 'hips_lin' ? `HIPS on the IMPROVE TOR response (µg/m³, ${mac.label})` : xk === 'aeth_ir' ? AETH_LABEL : 'Inferred comparator (µg/m³): HIPS × IMPROVE TOR/HIPS'
  const lambda = xk === 'hips_bc' ? demingLambda(mac.mac, num(d.settings?.LAMBDA_ADDIS) || 2.96) : xk === 'hips_lin' ? demingLambda(mac.mac, num(d.settings?.LAMBDA_HIPS_LIN) || 0.1) : xk === 'aeth_ir' ? num(d.settings?.LAMBDA_AETH) || 1 : num(d.settings?.LAMBDA_ADDIS) || 2.96
  const pointsOf = (m: string, rows: Record<string, any>[]): CrossPoint[] => rows
    .filter((r) => Number.isFinite(num(r[xk])) && Number.isFinite(num(r[m])) && lg.show(String(r.season)))
    .map((r) => ({
      x: num(r[xk]), y: num(r[m]), color: SEASON_COLOR[r.season] ?? INK.muted, id: baseFilterId(String(r.filter_id)), season: String(r.season), seasonColor: SEASON_COLOR[r.season],
      opacity: lg.dim(String(r.season)),
      tip: [`${r.filter_id} · ${r.season} · ${r.date ?? 'no date'} · lot ${r.lot ?? '?'}`, `${label(m)}: ${f2(r[m])} µg/m³`, `HIPS BC ${f2(r.hips_bc)} (${mac.label}), TOR-equivalent ${f2(r.tor_equiv)}, aethalometer ${f2(r.aeth_ir)}`],
      rec: { id: baseFilterId(String(r.filter_id)), source: 'Continuum and the baseline · Addis', site: 'ETAD', date: r.date ?? undefined, season: String(r.season),
        fields: [[`FTIR EC, ${SHORT[m]}`, f2(r[m])], ['HIPS BC', f2(r.hips_bc)], ['TOR-equivalent', f2(r.tor_equiv)], ['Aethalometer IR BCc', f2(r.aeth_ir)], ['IMPROVE TOR/HIPS here', f2(r.exp_tor_hips)]] as [string, string][] },
    }))
  const panels = (view === 'pair' ? [before, after] : models).map((m, i) => ({
    key: m, title: view === 'pair' ? `${i === 0 ? 'Before' : 'After'}: ${label(m)}` : label(m), points: pointsOf(m, counted), marked: pointsOf(m, marked),
  }))
  return (
    <ChartFrame id="xc-addis" source={source('AddisBeforeAfter')} provenance={runLine(d)}
      title={`Addis before and after — FTIR EC against ${X_TITLE[xk]}`}
      evidence={`Exploratory target comparison · ${new Set(counted.map((r) => baseFilterId(String(r.filter_id)))).size} counted physical filters; plotted pairs depend on available values · belg_feb · ${xk === 'hips_bc' ? mac.label : xk === 'aeth_ir' ? 'MA350 IR BCc' : 'transferred model, not measured Addis TOR'}`}
      subtitle={`Each dot is an Addis filter, coloured by season (belg_feb). Every panel is on the same range with the dashed 1:1 line and a Deming fit (λ ${f2(lambda)}). Before is the production VIBES background, after is VIBES rerun on Colab on the PTFE-anchored spectra; pick any two models (the pickers also hold the 3 Nov intercept fixes), or show every baseline. Switch the x axis to the TOR yardstick to see what a TOR-trained model can be expected to reach, or to the aethalometer, a separate instrument, to see which one follows Addis's day-to-day BC: there the slope carries the aethalometer's own calibration, so read the intercept and the straightness, not the slope. HIPS, TOR response corrects HIPS with a curve fitted on IMPROVE TOR alone, without Addis, the aethalometer or FTIR; read its intercept, because its level assumes MAC 10 holds at Addis.`}
      controls={<>
        <Segmented label="View" value={view} options={['pair', 'all'] as const} optionLabel={(o) => o === 'pair' ? 'Before / after' : 'Every baseline'} onChange={setView} />
        {view === 'pair' && <>
          <Select label="Before" value={before} options={pickable} optionLabel={(o: string) => SHORT[o] ?? o} onChange={setBefore} />
          <Select label="After" value={after} options={pickable} optionLabel={(o: string) => SHORT[o] ?? o} onChange={setAfter} />
        </>}
        <Segmented label="Against" value={xk} options={XKEYS} optionLabel={(o) => X_SHORT[o]} onChange={setXk} />
        <CrossplotBrushControls brush={[brush, setBrush]} />
      </>}>
      <FacetGrid items={panels} minCell={300} maxCols={view === 'pair' ? 2 : 3} itemKey={(p) => p.key} share render={(p, w) =>
        <Crossplot points={p.points} marked={p.marked} title={p.title} xLabel={xLabel} yLabel="FTIR EC (µg/m³)" brushable brush={brush} maxSide={Math.min(w - 70, 460)} errorRatio={lambda} />} />
      <Legend {...lg.props} items={SEASONS.map((s) => ({ label: s, color: SEASON_COLOR[s] }))} />
      <ExcludedNote count={nOut} drawn />
    </ChartFrame>
  )
}

function AddisFits({ d }: { d: ContinuumData }) {
  const m = useMac()
  const [xk, setXk] = useSearchParam<XKey>('xc-fits.x', 'hips_bc', XKEYS)
  const fits = rowsOf<Fit>(d.addis_fits).filter((r) => r.x === xk)
  const fixes = new Set<string>(d.settings?.fix_models ?? [])
  // saved at MAC 10: against HIPS, slope and ratio scale by MAC/10 and the intercept stays (lib/mac); the TOR yardstick has no MAC
  const sc = (v: number) => (onMac(xk) ? (m.slope(v) as number) : v)
  const rows = fits.map((r) => ({
    key: r.model, label: r.model === 'HIPS BC' ? 'HIPS BC itself (MAC 10)' : `${fixes.has(r.model) ? '3 Nov fix: ' : ''}${label(r.model)}`,
    color: r.model === 'HIPS BC' ? INK.text : MODEL_COLOR[r.model], bold: r.model === AFTER || r.model === BEFORE || r.model === 'HIPS BC', hollow: fixes.has(r.model),
    values: { intercept: { v: r.intercept }, slope: { v: sc(r.slope) }, ratio: { v: sc(r.ratio) } },
    tip: [label(r.model), `Deming ${coef(sc(r.slope))}x ${coefSigned(r.intercept)} µg/m³, R² ${f2(r.r2)}`, `median ÷ ${X_SHORT[xk]} ${f2(sc(r.ratio))}`, `n ${r.n}`],
  }))
  return (
    <ChartFrame id="xc-fits" source={source('AddisFits')} provenance={runLine(d)}
      title="Addis fits for every baseline"
      subtitle={`Deming fit of each model's Addis FTIR EC against ${xk === 'hips_bc' ? `HIPS BC (${m.label}; another MAC scales the slope and ratio, not the intercept)` : xk === 'hips_lin' ? `HIPS on the IMPROVE TOR response (${m.label}; read the intercept)` : xk === 'aeth_ir' ? 'the aethalometer IR BCc (λ 1; the slope and ratio carry the aethalometer\'s calibration, so only the intercept should be near its ideal 0; HIPS itself is the bold black row)' : 'the TOR yardstick (no MAC)'}, saved by the exporter on the filters that count. A model on the line has intercept 0, slope 1 and ratio 1. Hollow dots are the 3 Nov label-free intercept fixes.`}
      controls={<Segmented label="Against" value={xk} options={XKEYS} optionLabel={(o) => X_SHORT[o]} onChange={setXk} />}>
      <Forest rows={rows} labelWidth={340} panels={[{ key: 'intercept', label: 'Deming intercept (µg/m³)', ref: 0, fmt: coefSigned }, { key: 'slope', label: 'Deming slope', ref: 1, fmt: coef }, { key: 'ratio', label: 'median ratio', ref: 1, fmt: f2 }]} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

function ImproveBeforeAfter({ d }: { d: ContinuumData }) {
  const models: string[] = (d.settings?.models ?? []).filter((m: string) => m !== 'published' && m in d.improve_filters)
  const [before, setBefore] = useSearchParam<string>('xc-imp.before', BEFORE, models)
  const [after, setAfter] = useSearchParam<string>('xc-imp.after', AFTER, models)
  const [only, setOnly] = useSearchParam<'low' | 'all'>('xc-imp.set', 'low', ['low', 'all'])
  const lg = useLegend()
  const rows = useMemo(() => rowsOf(d.improve_filters), [d])
  const groupOf = (r: Record<string, any>) => (num(r.ocec) < 2.5 ? 'OC/EC < 2.5' : 'OC/EC ≥ 2.5')
  const GROUP_COLOR: Record<string, string> = { 'OC/EC < 2.5': '#d35400', 'OC/EC ≥ 2.5': '#8a8f98' }
  const pts = (m: string): CrossPoint[] => rows
    .filter((r) => (only === 'all' || num(r.ocec) < 2.5) && num(r.tor_ug) < 150 && Number.isFinite(num(r[m])) && lg.show(groupOf(r)))
    .map((r) => ({ x: num(r.tor_ug), y: num(r[m]), color: GROUP_COLOR[groupOf(r)], opacity: 0.7 * lg.dim(groupOf(r)), site: groupOf(r),
      tip: [`IMPROVE ${r.filter_id} · ${r.site} · ${r.date}`, `out-of-site ${SHORT[m]}: ${f2(r[m])} µg`, `TOR EC ${f2(r.tor_ug)} µg, HIPS ${f2(r.hips_ug)} µg, OC/EC ${f2(r.ocec)}`] }))
  const panels = [before, after].map((m, i) => ({ key: m, title: `${i === 0 ? 'Before' : 'After'}: ${label(m)}`, points: pts(m) }))
  const n = panels[0]?.points.length ?? 0
  return (
    <ChartFrame id="xc-improve" source={source('ImproveBeforeAfter')} provenance={runLine(d)}
      title="IMPROVE before and after — held-out heavy filters against TOR"
      subtitle={`IMPROVE filters with HIPS above 10 µg per filter (${n} drawn), each predicted by a model that never saw its site (5-fold by site), in µg per filter against TOR EC. The orange low-OC/EC filters are the ones that look most like Addis. A baseline change that helps Addis should not hurt these.`}
      controls={<>
        <Select label="Before" value={before} options={models} optionLabel={(o: string) => SHORT[o] ?? o} onChange={setBefore} />
        <Select label="After" value={after} options={models} optionLabel={(o: string) => SHORT[o] ?? o} onChange={setAfter} />
        <Segmented label="Filters" value={only} options={['low', 'all'] as const} optionLabel={(o) => o === 'low' ? 'OC/EC < 2.5' : 'every heavy filter'} onChange={setOnly} />
      </>}>
      <FacetGrid items={panels} minCell={300} maxCols={2} itemKey={(p) => p.key} share render={(p, w) =>
        <Crossplot points={p.points} title={p.title} xLabel="TOR EC (µg per filter)" yLabel="FTIR EC, out of site (µg per filter)" maxSide={Math.min(w - 70, 460)} grouping={only === 'all' ? 'site' : 'none'} />} />
      {only === 'all' && <Legend {...lg.props} items={Object.entries(GROUP_COLOR).map(([k, c]) => ({ label: k, color: c }))} />}
      <Note>TOR above 150 µg (five wildfire filters at TOR/HIPS 6–8) is left out of the fits and the plot. IMPROVE ids don't open a gallery record.</Note>
    </ChartFrame>
  )
}

interface Curve { model: string; k: number; what: string; lo: number; hi: number; median: number; n: number }

function MatchedLoading({ d }: { d: ContinuumData }) {
  const curves = rowsOf<Curve>(d.curves)
  const models = ['raw', 'spline', 'vibes', 'anchored', 'vibes_anchorhp', 'vibes_deflate'].filter((m) => curves.some((c) => c.model === m))
  const [picked, setPicked] = useSearchParam<string>('xc-match.models', 'vibes,vibes_anchorhp')
  const show = picked.split(',').filter((m) => models.includes(m))
  const mid = (c: Curve) => Math.sqrt(Math.max(c.lo, 0.5) * c.hi)
  const series = (m: string): SweepSeries[] => {
    const one = (what: string, id: string, lab: string, color: string, dash?: string): SweepSeries => ({
      id: `${m}|${id}`, label: lab, color, dash, group: id,
      points: curves.filter((c) => c.model === m && c.what === what && c.k === 18 && c.n >= 5).map((c) => ({ x: mid(c), y: c.median, tip: [`${lab}`, `HIPS ${c.lo}–${c.hi} µg: ${f2(c.median)} (n ${c.n})`] })),
    })
    const pub = curves.filter((c) => c.model === 'published' && c.n >= 5)
    return [one('IMPROVE low OC/EC FTIR/HIPS', 'implow', 'IMPROVE OC/EC < 2.5 (out of site)', '#d35400'),
      one('IMPROVE FTIR/HIPS', 'imp', 'IMPROVE, all (out of site)', '#9aa0a6'),
      one('Addis FTIR/HIPS', 'addis', 'Addis', MODEL_COLOR[`${m} k18`] ?? INK.accent, '5 3'),
      { id: `${m}|pub`, label: 'Addis published', color: INK.text, dash: '2 3', group: 'pub', points: pub.map((c) => ({ x: mid(c), y: c.median, tip: ['Addis published', `HIPS ${c.lo}–${c.hi} µg: ${f2(c.median)} (n ${c.n})`] })) }]
  }
  return (
    <ChartFrame id="xc-matched" source={source('MatchedLoading')} provenance={runLine(d)}
      title="Same loading, same composition — Addis against IMPROVE's own heavy filters"
      subtitle="Median FTIR EC ÷ HIPS BC (MAC 10) by HIPS loading per filter, k 18. IMPROVE is out of site. If Addis is just a heavy, low-OC/EC filter, the dashed Addis line should sit on the orange IMPROVE line. With raw and anchored baselines it nearly does; with production VIBES and the spline it falls far below."
      controls={<Select label="Models" value={show.join(',')} options={[...new Set(['vibes,vibes_anchorhp', 'raw,spline', 'vibes,anchored', 'vibes_deflate,vibes_anchorhp', show.join(',')])].filter(Boolean)}
        optionLabel={(o: string) => o.split(',').map((m) => SHORT[`${m} k18`] ?? m).join(' vs ')} onChange={setPicked} />}>
      <FacetGrid items={show} minCell={320} maxCols={2} itemKey={(m) => m} title={(m) => label(`${m} k18`)} render={(m, w) =>
        <SweepPlot series={series(m)} xScale="log" xDomain={[0.6, 90]} yDomain={[0.2, 2.4]} reference={1} xLabel="HIPS BC (µg per filter)" yLabel="median FTIR EC / HIPS BC" height={280} width={w} />} />
      <Legend items={[{ label: 'IMPROVE OC/EC < 2.5, out of site', color: '#d35400' }, { label: 'IMPROVE, all, out of site', color: '#9aa0a6' },
        { label: 'Addis, this model (dashed, in the model colour)', color: INK.accent }, { label: 'Addis, SPARTAN published (dotted)', color: INK.text }]} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

interface ContRow { prep: string; window: string; group: string; lo: number; hi: number; median: number; n: number }

function Continuum({ d }: { d: ContinuumData }) {
  const cb = rowsOf<ContRow>(d.continuum)
  const windows = [...new Set(cb.map((r) => String(r.window)))]
  const [win, setWin] = useSearchParam<string>('xc-cont.w', '1880', windows)
  const PREP: Record<string, string> = { raw: 'raw', spline: 'spline (AIRSpec)', vibes: 'VIBES 87 r70', anchored: 'PTFE-anchored', vibes_deflate: 'VIBES deflated', vibes_anchorhp: 'VIBES on anchored' }
  const preps = Object.keys(PREP).filter((p) => cb.some((r) => r.prep === p && String(r.window) === win))
  const mid = (r: ContRow) => Math.sqrt(Math.max(r.lo, 0.5) * r.hi)
  const series = (p: string): SweepSeries[] => ['IMPROVE lot 251', 'IMPROVE lot 248', 'Addis'].map((g) => ({
    id: `${p}|${g}`, label: g, color: g === 'Addis' ? '#e0a800' : g === 'IMPROVE lot 251' ? '#1f77b4' : '#7fb3d5', dash: g === 'Addis' ? undefined : '4 3',
    points: cb.filter((r) => r.prep === p && String(r.window) === win && r.group === g && r.n >= 5).map((r) => ({ x: mid(r), y: r.median * 1000, tip: [`${PREP[p]} · ${g}`, `HIPS ${r.lo}–${r.hi} µg: ${(r.median * 1000).toFixed(1)} ×10⁻³ (n ${r.n})`] })),
  }))
  const inj = rowsOf<{ variant: string; shape: string; value: number }>(d.injection)
  const variants = ['production (rank 70)', 'minus nu^1 direction', 'minus 1, nu, nu^2'].filter((v) => inj.some((r) => r.variant === v))
  const VARIANT: Record<string, string> = { 'production (rank 70)': 'production (87 blanks, rank 70)', 'minus nu^1 direction': 'ν¹ direction removed', 'minus 1, nu, nu^2': '1, ν, ν² removed' }
  const SHAPES: [string, string][] = [['nu^1 (BC-like)', 'BC-like ν¹ returned'], ['flat', 'flat returned'], ['nu^2 (scattering-like)', 'scattering-like ν² returned'], ['light-filter residual RMS', 'residual on light filters (RMS)']]
  const noise = rowsOf<{ window: string; lot: number; prep: string; noise_ug: number }>(d.noise)
  return (
    <ChartFrame id="xc-continuum" evidence="Saved 2–3 November study · broad spectral behavior by loading · synthetic shape response does not identify ambient chemistry; numerical convergence is a separate question" source={source('Continuum')} provenance={runLine(d)}
      title="The broad continuum each baseline keeps"
      subtitle={`Median absorbance (×10⁻³) in a band-free window (${win} cm⁻¹) by HIPS loading. Black carbon absorbs across the IR, so raw spectra rise with loading. The spline removes it by construction and production VIBES removes almost all of it. The PTFE-anchored backgrounds keep it while removing the PTFE and the lot offset. Each panel has its own y range.`}
      controls={<Segmented label="Window (cm⁻¹)" value={win} options={windows} onChange={setWin} />}>
      <FacetGrid items={preps} minCell={240} maxCols={3} itemKey={(p) => p} title={(p) => PREP[p]} render={(p, w) =>
        <SweepPlot series={series(p)} xScale="log" xDomain={[0.6, 90]} include={[0]} xLabel="HIPS BC (µg per filter)" yLabel="absorbance ×10⁻³" height={220} width={w} />} />
      <Legend items={[{ label: 'Addis', color: '#e0a800' }, { label: 'IMPROVE lot 251 (dashed)', color: '#1f77b4' }, { label: 'IMPROVE lot 248 (dashed)', color: '#7fb3d5' }]} />
      <table className="census-table">
        <thead><tr><th>VIBES background</th>{SHAPES.map(([k, l]) => <th key={k}>{l}</th>)}</tr></thead>
        <tbody>{variants.map((v) => <tr key={v}><td>{VARIANT[v]}</td>{SHAPES.map(([k]) => { const r = inj.find((x) => x.variant === v && x.shape === k)
          return <td key={k}>{r == null ? '–' : k.includes('RMS') ? r.value.toExponential(1) : `${(r.value * 100).toFixed(1)} %`}</td> })}</tr>)}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Window</th><th>Lot</th><th>Continuum scatter on light filters, raw − lot mean</th><th>PTFE-anchored</th></tr></thead>
        <tbody>{[...new Set(noise.map((r) => `${r.window}|${r.lot}`))].map((k) => { const [w, l] = k.split('|')
          const v = (p: string) => noise.find((r) => String(r.window) === w && String(r.lot) === l && r.prep === p)?.noise_ug
          return <tr key={k}><td>{w}</td><td>{l}</td><td>{f2(v('raw minus lot mean'))} µg EC</td><td>{f2(v('anchored'))} µg EC</td></tr> })}</tbody>
      </table>
      <Note>Injection test: a smooth continuum (0.05 absorbance at 2000 cm⁻¹) added to 16 light IMPROVE filters, real VIBES rerun with the saved 87-blank background. Noise: blank-to-blank scatter of the continuum on filters under 1 µg HIPS, in µg EC per filter. The IMPROVE median filter holds 3.7 µg, so the continuum is mostly noise in the training set; Addis filters hold about 34 µg.</Note>
    </ChartFrame>
  )
}

function Sites({ d }: { d: ContinuumData }) {
  const s = rowsOf<Record<string, any>>(d.sites).filter((r) => num(r.n) >= 15)
  const [set, setSet] = useSearchParam<'pair' | 'all'>('xc-sites.set', 'pair', ['pair', 'all'])
  const cols = (set === 'pair' ? ['published', BEFORE, AFTER] : ['published', 'vibes k18', 'vibes_anchorhp k18', 'anchored k18', 'raw k18', 'spline k18']).filter((c) => c in d.sites)
  const [sort, setSort] = useSearchParam<string>('xc-sites.sort', 'vibes k18', ['published', 'vibes k18', 'vibes_anchorhp k18', 'anchored k18', 'raw k18', 'spline k18'])
  const by = cols.includes(sort) ? sort : cols[0]
  const rows = [...s].sort((a, b) => num(a[by]) - num(b[by])).map((r) => ({
    key: r.code, label: `${r.code} (n ${r.n})`, bold: String(r.code).startsWith('ET'),
    values: Object.fromEntries(cols.map((c) => [c, { v: num(r[c]) }])),
    tip: [`${r.code}: ${r.n} filters, median HIPS ${f2(r.hips_ug)} µg, OC/BC ${f2(r.oc_over_bc)}`, `${Math.round(num(r.share_oc_bc_outside_improve) * 100)} % of filters have OC/BC outside IMPROVE's range`,
      ...cols.map((c) => `${SHORT[c] ?? c}: ${f2(r[c])}`)],
  }))
  return (
    <ChartFrame id="xc-sites" source={source('Sites')} provenance={runLine(d)}
      title="Every SPARTAN site against the TOR yardstick"
      subtitle="Median FTIR EC ÷ (HIPS × IMPROVE TOR/HIPS at the filter's loading and OC/BC), sites with 15 or more filters. 1 = reads the way TOR would. Bishoftu, Jaipur, Bujumbura and Dhaka under-read under every baseline; Addis, Santiago and Pretoria move most from VIBES (before) to VIBES anchored (after). Where OC/BC is outside IMPROVE's range (Addis, Bishoftu, Dhaka, Abu Dhabi), read the yardstick as ±20 %."
      controls={<>
        <Segmented label="Show" value={set} options={['pair', 'all'] as const} optionLabel={(o) => o === 'pair' ? 'Before / after' : 'Every baseline'} onChange={setSet} />
        <Select label="Sort by" value={cols.includes(sort) ? sort : cols[0]} options={cols} optionLabel={(o: string) => SHORT[o] ?? o} onChange={setSort} />
      </>}>
      <Forest rows={rows} labelWidth={110} rowHeight={20} panels={cols.map((c) => ({ key: c, label: SHORT[c] ?? c, ref: 1, fmt: f2, domain: [0.3, 1.7] as [number, number] }))} />
      <FixedSetNote what="the 2 Nov network fits (3,517 SPARTAN filters; HIPS > 2 µg)" />
    </ChartFrame>
  )
}

function Windows({ d }: { d: ContinuumData }) {
  const w = rowsOf<{ model: string; window: string; hi: number; lo: number; difference: number }>(d.windows)
  const [set, setSet] = useSearchParam<'pair' | 'all'>('xc-win.set', 'pair', ['pair', 'all'])
  const models = (set === 'pair' ? [bare(BEFORE), bare(AFTER)] : ['vibes', 'vibes_anchorhp', 'anchored', 'raw', 'spline', 'vibes_deflate']).filter((m) => w.some((r) => r.model === m))
  const vals = w.filter((r) => models.includes(r.model)).map((r) => r.difference).filter(Number.isFinite)
  const domain: [number, number] = [Math.min(...vals) - 0.05, Math.max(...vals) + 0.05]
  const wins = [...new Set(w.map((r) => r.window))]
  const rows = wins.map((win) => ({
    key: win, label: `${win} cm⁻¹`, bold: win === '3000-2800',
    values: Object.fromEntries(models.map((m) => { const r = w.find((x) => x.model === m && x.window === win); return [m, r ? { v: r.difference } : undefined] })),
  }))
  return (
    <ChartFrame id="xc-windows" source={source('Windows')} provenance={runLine(d)}
      title="Where in the spectrum Addis falls short"
      subtitle="Each model's prediction split into windows (coefficient × centred absorbance per µg HIPS, k 12): Addis minus matched IMPROVE filters (OC/EC < 2.5, HIPS 20–60 µg). Negative = Addis gets less EC from that window; every panel shares one axis. The aliphatic C–H stretch (3000–2800) is the largest deficit in every baseline; the anchored backgrounds recover part of it from the continuum windows (2800–2400, 4000–3700). Neighbouring windows offset each other, so read the pattern, not single bars."
      controls={<Segmented label="Show" value={set} options={['pair', 'all'] as const} optionLabel={(o) => o === 'pair' ? 'Before / after' : 'Every baseline'} onChange={setSet} />}>
      <Forest rows={rows} labelWidth={140} rowHeight={20} panels={models.map((m) => ({ key: m, label: SHORT[`${m} k18`] ?? m, ref: 0, fmt: (v: number) => v.toFixed(2), domain }))} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

interface AethSite { code: string; n: number; hips_ug: number; aeth_median: number; y: string; slope: number; intercept: number; intercept_rel: number; r2: number; loglog: number; rho_ratio_vs_aeth: number }
const AETH_Y: [string, string][] = [['HIPS BC (MAC 10)', 'HIPS BC'], ['SPARTAN published FTIR EC', 'published FTIR'], ['VIBES k18 (before)', 'VIBES (before)'],
  ['VIBES anchored k18 (after)', 'VIBES anchored (after)'], ['EPO k18', 'EPO']]
const pct = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(0)} %`

function AethCheck({ d }: { d: ContinuumData }) {
  const mac = useMac()
  const pickable: string[] = [...(d.settings?.models ?? []), ...(d.settings?.fix_models ?? [])]
  const [after, setAfter] = useSearchParam<string>('xc-aeth.after', AFTER, pickable)
  const lg = useLegend()
  const all = useMemo(() => rowsOf(d.addis_filters).filter((r) => Number.isFinite(num(r.aeth_ir)))
    .map((r): Record<string, any> => ({ ...r, hips_bc: mac.bc(num(r.hips_bc)) })), [d, mac])
  const { counted, marked, nOut } = useCounted(all, (r) => String(r.filter_id))
  const lambda = num(d.settings?.LAMBDA_AETH) || 1
  const pointsOf = (m: string, rows: Record<string, any>[]): CrossPoint[] => rows
    .filter((r) => Number.isFinite(num(r[m])) && lg.show(String(r.season)))
    .map((r) => ({
      x: num(r.aeth_ir), y: num(r[m]), color: SEASON_COLOR[r.season] ?? INK.muted, id: baseFilterId(String(r.filter_id)), season: String(r.season), seasonColor: SEASON_COLOR[r.season],
      opacity: lg.dim(String(r.season)),
      tip: [`${r.filter_id} · ${r.season} · ${r.date ?? 'no date'}`, `aethalometer IR BCc ${f2(r.aeth_ir)} µg/m³`, `${m === 'hips_bc' ? `HIPS BC (${mac.label})` : label(m)}: ${f2(r[m])} µg/m³`],
      rec: { id: baseFilterId(String(r.filter_id)), source: 'Continuum and the baseline · aethalometer check', site: 'ETAD', date: r.date ?? undefined, season: String(r.season),
        fields: [['Aethalometer IR BCc', f2(r.aeth_ir)], ['HIPS BC', f2(r.hips_bc)], [`FTIR EC, ${SHORT[m] ?? m}`, f2(r[m])]] as [string, string][] },
    }))
  const panels = [
    { key: 'hips_bc', title: `HIPS BC (${mac.label})`, y: 'HIPS BC (µg/m³)' },
    { key: BEFORE, title: `Before: ${label(BEFORE)}`, y: 'FTIR EC (µg/m³)' },
    { key: after, title: `After: ${label(after)}`, y: 'FTIR EC (µg/m³)' },
  ].map((p) => ({ ...p, points: pointsOf(p.key, counted), marked: pointsOf(p.key, marked) }))
  const sites = rowsOf<AethSite>(d.aeth_sites)
  const codes = [...new Set(sites.map((r) => r.code))].sort((a, b) => num(sites.find((r) => r.code === b)?.hips_ug) - num(sites.find((r) => r.code === a)?.hips_ug))
  return (
    <ChartFrame id="xc-aeth" source={source('AethCheck')} provenance={`run_intercept_fixes_20261103.py stage aeth · MA350 IR BCc, overlap-weighted daily bins (≥ 12 h; export_aethalometer.py) · Deming λ ${f2(lambda)}`}
      title="Whose intercept is it? HIPS and FTIR against the Addis aethalometer"
      subtitle={`The same Addis filters against the MA350 aethalometer on its own inlet (IR BCc over each filter's 24 h). It is a third instrument, so it can say which of HIPS and FTIR follows the day-to-day BC. Its calibration differs, so read the intercept and whether the cloud is straight through the origin, not the slope. FTIR on the PTFE-anchored background runs through the origin; HIPS sits on a large positive intercept and rises half as fast. Production VIBES has the negative intercept that the anchored background removed.`}
      controls={<>
        <Select label="After" value={after} options={pickable} optionLabel={(o: string) => SHORT[o] ?? o} onChange={setAfter} />
      </>}>
      <FacetGrid items={panels} minCell={280} maxCols={3} itemKey={(p) => p.key} share render={(p, w) =>
        <Crossplot points={p.points} marked={p.marked} title={p.title} xLabel={AETH_LABEL} yLabel={p.y} maxSide={Math.min(w - 70, 400)} errorRatio={lambda} />} />
      <Legend {...lg.props} items={SEASONS.map((s) => ({ label: s, color: SEASON_COLOR[s] }))} />
      <table className="census-table">
        <thead><tr><th>Site (median HIPS per filter)</th><th>n</th>{AETH_Y.map(([k, l]) => <th key={k}>{l}</th>)}</tr></thead>
        <tbody>{codes.map((c) => { const at = (y: string) => sites.find((r) => r.code === c && r.y === y); const first = sites.find((r) => r.code === c)
          return <tr key={c}><td>{c} ({f2(first?.hips_ug)} µg)</td><td>{first?.n}</td>{AETH_Y.map(([k]) => { const r = at(k)
            return <td key={k}>{r ? `${f2(r.loglog)} · ${pct(r.intercept_rel)}` : '–'}</td> })}</tr> })}</tbody>
      </table>
      <Note>Each cell: log-log slope against the aethalometer (1 = proportional) · Deming intercept as a share of the mean. At Addis HIPS rises as the aethalometer to the power 0.44 and its ratio to the aethalometer falls with the aethalometer (ρ −0.92), while FTIR's ratio doesn't (ρ 0.0). This pattern is consistent with HIPS range compression in these comparisons; differing calibration, timing and errors still limit causal attribution. The other sites show it too, but much more weakly (0.73–0.77). USPA filters are light and FTIR is noisy there (R² 0.2–0.3). Aethalometer days are legacy daily aggregates matched by overlap (unverified 9 AM bins); the timing was checked against HIPS on 27 Oct.</Note>
      <ExcludedNote count={nOut} drawn />
    </ChartFrame>
  )
}

interface SweepRow extends SourceScore { model: string; k: number; [m: string]: number | string | null | undefined }
const SWEEP_LABEL: Record<string, string> = {
  vibes: 'VIBES (before)', vibes_anchorhp: 'VIBES anchored (after)', epo: 'EPO', dipls_g10: 'di-PLS, Addis target (γ 10)', dipls_net: 'di-PLS, network target (γ 3)',
  mask_org: 'organic bands masked', cont_only: 'continuum windows only', iw: 'importance-weighted to Addis',
}
const SWEEP_COLOR: Record<string, string> = {
  vibes: '#d62728', vibes_anchorhp: '#8c564b', epo: '#e377c2', dipls_g10: '#1f9e89', dipls_net: '#6a51a3', mask_org: '#17becf', cont_only: '#9467bd', iw: '#bcbd22',
}
const METRIC: Record<string, { label: string; ref?: number; fmt: (v: number) => string; tick?: (v: number) => string }> = {
  intercept_hips_bc: { label: 'Deming intercept vs HIPS (µg/m³)', ref: 0, fmt: coefSigned },
  intercept_tor_equiv: { label: 'Deming intercept vs TOR yardstick (µg/m³)', ref: 0, fmt: coefSigned },
  intercept_aeth_rel: { label: 'intercept vs aethalometer (share of mean)', ref: 0, fmt: pct, tick: pct },
  loglog_aeth: { label: 'log-log slope vs aethalometer', ref: 1, fmt: f2 },
  slope_hips_bc: { label: 'Deming slope vs HIPS', ref: 1, fmt: coef },
  ratio_tor_equiv: { label: 'median ratio to TOR yardstick', ref: 1, fmt: f2 },
  improve_median_absolute_error: { label: 'IMPROVE median absolute error (µg/filter)', fmt: f2 },
  improve_heavy_lowocec_ratio: { label: 'IMPROVE heavy low-OC/EC: FTIR ÷ TOR', ref: 1, fmt: f2 },
}

function InterceptSweep({ d }: { d: ContinuumData }) {
  const rows: SweepRow[] = rowsOf<SweepRow>(d.sweep).map((r) => ({ ...r, improve_median_absolute_error: sweepMedianAbsoluteError(r, d.settings?.source_evaluation) }))
  const models = Object.keys(SWEEP_LABEL).filter((m) => rows.some((r) => r.model === m))
  const keys = Object.keys(METRIC)
  const [leftStored, setLeft] = useSearchParam<string>('xc-icpt.a', 'intercept_hips_bc', [...keys, 'improve_mae'])
  const [rightStored, setRight] = useSearchParam<string>('xc-icpt.b', 'intercept_aeth_rel', [...keys, 'improve_mae'])
  // Historical URLs used improve_mae for MedAE; preserve their meaning across the schema correction.
  const left = leftStored === 'improve_mae' ? 'improve_median_absolute_error' : leftStored
  const right = rightStored === 'improve_mae' ? 'improve_median_absolute_error' : rightStored
  const lg = useLegend(['dipls_net', 'iw', 'cont_only'].map((m) => SWEEP_LABEL[m]))
  const series = (metric: string): SweepSeries[] => models.filter((m) => lg.show(SWEEP_LABEL[m]) && !(m === 'epo' && metric.startsWith('improve_') && (!epoSourceVerified(d.settings?.source_evaluation) || !rows.some((r) => r.model === m && r.evaluation_status === 'corrected')))).map((m) => ({
    id: `${metric}|${m}`, label: SWEEP_LABEL[m], color: SWEEP_COLOR[m], group: SWEEP_LABEL[m], width: m === 'vibes' || m === 'vibes_anchorhp' ? 2.5 : 1.5,
    dash: m === 'vibes' ? '5 3' : undefined,
    points: rows.filter((r) => r.model === m).map((r) => ({ x: num(r.k), y: num(r[metric]), tip: [`${SWEEP_LABEL[m]}, k ${r.k}`, `${METRIC[metric].label}: ${METRIC[metric].fmt(num(r[metric]))}`,
      `vs HIPS ${coef(num(r.slope_hips_bc))}x ${coefSigned(num(r.intercept_hips_bc))}; vs aethalometer intercept ${pct(num(r.intercept_aeth_rel))}`, `IMPROVE median absolute error ${f2(num(r.improve_median_absolute_error))} µg/filter`] })),
  }))
  const kMax = d.settings?.k ?? 18
  const panels = [left, right]
  return (
    <ChartFrame id="xc-icpt" source={source('InterceptSweep')} provenance={`run_intercept_fixes_20261103.py · IMPROVE TOR EC only, 5-fold by site · channel pairs averaged · Addis HIPS never used to fit or choose`}
      title="Exploratory transfer candidates, by component count"
      evidence="Source: IMPROVE TOR; sweep heavy subset: HIPS-derived loading >20 µg, OC/EC <2.5. Target: optical comparators. Unverified EPO source metrics are withheld. Curves are exploratory, not independent model selection."
      subtitle="Every candidate is trained on IMPROVE TOR EC only, on the PTFE-anchored VIBES spectra. EPO removes the IMPROVE spectral signatures of OC, other PM, Fe and K. di-PLS aligns IMPROVE's spectral covariance with the unlabeled Addis (or whole-network) spectra. Importance weights up-weight Addis-like IMPROVE filters. The masked models drop bands. Against HIPS the intercept only gets smaller by giving up level (low k, continuum only). Whenever the level is right, the HIPS slope is 1.9–2.2 and the intercept −5 to −6 µg/m³, whatever the method. Against the aethalometer every anchored model is already proportional from k 10 on (intercept within 5 % of the mean); production VIBES is not (−17 to −28 %)."
      controls={<>
        <Select label="Left" value={left} options={keys} optionLabel={(o: string) => METRIC[o].label} onChange={setLeft} />
        <Select label="Right" value={right} options={keys} optionLabel={(o: string) => METRIC[o].label} onChange={setRight} />
      </>}>
      <FacetGrid items={panels} minCell={320} maxCols={2} itemKey={(m) => m} title={(m) => METRIC[m].label} render={(m, w) =>
        <SweepPlot series={series(m)} xDomain={[1, 25]} reference={METRIC[m].ref} xLabel="PLS components (k)" yLabel={METRIC[m].label} yFormat={METRIC[m].tick}
          height={280} width={w} dim={lg.dim} xMarkers={[{ at: kMax, label: `k ${kMax}` }]} />} />
      <Legend {...lg.props} items={models.map((m) => ({ label: SWEEP_LABEL[m], color: SWEEP_COLOR[m] }))} />
      <Note>Click a legend entry to show or hide a model. The historical EPO source curve is withheld until its fold-safe rerun is attached. Target curves use saved full-source models and optical comparisons. A component count chosen after inspecting these curves requires separate evaluation.</Note>
    </ChartFrame>
  )
}

interface GainRow { model: string; k: number; window: string; mean: number; slope_hips_bc: number; intercept_hips_bc: number; slope_tor_equiv: number; intercept_tor_equiv: number }

function WindowGain({ d }: { d: ContinuumData }) {
  const g = rowsOf<GainRow>(d.gain)
  const [what, setWhat] = useSearchParam<'slope_hips_bc' | 'slope_tor_equiv' | 'intercept_hips_bc'>('xc-gain.v', 'slope_hips_bc', ['slope_hips_bc', 'slope_tor_equiv', 'intercept_hips_bc'])
  const [k, setK] = useSearchParam<string>('xc-gain.k', '18', ['12', '18'])
  const models = ['vibes', 'vibes_anchorhp'].filter((m) => g.some((r) => r.model === m))
  const at = g.filter((r) => String(r.k) === k)
  const wins = [...new Set(at.map((r) => r.window))]
  const total = (m: string) => at.filter((r) => r.model === m).reduce((s, r) => s + num(r[what]), 0)
  const vals = at.map((r) => num(r[what])).concat(models.map(total)).filter(Number.isFinite)
  const domain: [number, number] = [Math.min(0, ...vals) - 0.1, Math.max(...vals) + 0.1]
  const NAME: Record<string, string> = { slope_hips_bc: 'OLS slope vs HIPS', slope_tor_equiv: 'OLS slope vs TOR yardstick', intercept_hips_bc: 'OLS intercept vs HIPS (µg/m³)' }
  const rows = [{ key: 'total', label: 'all windows (total)', bold: true, values: Object.fromEntries(models.map((m) => [m, { v: total(m) }])) },
    ...wins.map((w) => ({ key: w, label: w === 'mean term' ? 'IMPROVE mean (constant)' : `${w} cm⁻¹`, bold: ['3700-3400', '3000-2800', '1600-1500'].includes(w),
      values: Object.fromEntries(models.map((m) => { const r = at.find((x) => x.model === m && x.window === w); return [m, r ? { v: num(r[what]) } : undefined] })) }))]
  return (
    <ChartFrame id="xc-gain" source={source('WindowGain')} provenance="run_intercept_fixes_20261103.py stage attrib · Addis filters that count, µg/m³"
      title="Where the Addis day-to-day gain comes from"
      subtitle={`Each model's Addis prediction split into spectral windows (coefficient × centred absorbance). Each window's contribution is regressed on HIPS across the Addis filters, and the slopes add up to the model's OLS slope. Addis HIPS spans only about 3.4–6.8 µg/m³ (mean ${f2(d.settings?.hips_mean)}), so the intercept is the slope's mirror image: intercept ≈ −(slope − ratio) × mean. Production VIBES gets most of its extra gain from the PTFE bend and the edge of the C–F doublet (600–500, 1300–1250). On the anchored background, neighbouring windows trade large shares (O–H/N–H 3700–3400, C–H 3000–2800, 1600–1500 against 4000–3700 and 3400–3100): the continuum is read as differences between windows.`}
      controls={<>
        <Segmented label="Show" value={what} options={['slope_hips_bc', 'slope_tor_equiv', 'intercept_hips_bc'] as const} optionLabel={(o) => NAME[o]} onChange={setWhat} />
        <Segmented label="k" value={k} options={['12', '18']} onChange={setK} />
      </>}>
      <Forest rows={rows} labelWidth={170} rowHeight={20} panels={models.map((m) => ({ key: m, label: SHORT[`${m} k18`] ?? m, ref: 0, fmt: (v: number) => v.toFixed(2), domain }))} />
      <FixedSetNote what={FIXED} />
    </ChartFrame>
  )
}

interface HipsSite { code: string; n: number; median_hips_ug: number; predicted: number; rma_vs_ftir_ug: number; r_ftir_ug: number; rma_vs_pub_ug: number; loglog_vs_aeth: number | null }
interface HipsGrid { Ls: number; model: string; slope: number; intercept: number; intercept_rel: number; ratio: number; hips_c_loglog_aeth: number; hips_c_mean: number; hips_raw_mean: number }
interface HipsVol { group: string; reference: string; n: number; sites: number; median_hips_ug: number; sd_log_volume_within: number; coef: number; se: number }
const GRID_COLOR: Record<string, string> = { 'SPARTAN published': '#444', 'VIBES k18 (before)': '#d62728', 'VIBES anchored k18 (after)': '#8c564b', 'EPO k18': '#e377c2' }

function HipsSaturation({ d }: { d: ContinuumData }) {
  const sat = rowsOf<{ reference: string; Ls_ug: number; boot_p10: number; boot_p90: number; sites: string }>(d.hips_sat)
  const Ls = num(d.settings?.hips?.Ls_used)
  const lo = sat.reduce((s, r) => s + num(r.boot_p10), 0) / (sat.length || 1), hi = sat.reduce((s, r) => s + num(r.boot_p90), 0) / (sat.length || 1)
  const sites = rowsOf<HipsSite>(d.hips_sites).filter((r) => num(r.r_ftir_ug) >= 0.6)
  const grid = rowsOf<HipsGrid>(d.hips_grid).filter((r) => num(r.Ls) < 1e5 && num(r.Ls) >= 45)
  const [what, setWhat] = useSearchParam<'intercept' | 'slope' | 'hips_c_loglog_aeth'>('xc-hsat.y', 'intercept', ['intercept', 'slope', 'hips_c_loglog_aeth'])
  const raw = rowsOf<HipsGrid>(d.hips_grid).filter((r) => num(r.Ls) >= 1e5)
  const curve = (L: number, id: string, lab: string, dash?: string): SweepSeries => ({ id, label: lab, color: INK.muted, dash, marks: false, group: 'curve',
    points: [0.8, 1.5, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40].map((x) => ({ x, y: 1 - x / L })) })
  const left: SweepSeries[] = [
    curve(Ls, 'pred', `predicted 1 − loading/Ls (Ls ${Ls.toFixed(0)} µg)`), curve(lo, 'lo', `Ls ${lo.toFixed(0)} µg (bootstrap 10 %)`, '3 3'), curve(hi, 'hi', `Ls ${hi.toFixed(0)} µg (bootstrap 90 %)`, '3 3'),
    { id: 'ftir', label: 'HIPS vs anchored FTIR (per site)', color: '#8c564b', line: false, group: 'ftir',
      points: sites.map((r) => ({ x: num(r.median_hips_ug), y: num(r.rma_vs_ftir_ug), tip: [`${r.code} (n ${r.n}), median ${f2(r.median_hips_ug)} µg per filter`, `log-log HIPS on anchored FTIR ${f2(r.rma_vs_ftir_ug)} (r ${f2(r.r_ftir_ug)})`, `predicted ${f2(r.predicted)}`, `on published FTIR ${f2(r.rma_vs_pub_ug)}`] })) },
    { id: 'aeth', label: 'HIPS vs aethalometer', color: '#e0a800', line: false, r: 5, group: 'aeth',
      points: sites.filter((r) => r.loglog_vs_aeth != null && Number.isFinite(num(r.loglog_vs_aeth))).map((r) => ({ x: num(r.median_hips_ug), y: num(r.loglog_vs_aeth), tip: [`${r.code}: HIPS on aethalometer, log-log ${f2(r.loglog_vs_aeth)}`, `predicted ${f2(r.predicted)}`] })) },
  ]
  const models = Object.keys(GRID_COLOR).filter((m) => grid.some((r) => r.model === m))
  const YL: Record<string, string> = { intercept: 'Addis Deming intercept vs linearised HIPS (µg/m³)', slope: 'Addis Deming slope vs linearised HIPS (MAC 10, unrescaled)', hips_c_loglog_aeth: 'linearised HIPS vs aethalometer, log-log slope' }
  const right: SweepSeries[] = what === 'hips_c_loglog_aeth'
    ? [{ id: 'aethll', label: 'linearised HIPS vs aethalometer', color: '#e0a800', group: 'aethll', points: grid.filter((r) => r.model === models[0]).map((r) => ({ x: num(r.Ls), y: num(r.hips_c_loglog_aeth), tip: [`Ls ${num(r.Ls).toFixed(0)} µg`, `log-log on the aethalometer ${f2(r.hips_c_loglog_aeth)} (raw HIPS ${f2(raw[0]?.hips_c_loglog_aeth)})`] })) }]
    : models.map((m) => ({ id: m, label: m, color: GRID_COLOR[m], group: m, width: m.includes('anchored') ? 2.5 : 1.5,
      points: grid.filter((r) => r.model === m).map((r) => ({ x: num(r.Ls), y: num(r[what]), tip: [`${m}, Ls ${num(r.Ls).toFixed(0)} µg`, `Deming ${coef(num(r.slope))}x ${coefSigned(num(r.intercept))} µg/m³`, `uncorrected: ${coef(num(raw.find((x) => x.model === m)?.slope))}x ${coefSigned(num(raw.find((x) => x.model === m)?.intercept))}`] })) }))
  const vol = rowsOf<HipsVol>(d.hips_volume)
  const stress = rowsOf<{ group: string; sites: string; reference: string; control: string; coef: number; se: number; n: number }>(d.hips_stress)
  const groups = [...new Set(vol.map((r) => r.group))]
  return (
    <ChartFrame id="xc-hsat" source={source('HipsSaturation')} provenance={`run_intercept_fixes_20261103.py stage hips · Ls fitted on ${sat[0]?.sites ?? ''} (Ethiopia left out), site bootstrap`}
      title="Is it filter loading? What the sampled-volume test shows, and where it fails"
      subtitle={`A filter that drew more air at the same concentration carries more BC, so sampled volume can in principle separate loading from concentration. At heavy sites more air goes with a lower HIPS/FTIR, and a curve HIPS = true / (1 + true/Ls), fitted with Ethiopia left out, gives Ls ≈ ${Ls.toFixed(0)} µg per filter (80 % ${lo.toFixed(0)}–${hi.toFixed(0)}). Left: that curve against each site's HIPS compression. Right: Addis against HIPS linearised with each Ls. Read this as a what-if. The second table shows the volume signal comes mostly from Delhi and Bhaktapur, where volume moves with concentration and sampling period. Where volume is independent of concentration, the effect shrinks towards zero and Johannesburg shows none, so the close match at Addis can't be counted as proof.`}
      controls={<Segmented label="Right panel" value={what} options={['intercept', 'slope', 'hips_c_loglog_aeth'] as const} optionLabel={(o) => ({ intercept: 'intercept', slope: 'slope', hips_c_loglog_aeth: 'vs aethalometer' })[o]} onChange={setWhat} />}>
      <FacetGrid items={['sites', 'addis']} minCell={340} maxCols={2} itemKey={(k) => k} title={(k) => (k === 'sites' ? 'HIPS compression by site loading' : 'Addis against linearised HIPS, by Ls')} render={(k, w) => k === 'sites'
        ? <SweepPlot series={left} xScale="log" xDomain={[0.7, 45]} yDomain={[0.2, 1.7]} reference={1} xLabel="site median HIPS loading (µg per filter, MAC 10)" yLabel="log-log slope of HIPS (1 = proportional)" height={300} width={w} />
        : <SweepPlot series={right} xScale="log" xDomain={[45, 700]} reference={what === 'intercept' ? 0 : 1} xLabel="saturation constant Ls (µg per filter)" yLabel={YL[what]} height={300} width={w}
            xMarkers={[{ at: Ls, label: `Ls ${Ls.toFixed(0)}` }, { at: lo, label: '10 %' }, { at: hi, label: '90 %' }]} />} />
      <Legend items={[{ label: 'predicted from Ls (dashed: bootstrap 10 / 90 %)', color: INK.muted }, { label: 'HIPS vs anchored FTIR, per site (r ≥ 0.6)', color: '#8c564b' }, { label: 'HIPS vs aethalometer', color: '#e0a800' },
        ...(what === 'hips_c_loglog_aeth' ? [] : models.map((m) => ({ label: `Addis: ${m}`, color: GRID_COLOR[m] })))]} />
      <table className="census-table">
        <thead><tr><th>Within-site effect of log volume on log(HIPS / FTIR)</th><th>Filters (sites)</th><th>Median HIPS</th><th>Volume spread</th>{[...new Set(vol.map((r) => r.reference))].map((r) => <th key={r}>vs {r}</th>)}</tr></thead>
        <tbody>{groups.map((g) => { const rs = vol.filter((r) => r.group === g)
          return <tr key={g}><td>{g}</td><td>{rs[0]?.n} ({rs[0]?.sites})</td><td>{f2(rs[0]?.median_hips_ug)} µg</td><td>{f2(rs[0]?.sd_log_volume_within)}</td>{rs.map((r) => <td key={r.reference}>{coefSigned(r.coef)} ± {f2(r.se)}</td>)}</tr> })}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Stress test: effect of log volume</th><th>Sites</th><th>Control</th><th>vs VIBES anchored</th><th>vs SPARTAN published</th></tr></thead>
        <tbody>{stress.filter((r) => r.reference === 'VIBES anchored k18').map((r) => { const o = stress.find((x) => x.group === r.group && x.control === r.control && x.reference !== r.reference)
          return <tr key={r.group + r.control}><td>{r.group}</td><td>{r.sites}</td><td>{r.control}</td><td>{coefSigned(r.coef)} ± {f2(r.se)}</td><td>{o ? `${coefSigned(o.coef)} ± ${f2(o.se)}` : '–'}</td></tr> })}</tbody>
      </table>
      <Note>Coefficient of log sampled volume at fixed log concentration, with site fixed effects; volume spread is the within-site SD of log volume. Negative = more loading, lower HIPS. Sites are grouped by their median HIPS, never by the filter's own HIPS (that would select on the numerator). Addis volumes barely vary (SD 0.016), so Addis can't test this itself. Light sites lean positive, and the reason isn't known. A saturation of Ls 64 predicts about −0.25 at these sites; the independent-volume sites give −0.08 ± 0.08 and −0.01 ± 0.06 with the strictest control. The same loading rise appears in IMPROVE TOR/HIPS above 20 µg, about as predicted (×1.43 from 5–15 to 30–70 µg TOR in low-OC/EC filters with OP {'>'} 0, against ×1.53 predicted by Ls); the TOR yardstick therefore already carries part of the saturation.</Note>
    </ChartFrame>
  )
}

const SITE_NAME: Record<string, string> = { USPA: 'Pasadena', CHTS: 'Beijing', INDH: 'Delhi', ETAD: 'Addis Ababa' }
const SITE_HUE: Record<string, string> = { USPA: '#1f77b4', CHTS: '#2ca02c', INDH: '#9467bd', ETAD: '#d35400' }
interface Hold { trained_on: string; n: number; model: string; p1: number; p2: number; addis_bias_log: number; addis_rmse_log: number; predicted_loglog: number; observed_loglog: number }

function HipsCurve({ d }: { d: ContinuumData }) {
  const lg = useLegend()
  const pts = rowsOf<Record<string, any>>(d.tau_points)
  const hold = rowsOf<Hold>(d.tau_holdout)
  const main = hold.filter((r) => r.trained_on === hold[0]?.trained_on)
  const P = (m: string) => main.find((r) => r.model === m)
  const prop = P('proportional'), sat = P('saturating with loading')
  const points: CrossPoint[] = pts.filter((r) => lg.show(SITE_NAME[r.code])).map((r) => ({ x: num(r.aeth_ug), y: num(r.tau), color: SITE_HUE[r.code], opacity: 0.75 * lg.dim(SITE_NAME[r.code]), site: SITE_NAME[r.code],
    id: baseFilterId(String(r.filter_id)), tip: [`${r.filter_id} · ${SITE_NAME[r.code]} · ${r.date}`, `HIPS optical depth τ ${f2(r.tau)}`, `aethalometer ${f2(r.aeth)} µg/m³ × ${f2(r.volume_m3)} m³ = ${f2(r.aeth_ug)} µg per filter`, `HIPS transmittance t ${f2(r.HIPS_t)}, reflectance r ${f2(r.HIPS_r)}`] }))
  const xs = Array.from({ length: 60 }, (_, i) => (i + 1) * 2.4)
  const rel = (v: number) => pct(Math.exp(v) - 1)
  return (
    <ChartFrame id="xc-tau" source={source('HipsCurve')} provenance="run_intercept_fixes_20261103.py stage stress · HIPS_tau from Filter Data/unified_filter_dataset.pkl · MA350 IR BCc × volume"
      title="HIPS bends over: optical depth against the aethalometer, with Addis held out"
      subtitle={`HIPS optical depth τ (before any MAC) against BC per filter from the aethalometer, at the four sites with both. If HIPS were proportional, every site would lie on the dashed straight line. The solid curve was fitted on Beijing, Pasadena and Delhi only (τ = aL / (1 + aL/τ∞), τ∞ ${f2(sat?.p2)}), so Addis is a true prediction. It carries Addis far better than the line (${rel(sat?.addis_bias_log ?? NaN)} against ${rel(prop?.addis_bias_log ?? NaN)}), and it means the Addis "floor" could be the top of one shared curve rather than something only Addis does. It also shifts the earlier reading that the Addis relation is linear: that test used FTIR EC on x; with the aethalometer on x the quadratic wins (ΔAIC ${num(d.settings?.stress?.addis_tau_dAIC_quadratic_minus_linear).toFixed(1)}).`}>
      <div style={{ maxWidth: 560, margin: '0 auto' }}>
        <Crossplot points={points} title="HIPS optical depth vs aethalometer loading" xLabel="aethalometer BC (µg per filter)" yLabel="HIPS optical depth τ" fit={false} sameScale={false} grouping="none" maxSide={480}
          foreground={(x, y) => <>
            {prop && <path d={`M${x(0)},${y(0)}L${x(70)},${y(num(prop.p1) * 70)}`} fill="none" stroke={INK.muted} strokeWidth={1.5} strokeDasharray="5 4" />}
            {sat && <path d={`M${x(0)},${y(0)}` + xs.map((v) => `L${x(v)},${y((num(sat.p1) * v) / (1 + (num(sat.p1) * v) / num(sat.p2)))}`).join('')} fill="none" stroke={INK.text} strokeWidth={2} />}
          </>} />
      </div>
      <Legend {...lg.props} items={Object.keys(SITE_NAME).map((c) => ({ label: SITE_NAME[c], color: SITE_HUE[c] }))} />
      <table className="census-table">
        <thead><tr><th>Trained on (Addis held out)</th><th>Model</th><th>Addis bias</th><th>Addis scatter (log)</th><th>Predicted log-log</th><th>Observed</th></tr></thead>
        <tbody>{hold.map((r) => <tr key={r.trained_on + r.model}><td>{r.trained_on} ({r.n})</td><td>{r.model}</td><td>{rel(r.addis_bias_log)}</td><td>{f2(r.addis_rmse_log)}</td><td>{f2(r.predicted_loglog)}</td><td>{f2(r.observed_loglog)}</td></tr>)}</tbody>
      </table>
      <Note>The straight line loses in every training set. Loading against concentration doesn't resolve: loading wins on two training sets and concentration on the third, and the curve moves a lot with which sites go in (τ∞ 1.1 to 3.7). Only Delhi has a different volume, and it has just {pts.filter((r) => r.code === 'INDH').length} filters here. Beijing reads about 1.8× Pasadena against the aethalometer at the light end, so the aethalometer's scale isn't the same at every site. Addis's split was never fixed and the three training sites' were; BCc holds up across those fixes against HIPS (see the flow check), so that isn't the main worry here, the site-to-site scale is. Treat the curve as supporting evidence; the IMPROVE TOR response doesn't use the aethalometer at all.</Note>
    </ChartFrame>
  )
}

const num2 = (v: unknown, d = 2) => (Number.isFinite(num(v)) ? num(v).toFixed(d) : '–')

function WhereNext({ d }: { d: ContinuumData }) {
  const verified = epoSourceVerified(d.settings?.source_evaluation)
  return <ChartFrame id="xc-next" source={source('WhereNext')} exportable={false}
    title="Close this stage with a reproducible account of the evidence"
    evidence="Existing-data investigation · no independently validated Addis EC product selected">
    <div className="closeout-table-scroll"><table className="census-table">
      <thead><tr><th>Closeout task</th><th>Current evidence</th><th>Acceptance criterion</th></tr></thead>
      <tbody>
        <tr><td>Source evaluation</td><td>{verified ? 'Corrected retrospective source-score receipt attached.' : 'Historical EPO source scores withheld; fold-safe rerun pending.'}</td><td>Training-only EPO in each site fold; frozen membership; held-out-label perturbation leaves predictions unchanged; common physical filters and coverage reported.</td></tr>
        <tr><td>Baseline and numerical choice</td><td>Anchoring is a candidate; Newton addresses convergence.</td><td>Preserve each version and its downstream scope. Complete season-composition and fingerprint checks before promoting numerical v2 into those conclusions.</td></tr>
        <tr><td>Seasonal interpretation</td><td>Seasonal differences persist in several specified comparisons.</td><td>Name the cohort, reference and February convention. Keep Bega, Belg and Kiremt separate without deriving unvalidated seasonal correction factors.</td></tr>
        <tr><td>Optical response</td><td>The IMPROVE-derived response changes the Addis comparison.</td><td>Show raw HIPS and transferred-response scenarios together. Treat the fitted bend as an empirical model feature, not a certified instrument cutoff.</td></tr>
        <tr><td>Reproducible release</td><td>The six-figure section reuses the current saved exports.</td><td>Freeze source/data identities; pass the declared checks; restore a representative figure; verify a fresh-context link and completed export.</td></tr>
      </tbody>
    </table></div>
    <Note>Absolute Addis EC accuracy remains unresolved. Independent Ethiopian TOR, full PMF qualification and further acceleration are separate follow-on work. The future TOR rule below is a proposal; its acceptance limits and assumptions still require agreement.</Note>
  </ChartFrame>
}

function PassRuleProposal({ d }: { d: ContinuumData }) {
  const sim = rowsOf<{ model: string; n_filters: number; true_level: number; pass: number; fail: number; inconclusive: number }>(d.rule_sim).filter((r) => r.model === 'VIBES anchored k18 (after)')
  const q = (n: number, lvl: number) => sim.find((r) => r.n_filters === n && r.true_level === lvl)
  const share = (v?: number) => (v == null ? '–' : `${Math.round(v * 100)} %`)
  const T = d.settings?.tor ?? {}
  const Step = ({ a, b, c }: { a: string; b: React.ReactNode; c: React.ReactNode }) => <tr><td><strong>{a}</strong></td><td>{b}</td><td>{c}</td></tr>
  return (
    <ChartFrame id="xc-proposal" source={source('PassRuleProposal')} exportable={false} provenance="Proposed 3 Nov 2026, not agreed · simulation in run_hips_response_20261103.py stage tor · docs/open-items.md"
      title="Proposal: the pass rule for the Addis city TOR filters"
      subtitle="Status: proposed, not agreed. The team should confirm or change this before anyone reads a filter against FTIR. The saved export contains no Addis city TOR pairs; the rule remains a proposal and must be agreed before evaluation. Once the first ratio is read, any change has to be reported as a change.">
      <table className="census-table">
        <thead><tr><th>Step</th><th>Proposed</th><th>Why</th></tr></thead>
        <tbody>
          <Step a="1. Freeze the setup" b="Before reading any ratio, fix the primary baseline, the baselines shown beside it, k 18 and the spectrum-to-filter map." c="A baseline or map picked after seeing the answer isn't a test. Adama's map is still provisional, so this step has real work in it." />
          <Step a="2. Check composition" b="List OC, EC and OC/EC for every filter, and confirm each TOR filter is paired with a PTFE filter from the same sampling." c="It shows whether these filters look like the Addis filters the question is about, without touching FTIR." />
          <Step a="3. Apply the gates" b={`Count only filters with OC/EC below ${T.ocec_limit ?? 2.5} and TOR EC of ${T.gate_ug ?? 15} µg per filter or more.`} c={`Light, organic filters can't test a heavy-filter shortfall. All ${T.adama_filters ?? 5} Adama pairs stop here.`} />
          <Step a="4. Require enough filters" b="At least 10 eligible filters. Below that, the verdict is “not eligible”." c={`With 5 filters, a true level of 1.0 passes only ${share(q(5, 1)?.pass)} of the time. With 10 it passes ${share(q(10, 1)?.pass)}.`} />
          <Step a="5. Read the level" b="Median FTIR ÷ TOR with a 90 % bootstrap interval, every baseline side by side. Nothing is refitted to TOR." c="One number per baseline, with its uncertainty, and no tuning on the only direct test we have." />
        </tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Verdict</th><th>When</th><th>What we do</th></tr></thead>
        <tbody>
          <Step a="Pass" b="The whole interval sits inside 0.8–1.25." c="Would support the declared level criterion in this eligible paired sample; wider target accuracy, seasons and lots require their own evidence." />
          <Step a="Fail low" b="The whole interval sits below 0.8." c="FTIR really reads low on heavy Addis filters, and HIPS compression doesn't explain it away. Stop and rework the transfer." />
          <Step a="Fail high" b="The whole interval sits above 1.25." c="The baseline overshoots, as anchored did on Adama (lot 245). Check the lot's blanks before anything else." />
          <Step a="Inconclusive" b="The interval crosses 0.8 or 1.25." c="No level claim either way. Report the interval and get more filters." />
          <Step a="Not eligible" b="Fewer than 10 filters pass the gates." c="Don't read a level. Use the filters as a low-loading check only, as with Adama." />
        </tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>For the team to decide</th><th>Proposed</th><th>What the choice changes</th></tr></thead>
        <tbody>
          <Step a="Primary baseline" b="Anchored, with production VIBES and EPO as candidates for review." c="Choose and freeze the primary model before target outcomes are examined. EPO's historical source score cannot inform this choice until the fold-safe rerun is verified." />
          <Step a="Pass band" b="0.8–1.25, equal either side of 1 as a ratio." c="A tighter band means more inconclusive results. A wider one claims less." />
          <Step a="Gates" b={`${T.gate_ug ?? 15} µg per filter and OC/EC below ${T.ocec_limit ?? 2.5}.`} c="Loosening them gives more filters, but each one tests the heavy-filter question less well." />
          <Step a="Number of filters" b="10 at minimum, 20 if we can get them." c={`20 raises the pass rate at a true 1.0 from ${share(q(10, 1)?.pass)} to ${share(q(20, 1)?.pass)}. It doesn't help a level near a band edge, which stays inconclusive either way.`} />
          <Step a="Failing high" b="Added on 3 Nov. The first draft could only fail low." c="Without it, a baseline reading far above TOR would come back inconclusive rather than failed." />
        </tbody>
      </table>
      <Note>This tests the level at one site and nothing else. Simulated on {T.improve_eligible ?? '?'} eligible IMPROVE filters with 10 filters per draw: a true level of 1.0 passes {share(q(10, 1)?.pass)}, 0.6 fails {share(q(10, 0.6)?.fail)}, and 0.8 is inconclusive {share(q(10, 0.8)?.inconclusive)}. Those simulated operating rates are conditional on this source scatter and proposal; they do not establish sufficient target precision or an agreed sample size. The simulation uses IMPROVE's scatter, which may be smaller than a new site's. Every case is in the section on the filters we hold, further down.</Note>
    </ChartFrame>
  )
}

function WhyAnchored({ d }: { d: ContinuumData }) {
  const sp = d.spectra_story as Record<string, number[]>
  const [loading, setLoading] = useSearchParam<'heavy' | 'light'>('xc-why.load', 'heavy', ['heavy', 'light'])
  const trace = (key: string, label: string, color: string, dash?: string, width = 1.5): SpectrumSeries => ({ id: key, wn: sp.wn, y: sp[key], color, label, dash, width, role: 'line' })
  const left = [trace(`raw_${loading}`, 'raw spectrum', INK.muted)]
  const right = [trace(`vibes_${loading}`, 'production VIBES', '#d62728'), trace(`anchored_${loading}`, 'VIBES on PTFE-anchored', '#8c564b', undefined, 2)]
  const fits = rowsOf<Fit>(d.addis_fits)
  const F = (m: string, x: XKey) => fits.find((r) => r.model === m && r.x === x)
  const sb = rowsOf<SourceScore>(d.scoreboard)
  const ad = rowsOf<{ model: string; median_unflagged: number }>(d.adama_summary)
  const inj = rowsOf<{ variant: string; shape: string; value: number }>(d.injection)
  const kept = inj.find((r) => r.variant === 'production (rank 70)' && r.shape === 'nu^1 (BC-like)')?.value
  const board: [string, string, string, string, string][] = [
    ['raw', 'raw k18', 'raw k18', 'keeps all of it, plus the PTFE and lot offsets', 'raw'],
    ['spline', 'spline k18', '', 'removes it by construction', 'spline (AIRSpec)'],
    ['vibes', 'vibes k18', 'VIBES k18 (before)', `gives back ${kept != null ? (kept * 100).toFixed(1) : '<1'} %`, 'production VIBES'],
    ['vibes_deflate', 'vibes_deflate k18', '', 'kept, but lot differences flood in', 'VIBES, smooth shapes removed'],
    ['vibes_anchorhp', 'vibes_anchorhp k18', 'VIBES anchored k18 (after)', 'kept', 'VIBES on PTFE-anchored'],
    ['epo', 'epo k18', 'EPO k18', 'kept', 'anchored + EPO'],
  ]
  return (
    <ChartFrame id="xc-why" source={source('WhyAnchored')} provenance="run_hips_response_20261103.py stage options · workflows/vibes_continuum_20261102.py · k 18"
      title="The anchored background and candidate tradeoffs"
      evidence="Fixed 18 components · heavy source subset: 15 ≤ TOR < 150 µg/filter, OC/EC < 2.5 · source reference: IMPROVE TOR; Addis references: optical or inferred; Adama: provisional paired TOR. EPO source metrics require verified rerun provenance."
      subtitle="Loaded filters can carry broad spectral structure. A broad continuum is a candidate source of carbon-related information; the controls show how preprocessing changes broad shapes without establishing their chemical origin."
      controls={<Segmented label="Addis filters" value={loading} options={['heavy', 'light'] as const} optionLabel={(o) => (o === 'heavy' ? 'heaviest third' : 'lightest third')} onChange={setLoading} />}>
      <FindingsList>
        <li><strong>The problem.</strong> Production VIBES models each spectrum as blank mean + blank components + analyte. The blank components include smooth ramps, so the fit can absorb the BC continuum into the background.</li>
        <li><strong>The idea.</strong> A smooth continuum has no sharp features, but the PTFE bands do. So the blank components are scored only on the high-passed PTFE channels. PTFE and its lot differences are predicted from the bands and removed; the continuum can't move those scores, so it stays.</li>
        <li><strong>Then real VIBES.</strong> VIBES was rerun on Colab on the anchored spectra, on all 16,143 spectra, with no failures. A second variant that just removed smooth shapes from VIBES was tried and lost.</li>
      </FindingsList>
      <FacetGrid items={['raw', 'corrected']} minCell={340} maxCols={2} itemKey={(k) => k} title={(k) => (k === 'raw' ? 'Median raw spectrum' : 'After each background')} render={(k, w) =>
        <SpectrumPlot series={k === 'raw' ? left : right} window={[4000, 500]} width={w} height={260} yLabel="absorbance" zeroLine={k !== 'raw'} yIgnore={[[1050, 1300]]} regions={[]} />} />
      <Legend items={[{ label: 'production VIBES', color: '#d62728' }, { label: 'VIBES on PTFE-anchored', color: '#8c564b' }]} />
      <table className="census-table">
        <thead><tr><th>Baseline (k 18)</th><th>Broad continuum</th><th>IMPROVE median absolute error (µg/filter)</th><th>IMPROVE heavy low-OC/EC, FTIR ÷ TOR</th><th>Addis intercept vs aethalometer</th><th>Addis ÷ inferred TOR comparator</th><th>Adama ÷ TOR (3 provisional pairs)</th></tr></thead>
        <tbody>{board.map(([key, fitKey, adamaKey, cont, name]) => { const s = sb.find((r) => r.model === key)
          return <tr key={key}><td>{name}</td><td>{cont}</td><td>{key === 'epo' && medianAbsoluteError(s, d.settings?.source_evaluation) == null ? 'Withheld: rerun required' : num2(medianAbsoluteError(s, d.settings?.source_evaluation))}</td><td>{num2(sourceScoreValue(s, 'improve_heavy_ratio', d.settings?.source_evaluation))}</td><td>{coefSigned(F(fitKey, 'aeth_ir')?.intercept)} µg/m³</td><td>{num2(F(fitKey, 'tor_equiv')?.ratio)}</td><td>{num2(ad.find((r) => r.model === adamaKey)?.median_unflagged)}</td></tr> })}</tbody>
      </table>
      <Note>Anchoring changes the saved optical comparison and preserves more broad structure. The small, provisional Adama comparison raises a transfer or lot question. The historical EPO source score is withheld; saved all-source target predictions remain exploratory. None of these comparisons establishes the absolute Addis EC level.</Note>
    </ChartFrame>
  )
}

function AethFlow({ d }: { d: ContinuumData }) {
  const fl = rowsOf<{ code: string; n: number; flow_ratio: number; p10: number; p90: number; f1_share: number; bcc_over_bc1: number; flow_status: string }>(d.aeth_flow)
  const sp = rowsOf<{ code: string; x: string; y: string; loglog: number }>(d.aeth_spots)
  const co = rowsOf<{ code: string; x: string; y: string; controls: string; n: number; ratio_p10: number; ratio_p90: number; pct_over_range: number; pct_lo: number; pct_hi: number }>(d.aeth_flow_coef)
  const C = (c: string, x: string, y: string, controls = 'loading') => co.find((r) => r.code === c && r.x === x && r.y === y && r.controls === controls)
  const whole = (v?: number) => (v == null || !Number.isFinite(v) ? '–' : `${Math.round(v) > 0 ? '+' : Math.round(v) < 0 ? '−' : ''}${Math.abs(Math.round(v))}`)
  const signed = (v?: number) => (v == null || !Number.isFinite(v) ? '–' : `${whole(v)} %`)
  const band = (r?: { pct_over_range: number; pct_lo: number; pct_hi: number }) => (r ? `${signed(r.pct_over_range)} (${whole(r.pct_lo)} to ${whole(r.pct_hi)})` : '–')
  const fx = rowsOf<{ months: string; era: string; n: number; flow_ratio: number; hips_over_bcc: number; hips_over_bc1: number; published_over_bcc: number; published_over_bc1: number }>(d.aeth_fix)
  const t = d.settings?.flow_tests ?? {}
  const v = (c: string, x: string, y: string) => sp.find((r) => r.code === c && r.x === x && r.y === y)
  const order = ['ETAD', 'INDH', 'CHTS', 'USPA'].filter((c) => fl.some((r) => r.code === c))
  return (
    <ChartFrame id="xc-flow" source={source('AethFlow')} exportable={false} provenance="run_hips_response_20261103.py stage flow · Flow1 / Flow2 and spot BC from the daily aggregates, overlap-weighted to each filter"
      title="Can we trust the aethalometer? The flow check"
      subtitle="The MA350 flow split should be 1:3 (Flow2:Flow1, or 75:25). Beijing and Pasadena were moved from about 55:45 towards it, Delhi's ratio climbs through its record, and the Addis instrument (MA350-0238) stayed at 55:45. That is why this page uses BCc: the dual-spot compensated record is what's meant to carry the data across a split like this. So the test is whether BCc holds up, and it does. It removes about half of the flow effect inside Addis, and HIPS ÷ BCc stays flat across the fixes elsewhere. HIPS also stays compressed on BCc, spot 1 and spot 2 alike.">
      <table className="census-table">
        <thead><tr><th>Site</th><th>Filters</th><th>Flow1/Flow2 (10–90 %)</th><th>Flow1 share</th><th>Status</th><th>HIPS log-log: BCc · BC1 · BC2</th><th>Anchored FTIR log-log: BCc · BC1 · BC2</th></tr></thead>
        <tbody>{order.map((c) => { const f = fl.find((r) => r.code === c)!
          const three = (y: string) => ['BCc', 'BC1', 'BC2'].map((x) => num2(v(c, x, y)?.loglog)).join(' · ')
          return <tr key={c}><td>{SITE_NAME[c]}</td><td>{f.n}</td><td>{num2(f.flow_ratio)} ({num2(f.p10)}–{num2(f.p90)})</td><td>{num2(f.f1_share, 0)} %</td><td>{f.flow_status}</td><td>{three('hips')}</td><td>{three('anch')}</td></tr> })}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Change in filter ÷ aethalometer across each site's flow range (90 %), adjusted for loading</th><th>Flow1/Flow2</th><th>HIPS ÷ BCc</th><th>HIPS ÷ BC1</th><th>Published FTIR ÷ BCc</th><th>Published FTIR ÷ BC1</th><th>Anchored FTIR ÷ BCc</th><th>Anchored FTIR ÷ BC1</th><th>BCc ÷ BC1</th></tr></thead>
        <tbody>{['ETAD', 'USPA', 'INDH', 'CHTS'].map((c) => {
          const h = C(c, 'BCc', 'hips'), k = t.compensation_vs_flow?.[c]
          return <tr key={c}><td>{SITE_NAME[c]}</td><td>{num2(h?.ratio_p10)}–{num2(h?.ratio_p90)}</td><td>{band(h)}</td><td>{band(C(c, 'BC1', 'hips'))}</td><td>{band(C(c, 'BCc', 'pub'))}</td><td>{band(C(c, 'BC1', 'pub'))}</td><td>{band(C(c, 'BCc', 'anch'))}</td><td>{band(C(c, 'BC1', 'anch'))}</td><td>{k ? `${signed(k[0])} (${whole(k[1])} to ${whole(k[2])})` : '–'}</td></tr> })}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Pasadena, the only site with filters both sides of its fix</th><th>Filters</th><th>Flow1/Flow2</th><th>HIPS ÷ BCc</th><th>HIPS ÷ BC1</th><th>Published FTIR ÷ BCc</th><th>Published FTIR ÷ BC1</th></tr></thead>
        <tbody>{fx.map((r) => <tr key={r.months + r.era}><td>{r.era}, {r.months}</td><td>{r.n}</td><td>{num2(r.flow_ratio)}</td><td>{num2(r.hips_over_bcc)}</td><td>{num2(r.hips_over_bc1)}</td><td>{num2(r.published_over_bcc)}</td><td>{num2(r.published_over_bc1)}</td></tr>)}</tbody>
      </table>
      <Note>BCc does its job. Inside Addis, going from spot 1 to BCc takes the flow effect against published FTIR from {signed(C('ETAD', 'BC1', 'pub')?.pct_over_range)} to {signed(C('ETAD', 'BCc', 'pub')?.pct_over_range)}, against anchored FTIR from {signed(C('ETAD', 'BC1', 'anch')?.pct_over_range)} to {signed(C('ETAD', 'BCc', 'anch')?.pct_over_range)}, and against HIPS from {signed(C('ETAD', 'BC1', 'hips')?.pct_over_range)} to {signed(C('ETAD', 'BCc', 'hips')?.pct_over_range)}. The compensation moves that way with no filter involved: BCc ÷ BC1 changes by {signed(t.compensation_vs_flow?.ETAD?.[0])} over the same range. Across the fixes, HIPS ÷ BCc doesn't move: {signed(C('USPA', 'BCc', 'hips')?.pct_over_range)} at Pasadena, {signed(C('INDH', 'BCc', 'hips')?.pct_over_range)} at Delhi, {signed(C('CHTS', 'BCc', 'hips')?.pct_over_range)} at Beijing, each with a wide interval.</Note>
      <Note>What's left. The Pasadena table shows HIPS ÷ BCc going from {num2(fx.find((r) => r.era === 'before fix')?.hips_over_bcc)} to {num2(fx.find((r) => r.era === 'after fix')?.hips_over_bcc)}, but those are raw medians from only {fx.find((r) => r.era === 'before fix')?.n} filters before the fix, and the shift goes once loading is accounted for. BCc still shows {signed(C('ETAD', 'BCc', 'hips')?.pct_over_range)} against HIPS inside Addis, so the effect there isn't zero, though it's small and can't make a factor-of-two compression. FTIR moves in opposite directions at Pasadena (published {signed(C('USPA', 'BCc', 'pub')?.pct_over_range)}, anchored {signed(C('USPA', 'BCc', 'anch')?.pct_over_range)}), which points to date or filter lot, not the split. Addis also relies on a large compensation (BCc ÷ BC1 {num2(t.bcc_over_bc1?.[0])}–{num2(t.bcc_over_bc1?.[2])}, Pasadena about {num2(fl.find((r) => r.code === 'USPA')?.bcc_over_bc1)}), and Beijing reads low against everything for a reason flow doesn't explain. So BCc is good for shape, while its absolute level stays a calibration question.</Note>
    </ChartFrame>
  )
}

const CURVE_COLOR: Record<string, string> = {
  'IMPROVE: HIPS / TOR': '#1b1b1b', 'IMPROVE: HIPS / TOR, OC/EC < 2.5': '#d35400', 'IMPROVE: HIPS / FTIR, out of site': '#7f8c8d',
  'SPARTAN without Ethiopia: HIPS / anchored FTIR': '#8c564b', 'SPARTAN without Ethiopia: HIPS / published FTIR': '#1f77b4',
}

function HipsNetwork({ d }: { d: ContinuumData }) {
  const rows = rowsOf<{ curve: string; lo: number; hi: number; n: number; ratio: number; ratio_lo: number; ratio_hi: number }>(d.tau_curves)
  const names = Object.keys(CURVE_COLOR).filter((c) => rows.some((r) => r.curve === c))
  const lg = useLegend()
  const fit = rowsOf<{ fit: string; n: number; sites: number; tau0: number; tau0_lo: number; tau0_hi: number; c: number; c_lo: number; c_hi: number }>(d.response_fit)
  const sites = rowsOf<{ code: string; n: number; tau_median: number; share_over_0p3: number; share_over_1: number; correction_median: number; raw_intercept_rel: number; intercept_rel: number }>(d.response_sites)
  const ra = rowsOf<{ fit: string; model: string; raw_slope: number; raw_intercept: number; slope: number; intercept: number; raw_ratio: number; ratio: number; correction_median: number; hips_loglog_aeth: number; raw_loglog_aeth: number }>(d.response_addis)
  const series: SweepSeries[] = names.filter((c) => lg.show(c)).map((c) => ({ id: c, label: c, color: CURVE_COLOR[c], group: c, width: c === names[0] ? 2.5 : 1.5, interval: 'band',
    points: rows.filter((r) => r.curve === c && r.n >= 5).map((r) => ({ x: Math.sqrt(Math.max(r.lo, .05) * r.hi), y: r.ratio, lo: r.ratio_lo, hi: r.ratio_hi, tip: [c, `τ ${r.lo}–${r.hi}: ${f2(r.ratio)} of the 0.2–0.3 value (n ${r.n})`] })) }))
  const p = fit[0]
  const resp = p ? [{ id: 'fit', label: 'response fitted to IMPROVE TOR', color: INK.accent, dash: '6 4', group: 'fit', marks: false, points: [.05, .1, .2, num(p.tau0), .4, .6, .8, 1, 1.3, 1.6, 2.2].map((x) => ({ x, y: 1 / (1 + Math.max(x - num(p.tau0), 0) / num(p.c)) })) } as SweepSeries] : []
  const heavy = [...sites].sort((a, b) => num(b.tau_median) - num(a.tau_median)).slice(0, 12)
  const R = d.settings?.response ?? {}
  return (
    <ChartFrame id="xc-network" evidence="IMPROVE-derived response sensitivity · x = geometric mean of HIPS and reference optical depth · transferred model, not Addis TOR or an instrument reliability cutoff" source={source('HipsNetwork')} provenance={`run_hips_response_20261103.py stage network · within sites · bins on √(τ HIPS × τ reference), so neither side is selected on · shaded 90 %`}
      title="How far can HIPS be trusted? Every SPARTAN and IMPROVE filter"
      subtitle={`Each curve is HIPS ÷ a reference at one optical depth, relative to its value at τ 0.2–0.3, compared within each site. The fit assumes an approximately flat low-depth response as its control. Above it, HIPS falls behind in both networks, against thermal TOR as well as FTIR. The black IMPROVE TOR curve uses no FTIR and no aethalometer. Fitted to that alone: the empirical response bends at the geometric-mean optical-depth coordinate ${num2(p?.tau0)} (80 % ${num2(p?.tau0_lo)}–${num2(p?.tau0_hi)}).`}>
      <SweepPlot series={[...series, ...resp]} xScale="log" xDomain={[.06, 2.4]} yDomain={[.3, 1.25]} reference={1} xLabel="optical depth τ (geometric mean of HIPS and the reference)" yLabel="HIPS ÷ reference, relative to τ 0.2–0.3" height={340} xMarkers={[{ at: 0.3, label: 'τ 0.3' }, { at: 0.96, label: 'Addis median' }]} dim={lg.dim} />
      <Legend {...lg.props} items={names.map((c) => ({ label: c, color: CURVE_COLOR[c] }))} />
      <table className="census-table">
        <thead><tr><th>Addis, k 18 (fit used)</th><th>HIPS correction</th><th>HIPS log-log vs aethalometer</th><th>Against raw HIPS</th><th>Against corrected HIPS</th><th>Median FTIR ÷ corrected HIPS</th></tr></thead>
        <tbody>{ra.map((r) => <tr key={r.fit + r.model}><td>{r.model} ({r.fit})</td><td>×{num2(r.correction_median)}</td><td>{num2(r.raw_loglog_aeth)} → {num2(r.hips_loglog_aeth)}</td><td>{coef(r.raw_slope)}x {coefSigned(r.raw_intercept)}</td><td>{coef(r.slope)}x {coefSigned(r.intercept)}</td><td>{num2(r.ratio)}</td></tr>)}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Site (heaviest 12)</th><th>Filters</th><th>Median optical depth</th><th>Above 0.3</th><th>Above 1</th><th>HIPS correction</th><th>FTIR intercept share, raw → corrected</th></tr></thead>
        <tbody>{heavy.map((r) => <tr key={r.code}><td>{r.code}</td><td>{r.n}</td><td>{num2(r.tau_median)}</td><td>{Math.round(num(r.share_over_0p3) * 100)} %</td><td>{Math.round(num(r.share_over_1) * 100)} %</td><td>×{num2(r.correction_median)}</td><td>{pct(num(r.raw_intercept_rel))} → {pct(num(r.intercept_rel))}</td></tr>)}</tbody>
      </table>
      <Note>The response assumes proportionality below its fitted bend. This empirical feature is not an instrument-certified reliability boundary or an operational rule for flagging raw HIPS optical depth. Across sites with median τ above 0.35, the typical intercept share falls from {pct(num(R.heavy_abs_intercept_raw)).replace('+', '')} to {pct(num(R.heavy_abs_intercept_corrected)).replace('+', '')}. Two limits: the level of corrected HIPS assumes MAC 10 holds, which Adama questions, and TOR itself may misbehave on very heavy filters, so the fitted curve is less constrained above τ 1 (IMPROVE has {R.improve_n_over_1 ?? '?'} filters above that). The low-OC/EC fit is closer to Addis's composition and corrects less. The correction also overshoots at Johannesburg, Pretoria and Mexico City (intercept share +27 to +34 %), and it takes the Addis FTIR slope below 1, so one curve doesn't fit every site and the shape isn't fully settled.</Note>
    </ChartFrame>
  )
}

function TorTest({ d }: { d: ContinuumData }) {
  const a = rowsOf<Record<string, any>>(d.adama)
  const sum = rowsOf<{ model: string; median_all: number; median_unflagged: number; mapped_lo: number; mapped_hi: number; mean_diff_ug_unflagged: number }>(d.adama_summary)
  const sim = rowsOf<{ model: string; n_filters: number; true_level: number; pass: number; fail: number; inconclusive: number; observed_median: number }>(d.rule_sim).filter((r) => r.model === 'VIBES anchored k18 (after)')
  const T = d.settings?.tor ?? {}
  const models = ['VIBES k18 (before)', 'VIBES anchored k18 (after)', 'EPO k18', 'raw k18']
  const p = (v: number) => `${Math.round(num(v) * 100)} %`
  return (
    <ChartFrame id="xc-tortest" source={source('TorTest')} exportable={false} provenance="run_hips_response_20261103.py stage tor · Adama batch 54 · spectrum map provisional · real VIBES on the anchored spectra"
      title="Testing the pass rule: the TOR filters we actually hold"
      subtitle={`This saved export contains no Addis city TOR pairs. We hold ${T.adama_filters ?? 5} Adama pairs (a PTFE filter and a date-paired quartz TOR filter). Under the proposed rule they aren't a valid test: 0 of 5 reach ${T.gate_ug ?? 15} µg per filter (the heaviest is ${num2(T.adama_max_tor_ug, 1)} µg) and their OC/EC is about ${num2(T.adama_median_ocec, 1)}, not below 2.5. So the rule returns “not eligible”, which is the right answer. They're still worth reading as a low-loading check, and to inspect the proposal's scope.`}>
      <table className="census-table">
        <thead><tr><th>Date</th><th>PTFE</th><th>Flag</th><th>Optical depth</th><th>TOR EC</th><th>HIPS BC</th><th>Fabs ÷ TOR</th>{models.map((m) => <th key={m}>{m}</th>)}</tr></thead>
        <tbody>{a.map((r) => <tr key={r.ptfe_id}><td>{r.date}</td><td>{r.ptfe_id}</td><td>{r.flag ?? '–'}</td><td>{num2(r.tau)}</td><td>{num2(r.ec_tr_ugm3)}</td><td>{num2(r.hips_bc)}</td><td>{num2(r.fabs_over_tor, 1)}</td>{models.map((m) => <td key={m}>{num2(r[m])}</td>)}</tr>)}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Model ÷ TOR</th><th>Median, 3 unflagged</th><th>Across every spectrum map</th><th>Mean difference</th></tr></thead>
        <tbody>{sum.map((r) => <tr key={r.model}><td>{r.model}</td><td>{num2(r.median_unflagged)}</td><td>{num2(r.mapped_lo)}–{num2(r.mapped_hi)}</td><td>{coefSigned(r.mean_diff_ug_unflagged)} µg per filter</td></tr>)}</tbody>
      </table>
      <table className="census-table">
        <thead><tr><th>Rule on simulated draws (IMPROVE, {T.improve_eligible ?? '?'} eligible filters, out of site)</th><th>Pass</th><th>Fail</th><th>Inconclusive</th></tr></thead>
        <tbody>{sim.map((r) => <tr key={`${r.n_filters}${r.true_level}`}><td>{r.n_filters} filters, true level {num2(r.true_level, 1)}</td><td>{p(r.pass)}</td><td>{p(r.fail)}</td><td>{p(r.inconclusive)}</td></tr>)}</tbody>
      </table>
      <Note>What Adama does tell us: all five filters give Fabs ÷ TOR near {num2(T.fabs_over_tor_median, 1)} m²/g, well above the assumed 10, at optical depths from {num2(T.adama_tau_range?.[0])} to {num2(T.adama_tau_range?.[1])}. That doesn't depend on the spectrum map. If it holds at Addis, HIPS at MAC 10 overstates TOR EC at low optical depth, which offsets part of the compression. It's five filters, on another site and sampler, with date-paired quartz, so it's a lead rather than a result. What the simulation tells us: 10 filters give a true 1.0 and a true 0.6 clear, different verdicts. A level near a band edge (0.8 or 1.3) mostly comes back inconclusive, and 20 filters don't change that, because the interval then sits on the edge. That's the rule declining to call a borderline level, not a shortage of filters.</Note>
    </ChartFrame>
  )
}

function SeasonAeth({ d }: { d: ContinuumData }) {
  const sea = rowsOf<{ y: string; n: number; loglog: number; bega_vs_belg: number; bega_se: number; kiremt_vs_belg: number; kiremt_se: number }>(d.season_aeth)
  const lot = rowsOf<{ model: string; lot: string; season: string; n: number; ratio: number }>(d.lot_vs_aeth).filter((r) => r.model === 'VIBES anchored k18 (after)')
  const rows = sea.map((r) => ({ key: r.y, label: r.y, bold: r.y.includes('anchored') || r.y === 'HIPS BC', color: r.y === 'HIPS BC' ? INK.text : undefined,
    values: { bega: { v: num(r.bega_vs_belg) * 100, lo: (num(r.bega_vs_belg) - 1.645 * num(r.bega_se)) * 100, hi: (num(r.bega_vs_belg) + 1.645 * num(r.bega_se)) * 100 },
      kiremt: { v: num(r.kiremt_vs_belg) * 100, lo: (num(r.kiremt_vs_belg) - 1.645 * num(r.kiremt_se)) * 100, hi: (num(r.kiremt_vs_belg) + 1.645 * num(r.kiremt_se)) * 100 } } }))
  return (
    <ChartFrame id="xc-season-aeth" source={source('SeasonAeth')} provenance="run_intercept_fixes_20261103.py stage stress · log y on log aethalometer + season · bars 90 %"
      title="Seasonal contrasts against the aethalometer"
      evidence={`Bega Oct–Jan; Belg Feb–May (reference); Kiremt Jun–Sep · saved per-model n: ${sea.map((r) => `${r.y} ${r.n}`).join("; ")} · optical comparison, 90% intervals`}
      subtitle="The Season page measures the season effect against the TOR yardstick, which is built from HIPS. Since HIPS is compressed, that effect might have been a side effect of HIPS. So this repeats it with the aethalometer as the yardstick: after taking out each day's aethalometer BC, does a season still read high or low? FTIR does, and HIPS doesn't. This supports descriptive seasonal structure with a second optical comparator; it does not identify the cause or validate seasonal EC corrections.">
      <Forest rows={rows} labelWidth={240} panels={[{ key: 'bega', label: 'Bega vs Belg (%)', ref: 0, fmt: (v: number) => `${v.toFixed(0)} %`, domain: [-30, 30] }, { key: 'kiremt', label: 'Kiremt vs Belg (%)', ref: 0, fmt: (v: number) => `${v.toFixed(0)} %`, domain: [-30, 30] }]} />
      <Note>Lot check for anchored FTIR ÷ aethalometer: {lot.map((r) => `lot ${String(r.lot).replace('.0', '')} ${r.season} ${num2(r.ratio)} (n ${r.n})`).join(', ')}. Bega reads low in both lots, so it isn't only the lot 248 confound.</Note>
    </ChartFrame>
  )
}
