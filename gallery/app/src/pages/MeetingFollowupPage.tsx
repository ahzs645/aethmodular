import { RegressionLines, RegressionReadout, RegressionBand, RegressionGroupLines, RegressionGroupReadouts, regressionGroupReadoutLayout } from '@/components/RegressionOverlay'
import { groupedRegressions } from '@/lib/siteRegression'
import { monthBlocks } from '@/lib/regressionCI'
import { regression, type RegressionStats } from '@/lib/stats'
import { EstimatorViews, FIT_STYLE, useEstimator, useRegressions, fitsFor, FitGroupingControl } from '@/lib/regressionPrefs'
import { useId, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Select, Segmented } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { FullRangeVibesStatus } from '@/components/FullRangeVibesStatus'
import { MeetingCalibrationDetails } from '@/components/MeetingCalibrationDetails'
import { SeasonalComposition } from '@/components/SeasonalComposition'
import { SpecCurve } from '@/charts/calibration/SpecCurve'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { baseFilterId, useHighlight, type FilterRecord } from '@/lib/highlight'
import { FONT, INK, RAMP_DIVERGING } from '@/lib/theme'
import { ColorLegend } from '@/components/ColorLegend'
import { CV_PROTOCOL_SHORT, PREPROCESSING } from '@/lib/labels'
import { BASE as SIM_BASE, Traces, fetchBinary } from '@/lib/similarity'
import type { FilterRow, MetaFile } from '@/lib/types'

/**
 * Click-through for the 2026-09-17 Ann/Satoshi meeting follow-up. Every number
 * comes from research/ftir_hips_chem/output/tables/meeting_followup_20260917
 * via gallery/data/export_meeting_followup.py. Crossplot overlays refit complete
 * pairs with the saved error ratio; the underlying calibrations stay fixed.
 */

const URL_BASE = `${import.meta.env.BASE_URL}data/`
type Row = Record<string, any>
type Cols = Record<string, (number | string | null)[]>
interface MeetingData {
  schema_version: number
  generated_from: string
  summary: Row
  season_colors: Record<string, string>
  groups: string[]
  seasons: string[]
  fits: Row[]
  stats: Row[]
  analogs: Cols
  addis: Cols
  repeats: Cols
  stitched_repeats: Cols
  op_summary: Row[]
  op_terciles: Row[]
  op_pool_rho: number
  reproduction: Row[]
  overlap: Row[]
  mac_hist: Record<string, any>
  spectra: Record<string, any>
  grid: Cols | null
  grid_n: number
  grid_summary: Row[]
  headlines: Record<string, Record<string, [number, number, number]>>
}

/** Baselines in the export (AIRSpec, VIBES, and the full-range VIBES variants once assembled); set when the data load. */
let METHODS: string[] = ['AIRSpec', 'VIBES']
/** Baseline picker: data keys stay AIRSpec/VIBES/…, readers see the agreed names. */
function MethodSeg({ value, onChange, label = 'Baseline' }: { value: string; onChange: (v: string) => void; label?: string }) {
  const names = METHODS.map((m) => PREPROCESSING[m] ?? m)
  return <Segmented label={label} value={(PREPROCESSING[value] ?? value) as any} options={names as any}
    onChange={(v: string) => onChange(METHODS[names.indexOf(v)] ?? 'AIRSpec')} />
}
const methodName = (m: string) => PREPROCESSING[m] ?? m
const MODELS = ['all-IMPROVE lot 251', 'all-IMPROVE both lots (frozen Colab)'] as const
const modelLabel = (value: string) => ({
  'all-IMPROVE lot 251': 'Calibration pool · lot 251',
  'all-IMPROVE both lots (frozen Colab)': 'Full calibration pool · both lots',
  'seasonal calibration': 'Seasonal calibration',
}[value] ?? value)
function ModelSeg({label, value, options, onChange}: {label:string; value:string; options:readonly string[]; onChange:(v:string)=>void}) {
  return <Segmented label={label} value={modelLabel(value)} options={options.map(modelLabel)} onChange={v=>onChange(options.find(o=>modelLabel(o)===v)??options[0])} />
}
const MAC_LABEL = 'HIPS Fabs / MAC 10 (µg/m³)'
const f2 = (v: any) => (v == null || !Number.isFinite(+v) ? '—' : (+v).toFixed(2))
const signed = (v: any) => (v == null || !Number.isFinite(+v) ? '—' : `${+v >= 0 ? '+' : '−'}${Math.abs(+v).toFixed(2)}`)
const shortSeason = (g: string) => g.split(' ')[0]


export function MeetingFollowupPage({ galleryRows = [], galleryMeta }: { galleryRows?: FilterRow[]; galleryMeta?: MetaFile } = {}) {
  const [data, setData] = useState<MeetingData | null>(null)
  const [error, setError] = useState('')
  const [loadedVersion, setLoadedVersion] = useState<string | null>(null)
  const [archiveMode, setArchiveMode] = useSearchParam<string>('marchive', 'hide', ['hide', 'show'])
  const [version, setVersion] = useSearchParam<string>('mversion', 'current', ['current', 'earlier'])
  const showArchived = version === 'earlier' && archiveMode === 'show'
  useEffect(() => {
    const c = new AbortController()
    setData(null); setError('')
    fetch(URL_BASE + (version === 'current' ? 'meeting-r70/' : 'meeting/') + 'followup_20260917.json', { signal: c.signal })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: MeetingData) => { if (d.schema_version !== 1 || (version === 'current' && d.summary.background_rank !== 70)) throw new Error('unsupported data version'); setData(d); setLoadedVersion(version) })
      .catch((e) => { if (e.name !== 'AbortError') setError(String(e)) })
    return () => c.abort()
  }, [version])
  if (error) return <ChartFrame title="Meeting follow-up" exportable={false}><Empty>Could not load the export: {error}. Run <code>uv run --no-sync python gallery/data/export_meeting_followup.py</code>.</Empty></ChartFrame>
  if (!data || loadedVersion !== version) return <p>Loading the follow up meeting…</p>
  METHODS = (data.summary.methods ?? ['AIRSpec', 'VIBES']).filter((m: string) => showArchived || m !== 'VIBES')
  return <MeetingView key={version + String(showArchived)} data={data} galleryRows={galleryRows} galleryMeta={galleryMeta}
    version={version} onVersion={setVersion} showArchived={showArchived} onShowArchived={v => setArchiveMode(v ? 'show' : 'hide')} />
}

// ------------------------------------------------------------------ data helpers
function useIndex(cols: Cols, keys: string[]) {
  return useMemo(() => {
    const idx = new Map<string, number[]>()
    const n = cols[keys[0]].length
    for (let i = 0; i < n; i++) {
      const k = keys.map((c) => String(cols[c][i])).join('|')
      const arr = idx.get(k)
      if (arr) arr.push(i)
      else idx.set(k, [i])
    }
    return idx
  }, [cols, keys])
}

function statRow(stats: Row[], q: Row): Row | undefined {
  return stats.find((s) => Object.entries(q).every(([k, v]) => s[k] === v))
}

interface Pt { x: number; y: number; color: string; hollow?: boolean; tip: string[]; id?: string; rec?: FilterRecord }

// ------------------------------------------------------------------ page
function MeetingView({ data, galleryRows, galleryMeta, version, onVersion, showArchived, onShowArchived }: {
  data: MeetingData; galleryRows: FilterRow[]; galleryMeta?: MetaFile; version: string; onVersion: (v: string) => void; showArchived: boolean; onShowArchived: (v: boolean) => void
}) {
  const hl = useHighlight()
  // a clicked filter stays pinned on every chart until empty plot space is clicked or Esc
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') hl.setPinned(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [hl])
  return (
    <PageToc>
      <ExplainOnly><FullRangeVibesStatus /></ExplainOnly>
      <ChartFrame title="Analysis version" exportable={false} controls={<Select label="VIBES analysis" value={version} options={['current', 'earlier']} onChange={onVersion} optionLabel={v => v === 'current' ? 'Completed · rank 70' : 'Archived · preliminary rank 30'} />}>
        <p>{version === 'current' ? 'All seasonal, combined-set, calibration and specification-grid VIBES models below use the completed rank-70 correction. Full and shared prediction windows are separate options.' : 'Earlier seasonal and specification results retain their original preliminary rank-30 corrections. These archived models have not been overwritten.'}</p>
      </ChartFrame>
      <ExplainOnly>
      <ChartFrame title="Baseline versions" exportable={false}>
        <p>The VIBES fit restricted to 4000–1425 cm⁻¹ is archived. Current VIBES variants fit the full available 4000–500 cm⁻¹ grid. The narrower prediction variant is a sensitivity comparison: it uses the corrected 4000–1425 channels to test the effect of adding lower-wavenumber information. Archived results and their mixed specification grid remain available below.</p>
      </ChartFrame></ExplainOnly>
      {version === 'earlier' && <label className="control"><input type="checkbox" checked={showArchived} onChange={e => onShowArchived(e.target.checked)} /> Show archived 4000–1425 fit</label>}
      <CohortCoverage data={data} galleryRows={galleryRows} />
      <SeasonalComposition rows={galleryRows} meta={galleryMeta} />
      <ExplainOnly><Answers data={data} /></ExplainOnly>
      <ExplainOnly><Levers data={data} /></ExplainOnly>
      <EstimatorViews>{method => <StartingPoint data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews available={['deming', 'york']}>{() => <RepeatStrip data={data} />}</EstimatorViews>
      <EstimatorViews>{method => <ThreePlots data={projectMeeting(data, method)} />}</EstimatorViews>
      <BaselineCompare data={data} />
      <MacEnvelope data={data} />
      <ExplainOnly><OpStep data={data} /></ExplainOnly>
      <EstimatorViews>{method => <Seasonality data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews>{method => <Stitched data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews available={['deming', 'york']}>{() => <SlopeInterceptMap data={data} />}</EstimatorViews>
      <EstimatorViews>{method => <Combined data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews available={['deming', 'york']}>{() => <SpecCurve
        id="meeting-spec-curve"
        src={data.summary.spec_curve_export ?? 'meeting/spec_curve_20260917.json'}
        title={`Specification curve — ${showArchived ? 'active and archived' : 'active'} baselines from the meeting grid`}
        excludePreprocessing={showArchived ? [] : ['vibes']}
        defaultComparison="all"
        subtitle={(f) => `All ${f ? f.n.toLocaleString() + ' ' : ''}baselined Addis specifications after the 17 Sep levers (spline and VIBES baselines; CO₂ and >3600 cm⁻¹ excluded from selection; site-grouped or interleaved cross-validation; cohort sizes every 5 filters; number of PLS factors by each protocol's rule), sorted by Deming intercept at MAC 10, with the analytic choices underneath: a darker strip means more of the specifications at that position used the choice. Click a choice to keep only specifications that use it; click several to intersect. Red points use the selected highlight criteria; by default they pass every saved guardrail${f ? ': ' + f.guardrails : ''}.`}
        provenance={version === 'current' ? 'Completed rank-70 migration · category_grid.jsonl · original specification guardrails' : 'Archived meeting_followup_20260917/category_grid.jsonl · original specification guardrails'}
      />}</EstimatorViews>
      <EstimatorViews available={['deming', 'york']}>{() => <GridStep data={data} />}</EstimatorViews>
      <ExplainOnly><NextSteps data={data} /></ExplainOnly>
    </PageToc>
  )
}

// ------------------------------------------------------------------ 1. answers
function Answers({ data }: { data: MeetingData }) {
  const h = (m: string, g: string) => data.headlines[`${m}|${g}`]
  const band = (t: [number, number, number]) => <>{f2(t[0])} <span style={{ color: INK.muted }}>[{f2(t[1])}–{f2(t[2])}]</span></>
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Answers" title="Short answers — medians over 100 site-grouped cross-validation splits per season, spline baseline, CO₂ + >3600 cm⁻¹ excluded from selection" exportable={false}>
      <div style={{ overflowX: 'auto' }}>
        <table className="census-table">
          <thead><tr><th>Season</th><th>Test set (Addis): FTIR vs Fabs/10</th><th>IMPROVE cross-validation: FTIR vs Fabs/10</th><th>IMPROVE cross-validation: TOR EC vs Fabs/10</th><th>IMPROVE cross-validation: FTIR vs TOR</th><th>PLS factors</th></tr></thead>
          <tbody>{data.seasons.map((g) => {
            const r = h('AIRSpec', g)
            return <tr key={g}><td style={{ color: data.season_colors[g], fontWeight: 600 }}>{g}</td><td>{band(r.addis_slope)}</td><td>{band(r.analog_ftir_fabs)}</td><td>{band(r.analog_tor_fabs)}</td><td>{band(r.analog_ftir_tor)}</td><td>{band(r.k)}</td></tr>
          })}</tbody>
        </table>
      </div>
      <ol style={{ lineHeight: 1.55, fontSize: 14 }}>
        <li><strong>The analogs are mispredicted the way Addis is: same direction, a bit less strongly.</strong> Cross-validation analog slope vs Addis test set slope: {data.seasons.map((g, j) => <span key={g}>{j ? '; ' : ''}{shortSeason(g)} {f2(h('AIRSpec', g).analog_ftir_fabs[0])} vs {f2(h('AIRSpec', g).addis_slope[0])}</span>)}. That is Satoshi's guess: a similar but smaller systematic bias.</li>
        <li><strong>In Dry, part of the bias is already in the references.</strong> For Dry analogs, TOR EC vs Fabs/10 has slope {f2(h('AIRSpec', 'Dry (Oct-Feb)').analog_tor_fabs[0])} (TOR low for its absorption), and FTIR under-reads TOR on top of that ({f2(h('AIRSpec', 'Dry (Oct-Feb)').analog_ftir_tor[0])}). Training on TOR EC bakes the first part in. In Belg and Kiremt, TOR vs Fabs/10 is above 1 ({f2(h('AIRSpec', 'Belg (Mar-May)').analog_tor_fabs[0])}, {f2(h('AIRSpec', 'Kiremt (Jun-Sep)').analog_tor_fabs[0])}), so there the shortfall is FTIR under-reading TOR.</li>
        <li><strong>OP does not explain the TOR–HIPS spread.</strong> Across the IMPROVE pool the MAC ratio is flat in OP fraction (Spearman ρ = {f2(data.op_pool_rho)}); see the OP section.</li>
        <li><strong>The seasonal calibrations are fragile.</strong> Swapping ~12 of 500 analogs moves Kiremt from k = 18 to k = 5; the 10–90 % bands in the table are the honest spread.</li>
        {METHODS.includes('VIBES') && <li><strong>In the archived truncated VIBES fit, the analogs stop mirroring Addis.</strong> Addis slopes ({data.seasons.map((g, j) => <span key={g}>{j ? ', ' : ''}{shortSeason(g)} {f2(h('VIBES', g).addis_slope[0])}</span>)}) no longer track its cross-validation analogs ({data.seasons.map((g, j) => <span key={g}>{j ? ', ' : ''}{f2(h('VIBES', g).analog_ftir_fabs[0])}</span>)}): the wet seasons overshoot 1 while Dry stays low.</li>}
      </ol>
      <Note>Deming error ratios come from measurement uncertainties: Addis λ* {f2(data.summary.lambda)} (HIPS_Uncertainty vs IMPROVE cross-validation RMSE); IMPROVE FTIR panels use each calibration's cross-validation RMSE against a {(100 * (data.summary.ref_rel_uncertainty ?? 0)).toFixed(1)} % reference uncertainty (ETAD HIPS, the only one on file), so λ is 50–100: almost all the error is assigned to FTIR. Analog concentrations are low (median TOR EC 0.1–0.4 µg/m³) against Addis Fabs/10 of 4–6 µg/m³. Fabs/10 is an optical proxy, not thermal EC.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. levers
function Levers({ data }: { data: MeetingData }) {
  const s = data.summary
  const base = statRow(data.stats, { fit_id: 'allimprove|AIRSpec|all-IMPROVE both lots (frozen Colab)', population: 'addis', eval_group: 'All Addis' })
  return (
    <>
      <ChartFrame title="Levers fixed at the meeting — and the meeting numbers reproduced" exportable={false}>
        <table className="census-table"><tbody>
          <tr><td>Spectra</td><td>Baseline-corrected only: spline baseline (AIRSpec) and VIBES baseline in parallel (raw spectra and second derivative dropped)</td></tr>
          <tr><td>Selection exclusions</td><td>CO₂ {s.locked_mask?.[0]}–2500 cm⁻¹ and everything above {s.locked_mask?.[1]} cm⁻¹, same for both baselines</td></tr>
          <tr><td>Cross-validation</td><td>5-site grouped (site-disjoint 80/20 split) and 10-fold interleaved (the app)</td></tr>
          <tr><td>Analogs</td><td>Top {s.n_analogs} IMPROVE filters by Pearson r to each season's median Addis spectrum</td></tr>
          <tr><td>Crossplots</td><td>FTIR EC vs HIPS Fabs / <strong>MAC 10</strong>, Deming with λ* = {f2(s.lambda)} (HIPS uncertainty vs IMPROVE cross-validation RMSE)</td></tr>
          <tr><td>Addis set</td><td>{String((s.notes ?? [])[0] ?? '')}</td></tr>
        </tbody></table>
      </ChartFrame>
      <ChartFrame title="The meeting's numbers reproduce — slide values vs refits here of the same cohorts" exportable={false}>
        <table className="census-table">
          <thead><tr><th>Group</th><th>Slide k</th><th>Slide slope / intercept (n)</th><th>Refit here (n)</th><th>Cohort filters in frozen run</th></tr></thead>
          <tbody>{data.reproduction.map((r) => <tr key={r.group}><td>{r.group}</td><td>{r.slide_k}</td><td>{f2(r.slide_slope)} / {signed(r.slide_intercept)} ({r.slide_n_addis})</td><td>{f2(r.here_slope)} / {signed(r.here_intercept)} ({r.here_n_addis})</td><td>{r.cohort_in_frozen_run} / 500</td></tr>)}
            {base && <tr><td>Base case (all-IMPROVE, frozen)</td><td>{base.k}</td><td>0.47 / −0.78 (meeting)</td><td>{f2(base.deming_slope)} / {signed(base.deming_intercept)} ({base.n})</td><td>full pool</td></tr>}
          </tbody>
        </table>
        <Note>Differences come from the six Addis filters absent from the frozen run and the 6–12 analogs per season that failed the run's eligibility audit. Re-selecting from scratch with the same recipe moves Kiremt's k from 18 to 5, which is why later steps report 100 repeated splits.</Note>
      </ChartFrame>
    </>
  )
}

function CohortCoverage({ data, galleryRows }: { data: MeetingData; galleryRows: FilterRow[] }) {
  const gallery = galleryRows.filter(r => r.site === 'Addis Ababa' && !r.excluded)
  const a = data.addis
  const indices = a.fit_id.map((_, i) => i)
  const ids = (ii: number[]) => new Set(ii.map(i => baseFilterId(String(a.ExternalFilterId[i]))))
  const complete = (i: number) => typeof a.fabs10[i] === 'number' && Number.isFinite(a.fabs10[i]) &&
    typeof a.ftir_ec_ugm3[i] === 'number' && Number.isFinite(a.ftir_ec_ugm3[i])
  const evaluation = ids(indices)
  const predicted = ids(indices.filter(i => !String(a.fit_id[i]).startsWith('deployed|') && complete(i)))
  const deployed = ids(indices.filter(i => String(a.fit_id[i]).startsWith('deployed|') && complete(i)))
  const shared = gallery.filter(r => evaluation.has(r.id))
  const galleryOnly = gallery.filter(r => !evaluation.has(r.id))
  const dateRange = (dates: (string | number | null)[]) => {
    const sorted = dates.filter((d): d is string => typeof d === 'string').map(d => d.slice(0, 10)).sort()
    return sorted.length ? `${sorted[0]} to ${sorted[sorted.length - 1]}` : 'No dated filters'
  }
  const galleryPairs = gallery.filter(r => typeof r['HIPS BC'] === 'number' && Number.isFinite(r['HIPS BC']) &&
    typeof r['EC (FTIR)'] === 'number' && Number.isFinite(r['EC (FTIR)'])).length
  return <ChartFrame title="Addis filter coverage" exportable={false}>
    <table className="placement">
      <thead><tr><th>Data source</th><th>Filters</th><th>FTIR–HIPS pairs</th><th>Sampling window</th></tr></thead>
      <tbody>
        <tr><td>Gallery · existing FTIR product, after exclusions</td><td>{gallery.length}</td><td>{galleryPairs}</td><td>{dateRange(gallery.map(r => r.date))}</td></tr>
        <tr><td>Meeting test set · new FTIR predictions</td><td>{evaluation.size}</td><td>{predicted.size}</td><td>{dateRange(a.date)}</td></tr>
        <tr><td>Existing SPARTAN product within evaluation</td><td>{evaluation.size}</td><td>{deployed.size}</td><td>{dateRange(indices.filter(i => String(a.fit_id[i]).startsWith('deployed|') && complete(i)).map(i => a.date[i]))}</td></tr>
      </tbody>
    </table>
    <p className="chart-note">{data.summary.addis_source_n != null && <>The frozen meeting run contains {evaluation.size} of {data.summary.addis_source_n} source evaluation filters. </>}The gallery and evaluation share {shared.length} physical filters after matching base filter IDs. The evaluation includes {evaluation.size - shared.length} additional filters; {galleryOnly.length} gallery filter{galleryOnly.length === 1 ? '' : 's'} {galleryOnly.length === 1 ? 'is' : 'are'} absent from the frozen spectral run{galleryOnly.length ? ` (${galleryOnly.map(r => r.id).join(', ')})` : ''}.</p>
    <p className="chart-note">The starting-point plot uses {deployed.size} complete pairs and omits {evaluation.size - deployed.size} evaluation filters missing the existing FTIR product. New-prediction plots use {predicted.size} pairs. Missing values are omitted from each comparison, never plotted as zero. Addis HIPS is an optical comparison, not an independent thermal EC reference.</p>
  </ChartFrame>
}

// ------------------------------------------------------------------ scatter
/** A 1:1 crossplot: square plot area, identical x and y ranges, so the dashed line is the true diagonal. */
function Scatter({ points, stats, xLabel, yLabel, title, maxSide = 360, extraLines = [], demingLambda }: {
  points: Pt[]; stats?: Row; xLabel: string; yLabel: string; title: string; maxSide?: number
  demingLambda?: number
  extraLines?: { slope: number; intercept: number; color: string; label: string; dash?: string; method: import('@/lib/regressionMethods').Estimator; stats: RegressionStats | null }[]
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const hl = useHighlight()
  const m = { top: 24, right: 14, bottom: 46, left: 56 }
  const side = Math.max(160, Math.min(width - m.left - m.right, maxSide))
  const iw = side, ih = side, w = side + m.left + m.right, height = side + m.top + m.bottom
  const clipId = useId()
  const validPoints = points.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y) && p.x != null && p.y != null)
  const { grouping, methods, showCI } = useRegressions()
  const multiSite = new Set(validPoints.map(p => p.rec?.site)).size > 1
  const siteFits = multiSite && grouping === 'site' && methods.length > 0
  const fitGroups = useMemo(() => groupedRegressions(points, grouping, {
    x: p => p.x, y: p => p.y, site: p => p.rec?.site ?? 'Unknown site', color: p => p.color,
    options: members => ({ errorsInVariables: true, lambda: stats?.deming_lambda ?? demingLambda ?? 1,
      bootstrapGroups: monthBlocks(members.map(p => p.rec ?? {})) }),
  }), [points, grouping, stats?.deming_lambda, demingLambda])
  const st = regression(validPoints.map(p => p.x), validPoints.map(p => p.y), { errorsInVariables: true, lambda: stats?.deming_lambda ?? demingLambda ?? 1, bootstrapGroups: monthBlocks(validPoints.map(p => p.rec ?? {})) })
  const xs = validPoints.map((p) => p.x), ys = validPoints.map((p) => p.y)
  const hi = (d3.max([...xs, ...ys]) ?? 1) * 1.05
  const lo = Math.min(0, d3.min([...xs, ...ys]) ?? 0)
  const dom = d3.scaleLinear().domain([lo, hi]).nice().domain()
  const x = d3.scaleLinear().domain(dom).range([0, iw])
  const y = d3.scaleLinear().domain(dom).range([ih, 0])
  const [x0, x1] = x.domain()
  const line = (s: number, b: number) => ({ x1: x(x0), y1: y(s * x0 + b), x2: x(x1), y2: y(s * x1 + b) })
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      {multiSite && <FitGroupingControl />}
      <svg width={w} height={height} fontFamily={FONT.family}>
        <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>{title}</text>
        <g transform={`translate(${m.left},${m.top})`}>
          <defs><clipPath id={clipId}><rect width={iw} height={ih} /></clipPath></defs>
          <YAxis scale={y} x={0} label={yLabel} tickCount={5} gridWidth={iw} labelX={-42} format={(v) => String(+v.toFixed(2))} />
          <XAxis scale={x} y={ih} label={xLabel} tickCount={5} format={(v) => String(+v.toFixed(2))} labelDy={38} />
          <g clipPath={`url(#${clipId})`}>
            <rect width={iw} height={ih} fill="transparent" onClick={() => hl.setPinned(null)} />
            <line {...line(1, 0)} stroke={INK.identity} strokeDasharray="4 3" pointerEvents="none" />
            {validPoints.map((p, i) => {
              const key = p.id ? baseFilterId(p.id) : null
              const focus = !!key && (key === hl.pinnedId || key === hl.hoverId)
              const dim = !!(hl.pinnedId || hl.hoverId) && !focus
              return (
                <circle key={i} cx={x(p.x)} cy={y(p.y)} r={focus ? 6 : 3.2} fill={p.hollow ? 'none' : p.color}
                  stroke={focus ? INK.text : p.color} strokeWidth={focus ? 2 : p.hollow ? 1 : 0.4}
                  fillOpacity={dim ? 0.25 : 0.7} strokeOpacity={dim ? 0.3 : 1} style={{ cursor: p.rec ? 'pointer' : 'default' }}
                  onMouseEnter={(e) => { tip.show(e, p.rec ? [...p.tip, 'click to pin it on every chart and open its record'] : p.tip); if (key) hl.setHover(key) }}
                  onMouseLeave={() => { tip.hide(); hl.setHover(null) }}
                  onClick={() => { if (p.rec) hl.openRecord(p.rec); if (key) hl.setPinned(key) }} />
              )
            })}
            {siteFits ? <RegressionGroupLines groups={fitGroups} x={x} y={y} width={iw} height={ih} />
              : <RegressionLines stats={st} x={x} y={y} domain={x.domain()} width={iw} height={ih} />}
            {extraLines.map((l) => <g key={l.label}><RegressionBand stats={l.stats} method={l.method} color={l.color} x={x} y={y} domain={x.domain()} /><line {...line(l.slope, l.intercept)} stroke={l.color} strokeWidth={1.8} strokeDasharray={l.dash} /></g>)}
          </g>
          {!siteFits && <RegressionReadout stats={st} x={8} y={6} />}
        </g>
      </svg>
      {siteFits && <details><summary>Per-site fit statistics ({fitGroups.length} sites)</summary>
        <svg width={w} height={regressionGroupReadoutLayout(fitGroups.length, w, methods, showCI).height}>
          <RegressionGroupReadouts groups={fitGroups} width={w} />
        </svg>
        <p className="chart-note">Fits use each site's paired filters and observed x range. Sites with fewer than three pairs or constant values have no fitted line. Point colours retain the selected meeting grouping.</p>
      </details>}
      {tip.node}
    </div>
  )
}

function Grid({ children, cols = 2 }: { children: ReactNode; cols?: number }) {
  return <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(${cols > 2 ? 260 : 320}px, 1fr))`, gap: 14 }}>{children}</div>
}

type KeyItem = { label: string; color: string; kind?: 'dot' | 'hollow' | 'line' | 'dash' | 'bar' | 'faded' }
/** A static key in the gallery's legend style: what each colour, mark and line means. */
function Key({ items, children }: { items: KeyItem[]; children?: ReactNode }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label} className="legend-item">
          <svg width={20} height={12} aria-hidden="true">
            {(it.kind ?? 'dot') === 'dot' && <circle cx={10} cy={6} r={4} fill={it.color} fillOpacity={0.8} />}
            {it.kind === 'hollow' && <circle cx={10} cy={6} r={4} fill="none" stroke={it.color} strokeWidth={1.2} />}
            {it.kind === 'line' && <line x1={1} x2={19} y1={6} y2={6} stroke={it.color} strokeWidth={2.2} />}
            {it.kind === 'dash' && <line x1={1} x2={19} y1={6} y2={6} stroke={it.color} strokeWidth={1.4} strokeDasharray="4 3" />}
            {it.kind === 'bar' && <rect x={1} y={3} width={18} height={6} fill={it.color} />}
            {it.kind === 'faded' && <rect x={1} y={1} width={18} height={10} fill={it.color} opacity={0.3} />}
          </svg>
          <SavedFitLabel text={it.label} />
        </span>
      ))}
      {children}
    </div>
  )
}

function useAnalogPoints(data: MeetingData) {
  const idx = useIndex(data.analogs, ['model', 'method', 'group'])
  return (model: string, method: string, group: string, color: (i: number) => string, y: 'ftir' | 'tor', x: 'fabs' | 'tor', only?: 'test') => {
    const a = data.analogs
    const rows = (idx.get(`${model}|${method}|${group}`) ?? []).filter((i) => !only || a.role[i] === only)
    const val = { ftir: a.ftir_ec_ugm3, tor: a.tor_ec_ugm3, fabs: a.fabs10 }
    return rows.filter((i) => val[x][i] != null && val[y][i] != null).map((i): Pt => ({
      x: val[x][i] as number, y: val[y][i] as number, color: color(i), hollow: a.role[i] !== 'test',
      id: `improve:${a.filter_id[i]}`,
      rec: {
        id: `improve:${a.filter_id[i]}`, source: `meeting follow-up · ${method} · ${model} · analogs of ${group}`,
        site: String(a.Site[i]), date: a.date?.[i] != null ? String(a.date[i]).slice(0, 10) : undefined,
        fields: [
          ['role', a.role[i] === 'test' ? 'held out (site-disjoint)' : 'training'],
          ['IMPROVE lot', a.lot[i]],
          ['TOR EC (µg/m³)', a.tor_ec_ugm3[i]],
          ['HIPS Fabs / MAC 10 (µg/m³)', a.fabs10[i]],
          ['predicted FTIR EC (µg/m³)', a.ftir_ec_ugm3[i]],
          ['FTIR − TOR (µg/m³)', a.ftir_ec_ugm3[i] != null && a.tor_ec_ugm3[i] != null ? (a.ftir_ec_ugm3[i] as number) - (a.tor_ec_ugm3[i] as number) : null],
          ['Fabs / TOR EC (m²/g)', a.fabs10[i] != null && a.tor_ec_ugm3[i] ? (10 * (a.fabs10[i] as number)) / (a.tor_ec_ugm3[i] as number) : null],
          ['OP fraction, OPTR / (EC + OPTR)', a.op_tor_frac[i]],
          ['TOR OC/EC', a.tor_oc_ec[i]],
          ['TOR EC loading (µg/filter)', a.y[i]],
        ],
      },
      tip: [`IMPROVE ${a.Site[i]} · filter ${a.filter_id[i]}`, `${a.role[i] === 'test' ? 'held out (test split)' : 'seen in training'}`,
        `TOR EC ${f2(a.tor_ec_ugm3[i])} · Fabs/10 ${f2(a.fabs10[i])} · FTIR ${f2(a.ftir_ec_ugm3[i])}`, `OP fraction ${f2(a.op_tor_frac[i])} · OC/EC ${f2(a.tor_oc_ec[i])}`],
    }))
  }
}

function useAddisPoints(data: MeetingData) {
  const idx = useIndex(data.addis, ['fit_id'])
  return (fitId: string, season: string | 'all', color?: (s: string) => string) => {
    const a = data.addis
    return (idx.get(fitId) ?? []).filter((i) => typeof a.fabs10[i] === 'number' && Number.isFinite(a.fabs10[i]) && typeof a.ftir_ec_ugm3[i] === 'number' && Number.isFinite(a.ftir_ec_ugm3[i]) && (season === 'all' || season === 'All Addis' || a.season[i] === season)).map((i): Pt => ({
      x: a.fabs10[i] as number, y: a.ftir_ec_ugm3[i] as number,
      color: color ? color(String(a.season[i])) : data.season_colors[String(a.season[i])] ?? INK.axis,
      tip: [`${a.ExternalFilterId?.[i] ?? 'Addis'} · ${String(a.date[i]).slice(0, 10)}`, String(a.season[i]), `Fabs/10 ${f2(a.fabs10[i])} · FTIR ${f2(a.ftir_ec_ugm3[i])}`],
      id: String(a.ExternalFilterId?.[i] ?? `ETAD media ${a.MediaId[i]}`),
      rec: {
        id: String(a.ExternalFilterId?.[i] ?? `ETAD media ${a.MediaId[i]}`), source: `meeting follow-up · ${fitId.split('|').slice(1).join(' · ')}`,
        site: 'Addis Ababa', date: String(a.date[i]).slice(0, 10),
        fields: [
          ['season', a.season[i]],
          ['calibration', fitId.split('|').slice(1).join(' · ')],
          ['HIPS Fabs / MAC 10 (µg/m³)', a.fabs10[i]],
          ['predicted FTIR EC (µg/m³)', a.ftir_ec_ugm3[i]],
          ['residual, FTIR − Fabs/10 (µg/m³)', (a.ftir_ec_ugm3[i] as number) - (a.fabs10[i] as number)],
          ['MediaId', a.MediaId[i]],
        ],
      },
    }))
  }
}

function AcrossSeasons({ data, fitPrefix, seasonal }: { data: MeetingData; fitPrefix: string; seasonal: boolean }) {
  return (
    <table className="census-table" style={{ marginTop: 10 }}>
      <thead><tr><th>Analogs of</th><th>IMPROVE cross-validation: FTIR vs Fabs/10</th><th>IMPROVE cross-validation: TOR vs Fabs/10</th><th>IMPROVE cross-validation: FTIR vs TOR</th><th>Test set (Addis): FTIR vs Fabs/10</th></tr></thead>
      <tbody>{data.groups.map((g) => {
        const q = (population: string, panel: string, extra: Row = {}) => seasonal
          ? statRow(data.stats, population === 'addis' ? { fit_id: `${fitPrefix}|${g}`, population, eval_group: g } : { fit_id: `${fitPrefix}|${g}`, population, panel })
          : statRow(data.stats, { fit_id: fitPrefix, group: g, population, panel, ...extra })
        const cell = (s?: Row) => s ? <>{f2(s.deming_slope)} / {signed(s.deming_intercept)} <span style={{ color: INK.muted }}>R² {f2(s.R2)}, n {s.n}</span></> : '—'
        return <tr key={g}><td style={{ color: data.season_colors[g], fontWeight: 600 }}>{g}</td><td>{cell(q('analog_test', 'ftir_vs_fabs'))}</td><td>{cell(q('analog_test', 'tor_vs_fabs'))}</td><td>{cell(q('analog_test', 'ftir_vs_tor'))}</td><td>{cell(q('addis', 'ftir_vs_fabs', { eval_group: g }))}</td></tr>
      })}</tbody>
    </table>
  )
}

interface SimMeta { bin_wn: number[]; filters: { sample_id: string[] }; rank_cap?: number; background_rank?: number; presentation_status?: string }
let simMetaPromise: Promise<SimMeta> | null = null

/** Per-filter binned spectra from the Spectral similarity export (same frozen run, same ids). */
function useTraces(method: string) {
  const [state, setState] = useState<{ method: string; meta: SimMeta; traces: Traces; index: Map<string, number> } | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let live = true
    simMetaPromise ??= fetch(SIM_BASE + 'similarity.json').then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
    // the full-range VIBES traces share the filter table but have their own (4000-500) bin grid
    const full = method.startsWith('VIBES-full')
    const source = method.endsWith('-r70') ? 'VIBES-full-r70' : full ? 'VIBES-full' : method
    const gridPromise: Promise<Partial<SimMeta> | null> = full
      ? fetch(SIM_BASE + `traces_${source}.json`).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      : Promise.resolve(null)
    Promise.all([simMetaPromise, fetchBinary(`traces_${source}.bin`), gridPromise])
      .then(([meta, buf, grid]) => {
        if (!live) return
        const ids = meta.filters.sample_id
        const m2 = grid ? { ...meta, ...grid } : meta
        setState({ method, meta: m2, traces: new Traces(buf, ids.length, m2.bin_wn.length), index: new Map(ids.map((id, i) => [id, i])) })
      })
      .catch((e) => { if (live) setError(String(e)) })
    return () => { live = false }
  }, [method])
  return { value: state && state.method === method ? state : null, error }
}

/**
 * Ann (23 Sep): compare the two baselines only where both exist (4000-1425 cm-1),
 * and show the full-range VIBES on its own. The comparison can also overlay
 * each method on its available grid across the full range, without extrapolating.
 * Uses the pinned filter, or the first Addis filter until one is pinned.
 */
function BaselineCompare({ data }: { data: MeetingData }) {
  const hl = useHighlight()
  const [layout, setLayout] = useSearchParam<'Stacked' | 'Side by side'>('baselineLayout', 'Stacked', ['Stacked', 'Side by side'])
  const [range, setRange] = useSearchParam<'Shared (4000–1425)' | 'Full (4000–500)'>('baselineRange', 'Shared (4000–1425)', ['Shared (4000–1425)', 'Full (4000–500)'])
  const [hiddenValue, setHiddenValue] = useSearchParam<string>('baselineHidden', 'VIBES-full')
  const hidden = new Set(hiddenValue.split('|').filter(Boolean))
  const toggleSeries = (method: string) => {
    const next = new Set(hidden)
    if (next.has(method)) next.delete(method)
    else next.add(method)
    setHiddenValue([...next].sort().join('|') || 'none')
  }
  const lowerBound = range === 'Full (4000–500)' ? 500 : 1425
  const stacked = layout === 'Stacked'
  const air = useTraces('AIRSpec'), vib = useTraces('VIBES'), full = useTraces('VIBES-full'), completed = useTraces('VIBES-full-r70')
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const etadOf = useMemo(() => {
    const m = new Map<string, string>(), d = data.addis
    for (let i = 0; i < d.fit_id.length; i++) if (d.ExternalFilterId?.[i]) m.set(baseFilterId(String(d.ExternalFilterId[i])), `etad:${d.MediaId[i]}`)
    return m
  }, [data])
  const firstAddis = `etad:${data.addis.MediaId[0]}`
  const key = hl.pinnedId
  const sid = key ? (key.startsWith('improve:') ? key : etadOf.get(key) ?? firstAddis) : firstAddis
  const hasFull = METHODS.includes('VIBES-full')
  const series = [
    { method: 'AIRSpec', name: PREPROCESSING.AIRSpec, color: '#2C6E9E', t: air.value, error: air.error },
    { method: 'VIBES-full-r70', name: 'VIBES baseline (4000–500, completed rank 70)', color: '#d95f0e', t: completed.value, error: completed.error },
    ...(METHODS.includes('VIBES') ? [{ method: 'VIBES', name: PREPROCESSING.VIBES, color: '#9467bd', t: vib.value, error: vib.error }] : []),
    ...(hasFull ? [{ method: 'VIBES-full', name: PREPROCESSING['VIBES-full'], color: '#7A4FA3', t: full.value, error: full.error }] : []),
  ]
  const visibleSeries = series.filter((s2) => !hidden.has(s2.method))
  const get = (t: (typeof series)[number]['t']) => {
    if (!t) return null
    const k = t.index.get(sid)
    return k == null ? null : { wn: t.meta.bin_wn, v: t.traces.of(k) }
  }
  const panel = (title: string, lines: { name: string; color: string; wn: number[]; v: Float32Array }[], lo: number, w: number, caption?: string) => {
    const m = { top: caption ? 40 : 22, right: 12, bottom: 44, left: 58 }, h = caption ? 298 : 280, iw = w - m.left - m.right, ih = h - m.top - m.bottom
    const vals = lines.flatMap((l) => Array.from(l.v).filter((_, i) => l.wn[i] >= lo))
    const x = d3.scaleLinear().domain([4000, lo]).range([0, iw])
    const y = d3.scaleLinear().domain([Math.min(0, d3.min(vals) ?? 0), d3.max(vals) ?? 0.01]).range([ih, 0]).nice()
    return (
      <svg width={w} height={h} fontFamily={FONT.family} role="img" aria-label={title}>
        <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>{title}</text>
        {caption && <text x={m.left} y={30} fontSize={11} fill={INK.muted}>{caption}</text>}
        <g transform={`translate(${m.left},${m.top})`}>
          <YAxis scale={y} x={0} label="Absorbance" tickCount={4} labelX={-46} format={(v) => String(+v.toFixed(3))} />
          <XAxis scale={x} y={ih} label="Wavenumber (cm⁻¹)" tickCount={6} labelDy={36} />
          <line x1={0} x2={iw} y1={y(0)} y2={y(0)} stroke={INK.identity} strokeDasharray="3 3" />
          {lines.map((l) => (
            <path key={l.name} aria-label={l.name} d={d3.line<number>().defined((_, i) => l.wn[i] >= lo && Number.isFinite(l.v[i])).x((_, i) => x(l.wn[i])).y((_, i) => y(l.v[i]))(Array.from(l.v)) ?? ''}
              fill="none" stroke={l.color} strokeWidth={1.5} />
          ))}
        </g>
      </svg>
    )
  }
  const lines = visibleSeries.map((s2) => { const g = get(s2.t); return g ? { name: s2.name, color: s2.color, ...g } : null }).filter(Boolean) as { name: string; color: string; wn: number[]; v: Float32Array }[]
  const completeLine = lines.find(l => l.name === 'VIBES baseline (4000–500, completed rank 70)')
  const fullLine = completeLine ?? lines.find((l) => l.name === PREPROCESSING['VIBES-full'])
  const fullCaption = completeLine ? 'Completed · background rank 70' : full.value?.meta.rank_cap == null ? undefined : `Preliminary · background rank cap ${full.value.meta.rank_cap}`
  const w = Math.max(320, width || 700)
  const sharedWidth = stacked || !fullLine ? w : Math.max(320, Math.min((w - 14) * 0.6, 760))
  const fullWidth = stacked ? w : Math.max(300, Math.min((w - 14) * 0.4, 520))
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › BaselineCompare"
      title={`Baselines compared — one filter (${key ?? 'example: first Addis filter'}), spline vs VIBES where both exist, and the full-range VIBES`}
      controls={<>
        <Segmented label="Comparison range" value={range} options={['Shared (4000–1425)', 'Full (4000–500)']} onChange={setRange} />
        <Segmented label="Plot layout" value={layout} options={['Stacked', 'Side by side']} onChange={setLayout} />
        {series.map((s2) => <label key={s2.method} className="control">
          <input type="checkbox" checked={!hidden.has(s2.method)} onChange={() => toggleSeries(s2.method)} />
          <span style={{ color: s2.color }}>{s2.name}</span>
        </label>)}
      </>}
      subtitle={`The main panel overlays the selected baselines down to ${lowerBound} cm⁻¹. The archived truncated fit appears only when enabled. A separate full-range VIBES panel appears when that method is selected. Click any point or trace on this page to pin a filter here.`}>
      <div ref={ref}>
        {!lines.length ? <Empty>{!visibleSeries.length ? 'Select a baseline above to display its spectrum.' : visibleSeries.some((s2) => s2.error) ? `Could not load spectra: ${visibleSeries.find((s2) => s2.error)?.error}` : visibleSeries.some((s2) => !s2.t) ? 'Loading spectra…' : 'The selected baselines have no spectrum for this filter.'}</Empty> : (
          <div className="plot-groups" style={stacked ? { flexDirection: 'column' } : undefined}>
            <div style={{ flex: '0 0 auto', width: sharedWidth, minWidth: 0 }}>{panel(lowerBound === 500 ? '4000–500 cm⁻¹: selected baselines overlaid' : '4000–1425 cm⁻¹: shared comparison range', lines, lowerBound, sharedWidth, fullLine ? fullCaption : undefined)}</div>
            {fullLine && <div style={{ flex: '0 0 auto', width: fullWidth, minWidth: 0 }}>{panel('4000–500 cm⁻¹: VIBES baseline only', [fullLine], 500, fullWidth, fullCaption)}</div>}
          </div>
        )}
        <Key items={visibleSeries.map((s2) => ({ label: s2.name, color: s2.color, kind: 'line' as const }))} />
        {lowerBound === 500 && !hidden.has('AIRSpec') && <Note>The spline baseline ends near 1425 cm⁻¹; no spline values are extrapolated below its available range.</Note>}
      </div>
      <ExplainOnly><Note>Traces are 8-channel (~10 cm⁻¹) means from the gallery's spectral export, for display only. {sid}</Note></ExplainOnly>
    </ChartFrame>
  )
}

const SPEC_MODES = ['individual filters', 'median + IQR'] as const
/** pooled Addis in the spectra (seasons keep their own colours) and the IMPROVE median line */
const ADDIS_ALL = '#e0a800'
const IMPROVE_LINE = '#7b8794'
const TOP_N = ['50', '100', '250', '500'] as const

/**
 * The season's IMPROVE analogs and Addis filters as spectra: every filter, or the
 * median and IQR. The pinned (clicked) filter, and whichever filter is hovered on
 * any chart, is drawn on top in black, so a crossplot point can be read as a spectrum.
 */
function SpectraPanel({ data, method, group }: { data: MeetingData; method: string; group: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const hl = useHighlight()
  const [mode, setMode] = useSearchParam<string>('mspec', 'individual filters', SPEC_MODES)
  const [topN, setTopN] = useSearchParam<string>('mspecn', '100', TOP_N)
  const tr = useTraces(method)
  // one Addis colour per season; the pooled 'All Addis' summary is yellow so it never reads as the grey analogs
  const col = group === 'All Addis' ? ADDIS_ALL : data.season_colors[group]

  // the season's analogs in selection order (rank 1 first), and its Addis filters
  const { improveIds, addisRows, etadOf } = useMemo(() => {
    const a = data.analogs, d = data.addis
    const improveIds: string[] = []
    for (let i = 0; i < a.model.length; i++)
      if (a.model[i] === 'seasonal calibration' && a.method[i] === method && a.group[i] === group) improveIds.push(`improve:${a.filter_id[i]}`)
    const fid = `seasonal|${method}|locked|${group}`
    const addisRows: { sid: string; key: string; label: string; season: string }[] = []
    const etadOf = new Map<string, string>()
    for (let i = 0; i < d.fit_id.length; i++) {
      const ext = String(d.ExternalFilterId?.[i] ?? '')
      const sid = `etad:${d.MediaId[i]}`
      if (ext) etadOf.set(baseFilterId(ext), sid)
      if (d.fit_id[i] === fid && (group === 'All Addis' || d.season[i] === group))
        addisRows.push({ sid, key: ext ? baseFilterId(ext) : sid, label: `${ext || sid} · ${String(d.date[i]).slice(0, 10)}`, season: String(d.season[i]) })
    }
    return { improveIds, addisRows, etadOf }
  }, [data, method, group])

  const toSid = (key: string | null) => (key == null ? null : key.startsWith('improve:') ? key : etadOf.get(key) ?? null)
  const pinnedSid = toSid(hl.pinnedId), hoverSid = toSid(hl.hoverId)

  const m = { top: 24, right: 14, bottom: 46, left: 60 }, height = 300
  const w = Math.max(320, width), iw = w - m.left - m.right, ih = height - m.top - m.bottom
  if (tr.error) return <Empty>Could not load the per-filter spectra: {tr.error}</Empty>
  if (!tr.value) return <div ref={ref}><Empty>Loading per-filter spectra…</Empty></div>
  const { meta, traces, index } = tr.value
  const wn = meta.bin_wn
  const wnFloor = method.includes('full-cut') ? 1425 : 0
  const shownImp = improveIds.slice(0, Number(topN))
  const get = (sid: string) => { const k = index.get(sid); return k == null ? null : traces.of(k) }
  const impT = shownImp.map((sid) => ({ sid, v: get(sid) })).filter((t) => t.v) as { sid: string; v: Float32Array }[]
  const addT = addisRows.map((r) => ({ ...r, v: get(r.sid) })).filter((t) => t.v) as { sid: string; key: string; label: string; season: string; v: Float32Array }[]
  const qImp = data.spectra[`${method}|${group}|improve`] as number[][] | undefined
  const qAdd = data.spectra[`${method}|${group}|addis`] as number[][] | undefined
  const qwn: number[] = data.spectra[`wn|${method}`] ?? data.spectra.wn
  const pool = mode === 'individual filters'
    ? [...impT, ...addT].flatMap((t) => [d3.quantile(t.v, 0.995) ?? 0, d3.min(t.v) ?? 0])
    : [...(qImp ? [...qImp[1], ...qImp[3]] : []), ...(qAdd ? [...qAdd[1], ...qAdd[3]] : [])]
  const pin = pinnedSid ? get(pinnedSid) : null
  const hov = hoverSid && hoverSid !== pinnedSid ? get(hoverSid) : null
  const x = d3.scaleLinear().domain([d3.max(wn) ?? 4000, Math.max(wnFloor, d3.min(wn) ?? 1400)]).range([0, iw])
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(pool) ?? 0), Math.max(d3.max(pool) ?? 0.01, pin ? d3.max(pin) ?? 0 : 0)]).range([ih, 0]).nice()
  const path = (v: ArrayLike<number>, grid: number[] = wn) => d3.line<number>().defined((_, i) => grid[i] >= wnFloor).x((_, i) => x(grid[i])).y((_, i) => y(v[i]))(Array.from(v)) ?? ''
  const area = (q: number[][]) => d3.area<number>().x((_, i) => x(qwn[i])).y0((_, i) => y(q[1][i])).y1((_, i) => y(q[3][i]))(qwn) ?? ''
  const enter = (e: React.MouseEvent, key: string, lines: string[]) => { tip.show(e, [...lines, 'click to pin it on every chart']); hl.setHover(key) }
  const leave = () => { tip.hide(); hl.setHover(null) }
  const wnMax = d3.max(wn) ?? 4000
  return (
    <div ref={ref} style={{ marginTop: 10 }}>
      <div className="frame-controls" style={{ padding: 0 }}>
        <Segmented label="Spectra" value={mode as any} options={SPEC_MODES} onChange={setMode as any} />
        {mode === 'individual filters' && <Segmented label="Most similar analogs" value={topN as any} options={TOP_N} onChange={setTopN as any} />}
      </div>
      <svg width={w} height={height} fontFamily={FONT.family}>
        <defs><pattern id="oct1-hatch" width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width={6} height={6} fill="#f6f1e7" /><line x1={0} y1={0} x2={0} y2={6} stroke="#d9c9a3" strokeWidth={2} /></pattern></defs>
        <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>
          Spectra: {mode === 'individual filters' ? `${impT.length} most similar IMPROVE analogs and ${addT.length} Addis filters` : 'median and IQR'}
        </text>
        <g transform={`translate(${m.left},${m.top})`}>
          <rect x={x(2500)} width={x(1800) - x(2500)} height={ih} fill="#eef0f3" />
          <rect x={x(wnMax)} width={x(3600) - x(wnMax)} height={ih} fill="#eef0f3" />
          {/* also dropped by the region agreed 1 Oct (similarity only; PLS still sees them) */}
          {x.domain()[1] < 1050 && <rect x={x(1250)} width={x(1050) - x(1250)} height={ih} fill="url(#oct1-hatch)" />}
          {x.domain()[1] < 600 && <rect x={x(600)} width={x(x.domain()[1]) - x(600)} height={ih} fill="url(#oct1-hatch)" />}
          <rect width={iw} height={ih} fill="transparent" onClick={() => hl.setPinned(null)} />
          <YAxis scale={y} x={0} label="Absorbance" tickCount={4} labelX={-48} format={(v) => String(+v.toFixed(3))} />
          <XAxis scale={x} y={ih} label="Wavenumber (cm⁻¹)" tickCount={6} labelDy={38} />
          {mode === 'individual filters' ? (
            <>
              {impT.map((t) => (
                <path key={t.sid} d={path(t.v)} fill="none" stroke="#8a8f98" strokeOpacity={0.18} strokeWidth={0.8} style={{ cursor: 'pointer' }}
                  onMouseEnter={(e) => enter(e, t.sid, [t.sid, 'IMPROVE analog'])} onMouseLeave={leave} onClick={() => hl.setPinned(t.sid)} />
              ))}
              {addT.map((t) => (
                <path key={t.sid} d={path(t.v)} fill="none" stroke={data.season_colors[t.season] ?? col} strokeOpacity={0.4} strokeWidth={0.9} style={{ cursor: 'pointer' }}
                  onMouseEnter={(e) => enter(e, t.key, [t.label, `Addis · ${t.season}`])} onMouseLeave={leave} onClick={() => hl.setPinned(t.key)} />
              ))}
            </>
          ) : (
            <>
              {qImp && <><path d={area(qImp)} fill="#8a8f98" fillOpacity={0.25} /><path d={path(qImp[2], qwn)} stroke={IMPROVE_LINE} fill="none" strokeWidth={1.4} /></>}
              {qAdd && <><path d={area(qAdd)} fill={col} fillOpacity={0.25} /><path d={path(qAdd[2], qwn)} stroke={col} fill="none" strokeWidth={1.5} /></>}
            </>
          )}
          {hov && <path d={path(hov)} fill="none" stroke={INK.text} strokeOpacity={0.55} strokeWidth={1.6} strokeDasharray="5 3" pointerEvents="none" />}
          {pin && <path d={path(pin)} fill="none" stroke={INK.text} strokeWidth={2.2} pointerEvents="none" />}
        </g>
      </svg>
      <Key items={[
        { label: mode === 'individual filters' ? `IMPROVE analogs, ranks 1–${impT.length}` : 'IMPROVE analogs (500): median, IQR', color: '#8a8f98', kind: mode === 'individual filters' ? 'line' : 'bar' },
        ...(mode === 'individual filters'
          ? (group === 'All Addis' ? data.seasons : [group]).map((g) => ({ label: `Addis ${g} filters`, color: data.season_colors[g], kind: 'line' as const }))
          : [{ label: `Addis ${group === 'All Addis' ? 'all seasons' : shortSeason(group)}: median, IQR`, color: col, kind: 'bar' as const }]),
        { label: `pinned filter${pinnedSid ? ` (${hl.pinnedId})` : ': none, click a point or trace'}`, color: INK.text, kind: 'line' },
        { label: 'hovered filter', color: INK.text, kind: 'dash' },
        { label: 'excluded from analog selection only (CO₂, >3600 cm⁻¹)', color: '#dfe3e8', kind: 'bar' },
        ...(x.domain()[1] < 1050 ? [{ label: 'also excluded by the region agreed 1 Oct (<600, 1050–1250 PTFE); see the Satoshi 2 Oct view in Meeting', color: '#e6d8b5', kind: 'bar' as const }] : []),
      ]} />
      {tip.node}
    </div>
  )
}

const REPEAT_METRICS = ['slope', 'intercept'] as const

function RepeatStrip({ data }: { data: MeetingData }) {
  const [method, setMethod] = useSearchParam<string>('mrmethod', 'AIRSpec', METHODS)
  // 1 Oct: Ann asked for the intercept beside the slope, to see whether the IMPROVE
  // analogs also carry the large negative Addis intercept.
  const [metric, setMetric] = useSearchParam<(typeof REPEAT_METRICS)[number]>('mrmetric', 'slope', REPEAT_METRICS)
  const isSlope = metric === 'slope'
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const measures = [
    { base: 'addis', label: 'Test set (Addis): FTIR vs Fabs/10' },
    { base: 'ftir_vs_fabs', label: 'IMPROVE cross-validation: FTIR vs Fabs/10' },
    { base: 'tor_vs_fabs', label: 'IMPROVE cross-validation: TOR EC vs Fabs/10' },
    { base: 'ftir_vs_tor', label: 'IMPROVE cross-validation: FTIR vs TOR EC' },
  ].map((ms) => ({ ...ms, key: `${ms.base}_${metric}` }))
  const r = data.repeats
  const rows: { g: string; m: string; label: string; vals: number[] }[] = []
  for (const g of data.groups) for (const ms of measures) {
    const vals: number[] = []
    for (let i = 0; i < r.method.length; i++) if (r.method[i] === method && r.group[i] === g && r[ms.key][i] != null) vals.push(r[ms.key][i] as number)
    rows.push({ g, m: ms.base, label: ms.label, vals: vals.sort((a, b) => a - b) })
  }
  const rowH = 17, gap = 12, left = 250, right = 20
  const h = rows.length * rowH + data.groups.length * gap + 62
  const w = Math.max(560, width)
  const all = rows.flatMap((r) => [d3.quantile(r.vals, 0.05) ?? 0, d3.quantile(r.vals, 0.95) ?? 1])
  const ref0 = isSlope ? 1 : 0
  const x = d3.scaleLinear().domain(isSlope ? [Math.min(0, d3.min(all) ?? 0), Math.max(1.6, d3.max(all) ?? 1)]
    : [Math.min(-0.5, d3.min(all) ?? 0), Math.max(0.5, d3.max(all) ?? 0)]).range([0, w - left - right]).nice()
  const colors: Record<string, string> = { addis: '#111827', ftir_vs_fabs: '#c026d3', tor_vs_fabs: '#0f766e', ftir_vs_tor: '#2171b5' }
  let yCursor = 0
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › RepeatStrip" title={`One separate calibration per season, built from that season's IMPROVE spectral analogs: are the analogs mispredicted the way Addis is? Deming ${metric}, median and 10–90 % over 100 splits`}
      subtitle={`Each block is its own calibration set: the 500 IMPROVE filters whose spectra correlate best with that season's median Addis spectrum ("All Addis" uses the all-season median). Each split holds about 20 % of those analogs out by whole site (cross-validation), picks the number of PLS factors on the rest, and predicts both the held-out IMPROVE filters and the season's Addis filters (the test set) with the same calibration.${isSlope ? '' : ' Intercepts are in µg/m³ of EC.'}`}
      controls={<><MethodSeg value={method} onChange={setMethod} /><Segmented label="Show" value={metric} options={REPEAT_METRICS} onChange={setMetric} /></>}>
      <div ref={ref}>
        <svg width={w} height={h} fontFamily={FONT.family}>
          <g transform={`translate(${left},10)`}>
            <line x1={x(ref0)} x2={x(ref0)} y1={0} y2={h - 52} stroke={INK.identity} strokeDasharray="4 3" />
            {data.groups.map((g) => {
              const block = rows.filter((r) => r.g === g)
              const top = yCursor
              yCursor += block.length * rowH + gap
              return (
                <g key={g} transform={`translate(0,${top})`}>
                  <text x={-left + 4} y={10} fontSize={12} fontWeight={600} fill={data.season_colors[g]}>{g}</text>
                  {block.map((r, j) => {
                    const q10 = d3.quantile(r.vals, 0.1) ?? NaN, q50 = d3.quantile(r.vals, 0.5) ?? NaN, q90 = d3.quantile(r.vals, 0.9) ?? NaN
                    const yy = j * rowH + 8
                    return (
                      <g key={r.m} onMouseEnter={(e) => tip.show(e, [`${g} · ${r.label} · ${metric}`, `median ${f2(q50)}`, `10–90 %: ${f2(q10)} – ${f2(q90)}`, `${r.vals.length} splits`])} onMouseLeave={tip.hide}>
                        <text x={-8} y={yy + 4} fontSize={10.5} textAnchor="end" fill={INK.muted}>{r.label}</text>
                        <line x1={x(q10)} x2={x(q90)} y1={yy} y2={yy} stroke={colors[r.m]} strokeWidth={3} strokeOpacity={0.45} />
                        <circle cx={x(q50)} cy={yy} r={4.5} fill={colors[r.m]} />
                      </g>
                    )
                  })}
                </g>
              )
            })}
            <XAxis scale={x} y={h - 52} label={isSlope ? 'Deming slope (1 = agreement)' : 'Deming intercept, µg/m³ (0 = agreement)'} tickCount={8} labelDy={34} />
          </g>
        </svg>
        <Key items={[
          ...measures.map((ms) => ({ label: ms.label, color: colors[ms.base] })),
          { label: 'median', color: INK.text },
          { label: '10–90 % of splits', color: INK.muted, kind: 'bar' as const },
          { label: isSlope ? 'slope 1 (agreement)' : 'intercept 0 (agreement)', color: INK.identity, kind: 'dash' as const },
        ]} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. three crossplots
const COLOR_BY = ['season colour', 'OP fraction', 'OC/EC'] as const
function ThreePlots({ data }: { data: MeetingData }) {
  const fit = FIT_STYLE[useEstimator()]
  const [method, setMethod] = useSearchParam<string>('m3method', 'AIRSpec', METHODS)
  const [season, setSeason] = useSearchParam<string>('mseason', 'Dry (Oct-Feb)', data.groups)
  const FAMS = ['seasonal calibration', ...MODELS] as const
  const [fam, setFam] = useSearchParam<string>('mfam', 'seasonal calibration', FAMS)
  const [colorBy, setColorBy] = useSearchParam<string>('mcolor', 'season colour', COLOR_BY)
  const analogPts = useAnalogPoints(data)
  const addisPts = useAddisPoints(data)
  const a = data.analogs
  const opScale = d3.scaleSequential(d3.interpolateViridis).domain([0.1, 0.8])
  const ocScale = d3.scaleSequentialLog(d3.interpolatePlasma).domain([1.5, 12])
  const col = data.season_colors[season]
  const color = (i: number) => colorBy === 'OP fraction' ? (a.op_tor_frac[i] == null ? INK.neutral : opScale(a.op_tor_frac[i] as number))
    : colorBy === 'OC/EC' ? (a.tor_oc_ec[i] == null ? INK.neutral : ocScale(Math.max(1.5, a.tor_oc_ec[i] as number))) : col
  const fitId = fam === 'seasonal calibration' ? `seasonal|${method}|locked|${season}` : `allimprove|${method}|${fam}`
  const calibrationFit = data.fits.find(fit => fit.fit_id === fitId)
  const q = (panel: string) => statRow(data.stats, fam === 'seasonal calibration'
    ? { fit_id: fitId, population: 'analog_test', panel } : { fit_id: fitId, group: season, population: 'analog_test', panel })
  const sAd = statRow(data.stats, fam === 'seasonal calibration' ? { fit_id: fitId, population: 'addis', eval_group: season } : { fit_id: fitId, group: season, population: 'addis', eval_group: season })
  const only = 'test' as const
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › ThreePlots" title="Three crossplots per season — TOR EC vs HIPS, FTIR vs TOR and FTIR vs HIPS on the IMPROVE analogs in cross-validation, beside the Addis test set from the same calibration and the spectra"
      subtitle="Only IMPROVE analogs from sites held out of the calibration set are drawn (cross-validation). 'Seasonal calibration' is one model per season from its 500 analogs; the all-IMPROVE models are Ann's reading of the current model (lot 251, or the frozen full-pool model for the selected analysis version)."
      controls={<>
        <MethodSeg value={method} onChange={setMethod} />
        <ModelSeg label="Calibration" value={fam} options={FAMS} onChange={setFam} />
        <Segmented label="Season" value={season} options={data.groups} onChange={setSeason} />
        <Segmented label="Colour" value={colorBy as any} options={COLOR_BY} onChange={setColorBy as any} />
      </>}>
      <MeetingCalibrationDetails fit={calibrationFit} />
      <div className="plot-groups">
        <section className="plot-group improve">
          <h3>
            IMPROVE analogs only
            <span>{fam === 'seasonal calibration'
              ? `cross-validation · ${season === 'All Addis' ? 'all-Addis' : shortSeason(season)} analogs from sites held out of the season's calibration set`
              : `cross-validation · the ${season === 'All Addis' ? 'all-Addis' : shortSeason(season)} analogs at sites held out of the ${fam} model's calibration set`}</span>
          </h3>
          <div className="plot-group-grid three">
            <Scatter title="TOR EC vs HIPS" points={analogPts(fam, method, season, color, 'tor', 'fabs', only)} stats={q('tor_vs_fabs')} xLabel={MAC_LABEL} yLabel="TOR EC (µg/m³)" />
            <Scatter title="FTIR EC vs TOR EC" points={analogPts(fam, method, season, color, 'ftir', 'tor', only)} stats={q('ftir_vs_tor')} xLabel="TOR EC (µg/m³)" yLabel="FTIR EC (µg/m³)" />
            <Scatter title="FTIR EC vs HIPS" points={analogPts(fam, method, season, color, 'ftir', 'fabs', only)} stats={q('ftir_vs_fabs')} xLabel={MAC_LABEL} yLabel="FTIR EC (µg/m³)" />
          </div>
        </section>
        <section className="plot-group addis">
          <h3>
            Test set: Addis
            <span>predicted by the same calibration · HIPS only (no TOR at Addis)</span>
          </h3>
          <div className="plot-group-grid">
            <Scatter title={`Addis ${season === 'All Addis' ? 'all seasons' : shortSeason(season)}: FTIR vs HIPS`} points={addisPts(fitId, season)} stats={sAd} demingLambda={data.summary.lambda} xLabel={MAC_LABEL} yLabel="FTIR EC (µg/m³)" />
          </div>
        </section>
      </div>
      <Key items={[
        ...(colorBy === 'season colour' ? [{ label: `IMPROVE analogs of ${season === 'All Addis' ? 'all Addis' : shortSeason(season)}, cross-validation sites`, color: col }] : []),
        ...(season === 'All Addis'
          ? data.seasons.map((g) => ({ label: `Addis ${g}`, color: data.season_colors[g] }))
          : [{ label: `Addis ${shortSeason(season)} filters`, color: col }]),
        { label: fit === FIT_STYLE.deming ? 'Deming fit (λ from measurement uncertainties)' : `${fit.label} fit`, color: fit.color, kind: fit.dash ? 'dash' : 'line' },
        { label: '1:1', color: INK.identity, kind: 'dash' },
      ]}>
        {colorBy === 'OP fraction' && <ColorLegend scale={opScale} label="IMPROVE analogs: OP fraction = OPTR / (EC + OPTR)" width={200} />}
        {colorBy === 'OC/EC' && <ColorLegend scale={ocScale} label="IMPROVE analogs: TOR OC/EC" width={200} format={(v) => v.toFixed(0)} />}
      </Key>
      <SpectraPanel data={data} method={method} group={season} />
      <ExplainOnly>
        <AcrossSeasons data={data} fitPrefix={fam === 'seasonal calibration' ? `seasonal|${method}|locked` : fitId} seasonal={fam === 'seasonal calibration'} />
      </ExplainOnly>
      {colorBy !== 'season colour' && <ExplainOnly><Note>{colorBy === 'OP fraction' ? 'OP fraction = OPTR / (EC + OPTR), viridis 0.1 (purple) → 0.8 (yellow).' : 'TOR OC/EC, log plasma 1.5 → 12.'} Grey: missing.</Note></ExplainOnly>}
      {q('tor_vs_fabs') && <ExplainOnly><Note>TOR EC vs Fabs/10 slope {f2(q('tor_vs_fabs')?.deming_slope)} implies MAC ≈ {f2(q('tor_vs_fabs')?.implied_mac_deming)} m²/g (Deming); median Fabs/EC ratio {f2(q('tor_vs_fabs')?.mac_ratio_median)}, at the {f2(q('tor_vs_fabs')?.mac_percentile_median)}th percentile of the IMPROVE pool.</Note></ExplainOnly>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 6. MAC envelope
function MacEnvelope({ data }: { data: MeetingData }) {
  const [method, setMethod] = useSearchParam<string>('mmmethod', 'AIRSpec', METHODS)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const hst = data.mac_hist
  const edges: number[] = hst.edges
  const w = Math.max(560, width), height = 320, m = { top: 20, right: 20, bottom: 48, left: 60 }
  const iw = w - m.left - m.right, ih = height - m.top - m.bottom
  const x = d3.scaleLog().domain([edges[0], edges[edges.length - 1]]).range([0, iw])
  const dens = (c: number[]) => { const n = d3.sum(c) || 1; return c.map((v) => v / n) }
  const series = [{ key: 'pool', label: `IMPROVE pool (n ${hst.pool_n})`, color: '#8a8f98', med: hst.pool_median },
    ...data.seasons.map((g) => ({ key: `${method}|${g}`, label: `${g} analogs`, color: data.season_colors[g], med: hst[`${method}|${g}|median`] }))]
  const ymax = d3.max(series.flatMap((s) => dens(hst[s.key] ?? []))) ?? 0.1
  const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([ih, 0])
  const path = (c: number[]) => d3.line<number>().x((_, i) => x(Math.sqrt(edges[i] * edges[i + 1]))).y((v) => y(v)).curve(d3.curveMonotoneX)(dens(c)) ?? ''
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › MacEnvelope" title="MAC envelope — where the analogs sit in the IMPROVE TOR-vs-HIPS envelope (Fabs / TOR EC)"
      subtitle="Ann: does TOR EC vs Fabs give a MAC near 10 for the analogs, or do these samples sit at the low-EC end of the envelope?"
      controls={<MethodSeg label="Baseline (selection space)" value={method} onChange={setMethod} />}>
      <div ref={ref}>
        <svg width={w} height={height} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={y} x={0} label="Share of filters" tickCount={4} gridWidth={iw} labelX={-46} format={(v) => (+v).toFixed(2)} />
            <XAxis scale={x as any} y={ih} label="Fabs / TOR EC (m²/g)" labelDy={38} tickCount={6} format={(v) => String(+(+v).toPrecision(2))} />
            <line x1={x(10)} x2={x(10)} y1={0} y2={ih} stroke={INK.fit} strokeDasharray="3 3" />
            <text x={x(10) + 4} y={10} fontSize={11}>MAC 10</text>
            {series.map((s) => <path key={s.key} d={path(hst[s.key] ?? [])} fill="none" stroke={s.color} strokeWidth={s.key === 'pool' ? 3 : 2} />)}
            {series.map((s) => s.med != null && <line key={s.key + 'm'} x1={x(s.med)} x2={x(s.med)} y1={ih} y2={ih - 12} stroke={s.color} strokeWidth={3} />)}
            <g transform={`translate(${iw - 230},12)`} fontSize={11}>
              {series.map((s, i) => <g key={s.key} transform={`translate(0,${i * 16})`}><rect width={12} height={3} y={4} fill={s.color} /><text x={18} y={9}>{s.label} · median {f2(s.med)}</text></g>)}
            </g>
          </g>
        </svg>
      </div>
      <ExplainOnly><Note>Filters with TOR EC ≤ 0.02 µg/m³ are excluded (ratio unstable). A higher ratio means TOR EC is low for its absorption.</Note></ExplainOnly>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 7. OP
function OpStep({ data }: { data: MeetingData }) {
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › OpStep" title="OP / charring — OP does not track where filters sit on TOR EC vs HIPS" exportable={false}
      subtitle="Ann: pull OPR/OPT and colour the crossplots by OP. OPTR and OPTT are already in the local TOR table; the individual EC2/EC3/OC1–OC4 fractions are not (portal pull needed). Colour by OP in the three-crossplot section.">
      <table className="census-table">
        <thead><tr><th>Set</th><th>n</th><th>Spearman ρ (OP fraction, log MAC ratio)</th><th>Median OP fraction</th><th>Median OPTT / EC</th><th>Median MAC ratio</th><th>Median OC/EC</th><th>Median TOR EC (µg/m³)</th></tr></thead>
        <tbody>{data.op_summary.map((r) => <tr key={r.method + r.group}><td>{r.method === 'pool' ? 'IMPROVE pool' : `${r.method} · ${r.group}`}</td><td>{r.n}</td><td>{f2(r.spearman_op_vs_log_mac)}</td><td>{f2(r.op_frac_median)}</td><td>{f2(r.optt_over_ec_median)}</td><td>{f2(r.mac_ratio_median)}</td><td>{f2(r.tor_oc_ec_median)}</td><td>{f2(r.tor_ec_ugm3_median)}</td></tr>)}</tbody>
      </table>
      <table className="census-table" style={{ marginTop: 12 }}>
        <thead><tr><th>Set</th><th>Low-OP third: median MAC</th><th>Mid</th><th>High</th></tr></thead>
        <tbody>{Array.from(new Set(data.op_terciles.map((r) => `${r.method}|${r.group}`))).map((k) => {
          const rows = data.op_terciles.filter((r) => `${r.method}|${r.group}` === k)
          const v = (t: string) => f2(rows.find((r) => r.op_tercile === t)?.mac_ratio_median)
          return <tr key={k}><td>{k.startsWith('pool') ? 'IMPROVE pool' : k.replace('|', ' · ')}</td><td>{v('low OP')}</td><td>{v('mid OP')}</td><td>{v('high OP')}</td></tr>
        })}</tbody>
      </table>
      <Note>OP fraction = OPTR / (EC + OPTR). Dry analogs do carry more OP than the wet-season analogs, but within every set the MAC ratio barely changes across OP terciles.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 8. seasonality under all-IMPROVE
/** One Addis test set crossplot coloured by season, with a Deming line per season and for all Addis. */
function SeasonCrossplot({ data, fitId, title }: { data: MeetingData; fitId: string; title: string }) {
  const { methods } = useRegressions()
  const addisPts = useAddisPoints(data)
  const lines = data.seasons.flatMap((g) => {
    const s = statRow(data.stats, { fit_id: fitId, population: 'addis', eval_group: g })
    const points = addisPts(fitId, g)
    const st = regression(points.map(p => p.x), points.map(p => p.y), { errorsInVariables: true, lambda: s?.deming_lambda ?? data.summary.lambda, bootstrapGroups: monthBlocks(points.map(p => p.rec ?? {})) })
    return fitsFor(st, methods).map(fit => ({ ...fit, stats: st, color: data.season_colors[g], label: `${g} · ${fit.label}` }))
  })
  const all = statRow(data.stats, { fit_id: fitId, population: 'addis', eval_group: 'All Addis' })
  return (
    <div>
      <Scatter title={title} points={addisPts(fitId, 'all')} stats={all} demingLambda={data.summary.lambda} extraLines={lines} xLabel={MAC_LABEL} yLabel="FTIR EC (µg/m³)" maxSide={480} />
      <Key items={[
        ...data.seasons.map((g) => {
          const s = statRow(data.stats, { fit_id: fitId, population: 'addis', eval_group: g })
          return { label: `${g}: ${s ? `${f2(s.deming_slope)}x ${signed(s.deming_intercept)}` : '—'}`, color: data.season_colors[g] }
        }),
        { label: `${FIT_STYLE[methods[0] ?? 'deming'].label} fit, all Addis`, color: FIT_STYLE[methods[0] ?? 'deming'].color, kind: FIT_STYLE[methods[0] ?? 'deming'].dash ? 'dash' : 'line' },
        { label: '1:1', color: INK.identity, kind: 'dash' },
      ]} />
    </div>
  )
}

function SeasonTable({ data, rows }: { data: MeetingData; rows: { label: string; fitId: string }[] }) {
  const { showCI } = useRegressions()
  const cell = (s?: Row) => s ? <>{f2(s.deming_slope)} / {signed(s.deming_intercept)}<br /><span style={{ color: INK.muted, fontSize: 11 }}>{showCI && <>95% slope {f2(s.slope_ci_low)}–{f2(s.slope_ci_high)} · </>}R² {f2(s.R2)} · n {s.n}</span></> : '—'
  return (
    <table className="census-table">
      <thead><tr><th>Addis test set</th>{rows.map((r) => <th key={r.label}>{r.label}</th>)}</tr></thead>
      <tbody>{data.groups.map((g) => (
        <tr key={g}><td style={{ color: data.season_colors[g], fontWeight: 600 }}>{g}</td>
          {rows.map((r) => <td key={r.label}>{cell(statRow(data.stats, { fit_id: r.fitId, population: 'addis', eval_group: g }))}</td>)}</tr>
      ))}</tbody>
    </table>
  )
}

function StartingPoint({ data }: { data: MeetingData }) {
  const { showCI } = useRegressions()
  const fitId = 'deployed|SPARTAN|current SPARTAN calibration'
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › StartingPoint"
      title="Starting point — the Addis test set with the calibration SPARTAN uses today: FTIR EC vs HIPS Fabs / MAC 10, by season"
      subtitle="The poster's reference plot: the deployed SPARTAN FTIR EC (all IMPROVE, raw spectra) for the Addis filters that have it. Every other calibration on this page is judged against this.">
      <Grid>
        <SeasonCrossplot data={data} fitId={fitId} title="Addis, current SPARTAN calibration" />
        <ExplainOnly><SeasonTable data={data} rows={[{ label: 'Current SPARTAN calibration', fitId }]} /></ExplainOnly>
      </Grid>
      <ExplainOnly><Note>Deming with λ* {f2(data.summary.lambda)}. {showCI && 'Ranges are 95% month-block bootstrap intervals where available.'} Only the Addis filters with a deployed SPARTAN value are drawn, so n is smaller than elsewhere on this page.</Note></ExplainOnly>
    </ChartFrame>
  )
}

function Seasonality({ data }: { data: MeetingData }) {
  const { showCI } = useRegressions()
  const fitName = FIT_STYLE[useEstimator()].label
  const [method, setMethod] = useSearchParam<string>('msmethod', 'AIRSpec', METHODS)
  const [model, setModel] = useSearchParam<string>('msmodel', MODELS[0], MODELS)
  const fitId = `allimprove|${method}|${model}`
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Seasonality" title="Addis by season, all-IMPROVE model — the full baseline-corrected Addis test set coloured by season, with a Deming line per season"
      controls={<>
        <MethodSeg value={method} onChange={setMethod} />
        <ModelSeg label="Model" value={model} options={MODELS} onChange={setModel} />
      </>}>
      <Grid>
        <SeasonCrossplot data={data} fitId={fitId} title="Addis, all-IMPROVE model, coloured by season" />
        <ExplainOnly><SeasonTable data={data} rows={[{ label: 'All-IMPROVE model', fitId }, { label: 'Own-season calibration', fitId: `stitched|${method}|season calibrations` }]} /></ExplainOnly>
      </Grid>
      <ExplainOnly><Note>{fitName} line: all Addis; season-coloured lines: Deming fit within each season (λ* {f2(data.summary.lambda)}). {showCI && 'Bands are pointwise 95% month-block bootstrap intervals where available.'}</Note></ExplainOnly>
    </ChartFrame>
  )
}

const q3 = (vals: number[]) => { const v = vals.filter(Number.isFinite).sort((a, b) => a - b); return [d3.quantile(v, 0.5) ?? NaN, d3.quantile(v, 0.1) ?? NaN, d3.quantile(v, 0.9) ?? NaN] }
function stitchedSpread(data: MeetingData, method: string) {
  const r = data.stitched_repeats
  const sl: number[] = [], ic: number[] = []
  for (let i = 0; i < r.method.length; i++) if (r.method[i] === method) { sl.push(r.addis_slope[i] as number); ic.push(r.addis_intercept[i] as number) }
  return { slope: q3(sl), intercept: q3(ic), n: sl.length }
}

function Stitched({ data }: { data: MeetingData }) {
  const fitName = FIT_STYLE[useEstimator()].label
  const [method, setMethod] = useSearchParam<string>('mtmethod', 'AIRSpec', METHODS)
  const fitId = `stitched|${method}|season calibrations`
  const sp = stitchedSpread(data, method)
  const all = statRow(data.stats, { fit_id: fitId, population: 'addis', eval_group: 'All Addis' })
  const ref = statRow(data.stats, { fit_id: `allimprove|${method}|all-IMPROVE lot 251`, population: 'addis', eval_group: 'All Addis' })
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Stitched"
      title="Seasons stitched — each season's Addis filters predicted by its own season calibration, all on one test set crossplot"
      subtitle={`Ann's top priority: calibrate each season on its own analogs, then put the three seasons back together. Each season's line is its own calibration's readout; the ${fitName} line is the combined Addis result.`}
      controls={<MethodSeg value={method} onChange={setMethod} />}>
      <Grid>
        <SeasonCrossplot data={data} fitId={fitId} title={`Addis, own-season calibrations (${methodName(method)})`} />
        <div style={{ fontSize: 13, lineHeight: 1.6, alignSelf: 'center' }}>
          <p style={{ margin: '0 0 8px' }}><strong>Stitched, all Addis:</strong> {fitName} {f2(all?.deming_slope)}x {signed(all?.deming_intercept)} (R² {f2(all?.R2)})</p>
          <p style={{ margin: '0 0 8px' }}><strong>Over 100 cross-validation splits:</strong> slope {f2(sp.slope[0])} [{f2(sp.slope[1])}–{f2(sp.slope[2])}], intercept {signed(sp.intercept[0])} [{f2(sp.intercept[1])} to {f2(sp.intercept[2])}]</p>
          <p style={{ margin: 0, color: INK.muted }}>For comparison, one all-IMPROVE model (lot 251): {f2(ref?.deming_slope)}x {signed(ref?.deming_intercept)}. Each season's own slope is low, but the three season calibrations sit at different levels, so putting them back together steepens the combined line and pushes its intercept further from zero.</p>
        </div>
      </Grid>
      <ExplainOnly><SeasonTable data={data} rows={[{ label: 'Own-season calibrations', fitId }, { label: 'All-IMPROVE lot 251', fitId: `allimprove|${method}|all-IMPROVE lot 251` }, { label: 'Current SPARTAN', fitId: 'deployed|SPARTAN|current SPARTAN calibration' }]} /></ExplainOnly>
    </ChartFrame>
  )
}

/**
 * Where every calibration lands on test set slope vs intercept, with the target
 * at slope 1, intercept 0. Grid configurations and the all-IMPROVE / SPARTAN fits
 * are read out on all Addis; the season calibrations are shown both on their own
 * season (what each one achieves) and stitched (what they achieve together).
 */
function SlopeInterceptMap({ data }: { data: MeetingData }) {
  const { showCI } = useRegressions()
  const [method, setMethod] = useSearchParam<string>('mimethod', 'AIRSpec', METHODS)
  const [gridSet, setGridSet] = useSearchParam<string>('migrid', 'passing', ['passing', 'all', 'none'])
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  type P = { x: number; y: number; color: string; r: number; shape: 'dot' | 'tri' | 'star' | 'sq' | 'x'; op: number; tip: string[]; ci?: [number, number, number, number] }
  const pts: P[] = []
  const g = data.grid
  const catColor = d3.scaleOrdinal<string>(d3.schemeTableau10)
  if (g && gridSet !== 'none') {
    for (let i = 0; i < g.method.length; i++) {
      if (g.method[i] !== method) continue
      const pass = g.mode[i] === 'site_heldout' && ((g.heldout_R2[i] as number | null) ?? 0) >= 0.85
      if (gridSet === 'passing' && !pass) continue
      pts.push({ x: g.all_slope[i] as number, y: g.all_intercept[i] as number, color: catColor(String(g.label[i])), r: 2.2, shape: 'dot', op: 0.35,
        tip: [String(g.label[i]), `cohort ${g.cutoff[i]} filters · ${CV_PROTOCOL_SHORT[String(g.mode[i])] ?? g.mode[i]} · ${g.k[i]} PLS factors`, `test set ${f2(g.all_slope[i])}x ${signed(g.all_intercept[i])}`] })
    }
  }
  const rp = data.repeats
  for (const season of data.seasons) {
    const sl: number[] = [], ic: number[] = []
    for (let i = 0; i < rp.method.length; i++) if (rp.method[i] === method && rp.group[i] === season) {
      sl.push(rp.addis_slope[i] as number); ic.push(rp.addis_intercept[i] as number)
      pts.push({ x: rp.addis_slope[i] as number, y: rp.addis_intercept[i] as number, color: data.season_colors[season], r: 2.4, shape: 'dot', op: 0.3,
        tip: [`${season} calibration, split ${rp.repeat[i]}`, 'read out on its own season only', `${f2(rp.addis_slope[i])}x ${signed(rp.addis_intercept[i])}`] })
    }
    pts.push({ x: q3(sl)[0], y: q3(ic)[0], color: data.season_colors[season], r: 8, shape: 'tri', op: 1,
      tip: [`${season} calibration: median of 100 splits`, 'read out on its own season only', `${f2(q3(sl)[0])}x ${signed(q3(ic)[0])}`] })
  }
  const st = data.stitched_repeats
  for (let i = 0; i < st.method.length; i++) if (st.method[i] === method)
    pts.push({ x: st.addis_slope[i] as number, y: st.addis_intercept[i] as number, color: '#6d28d9', r: 2.4, shape: 'dot', op: 0.35,
      tip: [`Seasons stitched, split ${st.repeat[i]}`, 'all Addis', `${f2(st.addis_slope[i])}x ${signed(st.addis_intercept[i])}`] })
  const sp = stitchedSpread(data, method)
  pts.push({ x: sp.slope[0], y: sp.intercept[0], color: '#6d28d9', r: 10, shape: 'star', op: 1, tip: ['Seasons stitched: median of 100 splits', 'all Addis', `${f2(sp.slope[0])}x ${signed(sp.intercept[0])}`] })
  const one = (fit: string, color: string, shape: P['shape'], name: string) => {
    const s = statRow(data.stats, { fit_id: fit, population: 'addis', eval_group: 'All Addis' })
    if (s) pts.push({ x: s.deming_slope, y: s.deming_intercept, color, r: 8, shape, op: 1,
      ci: [s.slope_ci_low, s.slope_ci_high, s.intercept_ci_low, s.intercept_ci_high].every(v => typeof v === 'number' && Number.isFinite(v)) ? [s.slope_ci_low, s.slope_ci_high, s.intercept_ci_low, s.intercept_ci_high] : undefined,
      tip: [name, 'all Addis', `${f2(s.deming_slope)}x ${signed(s.deming_intercept)}`] })
  }
  one(`allimprove|${method}|all-IMPROVE lot 251`, '#111827', 'sq', 'All-IMPROVE model (lot 251)')
  one('deployed|SPARTAN|current SPARTAN calibration', '#b91c1c', 'x', 'Current SPARTAN calibration')

  const xs = pts.map((p) => p.x).filter(Number.isFinite), ys = pts.map((p) => p.y).filter(Number.isFinite)
  const xDom = [Math.min(0, d3.quantile(xs.sort(d3.ascending), 0.005) ?? 0), Math.max(1.5, d3.quantile(xs, 0.995) ?? 2)]
  const yDom = [Math.min(-1, d3.quantile(ys.sort(d3.ascending), 0.005) ?? -5), Math.max(1, d3.quantile(ys, 0.995) ?? 1)]
  const m = { top: 20, right: 20, bottom: 50, left: 64 }
  const side = Math.max(300, Math.min((width || 600) - m.left - m.right, 560))
  const x = d3.scaleLinear().domain(xDom).range([0, side]).nice()
  const y = d3.scaleLinear().domain(yDom).range([side, 0]).nice()
  const cx = (v: number) => x(Math.max(x.domain()[0], Math.min(x.domain()[1], v)))
  const cy = (v: number) => y(Math.max(y.domain()[0], Math.min(y.domain()[1], v)))
  const mark = (p: P, key: number) => {
    const X = cx(p.x), Y = cy(p.y), r = p.r
    const ev = { onMouseEnter: (e: React.MouseEvent) => tip.show(e, p.tip), onMouseLeave: tip.hide }
    if (p.shape === 'dot') return <circle key={key} cx={X} cy={Y} r={r} fill={p.color} fillOpacity={p.op} {...ev} />
    if (p.shape === 'sq') return <rect key={key} x={X - r * 0.8} y={Y - r * 0.8} width={r * 1.6} height={r * 1.6} fill={p.color} stroke="white" {...ev} />
    if (p.shape === 'x') return <g key={key} {...ev}><line x1={X - r} x2={X + r} y1={Y - r} y2={Y + r} stroke={p.color} strokeWidth={3} /><line x1={X - r} x2={X + r} y1={Y + r} y2={Y - r} stroke={p.color} strokeWidth={3} /></g>
    if (p.shape === 'tri') return <path key={key} d={`M${X},${Y - r} L${X + r},${Y + r * 0.8} L${X - r},${Y + r * 0.8} Z`} fill={p.color} stroke="#111827" strokeWidth={1} {...ev} />
    return <path key={key} d={d3.symbol(d3.symbolStar, r * r * 3)() ?? ''} transform={`translate(${X},${Y})`} fill={p.color} stroke="white" {...ev} />
  }
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › SlopeInterceptMap"
      title="Slope vs intercept — where the season calibrations land among every configuration tried, against the target (slope 1, intercept 0)"
      subtitle="Test set (Addis) Deming slope and intercept at MAC 10. Grid points, the all-IMPROVE model, SPARTAN and the stitched seasons are read out on all Addis; each season calibration is also shown read out on its own season."
      controls={<>
        <MethodSeg value={method} onChange={setMethod} />
        <Segmented label="Grid configurations" value={gridSet as any} options={['passing', 'all', 'none'] as const} onChange={setGridSet as any} />
      </>}>
      <div ref={ref}>
        <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            <rect x={x(0.8)} width={x(1.2) - x(0.8)} y={y(1)} height={y(-1) - y(1)} fill={INK.good} fillOpacity={0.1} />
            <YAxis scale={y} x={0} label="Test set Deming intercept (µg/m³)" gridWidth={side} tickCount={6} labelX={-46} />
            <XAxis scale={x} y={side} label="Test set Deming slope" tickCount={6} labelDy={38} />
            <line x1={x(1)} x2={x(1)} y1={0} y2={side} stroke={INK.identity} strokeDasharray="4 3" />
            <line x1={0} x2={side} y1={y(0)} y2={y(0)} stroke={INK.identity} strokeDasharray="4 3" />
            {showCI && pts.filter(p => p.ci).map((p, i) => <g key={`ci-${i}`} data-fit-uncertainty="saved-coefficients" stroke={p.color} strokeOpacity={0.55} pointerEvents="none">
              <line x1={cx(p.ci![0])} x2={cx(p.ci![1])} y1={cy(p.y)} y2={cy(p.y)} />
              <line x1={cx(p.x)} x2={cx(p.x)} y1={cy(p.ci![2])} y2={cy(p.ci![3])} />
            </g>)}
            {pts.filter((p) => p.shape === 'dot').map(mark)}
            {pts.filter((p) => p.shape !== 'dot').map((p, i) => mark(p, 100000 + i))}
            <text x={x(1) + 6} y={y(0) - 6} fontSize={11} fill={INK.good}>target</text>
          </g>
        </svg>
        <Key items={[
          ...(gridSet !== 'none' ? [{ label: `grid configurations (${gridSet === 'passing' ? 'site-grouped, IMPROVE cross-validation R² ≥ 0.85' : 'all'}), coloured by cohort`, color: '#9aa5b1' }] : []),
          ...data.seasons.map((g2) => ({ label: `${g2} calibration, own season (▲ median, dots = splits)`, color: data.season_colors[g2] })),
          { label: 'seasons stitched, all Addis (★ median, dots = splits)', color: '#6d28d9' },
          { label: 'all-IMPROVE model (■)', color: '#111827', kind: 'bar' as const },
          { label: 'current SPARTAN (✕)', color: '#b91c1c', kind: 'bar' as const },
          ...(showCI ? [{ label: 'saved 95% marginal intervals; endpoints clipped at axes', color: INK.muted, kind: 'bar' as const }] : []),
          { label: 'target box: slope 0.8–1.2, |intercept| < 1', color: INK.good, kind: 'faded' as const },
        ]} />
        {tip.node}
      </div>
      <ExplainOnly><Note>Points beyond the axes are pinned to the edge. A season calibration read out on its own season spans a narrow Fabs range, so its slope and intercept are not comparable one-to-one with the all-Addis readouts; the stitched star is the like-for-like comparison.</Note></ExplainOnly>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 9. combined comparison (forest)
function Combined({ data }: { data: MeetingData }) {
  const { showCI } = useRegressions()
  const [method, setMethod] = useSearchParam<string>('mcmethod', 'AIRSpec', METHODS)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const cals = [
    { label: 'All-IMPROVE lot 251', fit: `allimprove|${method}|all-IMPROVE lot 251`, byGroup: true },
    { label: 'All-IMPROVE both lots (frozen)', fit: `allimprove|${method}|all-IMPROVE both lots (frozen Colab)`, byGroup: true },
    { label: 'Analogs of all Addis', fit: `seasonal|${method}|locked|All Addis`, byGroup: false },
    { label: 'Own-season analogs', fit: 'season', byGroup: false },
    { label: 'Three seasonal sets combined', fit: `combined|${method}|three seasonal sets combined`, byGroup: false },
    { label: 'VIP/Euclidean analogs (500)', fit: `combined|${method}|VIP/Euclidean analogs (500)`, byGroup: false },
  ]
  const rows: { g: string; cal: string; s: Row }[] = []
  for (const g of data.groups) for (const c of cals) {
    const fit = c.fit === 'season' ? `seasonal|${method}|locked|${g}` : c.fit
    const s = c.byGroup ? statRow(data.stats, { fit_id: fit, group: g, population: 'addis', eval_group: g }) : statRow(data.stats, { fit_id: fit, population: 'addis', eval_group: g })
    if (s) rows.push({ g, cal: c.label, s })
  }
  const rowH = 16, gap = 14, left = 260
  const w = Math.max(700, width), panelW = (w - left - 40) / 2
  const h = rows.length * rowH + data.groups.length * gap + 60
  const xs = d3.scaleLinear().domain([0, Math.max(1.5, d3.max(rows, (r) => (showCI ? r.s.slope_ci_high : null) ?? r.s.deming_slope) ?? 1.5)]).range([0, panelW - 20]).nice()
  const xi = d3.scaleLinear().domain([Math.min(-2, d3.min(rows, (r) => (showCI ? r.s.intercept_ci_low : null) ?? r.s.deming_intercept) ?? -2), Math.max(0.5, d3.max(rows, (r) => (showCI ? r.s.intercept_ci_high : null) ?? r.s.deming_intercept) ?? 0.5)]).range([0, panelW - 20]).nice()
  const calColor = d3.scaleOrdinal<string>().domain(cals.map((c) => c.label)).range(['#111827', '#6b7280', '#2171b5', '#c026d3', '#0f766e', '#d97706'])
  let cursor = 0
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Combined" title="Calibrations compared — Addis Deming slope and intercept by season for every calibration"
      subtitle="Ann: combine the three seasonal analog sets into one calibration and compare against the VIP/Euclidean analogs and the seasonally calibrated results."
      controls={<MethodSeg value={method} onChange={setMethod} />}>
      <div ref={ref}>
        <svg width={w} height={h} fontFamily={FONT.family}>
          <g transform={`translate(${left},24)`}>
            <text x={0} y={-8} fontSize={12} fontWeight={600}>Slope (1 = agreement)</text>
            <text x={panelW + 20} y={-8} fontSize={12} fontWeight={600}>Intercept (µg/m³, 0 = agreement)</text>
            <line x1={xs(1)} x2={xs(1)} y1={0} y2={h - 50} stroke={INK.identity} strokeDasharray="4 3" />
            <line x1={panelW + 20 + xi(0)} x2={panelW + 20 + xi(0)} y1={0} y2={h - 50} stroke={INK.identity} strokeDasharray="4 3" />
            {data.groups.map((g) => {
              const block = rows.filter((r) => r.g === g)
              const top = cursor
              cursor += block.length * rowH + gap
              return <g key={g} transform={`translate(0,${top})`}>
                <text x={-left + 4} y={10} fontSize={12} fontWeight={600} fill={data.season_colors[g]}>{g}</text>
                {block.map((r, j) => {
                  const yy = j * rowH + 8, c = calColor(r.cal)
                  return <g key={r.cal} onMouseEnter={(e) => tip.show(e, [`${g} · ${r.cal}`, `slope ${f2(r.s.deming_slope)}${showCI ? ` · 95% CI [${f2(r.s.slope_ci_low)}, ${f2(r.s.slope_ci_high)}]` : ''}`, `intercept ${signed(r.s.deming_intercept)}${showCI ? ` · 95% CI [${f2(r.s.intercept_ci_low)}, ${f2(r.s.intercept_ci_high)}]` : ''}`, `R² ${f2(r.s.R2)} · n ${r.s.n}`])} onMouseLeave={tip.hide}>
                    <text x={-8} y={yy + 4} fontSize={10.5} textAnchor="end" fill={INK.muted}>{r.cal}</text>
                    {showCI && r.s.slope_ci_low != null && r.s.slope_ci_high != null && <line data-fit-uncertainty="saved-slope" x1={xs(r.s.slope_ci_low)} x2={xs(r.s.slope_ci_high)} y1={yy} y2={yy} stroke={c} strokeWidth={2} strokeOpacity={0.5} />}
                    <circle cx={xs(r.s.deming_slope)} cy={yy} r={4} fill={c} />
                    <g transform={`translate(${panelW + 20},0)`}>
                      {showCI && r.s.intercept_ci_low != null && r.s.intercept_ci_high != null && <line data-fit-uncertainty="saved-intercept" x1={xi(r.s.intercept_ci_low)} x2={xi(r.s.intercept_ci_high)} y1={yy} y2={yy} stroke={c} strokeWidth={2} strokeOpacity={0.5} />}
                      <circle cx={xi(r.s.deming_intercept)} cy={yy} r={4} fill={c} />
                    </g>
                  </g>
                })}
              </g>
            })}
            <XAxis scale={xs} y={h - 50} tickCount={6} />
            <g transform={`translate(${panelW + 20},0)`}><XAxis scale={xi} y={h - 50} tickCount={6} /></g>
          </g>
        </svg>
        <Key items={[
          ...cals.map((c) => ({ label: c.label, color: calColor(c.label) })),
          ...(showCI ? [{ label: '95 % month-block bootstrap (where saved)', color: INK.muted, kind: 'bar' as const }] : []),
          { label: 'agreement (slope 1, intercept 0)', color: INK.identity, kind: 'dash' as const },
        ]} />
        {tip.node}
      </div>
      <ExplainOnly><Note>{showCI ? 'Bars: saved 95% month-block bootstrap intervals; missing intervals are not drawn.' : 'Fit uncertainty hidden; enable it in the sidebar to show saved intervals.'} VIP/Euclidean analogs exist only in spline-baselined space (the explorer's corrected-space ranking). Combined set: union of the three seasonal top-500 lists.</Note></ExplainOnly>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 10. grid heat map
const METRICS = ['intercept', 'slope', 'CV R²'] as const
const READOUTS = ['all', 'dry', 'belg', 'kiremt'] as const
function GridStep({ data }: { data: MeetingData }) {
  const [method, setMethod] = useSearchParam<string>('mgmethod', 'AIRSpec', METHODS)
  const [mode, setMode] = useSearchParam<string>('mmode', 'site_heldout', ['site_heldout', 'app'])
  const [metric, setMetric] = useSearchParam<string>('mmetric', 'intercept', METRICS)
  const [readout, setReadout] = useSearchParam<string>('mread', 'all', READOUTS)
  const [gate, setGate] = useState(true)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const g = data.grid
  if (!g) return <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › GridStep" title="Category grid"><Empty>The grid has not been exported yet.</Empty></ChartFrame>
  const idx: number[] = []
  for (let i = 0; i < g.method.length; i++) if (g.method[i] === method && g.mode[i] === mode) idx.push(i)
  const valueKey = metric === 'CV R²' ? 'heldout_R2' : `${readout}_${metric}`
  const cats = Array.from(new Set(idx.map((i) => String(g.label[i]))))
  const binW = 25
  const cutBin = (c: number) => Math.round(c / binW) * binW
  const bins = Array.from(new Set(idx.map((i) => cutBin(g.cutoff[i] as number)))).sort((a, b) => a - b)
  const cells = new Map<string, { v: number; n: number; pass: number; k: number[] }>()
  for (const i of idx) {
    const v = g[valueKey]?.[i] as number | null
    if (v == null) continue
    const key = `${g.label[i]}|${cutBin(g.cutoff[i] as number)}`
    const c = cells.get(key) ?? { v: 0, n: 0, pass: 0, k: [] }
    c.v += v; c.n += 1; c.k.push(g.k[i] as number)
    if (mode !== 'site_heldout' || ((g.heldout_R2[i] as number | null) ?? 0) >= 0.85) c.pass += 1
    cells.set(key, c)
  }
  const left = 230, top = 20
  const w = Math.max(600, width), cw = Math.max(3, (w - left - 20) / Math.max(1, bins.length)), chH = 26
  const h = cats.length * chH + 70
  const dom = metric === 'intercept' ? [-2.5, 0, 2.5] : metric === 'slope' ? [0, 1, 2] : [0.5, 0.85, 1]
  const color = d3.scaleLinear<string>().domain([dom[0], (dom[0] + dom[1]) / 2, dom[1], (dom[1] + dom[2]) / 2, dom[2]])
    .range(metric === 'CV R²' ? ['#f7f7f7', '#d1e5f0', '#67a9cf', '#2166ac', '#053061'] : [RAMP_DIVERGING[0], RAMP_DIVERGING[2], '#f7f7f7', RAMP_DIVERGING[4], RAMP_DIVERGING[6]]).clamp(true)
  return (
    <ChartFrame title="Category grid — prioritization categories re-run on baselined spectra: Addis readout by cohort size"
      subtitle={`${data.grid_n} fitted cohorts: category × cutoff (step 5) × baseline × CV. Cells average the fits inside each 25-filter bin. Replaces the intercept continuum plot.`}
      controls={<>
        <MethodSeg value={method} onChange={setMethod} />
        <Segmented label="Cross-validation protocol" value={CV_PROTOCOL_SHORT[mode] as any} options={[CV_PROTOCOL_SHORT.site_heldout, CV_PROTOCOL_SHORT.app] as any} onChange={(v: string) => setMode(v === CV_PROTOCOL_SHORT.app ? 'app' : 'site_heldout')} />
        <Segmented label="Colour" value={metric as any} options={METRICS} onChange={setMetric as any} />
        {metric !== 'CV R²' && <Segmented label="Test set (Addis) readout" value={readout as any} options={READOUTS} onChange={setReadout as any} />}
        {mode === 'site_heldout' && <label className="toggle"><input type="checkbox" checked={gate} onChange={(e) => setGate(e.target.checked)} /> Fade cells failing IMPROVE cross-validation R² ≥ 0.85</label>}
      </>}>
      <div ref={ref} style={{ overflowX: 'auto' }}>
        <svg width={w} height={h} fontFamily={FONT.family}>
          <g transform={`translate(${left},${top})`}>
            {cats.map((cat, r) => (
              <g key={cat} transform={`translate(0,${r * chH})`}>
                <text x={-8} y={chH / 2 + 4} textAnchor="end" fontSize={11.5} fill={INK.text}>{cat}</text>
                {bins.map((b, c) => {
                  const cell = cells.get(`${cat}|${b}`)
                  if (!cell) return null
                  const v = cell.v / cell.n
                  const faded = gate && mode === 'site_heldout' && metric !== 'CV R²' && cell.pass === 0
                  return <rect key={b} x={c * cw} y={1} width={cw - 0.5} height={chH - 2} fill={color(v)} opacity={faded ? 0.3 : 1} stroke={faded ? 'none' : '#1f2933'} strokeWidth={faded ? 0 : 0.3}
                    onMouseEnter={(e) => tip.show(e, [`${cat} · cutoff ≈ ${b}`, `${metric}${metric !== 'CV R²' ? ` (${readout})` : ''}: ${f2(v)}`, `${cell.n} fits · k ${d3.min(cell.k)}–${d3.max(cell.k)}`, mode === 'site_heldout' ? `${cell.pass}/${cell.n} pass IMPROVE cross-validation R² ≥ 0.85` : 'interleaved CV holds out no whole site: no cross-validation R² gate'])} onMouseLeave={tip.hide} />
                })}
              </g>
            ))}
            {bins.map((b, c) => (b % 250 === 0) && <text key={b} x={c * cw + cw / 2} y={cats.length * chH + 16} fontSize={10.5} textAnchor="middle" fill={INK.muted}>{b}</text>)}
            <text x={(bins.length * cw) / 2} y={cats.length * chH + 36} fontSize={12} fontWeight={600} textAnchor="middle">Cohort size (IMPROVE filters)</text>
          </g>
        </svg>
        <Key items={mode === 'site_heldout' && gate && metric !== 'CV R²' ? [{ label: 'no fit in the bin passes IMPROVE cross-validation R² ≥ 0.85', color: '#9aa5b1', kind: 'faded' }] : []}>
          <ColorLegend scale={color} width={220}
            label={metric === 'intercept' ? `Test set (Addis) Deming intercept (µg/m³, ${readout})` : metric === 'slope' ? `Test set (Addis) Deming slope (${readout})` : 'IMPROVE cross-validation R² (calibration set)'}
            note={metric === 'intercept' ? 'white = 0' : metric === 'slope' ? 'white = 1' : undefined} />
        </Key>
        {tip.node}
      </div>
      {(data.grid_summary ?? []).map((gs) => <ExplainOnly key={gs.method}><Note>{methodName(gs.method)}, site-grouped cross-validation: {gs.passing} fits pass IMPROVE cross-validation R² ≥ 0.85; their all-Addis test set slopes span {f2(gs.slope_min)}–{f2(gs.slope_max)} and intercepts {signed(gs.intercept_min)} to {signed(gs.intercept_max)}; {gs.balanced} reach slope 0.8–1.2 with |intercept| &lt; 1.</Note></ExplainOnly>)}
      <ExplainOnly><Note>Colour scale: {metric === 'intercept' ? 'blue = negative intercept, red = positive, white = 0 (µg/m³, clamped at ±2.5)' : metric === 'slope' ? 'blue = slope below 1, white = 1, red = above (clamped 0–2)' : 'white 0.5 → dark blue 1.0'}. Single-cohort categories (entire pool, smoke 906) sit in one column.</Note></ExplainOnly>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 11. still open
function NextSteps({ data }: { data: MeetingData }) {
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › NextSteps" title="Still open after this pass" exportable={false}>
      <ul style={{ lineHeight: 1.6, fontSize: 14 }}>
        <li><strong>EC2, EC3 and OC1–OC4 fractions.</strong> Not in the local TOR table (only EC, EC1, OC, OPTR, OPTT). Needs a FED/IMPROVE portal query (IMPAER; EC1f–EC3f, OC1f–OC4f, OPf/OPTf) for the lot 248/251 sites.</li>
        <li><strong>Raw Addis spectra for Francois.</strong> VIBES already runs here on the whole pool; a run by Francois on a few Addis spectra would confirm the adapter matches his code.</li>
        <li><strong>Scoring decision.</strong> {(data.grid_summary ?? []).map((g) => <span key={g.method}>{methodName(g.method)}: {g.passing} site-grouped fits pass IMPROVE cross-validation R² ≥ 0.85, {g.balanced} of them have Addis slope 0.8–1.2 with |intercept| &lt; 1 (best balanced: {g.best_label} {g.best_cutoff}, {f2(g.best_slope)} / {signed(g.best_intercept)}). </span>)}Slope target, intercept target, or a range for both?</li>
        <li><strong>Why FTIR under-reads TOR on the cross-validation analogs</strong> (slope 0.5–0.8). Part is expected PLS shrinkage at low concentrations; whether any is composition needs a loading-matched test. Addis HIPS-equivalent loadings (~30–40 µg/filter) sit above most analogs (median 4–13 µg/filter).</li>
        <li><strong>Independent Addis EC.</strong> Every Addis number here is against Fabs/10.</li>
      </ul>
      <Note>Analysis tables: {data.generated_from}</Note>
    </ChartFrame>
  )
}

function projectMeeting(data: MeetingData, method: import('@/lib/regressionPrefs').Estimator): MeetingData {
  if (method !== 'ols') return data
  return { ...data, stats: data.stats.map(s => ({ ...s, deming_slope: s.ols_slope, deming_intercept: s.ols_intercept,
    slope_ci_low: null, slope_ci_high: null, intercept_ci_low: null, intercept_ci_high: null })) }
}

function SavedFitLabel({ text }: { text: string }) {
  return <>{text.replace(/Deming/g, FIT_STYLE[useEstimator()].label)}</>
}
