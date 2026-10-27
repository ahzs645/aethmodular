import { monthBlocks } from '@/lib/regressionCI'
import { robustDomain } from '@/lib/axes'
import { RefLine } from '@/components/RefLines'
import { regression } from '@/lib/stats'
import { EstimatorViews, FIT_STYLE, useEstimator, useRegressions, fitsFor } from '@/lib/regressionPrefs'
import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Select, Segmented, Toggle } from '@/components/ChartFrame'
import { BaselineOptionsProvider, BaselinePicker, CountPicker, SeasonPicker, clampCount, useBaselineOptions } from '@/components/Pickers'
import { PageToc } from '@/components/PageToc'
import { Crossplot, sharedCrossDomains, type CrossPoint } from '@/components/Crossplot'
import { Key as SharedKey } from '@/components/Key'
import { IntervalStrip } from '@/components/IntervalStrip'
import { BandCell, FitCell } from '@/components/TableCells'
import { ExportState } from '@/components/ExportState'
import { useExport } from '@/lib/data'
import { FullRangeVibesStatus } from '@/components/FullRangeVibesStatus'
import { MeetingCalibrationDetails } from '@/components/MeetingCalibrationDetails'
import { SeasonalComposition } from '@/components/SeasonalComposition'
import { SpecCurve } from '@/charts/calibration/SpecCurve'
import { XAxis, YAxis } from '@/components/Axes'
import { SpectrumPlot, type SpectrumBand, type SpectrumSeries } from '@/components/SpectrumPlot'
import { channelQuantiles, iqrRows } from '@/lib/spectra'
import { SweepPlot } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { PageChoiceControls, PageChoices, usePageChoice } from '@/lib/pageChoices'
import { SidebarControls } from '@/lib/pageSidebar'
import { baseFilterId, dotRim, useHighlight } from '@/lib/highlight'
import { ADDIS_ALL, BASELINE_COLOR, FONT, IMPROVE_GREY, INK, OCT1_HATCH, SPECTRAL_REGION, dotOpacity } from '@/lib/theme'
import { ANALOG_SELECTION_EXCLUDED, OCT1_ALSO_EXCLUDED } from '@/components/SpectralRegions'
import { coef, coefSigned, f2, fmt, signed } from '@/lib/format'
import { ALL_ADDIS, ANALYSIS_CALENDAR_NOTE, inSeasonGroup, isAllAddis, seasonPhrase, shortSeason } from '@/lib/seasons'
import { ColorLegend } from '@/components/ColorLegend'
import { CV_PROTOCOL_SHORT, PREPROCESSING, baselineName, ecAxis } from '@/lib/labels'
import { Traces, loadTraces } from '@/lib/similarity'
import type { FilterRow, MetaFile } from '@/lib/types'
import { AxisChoices, type AxisOption } from '@/components/AxisChoices'
import { withUnit } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'

/**
 * Click-through for the 2026-09-17 Ann/Satoshi meeting follow-up. Every number
 * comes from research/ftir_hips_chem/output/tables/meeting_followup_20260917
 * via gallery/data/export_meeting_followup.py. Crossplot overlays refit complete
 * pairs with the saved error ratio; the underlying calibrations stay fixed.
 */

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
  /** every held-out IMPROVE filter under the all-IMPROVE models (absent from older exports) */
  pool?: Cols | null
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

const methodName = (m: string) => PREPROCESSING[m] ?? m
const MODELS = ['all-IMPROVE lot 251', 'all-IMPROVE both lots (frozen Colab)'] as const
const modelLabel = (value: string) => ({
  'all-IMPROVE lot 251': 'All-IMPROVE model, lot 251',
  'all-IMPROVE both lots (frozen Colab)': 'All-IMPROVE model, both lots',
  'seasonal calibration': 'Seasonal calibration',
}[value] ?? value)
function ModelSeg({label, value, options, onChange}: {label:string; value:string; options:readonly string[]; onChange:(v:string)=>void}) {
  return <Segmented label={label} value={modelLabel(value)} options={options.map(modelLabel)} onChange={v=>onChange(options.find(o=>modelLabel(o)===v)??options[0])} />
}
const MAC_LABEL = ecAxis('hips')


export function MeetingFollowupPage({ galleryRows = [], galleryMeta, viewRows = [], viewMeta }: {
  galleryRows?: FilterRow[]; galleryMeta?: MetaFile
  /** the gallery's rows and meta under the reader's view (ratios, MAC, mass per filter, weather): the season plots' other axes */
  viewRows?: FilterRow[]; viewMeta?: MetaFile
} = {}) {
  const [archiveMode, setArchiveMode] = useSearchParam<string>('marchive', 'hide', ['hide', 'show'])
  const [version, setVersion] = useSearchParam<string>('mversion', 'current', import.meta.env.VITE_SHARE_TABS ? ['current'] : ['current', 'earlier'])
  const showArchived = version === 'earlier' && archiveMode === 'show'
  const { data, error } = useExport<MeetingData>((version === 'current' ? 'meeting-r70/' : 'meeting/') + 'followup_20260917.json',
    (d) => d.schema_version !== 1 || (version === 'current' && d.summary.background_rank !== 70) ? 'unsupported data version' : null)
  if (!data) return <ExportState title="Meeting follow-up" what="the meeting follow-up export" error={error} command="uv run --no-sync python gallery/data/export_meeting_followup.py" />
  const methods: string[] = (data.summary.methods ?? ['AIRSpec', 'VIBES']).filter((m: string) => showArchived || m !== 'VIBES')
  return <BaselineOptionsProvider methods={methods}><MeetingView key={version + String(showArchived)} data={data} galleryRows={galleryRows} galleryMeta={galleryMeta} viewRows={viewRows} viewMeta={viewMeta}
    version={version} onVersion={setVersion} showArchived={showArchived} onShowArchived={v => setArchiveMode(v ? 'show' : 'hide')} /></BaselineOptionsProvider>
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

type Pt = CrossPoint

// ------------------------------------------------------------------ page
function MeetingView({ data, galleryRows, galleryMeta, viewRows, viewMeta, version, onVersion, showArchived, onShowArchived }: {
  data: MeetingData; galleryRows: FilterRow[]; galleryMeta?: MetaFile; viewRows: FilterRow[]; viewMeta?: MetaFile
  version: string; onVersion: (v: string) => void; showArchived: boolean; onShowArchived: (v: boolean) => void
}) {
  const hl = useHighlight()
  const seasonAxes = useSeasonAxes(viewRows, viewMeta)
  const baselines = useBaselineOptions()
  // a clicked filter stays pinned on every chart until empty plot space is clicked or Esc
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') hl.setPinned(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [hl])
  return (
    <SeasonAxesCtx.Provider value={seasonAxes.value}>
    <AxisChoices noun="measurements" options={seasonAxes.options} x={seasonAxes.x} y={seasonAxes.y}>
    <PageToc>
      <ExplainOnly><FullRangeVibesStatus /></ExplainOnly>
      {/* share builds ship only the current rank-70 export (gallery/share/build_share.py) */}
      {/* the version switches sit in the sidebar; what each version is, is explanation (Explain on) */}
      <SidebarControls title="Analysis">
        {!import.meta.env.VITE_SHARE_TABS && <Segmented label="VIBES run" value={version} options={['current', 'earlier']} onChange={onVersion} optionLabel={v => v === 'current' ? 'Completed · rank 70' : 'Archived · preliminary rank 30'} />}
        {version === 'earlier' && <Toggle label="Show archived VIBES fit (4000–1425)" checked={showArchived} onChange={onShowArchived} />}
      </SidebarControls>
      {!import.meta.env.VITE_SHARE_TABS && <ExplainOnly><ChartFrame title="Analysis version" exportable={false}>
        <p>{version === 'current' ? 'All seasonal, combined-set, calibration and specification-grid VIBES models below use the completed rank-70 correction. Full and shared prediction windows are separate options.' : 'Earlier seasonal and specification results retain their original preliminary rank-30 corrections. These archived models have not been overwritten.'}</p>
      </ChartFrame></ExplainOnly>}
      <ExplainOnly>
      <ChartFrame title="Baseline versions" exportable={false}>
        <p>The VIBES fit restricted to 4000–1425 cm⁻¹ is archived. Current VIBES variants fit the full available 4000–500 cm⁻¹ grid. The narrower prediction variant is a sensitivity comparison: it uses the corrected 4000–1425 channels to test the effect of adding lower-wavenumber information. Archived results and their mixed specification grid remain available below.</p>
      </ChartFrame></ExplainOnly>
      <CohortCoverage data={data} galleryRows={galleryRows} />
      <SeasonalComposition rows={galleryRows} meta={galleryMeta} />
      <ExplainOnly><Answers data={data} /></ExplainOnly>
      <ExplainOnly><Levers data={data} /></ExplainOnly>
      <PageChoices panel={false} defs={[{ key: 'baseline', urlKey: 'mbase', label: 'Baseline', default: 'AIRSpec', options: baselines,
        legacy: ['mrmethod', 'm3method', 'mmmethod', 'msmethod', 'mtmethod', 'mimethod', 'mcmethod', 'mgmethod'],
        render: (v, set) => <BaselinePicker value={v} onChange={set} /> }]}>
      <SidebarControls title="For every chart on this page"><PageChoiceControls /></SidebarControls>
      <EstimatorViews>{method => <StartingPoint data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews fixed="deming">{() => <RepeatStrip data={data} />}</EstimatorViews>
      <EstimatorViews>{method => <ThreePlots data={projectMeeting(data, method)} />}</EstimatorViews>
      <BaselineCompare data={data} />
      <MacEnvelope data={data} />
      <ExplainOnly><OpStep data={data} /></ExplainOnly>
      <EstimatorViews>{method => <Seasonality data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews>{method => <Stitched data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews fixed="deming">{() => <SlopeInterceptMap data={data} />}</EstimatorViews>
      <EstimatorViews>{method => <Combined data={projectMeeting(data, method)} />}</EstimatorViews>
      <EstimatorViews available={['deming', 'york']}>{() => <SpecCurve
        id="meeting-spec-curve"
        src={data.summary.spec_curve_export ?? 'meeting/spec_curve_20260917.json'}
        title={`Specification curve — ${showArchived ? 'active and archived' : 'active'} baselines from the meeting grid`}
        excludePreprocessing={showArchived ? [] : ['vibes']}
        subtitle={(f) => `All ${f ? f.n.toLocaleString() + ' ' : ''}baselined Addis specifications after the 17 Sep levers (spline and VIBES baselines; CO₂ and >3600 cm⁻¹ excluded from selection; site-grouped or interleaved cross-validation; cohort sizes every 5 filters; number of PLS factors by each protocol's rule), sorted by Deming intercept at MAC 10, with the analytic choices underneath: a darker strip means more of the specifications at that position used the choice. Click a choice to keep only specifications that use it; click several to intersect. Red points use the selected highlight criteria; by default they pass every saved guardrail${f ? ': ' + f.guardrails : ''}.`}
        provenance={version === 'current' ? 'Completed rank-70 migration · category_grid.jsonl · original specification guardrails' : 'Archived meeting_followup_20260917/category_grid.jsonl · original specification guardrails'}
      />}</EstimatorViews>
      <EstimatorViews fixed="deming">{() => <GridStep data={data} />}</EstimatorViews>
      </PageChoices>
      <ExplainOnly><NextSteps data={data} /></ExplainOnly>
    </PageToc>
    </AxisChoices>
    </SeasonAxesCtx.Provider>
  )
}

// ------------------------------------------------------------------ the season crossplots' axes
/**
 * One x and one y for every season crossplot on the page (Starting point, Addis by season,
 * Seasons stitched), behind their axis titles (components/AxisChoices). By default the
 * calibration's FTIR EC against HIPS Fabs / MAC; any gallery measurement, ratio or weather field
 * of the same Addis filter can take either axis. The gallery's MAC and mass-per-filter choices
 * apply (Filters → units).
 */
const PRED = 'FTIR EC, this calibration'
const RESID = 'FTIR EC − HIPS BC'
const HIPS = 'HIPS BC'
const EC_LIKE = new Set([PRED, HIPS, 'EC (FTIR)', 'BC (public HIPS)'])
interface SeasonAxes { x: string; y: string; byId: Map<string, FilterRow>; meta?: MetaFile; mac: number; mass: boolean }
const SeasonAxesCtx = createContext<SeasonAxes | null>(null)

function useSeasonAxes(rows: FilterRow[], meta?: MetaFile) {
  const byId = useMemo(() => new Map(rows.filter((r) => r.code === 'ETAD').map((r) => [r.id, r])), [rows])
  const mass = meta?.basis === 'mass', mac = meta?.mac_value ?? 10
  const unit = mass ? 'µg/filter' : 'µg/m³'
  const options = useMemo<AxisOption[]>(() => [
    { key: PRED, group: 'This calibration', unit }, { key: RESID, group: 'This calibration', unit }, { key: HIPS, group: 'HIPS', unit },
    ...(meta?.field_groups ?? []).flatMap((g) => g.fields
      .filter((f) => f !== HIPS && meta!.fields.includes(f) && [...byId.values()].some((r) => typeof r[f] === 'number'))
      .map((f) => ({ key: f, group: g.label, unit: meta!.field_units?.[f] }))),
  ], [meta, byId, unit])
  const keys = options.map((o) => o.key)
  const x = useSearchParam<string>('mcx', HIPS, keys)
  const y = useSearchParam<string>('mcy', PRED, keys)
  const value = useMemo(() => ({ x: x[0], y: y[0], byId, meta, mac, mass }), [x, y, byId, meta, mac, mass])
  return { options, x, y, value }
}

// ------------------------------------------------------------------ 1. answers
function Answers({ data }: { data: MeetingData }) {
  const methods = useBaselineOptions()
  const h = (m: string, g: string) => data.headlines[`${m}|${g}`]
  // coefficients at the agreed 3 dp; PLS factor counts are whole numbers
  const band = (t: [number, number, number], whole = false) => whole
    ? <>{fmt(t[0], 0)} <span style={{ color: INK.muted }}>[{fmt(t[1], 0)}–{fmt(t[2], 0)}]</span></>
    : <BandCell median={t[0]} lo={t[1]} hi={t[2]} />
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Answers" title="Short answers — medians over 100 site-grouped cross-validation splits per season, spline baseline, CO₂ + >3600 cm⁻¹ excluded from selection" exportable={false}>
      <div style={{ overflowX: 'auto' }}>
        <table className="census-table">
          <thead><tr><th>Season</th><th>Test set (Addis): FTIR vs Fabs/10</th><th>IMPROVE cross-validation: FTIR vs Fabs/10</th><th>IMPROVE cross-validation: TOR EC vs Fabs/10</th><th>IMPROVE cross-validation: FTIR vs TOR</th><th>PLS factors</th></tr></thead>
          <tbody>{data.seasons.map((g) => {
            const r = h('AIRSpec', g)
            return <tr key={g}><td style={{ color: data.season_colors[g], fontWeight: 600 }}>{g}</td><td>{band(r.addis_slope)}</td><td>{band(r.analog_ftir_fabs)}</td><td>{band(r.analog_tor_fabs)}</td><td>{band(r.analog_ftir_tor)}</td><td>{band(r.k, true)}</td></tr>
          })}</tbody>
        </table>
      </div>
      <ol style={{ lineHeight: 1.55, fontSize: 14 }}>
        <li><strong>The analogs are mispredicted the way Addis is: same direction, a bit less strongly.</strong> Cross-validation analog slope vs Addis test set slope: {data.seasons.map((g, j) => <span key={g}>{j ? '; ' : ''}{shortSeason(g)} {coef(h('AIRSpec', g).analog_ftir_fabs[0])} vs {coef(h('AIRSpec', g).addis_slope[0])}</span>)}. That is Satoshi's guess: a similar but smaller systematic bias.</li>
        <li><strong>In Dry, part of the bias is already in the references.</strong> For Dry analogs, TOR EC vs Fabs/10 has slope {coef(h('AIRSpec', 'Dry (Oct-Feb)').analog_tor_fabs[0])} (TOR low for its absorption), and FTIR under-reads TOR on top of that ({coef(h('AIRSpec', 'Dry (Oct-Feb)').analog_ftir_tor[0])}). Training on TOR EC bakes the first part in. In Belg and Kiremt, TOR vs Fabs/10 is above 1 ({coef(h('AIRSpec', 'Belg (Mar-May)').analog_tor_fabs[0])}, {coef(h('AIRSpec', 'Kiremt (Jun-Sep)').analog_tor_fabs[0])}), so there the shortfall is FTIR under-reading TOR.</li>
        <li><strong>OP does not explain the TOR–HIPS spread.</strong> Across the IMPROVE pool the MAC ratio is flat in OP fraction (Spearman ρ = {f2(data.op_pool_rho)}); see the OP section.</li>
        <li><strong>The seasonal calibrations are fragile.</strong> Swapping ~12 of 500 analogs moves Kiremt from k = 18 to k = 5; the 10–90 % bands in the table are the honest spread.</li>
        {methods.includes('VIBES') && <li><strong>In the archived truncated VIBES fit, the analogs stop mirroring Addis.</strong> Addis slopes ({data.seasons.map((g, j) => <span key={g}>{j ? ', ' : ''}{shortSeason(g)} {coef(h('VIBES', g).addis_slope[0])}</span>)}) no longer track its cross-validation analogs ({data.seasons.map((g, j) => <span key={g}>{j ? ', ' : ''}{coef(h('VIBES', g).analog_ftir_fabs[0])}</span>)}): the wet seasons overshoot 1 while Dry stays low.</li>}
      </ol>
      <Note>Deming error ratios come from measurement uncertainties: Addis λ* {f2(data.summary.lambda)} (HIPS_Uncertainty vs IMPROVE cross-validation RMSE); IMPROVE FTIR panels use each calibration's cross-validation RMSE against a {(100 * (data.summary.ref_rel_uncertainty ?? 0)).toFixed(1)} % reference uncertainty (ETAD HIPS, the only one on file), so λ is 50–100: almost all the error is assigned to FTIR. Analog concentrations are low (median TOR EC 0.1–0.4 µg/m³) against Addis Fabs/10 of 4–6 µg/m³. Fabs/10 is an optical proxy, not thermal EC.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. levers
function Levers({ data }: { data: MeetingData }) {
  const s = data.summary
  const base = statRow(data.stats, { fit_id: 'allimprove|AIRSpec|all-IMPROVE both lots (frozen Colab)', population: 'addis', eval_group: ALL_ADDIS })
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
          <tbody>{data.reproduction.map((r) => <tr key={r.group}><td>{r.group}</td><td>{r.slide_k}</td><td>{coef(r.slide_slope)} / {coefSigned(r.slide_intercept)} ({r.slide_n_addis})</td><td>{coef(r.here_slope)} / {coefSigned(r.here_intercept)} ({r.here_n_addis})</td><td>{r.cohort_in_frozen_run} / 500</td></tr>)}
            {base && <tr><td>Base case (all-IMPROVE, frozen)</td><td>{base.k}</td><td>0.47 / −0.78 (meeting)</td><td>{coef(base.deming_slope)} / {coefSigned(base.deming_intercept)} ({base.n})</td><td>full pool</td></tr>}
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
    <p className="chart-note">{ANALYSIS_CALENDAR_NOTE}</p>
  </ChartFrame>
}

// ------------------------------------------------------------------ scatter
function Grid({ children, cols = 2 }: { children: ReactNode; cols?: number }) {
  return <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(${cols > 2 ? 260 : 320}px, 1fr))`, gap: 14 }}>{children}</div>
}

/** The page's keys relabel saved Deming lines with the active estimator (the browser refits them). */
const Key = (props: Parameters<typeof SharedKey>[0]) => <SharedKey relabelFits {...props} />

function useAnalogPoints(data: MeetingData) {
  const idx = useIndex(data.analogs, ['model', 'method', 'group'])
  return (model: string, method: string, group: string, color: (i: number) => string, y: 'ftir' | 'tor', x: 'fabs' | 'tor', only?: 'test', perFilter = false) => {
    const a = data.analogs
    const rows = (idx.get(`${model}|${method}|${group}`) ?? []).filter((i) => !only || a.role[i] === only)
    const val = { ftir: a.ftir_ec_ugm3, tor: a.tor_ec_ugm3, fabs: a.fabs10 }
    const k = (i: number) => (perFilter ? (a.vol?.[i] as number | null) ?? NaN : 1)
    return rows.filter((i) => val[x][i] != null && val[y][i] != null).map((i): Pt => ({
      x: (val[x][i] as number) * k(i), y: (val[y][i] as number) * k(i), color: color(i), hollow: a.role[i] !== 'test',
      id: `improve:${a.filter_id[i]}`,
      rec: {
        id: `improve:${a.filter_id[i]}`, source: `meeting follow-up · ${method} · ${model} · analogs of ${group}`,
        site: String(a.Site[i]), date: a.date?.[i] != null ? String(a.date[i]).slice(0, 10) : undefined,
        fields: [
          ['role', a.role[i] === 'test' ? 'held out (site-disjoint)' : 'training'],
          ['IMPROVE lot', a.lot[i]],
          ['TOR EC (µg/m³)', a.tor_ec_ugm3[i]],
          [MAC_LABEL, a.fabs10[i]],
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

/** Every held-out IMPROVE filter under one all-IMPROVE model: the population the analogs are drawn from. */
function usePoolPoints(data: MeetingData) {
  const empty: Cols = { fit_id: [] }
  const idx = useIndex(data.pool ?? empty, ['fit_id'])
  return (fitId: string, color: (op: number | null, ocec: number | null) => string, y: 'ftir' | 'tor', x: 'fabs' | 'tor', perFilter = false) => {
    const a = data.pool
    if (!a) return [] as Pt[]
    const val = { ftir: a.ftir_ec_ugm3, tor: a.tor_ec_ugm3, fabs: a.fabs10 }
    const k = (i: number) => (perFilter ? (a.vol[i] as number | null) ?? NaN : 1)
    return (idx.get(fitId) ?? []).filter((i) => val[x][i] != null && val[y][i] != null).map((i): Pt => ({
      x: (val[x][i] as number) * k(i), y: (val[y][i] as number) * k(i),
      color: color(a.op_tor_frac[i] as number | null, a.tor_oc_ec[i] as number | null),
      id: `improve:${a.filter_id[i]}`,
      rec: {
        id: `improve:${a.filter_id[i]}`, source: `meeting follow-up · ${fitId.split('|').slice(1).join(' · ')} · held-out IMPROVE`,
        site: String(a.Site[i]), date: a.date?.[i] != null ? String(a.date[i]).slice(0, 10) : undefined,
        fields: [
          ['role', 'held out (site-disjoint)'],
          ['IMPROVE lot', a.lot[i]],
          ['TOR EC (µg/m³)', a.tor_ec_ugm3[i]],
          [MAC_LABEL, a.fabs10[i]],
          ['predicted FTIR EC (µg/m³)', a.ftir_ec_ugm3[i]],
          ['sampled volume (m³)', a.vol[i]],
          ['OP fraction, OPTR / (EC + OPTR)', a.op_tor_frac[i]],
          ['TOR OC/EC', a.tor_oc_ec[i]],
        ],
      },
      tip: [`IMPROVE ${a.Site[i]} · filter ${a.filter_id[i]}`, 'held out (test split)',
        `TOR EC ${f2(a.tor_ec_ugm3[i])} · Fabs/10 ${f2(a.fabs10[i])} · FTIR ${f2(a.ftir_ec_ugm3[i])} µg/m³`, `OP fraction ${f2(a.op_tor_frac[i])} · OC/EC ${f2(a.tor_oc_ec[i])}`],
    }))
  }
}

/** an Addis test-set point, with the values the season axes can be built from */
type AddisPt = Pt & { fabs10: number; pred: number; vol: number }

function useAddisPoints(data: MeetingData) {
  const idx = useIndex(data.addis, ['fit_id'])
  return (fitId: string, season: string | 'all', color?: (s: string) => string, perFilter = false) => {
    const a = data.addis
    const k = (i: number) => (perFilter ? (a.vol?.[i] as number | null) ?? NaN : 1)
    return (idx.get(fitId) ?? []).filter((i) => typeof a.fabs10[i] === 'number' && Number.isFinite(a.fabs10[i]) && typeof a.ftir_ec_ugm3[i] === 'number' && Number.isFinite(a.ftir_ec_ugm3[i]) && inSeasonGroup(String(a.season[i]), season)).map((i): AddisPt => ({
      x: (a.fabs10[i] as number) * k(i), y: (a.ftir_ec_ugm3[i] as number) * k(i),
      fabs10: a.fabs10[i] as number, pred: a.ftir_ec_ugm3[i] as number, vol: (a.vol?.[i] as number | null) ?? NaN,
      color: color ? color(String(a.season[i])) : data.season_colors[String(a.season[i])] ?? INK.axis,
      tip: [`${a.ExternalFilterId?.[i] ?? 'Addis'} · ${String(a.date[i]).slice(0, 10)}`, String(a.season[i]), `Fabs/10 ${f2(a.fabs10[i])} · FTIR ${f2(a.ftir_ec_ugm3[i])}`],
      id: String(a.ExternalFilterId?.[i] ?? `ETAD media ${a.MediaId[i]}`),
      rec: {
        id: String(a.ExternalFilterId?.[i] ?? `ETAD media ${a.MediaId[i]}`), source: `meeting follow-up · ${fitId.split('|').slice(1).join(' · ')}`,
        site: 'Addis Ababa', date: String(a.date[i]).slice(0, 10),
        fields: [
          ['season', a.season[i]],
          ['calibration', fitId.split('|').slice(1).join(' · ')],
          [MAC_LABEL, a.fabs10[i]],
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
        const cell = (s?: Row) => s ? <FitCell separator="/" slope={s.deming_slope} intercept={s.deming_intercept} r2={s.R2} n={s.n} /> : '—'
        return <tr key={g}><td style={{ color: data.season_colors[g], fontWeight: 600 }}>{g}</td><td>{cell(q('analog_test', 'ftir_vs_fabs'))}</td><td>{cell(q('analog_test', 'tor_vs_fabs'))}</td><td>{cell(q('analog_test', 'ftir_vs_tor'))}</td><td>{cell(q('addis', 'ftir_vs_fabs', { eval_group: g }))}</td></tr>
      })}</tbody>
    </table>
  )
}

interface SimMeta { bin_wn: number[]; filters: { sample_id: string[] }; rank_cap?: number; background_rank?: number; presentation_status?: string }

/**
 * Per-filter spectra for one baseline (lib/similarity loadTraces, shared with the drawer and the
 * other pages); null fetches nothing (variants the current version does not show). The sensitivity
 * variant reads the full-range rank-70 file and is drawn from 1425 cm⁻¹ down.
 */
function useTraces(method: string | null) {
  const [state, setState] = useState<{ method: string; meta: SimMeta; traces: Traces; index: Map<string, number> } | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    if (method === null) return
    let live = true
    const file = method.endsWith('-r70') ? 'VIBES-full-r70' : method.startsWith('VIBES-full') ? 'VIBES-full' : method
    loadTraces(file)
      .then((t) => {
        if (!live) return
        const meta: SimMeta = { bin_wn: t.wn, filters: t.meta.filters, rank_cap: t.run?.rank_cap, background_rank: t.run?.background_rank, presentation_status: t.run?.presentation_status }
        setState({ method, meta, traces: t.traces, index: t.index })
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
  const methods = useBaselineOptions()
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
  const air = useTraces('AIRSpec'), completed = useTraces('VIBES-full-r70')
  const vib = useTraces(methods.includes('VIBES') ? 'VIBES' : null), full = useTraces(methods.includes('VIBES-full') ? 'VIBES-full' : null)
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
  const hasFull = methods.includes('VIBES-full')
  const series = [
    { method: 'AIRSpec', name: baselineName('AIRSpec'), color: BASELINE_COLOR.AIRSpec, t: air.value, error: air.error },
    { method: 'VIBES-full-r70', name: baselineName('VIBES-full-r70'), color: BASELINE_COLOR['VIBES-full-r70'], t: completed.value, error: completed.error },
    ...(methods.includes('VIBES') ? [{ method: 'VIBES', name: baselineName('VIBES'), color: BASELINE_COLOR.VIBES, t: vib.value, error: vib.error }] : []),
    ...(hasFull ? [{ method: 'VIBES-full', name: baselineName('VIBES-full'), color: BASELINE_COLOR['VIBES-full'], t: full.value, error: full.error }] : []),
  ]
  const visibleSeries = series.filter((s2) => !hidden.has(s2.method))
  const get = (t: (typeof series)[number]['t']) => {
    if (!t) return null
    const k = t.index.get(sid)
    return k == null ? null : { wn: t.meta.bin_wn, v: t.traces.of(k) }
  }
  const panel = (title: string, lines: { name: string; color: string; wn: number[]; v: Float32Array }[], lo: number, w: number, caption?: string) =>
    <SpectrumPlot width={w} height={280} window={[4000, lo]} includeZero zeroLine hover="readout"
      title={<span>{title}{caption && <span style={{ fontWeight: 400, color: INK.muted, fontSize: 11, marginLeft: 8 }}>{caption}</span>}</span>}
      series={lines.map((l) => ({ id: l.name, label: l.name, wn: l.wn, y: l.v, color: l.color, width: 1.5 }))} />
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
        <Segmented label="Spectrum range" value={range} options={['Shared (4000–1425)', 'Full (4000–500)']} onChange={setRange} />
        {/* side by side places the full-range panel beside the comparison: nothing to place without one */}
        {fullLine && <Segmented label="Plot layout" value={layout} options={['Stacked', 'Side by side']} onChange={setLayout} />}
        {series.map((s2) => <Toggle key={s2.method} label={s2.name} checked={!hidden.has(s2.method)} onChange={() => toggleSeries(s2.method)} />)}
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
const TOP_N = ['50', '100', '250', '500'] as const

/**
 * The season's IMPROVE analogs and Addis filters as spectra: every filter, or the
 * median and IQR. The pinned (clicked) filter, and whichever filter is hovered on
 * any chart, is drawn on top in black, so a crossplot point can be read as a spectrum.
 */
function SpectraPanel({ data, method, group }: { data: MeetingData; method: string; group: string }) {
  const hl = useHighlight()
  const [mode, setMode] = useSearchParam<string>('mspec', 'individual filters', SPEC_MODES)
  // any count up to the 500 stored analogs; the presets are the usual ones
  const [topRaw, setTopN] = useSearchParam<string>('mspecn', '100')
  const topN = clampCount(topRaw, 500, 100)
  const tr = useTraces(method)
  // one Addis colour per season; the pooled 'All Addis' summary is yellow so it never reads as the grey analogs
  const col = isAllAddis(group) ? ADDIS_ALL : data.season_colors[group]

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
      if (d.fit_id[i] === fid && inSeasonGroup(String(d.season[i]), group))
        addisRows.push({ sid, key: ext ? baseFilterId(ext) : sid, label: `${ext || sid} · ${String(d.date[i]).slice(0, 10)}`, season: String(d.season[i]) })
    }
    return { improveIds, addisRows, etadOf }
  }, [data, method, group])

  const toSid = (key: string | null) => (key == null ? null : key.startsWith('improve:') ? key : etadOf.get(key) ?? null)

  if (tr.error) return <Empty>Could not load the per-filter spectra: {tr.error}</Empty>
  if (!tr.value) return <div><Empty>Loading per-filter spectra…</Empty></div>
  const { meta, traces, index } = tr.value
  const wn = meta.bin_wn
  const wnFloor = method.includes('full-cut') ? 1425 : 0
  const shownImp = improveIds.slice(0, Number(topN))
  const get = (sid: string) => { const k = index.get(sid); return k == null ? null : traces.of(k) }
  const impT = shownImp.map((sid) => ({ sid, v: get(sid) })).filter((t) => t.v) as { sid: string; v: Float32Array }[]
  const addT = addisRows.map((r) => ({ ...r, v: get(r.sid) })).filter((t) => t.v) as { sid: string; key: string; label: string; season: string; v: Float32Array }[]
  // the export's quantiles cover all 500 analogs; for fewer, take them from the same traces the
  // individual view draws, so both views describe the same N most similar filters (Ann, 9 Oct)
  const qImp = topN === '500' || impT.length === 0 ? data.spectra[`${method}|${group}|improve`] as number[][] | undefined
    : channelQuantiles(impT.map((t) => t.v), [0.1, 0.25, 0.5, 0.75, 0.9])
  const qImpWn: number[] = topN === '500' || impT.length === 0 ? (data.spectra[`wn|${method}`] ?? data.spectra.wn) : Array.from(wn)
  const qAdd = data.spectra[`${method}|${group}|addis`] as number[][] | undefined
  const qwn: number[] = data.spectra[`wn|${method}`] ?? data.spectra.wn
  const individual = mode === 'individual filters'
  const series: SpectrumSeries[] = individual
    ? [...impT.map((t) => ({ id: t.sid, wn, y: t.v, color: IMPROVE_GREY, role: 'trace' as const, tip: [t.sid, 'IMPROVE analog'] })),
      ...addT.map((t) => ({ id: t.key, wn, y: t.v, color: data.season_colors[t.season] ?? col, role: 'trace' as const, opacity: 0.4, width: 0.9, tip: [t.label, `Addis · ${t.season}`] }))]
    : [...(qImp ? [{ id: 'improve-median', label: 'IMPROVE median', wn: qImpWn, y: iqrRows(qImp).mid, color: INK.muted, width: 1.4, hoverable: false }] : []),
      ...(qAdd ? [{ id: 'addis-median', label: 'Addis median', wn: qwn, y: iqrRows(qAdd).mid, color: col, width: 1.5, hoverable: false }] : [])]
  const bands: SpectrumBand[] = individual ? [] : [
    ...(qImp ? [{ id: 'improve', wn: qImpWn, ...iqrRows(qImp), color: IMPROVE_GREY, opacity: 0.25 }] : []),
    ...(qAdd ? [{ id: 'addis', wn: qwn, ...iqrRows(qAdd), color: col, opacity: 0.25 }] : [])]
  const lookup = (key: string) => { const sid = toSid(key), v = sid ? get(sid) : null; return v ? { wn, y: v } : null }
  const lo = Math.max(wnFloor, d3.min(wn) ?? 1400)
  const pinnedSid = toSid(hl.pinnedId)
  return (
    <div style={{ marginTop: 10 }}>
      <div className="frame-controls" style={{ padding: 0 }}>
        <Segmented label="Spectra" value={mode as any} options={SPEC_MODES} optionLabel={(v: string) => v === 'median + IQR' ? 'Median + IQR' : 'Individual filters'} onChange={setMode as any} />
        <CountPicker label="Most similar" value={topN} presets={TOP_N} max={500} onChange={setTopN} noun="analogs" />
      </div>
      {/* grey: dropped from analog selection; hatched: also dropped by the region agreed 1 Oct (similarity only; PLS still sees them) */}
      <SpectrumPlot series={series} bands={bands} window={[d3.max(wn) ?? 4000, lo]} height={286} includeZero
        regions={[...ANALOG_SELECTION_EXCLUDED, ...OCT1_ALSO_EXCLUDED]} hover={individual ? 'nearest' : 'readout'} highlight={hl} lookup={lookup}
        yFormat={(v) => String(+v.toFixed(3))}
        title={`Spectra: ${individual ? `${impT.length} most similar IMPROVE analogs and ${addT.length} Addis filters` : `median and IQR · ${topN === '500' ? 500 : impT.length} most similar IMPROVE analogs`}`} />
      <Key items={[
        { label: individual ? `IMPROVE analogs, ranks 1–${impT.length}` : `IMPROVE analogs, ranks 1–${topN === '500' ? 500 : impT.length}: median, IQR`, color: IMPROVE_GREY, kind: individual ? 'line' : 'bar' },
        ...(individual
          ? (isAllAddis(group) ? data.seasons : [group]).map((g) => ({ label: `Addis ${g} filters`, color: data.season_colors[g], kind: 'line' as const }))
          : [{ label: `Addis ${seasonPhrase(group)}: median, IQR`, color: col, kind: 'bar' as const }]),
        { label: `pinned filter${pinnedSid ? ` (${hl.pinnedId})` : ': none, click a point or trace'}`, color: INK.text, kind: 'line' },
        { label: 'hovered filter', color: INK.text, kind: 'dash' },
        { label: 'excluded from analog selection only (CO₂, >3600 cm⁻¹)', color: SPECTRAL_REGION.co2.fill, kind: 'bar' },
        ...(lo < 1050 ? [{ label: 'also excluded by the region agreed 1 Oct (<600, 1050–1250 PTFE); see the Satoshi 2 Oct view in Meeting', color: OCT1_HATCH.stroke, kind: 'bar' as const }] : []),
      ]} />
    </div>
  )
}

const REPEAT_METRICS = ['slope', 'intercept'] as const

function RepeatStrip({ data }: { data: MeetingData }) {
  const methods = useBaselineOptions()
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
  // 1 Oct: Ann asked for the intercept beside the slope, to see whether the IMPROVE
  // analogs also carry the large negative Addis intercept.
  const [metric, setMetric] = useSearchParam<(typeof REPEAT_METRICS)[number]>('mrmetric', 'slope', REPEAT_METRICS)
  const isSlope = metric === 'slope'
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
  const ref0 = isSlope ? 1 : 0
  const colors: Record<string, string> = { addis: INK.fit, ftir_vs_fabs: '#c026d3', tor_vs_fabs: '#0f766e', ftir_vs_tor: '#2171b5' }
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › RepeatStrip" title={`One separate calibration per season, built from that season's IMPROVE spectral analogs: are the analogs mispredicted the way Addis is? Deming ${metric}, median and 10–90 % over 100 splits`}
      subtitle={`Each block is its own calibration set: the 500 IMPROVE filters whose spectra correlate best with that season's median Addis spectrum ("All Addis" uses the all-season median). Each split holds about 20 % of those analogs out by whole site (cross-validation), picks the number of PLS factors on the rest, and predicts both the held-out IMPROVE filters and the season's Addis filters (the test set) with the same calibration.${isSlope ? '' : ' Intercepts are in µg/m³ of EC.'}`}>
      <div>
        <AxisChoices noun="readouts" x={[metric, (v) => setMetric(v as (typeof REPEAT_METRICS)[number])]} options={REPEAT_METRICS.map((m) => ({ key: m, label: `Deming ${m}`, group: 'Readout', roles: ['x'] }))}>
        <IntervalStrip labelWidth={250} rowHeight={17} blockGap={12}
          panels={[{ label: isSlope ? 'Deming slope (1 = agreement)' : 'Deming intercept, µg/m³ (0 = agreement)', reference: ref0, pick: true }]}
          blocks={data.groups.map((g) => ({ key: g, title: g, color: data.season_colors[g], rows: rows.filter((r) => r.g === g).map((r) => {
            const q10 = d3.quantile(r.vals, 0.1) ?? NaN, q50 = d3.quantile(r.vals, 0.5) ?? NaN, q90 = d3.quantile(r.vals, 0.9) ?? NaN
            return { key: r.m, label: r.label, tip: [`${g} · ${r.label} · ${metric}`, `median ${coef(q50)}`, `10–90 %: ${coef(q10)} – ${coef(q90)}`, `${r.vals.length} splits`],
              marks: [[{ lo: q10, hi: q90, mid: q50, color: colors[r.m] }]] }
          }) }))} />
        </AxisChoices>
        <Key items={[
          ...measures.map((ms) => ({ label: ms.label, color: colors[ms.base] })),
          { label: 'median', color: INK.text },
          { label: '10–90 % of splits', color: INK.muted, kind: 'bar' as const },
          { label: isSlope ? 'slope 1 (agreement)' : 'intercept 0 (agreement)', color: INK.axis, kind: 'line' as const },
        ]} />
      </div>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. three crossplots
// 9 Oct (Ann): IMPROVE stays grey so it never reads as an Addis season; label the baseline
// on the panels; flip between the analogs and every held-out IMPROVE filter; and show
// mass per filter, because IMPROVE samples ~4x more air than SPARTAN.
const COLOR_BY = ['grey', 'OP fraction', 'OC/EC'] as const
const POPULATIONS = ['analogs', 'all held-out IMPROVE'] as const
const UNITS = ['µg/m³', 'µg per filter'] as const
function ThreePlots({ data }: { data: MeetingData }) {
  const methods = useBaselineOptions()
  const fit = FIT_STYLE[useEstimator()]
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
  const [season, setSeason] = useSearchParam<string>('mseason', 'Dry (Oct-Feb)', data.groups)
  const FAMS = ['seasonal calibration', ...MODELS] as const
  const [fam, setFam] = useSearchParam<string>('mfam', 'seasonal calibration', FAMS)
  const [colorBy, setColorBy] = useSearchParam<string>('mcolor', 'grey', COLOR_BY)
  const [popPick, setPop] = useSearchParam<string>('mpop', 'analogs', POPULATIONS)
  const [units, setUnits] = useSearchParam<string>('munits', 'µg/m³', UNITS)
  const analogPts = useAnalogPoints(data)
  const poolPts = usePoolPoints(data)
  const addisPts = useAddisPoints(data)
  const a = data.analogs
  const seasonal = fam === 'seasonal calibration'
  // a seasonal calibration's held-out split is its own analogs; only the all-IMPROVE models have a wider one
  const canPool = !seasonal && !!data.pool
  const pop = canPool ? popPick : 'analogs'
  const pooled = pop === 'all held-out IMPROVE'
  const perFilter = units === 'µg per filter' && !!a.vol
  // each colouring keeps its own palette, picked from its colour ramp (kept in the URL)
  const opColours = useColourScheme('mthree-op', 'sequential', 'Viridis')
  const ocColours = useColourScheme('mthree-oc', 'sequential', 'Plasma')
  const opScale = d3.scaleSequential(d3.interpolateRgbBasis(opColours.stops)).domain([0.1, 0.8])
  const ocScale = d3.scaleSequentialLog(d3.interpolateRgbBasis(ocColours.stops)).domain([1.5, 12])
  const col = data.season_colors[season]
  const shade = (op: number | null, ocec: number | null) => colorBy === 'OP fraction' ? (op == null ? INK.neutral : opScale(op))
    : colorBy === 'OC/EC' ? (ocec == null ? INK.neutral : ocScale(Math.max(1.5, ocec))) : IMPROVE_GREY
  const color = (i: number) => shade(a.op_tor_frac[i] as number | null, a.tor_oc_ec[i] as number | null)
  const fitId = seasonal ? `seasonal|${method}|locked|${season}` : `allimprove|${method}|${fam}`
  const calibrationFit = data.fits.find(fit => fit.fit_id === fitId)
  const q = (panel: string) => statRow(data.stats, seasonal ? { fit_id: fitId, population: 'analog_test', panel }
    : pooled ? { fit_id: fitId, group: 'IMPROVE pool', population: 'pool_test', panel }
    : { fit_id: fitId, group: season, population: 'analog_test', panel })
  const sAd = statRow(data.stats, seasonal ? { fit_id: fitId, population: 'addis', eval_group: season } : { fit_id: fitId, group: season, population: 'addis', eval_group: season })
  const only = 'test' as const
  const improve = (y: 'ftir' | 'tor', x: 'fabs' | 'tor') => pooled ? poolPts(fitId, shade, y, x, perFilter) : analogPts(fam, method, season, color, y, x, only, perFilter)
  const unit = perFilter ? 'µg per filter' : 'µg/m³'
  const hipsLabel = ecAxis('hips', unit)
  const seasonName = isAllAddis(season) ? 'all-Addis' : shortSeason(season)
  const baselineTag = <em className="baseline-tag">{methodName(method)}</em>
  const torHips = improve('tor', 'fabs'), ftirTor = improve('ftir', 'tor'), ftirHips = improve('ftir', 'fabs')
  const addis = addisPts(fitId, season, undefined, perFilter)
  // all four panels on one EC range, always: the IMPROVE and Addis loadings compare by eye
  const shared = sharedCrossDomains([torHips, ftirTor, ftirHips, addis])
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › ThreePlots" title="Three crossplots per season — TOR EC vs HIPS, FTIR vs TOR and FTIR vs HIPS on held-out IMPROVE filters, beside the Addis test set from the same calibration and the spectra"
      subtitle="Only IMPROVE filters from sites held out of the calibration set are drawn (cross-validation). 'Seasonal calibration' is one model per season from its 500 analogs; the all-IMPROVE models are Ann's reading of the current model (lot 251, or the frozen full-pool model for the selected analysis version). For the all-IMPROVE models, the IMPROVE panels can show either the season's analogs or every held-out IMPROVE filter."
      controls={<>
        <ModelSeg label="Calibration" value={fam} options={FAMS} onChange={setFam} />
        <SeasonPicker value={season} groups={data.groups} onChange={setSeason} />
        {canPool && <Segmented label="IMPROVE filters" value={pop as any} options={POPULATIONS} onChange={setPop as any} />}
        {a.vol && <Segmented label="Units" value={units as any} options={UNITS} onChange={setUnits as any} />}
        <Select label="IMPROVE colour" value={colorBy} options={[...COLOR_BY]} onChange={setColorBy} optionLabel={(v) => (v === 'grey' ? 'Grey' : v)} />
      </>}>
      <MeetingCalibrationDetails fit={calibrationFit} />
      <div className="plot-groups">
        <section className="plot-group improve">
          <h3>
            {pooled ? 'Every held-out IMPROVE filter' : 'IMPROVE analogs only'} {baselineTag}
            <span>{seasonal
              ? `cross-validation · ${seasonName} analogs from sites held out of the season's calibration set`
              : pooled ? `cross-validation · all IMPROVE filters at sites held out of the ${modelLabel(fam).toLowerCase()} calibration set, not only the analogs`
              : `cross-validation · the ${seasonName} analogs at sites held out of the ${modelLabel(fam).toLowerCase()} calibration set`}</span>
          </h3>
          <div className="plot-group-grid three">
            <Crossplot title="TOR EC vs HIPS" points={torHips} errorRatio={q('tor_vs_fabs')?.deming_lambda} xLabel={hipsLabel} yLabel={ecAxis('tor', unit)} domain={shared.domain} />
            <Crossplot title="FTIR EC vs TOR EC" points={ftirTor} errorRatio={q('ftir_vs_tor')?.deming_lambda} xLabel={ecAxis('tor', unit)} yLabel={ecAxis('ftir', unit)} domain={shared.domain} />
            <Crossplot title="FTIR EC vs HIPS" points={ftirHips} errorRatio={q('ftir_vs_fabs')?.deming_lambda} xLabel={hipsLabel} yLabel={ecAxis('ftir', unit)} domain={shared.domain} />
          </div>
        </section>
        <section className="plot-group addis">
          <h3>
            Test set: Addis {baselineTag}
          </h3>
          <div className="plot-group-grid">
            <Crossplot title={`Addis ${seasonPhrase(season)}: FTIR vs HIPS`} points={addis} domain={shared.domain} errorRatio={sAd?.deming_lambda ?? data.summary.lambda} xLabel={hipsLabel} yLabel={ecAxis('ftir', unit)} />
          </div>
        </section>
      </div>
      <Key items={[
        ...(colorBy === 'grey' ? [{ label: pooled ? 'every held-out IMPROVE filter' : `IMPROVE analogs of ${isAllAddis(season) ? 'all Addis' : shortSeason(season)}, cross-validation sites`, color: IMPROVE_GREY }] : []),
        ...(isAllAddis(season)
          ? data.seasons.map((g) => ({ label: `Addis ${g}`, color: data.season_colors[g] }))
          : [{ label: `Addis ${shortSeason(season)} filters`, color: col }]),
        { label: fit === FIT_STYLE.deming ? 'Deming fit (λ from measurement uncertainties)' : `${fit.label} fit`, color: fit.color, kind: fit.dash ? 'dash' : 'line' },
        { label: '1:1', color: INK.identity, kind: 'dash' },
      ]}>
        {colorBy === 'OP fraction' && <ColorLegend scale={opScale} scheme={opColours.scheme} label="IMPROVE: OP fraction = OPTR / (EC + OPTR)" width={200} />}
        {colorBy === 'OC/EC' && <ColorLegend scale={ocScale} scheme={ocColours.scheme} label="IMPROVE: TOR OC/EC" width={200} format={(v) => v.toFixed(0)} />}
      </Key>
      {perFilter && <ExplainOnly><Note>Mass per filter = concentration × sampled air volume, filter by filter. IMPROVE samples about 33 m³ per filter, SPARTAN at Addis about 7 m³, so Addis is far less extreme in mass than in concentration: its median HIPS loading is about 9× the IMPROVE median (41× in µg/m³), though still above IMPROVE's 99th percentile. Fit lines are refitted on the converted points with the same error ratio.</Note></ExplainOnly>}
      <SpectraPanel data={data} method={method} group={season} />
      <ExplainOnly>
        <AcrossSeasons data={data} fitPrefix={seasonal ? `seasonal|${method}|locked` : fitId} seasonal={seasonal} />
      </ExplainOnly>
      {colorBy !== 'grey' && <ExplainOnly><Note>{colorBy === 'OP fraction' ? `OP fraction = OPTR / (EC + OPTR), ${opColours.name}${opColours.reverse ? ' reversed' : ''} 0.1 → 0.8.` : `TOR OC/EC, log ${ocColours.name}${ocColours.reverse ? ' reversed' : ''} 1.5 → 12.`} Grey: missing.</Note></ExplainOnly>}
      {q('tor_vs_fabs') && <ExplainOnly><Note>TOR EC vs Fabs/10 slope {coef(q('tor_vs_fabs')?.deming_slope)} implies MAC ≈ {f2(q('tor_vs_fabs')?.implied_mac_deming)} m²/g (Deming); median Fabs/EC ratio {f2(q('tor_vs_fabs')?.mac_ratio_median)}, at the {f2(q('tor_vs_fabs')?.mac_percentile_median)}th percentile of the IMPROVE pool.</Note></ExplainOnly>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 6. MAC envelope
function MacEnvelope({ data }: { data: MeetingData }) {
  const methods = useBaselineOptions()
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
  const hst = data.mac_hist
  const edges: number[] = hst.edges
  const dens = (c: number[]) => { const n = d3.sum(c) || 1; return c.map((v) => v / n) }
  const series = [{ key: 'pool', label: `IMPROVE pool (n ${hst.pool_n})`, color: IMPROVE_GREY, med: hst.pool_median },
    ...data.seasons.map((g) => ({ key: `${method}|${g}`, label: `${g} analogs`, color: data.season_colors[g], med: hst[`${method}|${g}|median`] }))]
  const ymax = d3.max(series.flatMap((s) => dens(hst[s.key] ?? []))) ?? 0.1
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › MacEnvelope" title="MAC envelope — where the analogs sit in the IMPROVE TOR-vs-HIPS envelope (Fabs / TOR EC)"
      subtitle="Ann: does TOR EC vs Fabs give a MAC near 10 for the analogs, or do these samples sit at the low-EC end of the envelope?"
      >
      <SweepPlot height={320} minWidth={560} xScale="log" xDomain={[edges[0], edges[edges.length - 1]]} yDomain={[0, ymax * 1.1]}
        xLabel="Fabs / TOR EC (m²/g)" yLabel="Share of filters" yFormat={(v) => v.toFixed(2)} xMarkers={[{ at: 10, label: 'MAC 10' }]}
        series={series.map((s) => ({ id: s.key, label: s.label, color: s.color, curve: 'monotone' as const, marks: false, interval: 'none' as const, width: s.key === 'pool' ? 3 : 2,
          points: dens(hst[s.key] ?? []).map((v, i) => ({ x: Math.sqrt(edges[i] * edges[i + 1]), y: v })) }))}>
        {/* each population's median as a tick on the x axis */}
        {({ x, ih }) => series.map((s) => s.med != null && <line key={s.key} x1={(x as d3.ScaleLogarithmic<number, number>)(s.med)} x2={(x as d3.ScaleLogarithmic<number, number>)(s.med)} y1={ih} y2={ih - 12} stroke={s.color} strokeWidth={3} />)}
      </SweepPlot>
      <Key items={series.map((s) => ({ label: `${s.label} · median ${f2(s.med)}`, color: s.color, kind: 'line' as const }))} />
      <ExplainOnly><Note>Filters with TOR EC ≤ 0.02 µg/m³ are excluded (ratio unstable). A higher ratio means TOR EC is low for its absorption. Ticks on the axis mark each population's median.</Note></ExplainOnly>
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
  // the page's season axes (useSeasonAxes); the saved fits describe only the default axes at MAC 10
  const ax = useContext(SeasonAxesCtx)
  const custom = !!ax && (ax.x !== HIPS || ax.y !== PRED || ax.mac !== 10 || ax.mass)
  const valueOf = (k: string, p: AddisPt) => {
    if (!ax) return k === HIPS ? p.fabs10 : p.pred
    const per = ax.mass ? p.vol : 1, bc = (p.fabs10 * 10) / ax.mac
    if (k === HIPS) return bc * per
    if (k === PRED) return p.pred * per
    if (k === RESID) return (p.pred - bc) * per
    const v = ax.byId.get(baseFilterId(p.id ?? ''))?.[k]
    return typeof v === 'number' ? v : NaN
  }
  const pts = (season: string) => (addisPts(fitId, season) as AddisPt[]).flatMap((p) => {
    if (!custom) return [p]
    const x = valueOf(ax!.x, p), y = valueOf(ax!.y, p)
    return Number.isFinite(x) && Number.isFinite(y) ? [{ ...p, x, y }] : []
  })
  const all = statRow(data.stats, { fit_id: fitId, population: 'addis', eval_group: ALL_ADDIS })
  // the saved error ratio belongs to FTIR EC vs Fabs / MAC 10; other axes are fitted with λ = 1
  const lambdaOf = (s?: Row) => (custom ? 1 : s?.deming_lambda ?? data.summary.lambda)
  const seasonFits = data.seasons.map((g) => {
    const s = statRow(data.stats, { fit_id: fitId, population: 'addis', eval_group: g })
    const points = pts(g)
    const st = regression(points.map(p => p.x), points.map(p => p.y), { errorsInVariables: true, lambda: lambdaOf(s), bootstrapGroups: monthBlocks(points.map(p => p.rec ?? {})) })
    return { g, saved: s, st }
  })
  const unit = ax?.mass ? 'µg/filter' : 'µg/m³'
  const sameScale = !ax || (EC_LIKE.has(ax.x) && EC_LIKE.has(ax.y))
  // Deming needs comparable axes; between different quantities (iron vs a residual) the lines are OLS
  const lineMethods = sameScale ? methods : ['ols' as const]
  const lines = seasonFits.flatMap(({ g, st }) => fitsFor(st, lineMethods).map(fit => ({ ...fit, stats: st, color: data.season_colors[g], label: `${g} · ${fit.label}` })))
  const labelOf = (k: string) => k === HIPS ? `HIPS Fabs / MAC ${ax?.mac ?? 10} (${unit})` : k === PRED ? `FTIR EC (${unit})`
    : k === RESID ? `FTIR EC − Fabs / MAC ${ax?.mac ?? 10} (${unit})` : withUnit(k, ax?.meta?.field_units ?? {})
  return (
    <div>
      {/* unlike axes: the season OLS lines only (a pooled Deming refit would need comparable axes) */}
      <Crossplot title={title} points={pts('all')} errorRatio={lambdaOf(all)} extraLines={lines} pick={!!ax} sameScale={sameScale} grouping={sameScale ? undefined : 'none'}
        xLabel={custom ? labelOf(ax!.x) : MAC_LABEL} yLabel={custom ? labelOf(ax!.y) : 'FTIR EC (µg/m³)'} maxSide={480} />
      <Key items={[
        ...seasonFits.map(({ g, saved, st }) => {
          const sl = !custom ? saved?.deming_slope : sameScale ? st?.demingSlope : st?.slope
          const ic = !custom ? saved?.deming_intercept : sameScale ? st?.demingIntercept : st?.intercept
          return { label: `${g}: ${sameScale ? '' : 'OLS '}${sl != null && Number.isFinite(sl) ? `${coef(sl)}x ${coefSigned(ic)}` : '—'}`, color: data.season_colors[g] }
        }),
        ...(sameScale ? [
          { label: `${FIT_STYLE[methods[0] ?? 'deming'].label} fit, all Addis`, color: FIT_STYLE[methods[0] ?? 'deming'].color, kind: (FIT_STYLE[methods[0] ?? 'deming'].dash ? 'dash' : 'line') as 'dash' | 'line' },
          { label: '1:1', color: INK.identity, kind: 'dash' as const },
        ] : []),
      ]} />
    </div>
  )
}

function SeasonTable({ data, rows }: { data: MeetingData; rows: { label: string; fitId: string }[] }) {
  const { showCI } = useRegressions()
  const cell = (s?: Row) => s ? <FitCell separator="/" stacked slope={s.deming_slope} intercept={s.deming_intercept} r2={s.R2} n={s.n} slopeCI={showCI ? [s.slope_ci_low, s.slope_ci_high] : null} /> : '—'
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
  const methods = useBaselineOptions()
  const { showCI } = useRegressions()
  const fitName = FIT_STYLE[useEstimator()].label
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
  const [model, setModel] = useSearchParam<string>('msmodel', MODELS[0], MODELS)
  const fitId = `allimprove|${method}|${model}`
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Seasonality" title="Addis by season, all-IMPROVE model — the full baseline-corrected Addis test set coloured by season, with a Deming line per season"
      controls={<>
        <ModelSeg label="Model" value={model} options={MODELS} onChange={setModel} />
      </>}>
      <Grid>
        <SeasonCrossplot data={data} fitId={fitId} title="Addis, all-IMPROVE model, coloured by season" />
        <ExplainOnly><SeasonTable data={data} rows={[{ label: 'All-IMPROVE model', fitId }, { label: 'Own-season calibration', fitId: `stitched|${method}|season calibrations` }]} /></ExplainOnly>
      </Grid>
      <ExplainOnly><Note>{fitName} line: all Addis; season-coloured lines: {fitName} fit within each season (λ* {f2(data.summary.lambda)}). {showCI && 'Bands are pointwise 95% month-block bootstrap intervals where available.'}</Note></ExplainOnly>
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
  const methods = useBaselineOptions()
  const fitName = FIT_STYLE[useEstimator()].label
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
  const fitId = `stitched|${method}|season calibrations`
  const sp = stitchedSpread(data, method)
  const all = statRow(data.stats, { fit_id: fitId, population: 'addis', eval_group: ALL_ADDIS })
  const ref = statRow(data.stats, { fit_id: `allimprove|${method}|all-IMPROVE lot 251`, population: 'addis', eval_group: ALL_ADDIS })
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Stitched"
      title="Seasons stitched — each season's Addis filters predicted by its own season calibration, all on one test set crossplot"
      subtitle={`Ann's top priority: calibrate each season on its own analogs, then put the three seasons back together. Each season's line is its own calibration's readout; the ${fitName} line is the combined Addis result.`}
      >
      <Grid>
        <SeasonCrossplot data={data} fitId={fitId} title={`Addis, own-season calibrations (${methodName(method)})`} />
        <div style={{ fontSize: 13, lineHeight: 1.6, alignSelf: 'center' }}>
          <p style={{ margin: '0 0 8px' }}><strong>Stitched, all Addis:</strong> {fitName} {coef(all?.deming_slope)}x {coefSigned(all?.deming_intercept)} (R² {coef(all?.R2)})</p>
          <p style={{ margin: '0 0 8px' }}><strong>Over 100 cross-validation splits:</strong> slope {coef(sp.slope[0])} [{coef(sp.slope[1])}–{coef(sp.slope[2])}], intercept {coefSigned(sp.intercept[0])} [{coef(sp.intercept[1])} to {coef(sp.intercept[2])}]</p>
          <p style={{ margin: 0, color: INK.muted }}>For comparison, one all-IMPROVE model (lot 251): {coef(ref?.deming_slope)}x {coefSigned(ref?.deming_intercept)}. Each season's own slope is low, but the three season calibrations sit at different levels, so putting them back together steepens the combined line and pushes its intercept further from zero.</p>
        </div>
      </Grid>
      <ExplainOnly><SeasonTable data={data} rows={[{ label: 'Own-season calibrations', fitId }, { label: 'All-IMPROVE model, lot 251', fitId: `allimprove|${method}|all-IMPROVE lot 251` }, { label: 'Current SPARTAN', fitId: 'deployed|SPARTAN|current SPARTAN calibration' }]} /></ExplainOnly>
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
  const methods = useBaselineOptions()
  const { showCI } = useRegressions()
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
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
      pts.push({ x: g.all_slope[i] as number, y: g.all_intercept[i] as number, color: catColor(String(g.label[i])), r: 2.2, shape: 'dot', op: dotOpacity('dimmed'),
        tip: [String(g.label[i]), `cohort ${g.cutoff[i]} filters · ${CV_PROTOCOL_SHORT[String(g.mode[i])] ?? g.mode[i]} · ${g.k[i]} PLS factors`, `test set ${coef(g.all_slope[i])}x ${coefSigned(g.all_intercept[i])}`] })
    }
  }
  const rp = data.repeats
  for (const season of data.seasons) {
    const sl: number[] = [], ic: number[] = []
    for (let i = 0; i < rp.method.length; i++) if (rp.method[i] === method && rp.group[i] === season) {
      sl.push(rp.addis_slope[i] as number); ic.push(rp.addis_intercept[i] as number)
      pts.push({ x: rp.addis_slope[i] as number, y: rp.addis_intercept[i] as number, color: data.season_colors[season], r: 2.4, shape: 'dot', op: dotOpacity('dimmed'),
        tip: [`${season} calibration, split ${rp.repeat[i]}`, 'read out on its own season only', `${coef(rp.addis_slope[i])}x ${coefSigned(rp.addis_intercept[i])}`] })
    }
    pts.push({ x: q3(sl)[0], y: q3(ic)[0], color: data.season_colors[season], r: 8, shape: 'tri', op: 1,
      tip: [`${season} calibration: median of 100 splits`, 'read out on its own season only', `${f2(q3(sl)[0])}x ${signed(q3(ic)[0])}`] })
  }
  const st = data.stitched_repeats
  for (let i = 0; i < st.method.length; i++) if (st.method[i] === method)
    pts.push({ x: st.addis_slope[i] as number, y: st.addis_intercept[i] as number, color: '#6d28d9', r: 2.4, shape: 'dot', op: dotOpacity('dimmed'),
      tip: [`Seasons stitched, split ${st.repeat[i]}`, 'all Addis', `${coef(st.addis_slope[i])}x ${coefSigned(st.addis_intercept[i])}`] })
  const sp = stitchedSpread(data, method)
  pts.push({ x: sp.slope[0], y: sp.intercept[0], color: '#6d28d9', r: 10, shape: 'star', op: 1, tip: ['Seasons stitched: median of 100 splits', 'all Addis', `${coef(sp.slope[0])}x ${coefSigned(sp.intercept[0])}`] })
  const one = (fit: string, color: string, shape: P['shape'], name: string) => {
    const s = statRow(data.stats, { fit_id: fit, population: 'addis', eval_group: ALL_ADDIS })
    if (s) pts.push({ x: s.deming_slope, y: s.deming_intercept, color, r: 8, shape, op: 1,
      ci: [s.slope_ci_low, s.slope_ci_high, s.intercept_ci_low, s.intercept_ci_high].every(v => typeof v === 'number' && Number.isFinite(v)) ? [s.slope_ci_low, s.slope_ci_high, s.intercept_ci_low, s.intercept_ci_high] : undefined,
      tip: [name, 'all Addis', `${coef(s.deming_slope)}x ${coefSigned(s.deming_intercept)}`] })
  }
  one(`allimprove|${method}|all-IMPROVE lot 251`, INK.fit, 'sq', 'All-IMPROVE model, lot 251')
  one('deployed|SPARTAN|current SPARTAN calibration', '#b91c1c', 'x', 'Current SPARTAN calibration')

  const xs = pts.map((p) => p.x).filter(Number.isFinite), ys = pts.map((p) => p.y).filter(Number.isFinite)
  const xDom = robustDomain(xs, 0.005, 0.995, { include: [0, 1.5], fallback: [0, 2] }).domain
  const yDom = robustDomain(ys, 0.005, 0.995, { include: [-1, 1], fallback: [-5, 1] }).domain
  const m = { top: 20, right: 20, bottom: 50, left: 64 }
  const side = Math.max(300, Math.min((width || 600) - m.left - m.right, 560))
  const x = d3.scaleLinear().domain(xDom).range([0, side]).nice()
  const y = d3.scaleLinear().domain(yDom).range([side, 0]).nice()
  const cx = (v: number) => x(Math.max(x.domain()[0], Math.min(x.domain()[1], v)))
  const cy = (v: number) => y(Math.max(y.domain()[0], Math.min(y.domain()[1], v)))
  const mark = (p: P, key: number) => {
    const X = cx(p.x), Y = cy(p.y), r = p.r
    const ev = { onMouseEnter: (e: React.MouseEvent) => tip.show(e, p.tip), onMouseLeave: tip.hide }
    if (p.shape === 'dot') return <circle key={key} cx={X} cy={Y} r={r} fill={p.color} fillOpacity={p.op} {...dotRim()} {...ev} />
    if (p.shape === 'sq') return <rect key={key} x={X - r * 0.8} y={Y - r * 0.8} width={r * 1.6} height={r * 1.6} fill={p.color} stroke={INK.panel} {...ev} />
    if (p.shape === 'x') return <g key={key} {...ev}><line x1={X - r} x2={X + r} y1={Y - r} y2={Y + r} stroke={p.color} strokeWidth={3} /><line x1={X - r} x2={X + r} y1={Y + r} y2={Y - r} stroke={p.color} strokeWidth={3} /></g>
    if (p.shape === 'tri') return <path key={key} d={`M${X},${Y - r} L${X + r},${Y + r * 0.8} L${X - r},${Y + r * 0.8} Z`} fill={p.color} stroke={INK.fit} strokeWidth={1} {...ev} />
    return <path key={key} d={d3.symbol(d3.symbolStar, r * r * 3)() ?? ''} transform={`translate(${X},${Y})`} fill={p.color} stroke={INK.panel} {...ev} />
  }
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › SlopeInterceptMap"
      title="Slope vs intercept — where the season calibrations land among every configuration tried, against the target (slope 1, intercept 0)"
      subtitle="Test set (Addis) Deming slope and intercept at MAC 10. Grid points, the all-IMPROVE model, SPARTAN and the stitched seasons are read out on all Addis; each season calibration is also shown read out on its own season."
      controls={<>
        <Segmented label="Grid configurations" value={gridSet as any} options={['passing', 'all', 'none'] as const} onChange={setGridSet as any} />
      </>}>
      <div ref={ref}>
        <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family} style={{ display: 'block', margin: '0 auto' }}>
          <g transform={`translate(${m.left},${m.top})`}>
            <rect x={x(0.8)} width={x(1.2) - x(0.8)} y={y(1)} height={y(-1) - y(1)} fill={INK.good} fillOpacity={0.1} />
            <YAxis scale={y} x={0} label="Test set Deming intercept (µg/m³)" gridWidth={side} tickCount={6} labelX={-46} />
            <XAxis scale={x} y={side} label="Test set Deming slope" tickCount={6} labelDy={38} />
            <RefLine orient="v" at={x(1)} from={0} to={side} />
            <RefLine orient="h" at={y(0)} from={0} to={side} />
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
          { label: 'all-IMPROVE model (■)', color: INK.fit, kind: 'bar' as const },
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
  const methods = useBaselineOptions()
  const { showCI } = useRegressions()
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
  const cals = [
    { label: 'All-IMPROVE model, lot 251', fit: `allimprove|${method}|all-IMPROVE lot 251`, byGroup: true },
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
  const slopeDomain: [number, number] = [0, Math.max(1.5, d3.max(rows, (r) => (showCI ? r.s.slope_ci_high : null) ?? r.s.deming_slope) ?? 1.5)]
  const interceptDomain: [number, number] = [Math.min(-2, d3.min(rows, (r) => (showCI ? r.s.intercept_ci_low : null) ?? r.s.deming_intercept) ?? -2), Math.max(0.5, d3.max(rows, (r) => (showCI ? r.s.intercept_ci_high : null) ?? r.s.deming_intercept) ?? 0.5)]
  const calColor = d3.scaleOrdinal<string>().domain(cals.map((c) => c.label)).range([INK.fit, '#6b7280', '#2171b5', '#c026d3', '#0f766e', '#d97706'])
  return (
    <ChartFrame source="gallery/app/src/pages/MeetingFollowupPage.tsx › Combined" title="Calibrations compared — Addis Deming slope and intercept by season for every calibration"
      subtitle="Ann: combine the three seasonal analog sets into one calibration and compare against the VIP/Euclidean analogs and the seasonally calibrated results."
      >
      <div>
        <IntervalStrip labelWidth={260} rowHeight={16} panels={[
          { heading: 'Slope (1 = agreement)', label: 'Deming slope', reference: 1, domain: slopeDomain },
          { heading: 'Intercept (µg/m³, 0 = agreement)', label: 'Deming intercept (µg/m³)', reference: 0, domain: interceptDomain },
        ]} blocks={data.groups.map((g) => ({ key: g, title: g, color: data.season_colors[g], rows: rows.filter((r) => r.g === g).map((r) => {
          const c = calColor(r.cal)
          return { key: r.cal, label: r.cal,
            tip: [`${g} · ${r.cal}`, `slope ${coef(r.s.deming_slope)}${showCI ? ` · 95% CI [${coef(r.s.slope_ci_low)}, ${coef(r.s.slope_ci_high)}]` : ''}`, `intercept ${coefSigned(r.s.deming_intercept)}${showCI ? ` · 95% CI [${coef(r.s.intercept_ci_low)}, ${coef(r.s.intercept_ci_high)}]` : ''}`, `R² ${coef(r.s.R2)} · n ${r.s.n}`],
            marks: [[{ lo: showCI ? r.s.slope_ci_low : null, hi: showCI ? r.s.slope_ci_high : null, mid: r.s.deming_slope, color: c, r: 4 }],
              [{ lo: showCI ? r.s.intercept_ci_low : null, hi: showCI ? r.s.intercept_ci_high : null, mid: r.s.deming_intercept, color: c, r: 4 }]] }
        }) }))} />
        <Key items={[
          ...cals.map((c) => ({ label: c.label, color: calColor(c.label) })),
          ...(showCI ? [{ label: '95 % month-block bootstrap (where saved)', color: INK.muted, kind: 'bar' as const }] : []),
          { label: 'agreement (slope 1, intercept 0)', color: INK.axis, kind: 'line' as const },
        ]} />
      </div>
      <ExplainOnly><Note>{showCI ? 'Bars: saved 95% month-block bootstrap intervals; missing intervals are not drawn.' : 'Fit uncertainty hidden; enable it in the sidebar to show saved intervals.'} VIP/Euclidean analogs exist only in spline-baselined space (the explorer's corrected-space ranking). Combined set: union of the three seasonal top-500 lists.</Note></ExplainOnly>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 10. grid heat map
const METRICS = ['intercept', 'slope', 'CV R²'] as const
const READOUTS = ['all', 'dry', 'belg', 'kiremt'] as const
const READOUT_LABEL: Record<string, string> = { all: 'All Addis', dry: 'Dry season', belg: 'Belg season', kiremt: 'Kiremt season' }
const METRIC_LABEL: Record<string, string> = { intercept: 'Intercept', slope: 'Slope', 'CV R²': 'Cross-validation R²' }
function GridStep({ data }: { data: MeetingData }) {
  const methods = useBaselineOptions()
  // the page's one baseline (PageChoices at the top of the page)
  const { value: method } = usePageChoice('baseline', methods)
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
  // intercept and slope diverge around agreement; CV R² is sequential: each with its own palette (lib/colourSchemes)
  const divColours = useColourScheme('category-grid', 'diverging')
  const seqColours = useColourScheme('category-grid-r2', 'sequential')
  const colours = metric === 'CV R²' ? seqColours : divColours
  const ds = divColours.stops, ss = seqColours.stops
  const color = d3.scaleLinear<string>().domain([dom[0], (dom[0] + dom[1]) / 2, dom[1], (dom[1] + dom[2]) / 2, dom[2]])
    .range(metric === 'CV R²' ? [ss[0], ss[2], ss[4], ss[6], ss[7]] : [ds[0], ds[2], ds[3], ds[4], ds[6]]).clamp(true)
  return (
    <ChartFrame title="Category grid — prioritization categories re-run on baselined spectra: Addis readout by cohort size"
      subtitle={`${data.grid_n} fitted cohorts: category × cutoff (step 5) × baseline × CV. Cells average the fits inside each 25-filter bin. Replaces the intercept continuum plot.`}
      controls={<>
        <Segmented label="Cross-validation protocol" value={CV_PROTOCOL_SHORT[mode] as any} options={[CV_PROTOCOL_SHORT.site_heldout, CV_PROTOCOL_SHORT.app] as any} onChange={(v: string) => setMode(v === CV_PROTOCOL_SHORT.app ? 'app' : 'site_heldout')} />
        <Segmented label="Cell" value={metric as any} options={METRICS} optionLabel={(v: string) => METRIC_LABEL[v] ?? v} onChange={setMetric as any} />
        {metric !== 'CV R²' && <Segmented label="Test set (Addis) readout" value={readout as any} options={READOUTS} optionLabel={(v: string) => READOUT_LABEL[v] ?? v} onChange={setReadout as any} />}
        {/* fading by CV R² is moot when the cells are already coloured by it */}
        {mode === 'site_heldout' && metric !== 'CV R²' && <Toggle label="Fade cells failing IMPROVE cross-validation R² ≥ 0.85" checked={gate} onChange={setGate} />}
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
                  return <rect key={b} x={c * cw} y={1} width={cw - 0.5} height={chH - 2} fill={color(v)} opacity={faded ? 0.3 : 1} stroke={faded ? 'none' : INK.text} strokeWidth={faded ? 0 : 0.3}
                    onMouseEnter={(e) => tip.show(e, [`${cat} · cutoff ≈ ${b}`, `${METRIC_LABEL[metric] ?? metric}${metric !== 'CV R²' ? ` (${READOUT_LABEL[readout] ?? readout})` : ''}: ${f2(v)}`, `${cell.n} fits · k ${d3.min(cell.k)}–${d3.max(cell.k)}`, mode === 'site_heldout' ? `${cell.pass}/${cell.n} pass IMPROVE cross-validation R² ≥ 0.85` : 'interleaved CV holds out no whole site: no cross-validation R² gate'])} onMouseLeave={tip.hide} />
                })}
              </g>
            ))}
            {bins.map((b, c) => (b % 250 === 0) && <text key={b} x={c * cw + cw / 2} y={cats.length * chH + 16} fontSize={10.5} textAnchor="middle" fill={INK.muted}>{b}</text>)}
            <text x={(bins.length * cw) / 2} y={cats.length * chH + 36} fontSize={12} fontWeight={600} textAnchor="middle">Cohort size (IMPROVE filters)</text>
          </g>
        </svg>
        <Key items={mode === 'site_heldout' && gate && metric !== 'CV R²' ? [{ label: 'no fit in the bin passes IMPROVE cross-validation R² ≥ 0.85', color: '#9aa5b1', kind: 'faded' }] : []}>
          <ColorLegend scale={color} scheme={colours.scheme} width={220}
            label={metric === 'intercept' ? `Test set (Addis) Deming intercept (µg/m³, ${READOUT_LABEL[readout] ?? readout})` : metric === 'slope' ? `Test set (Addis) Deming slope (${READOUT_LABEL[readout] ?? readout})` : 'IMPROVE cross-validation R² (calibration set)'}
            note={metric === 'intercept' ? 'white = 0' : metric === 'slope' ? 'white = 1' : undefined} />
        </Key>
        {tip.node}
      </div>
      {(data.grid_summary ?? []).map((gs) => <ExplainOnly key={gs.method}><Note>{methodName(gs.method)}, site-grouped cross-validation: {gs.passing} fits pass IMPROVE cross-validation R² ≥ 0.85; their all-Addis test set slopes span {coef(gs.slope_min)}–{coef(gs.slope_max)} and intercepts {coefSigned(gs.intercept_min)} to {coefSigned(gs.intercept_max)}; {gs.balanced} reach slope 0.8–1.2 with |intercept| &lt; 1.</Note></ExplainOnly>)}
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
        <li><strong>Scoring decision.</strong> {(data.grid_summary ?? []).map((g) => <span key={g.method}>{methodName(g.method)}: {g.passing} site-grouped fits pass IMPROVE cross-validation R² ≥ 0.85, {g.balanced} of them have Addis slope 0.8–1.2 with |intercept| &lt; 1 (best balanced: {g.best_label} {g.best_cutoff}, {coef(g.best_slope)} / {coefSigned(g.best_intercept)}). </span>)}Slope target, intercept target, or a range for both?</li>
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

