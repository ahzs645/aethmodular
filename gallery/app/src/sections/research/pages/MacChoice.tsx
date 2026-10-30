import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Figure } from '@/components/Figure'
import { Key } from '@/components/Key'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, histogramSeries } from '@/components/SweepPlot'
import { useChartOptions } from '@/lib/chartOptions'
import { useColourScheme } from '@/lib/colourSchemes'
import { f2, fmt, signed } from '@/lib/format'
import { slopeAt, useMac } from '@/lib/mac'
import { SidebarControls } from '@/lib/pageSidebar'
import { categoricalScale } from '@/lib/palettes'
import { FIT_STYLE, useRegressions } from '@/lib/regressionPrefs'
import type { Estimator } from '@/lib/regressionMethods'
import { regression } from '@/lib/stats'
import { num, type Row } from '@/lib/table'
import { INK } from '@/lib/theme'
import type { CalibRun, CalibRunsFile, FilterRow } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { quantile } from '../common/stats'
import { BASIS_NOTE, num as rowNum, useConcRows } from '../common/ui'
import { PageIntro, presetName, provenance, rows, useCalibRuns, useSpectral, Waiting } from './spectralCommon'

const MAC10 = '#2b6cb0', MAC6 = '#c05621'
const siteName = (runs: CalibRunsFile, t: string) => runs.targets[t]?.label.split(':')[0] ?? t
/** the explorer's calibration targets by SPARTAN site code, so the Filters panel's site pick chooses them */
const TARGET_OF: Record<string, string> = { ETAD: 'addis', CHTS: 'chts', ETBI: 'etbi', INDH: 'indh', USPA: 'uspa' }
/** the explorer saved Deming and OLS fits only: the Regression fits setting draws those of its choices */
const SAVED_FITS: readonly Estimator[] = ['deming', 'ols']
type EvalSet = 'fixed' | 'all'
/** the 1:1 chart's MAC: the one set under Filters, or a quick comparison */
const MACS = ['gallery', '6', '10', '17'] as const

/** the page's choices from the Filters panel: its calibration sites (Addis when none is picked), its filters, its fits */
interface MacView { targets: string[]; fallback: boolean; set: EvalSet; fits: Estimator[]; unsaved: Estimator[] }

/**
 * ftir_16 and ftir_19: MAC 6 or 10. HIPS enters every crossplot as x = Fabs/MAC, so the MAC rescales
 * x: each calibration keeps its intercept and R² and its slope scales by MAC/10 (lib/mac). The explorer
 * saved every preset's fits at MAC 6, 10 and 17 (calibration_runs.json), which obey that exactly, so the
 * page draws any MAC: the one set under Filters by default. ftir_19's six setups are refitted here from
 * its per-filter table; ftir_16's IMPROVE bridge says which MAC the network supports.
 */
export function MacChoice({ picked, rows: viewRows, meta }: SectionPageProps) {
  const concRows = useConcRows(viewRows, meta)
  const { data, error } = useSpectral('mac_choice')
  const { runs, error: runsError } = useCalibRuns()
  const [set, setSet] = useSearchParam<EvalSet>('mac.set', 'fixed', ['fixed', 'all'])
  const { methods } = useRegressions()

  if (!data || !runs) return <Waiting error={error || runsError} />
  const summary = rows(data, 'bridge_summary')
  const picks = Object.keys(runs.targets).filter((t) => [...picked].some((c) => TARGET_OF[c] === t))
  const view: MacView = { targets: picks.length ? picks : ['addis'], fallback: !picks.length, set,
    fits: methods.filter((m) => SAVED_FITS.includes(m)), unsaved: methods.filter((m) => !SAVED_FITS.includes(m)) }
  return <>
    <SidebarControls title="MAC page">
      <Segmented label="Evaluation filters" value={set} options={['fixed', 'all'] as const} onChange={setSet}
        optionLabel={(o) => (o === 'fixed' ? 'Fixed cohort' : 'All pairs')} />
    </SidebarControls>
    <PageIntro id="mac-intro" title="MAC 6 or 10 — the choice moves every slope and no intercept" data={data}
      notebooks="research/ftir_ec_phase3/ftir_16_mac_decision_prep, ftir_19_mac_effect_on_calibrations; ftir_31 §1.11 (the pivot with Deming)">
      <p>The 28 Oct meeting page §4 has the MAC by season. This page applies MAC 6 and 10 to every calibration the explorer exported, and sets them against the IMPROVE evidence.</p>
    </PageIntro>

    <MacOneToOne runs={runs} view={view} />

    <MacPivot runs={runs} view={view} />

    <SetupsAtMac rows={rows(data, 'setup_filters')} saved={rows(data, 'setups')} provenance={provenance(data, ['setup_filters', 'setups'])} />

    <ChartFrame id="mac-bridge" title="The IMPROVE bridge — at Addis-like composition the implied MAC is centred on 10"
      subtitle={`Implied MAC = HIPS Fabs ÷ TOR EC on matched IMPROVE filters, by decile of TOR OC/EC: median and IQR. ${summary.map((r) => `${r.population}: median ${f2(r.median)} (IQR ${f2(r.p25)}–${f2(r.p75)}, n = ${Number(r.n).toLocaleString()})`).join('; ')}. The upper tail at high OC/EC fits Fabs carrying non-EC absorption that scales with organics. IMPROVE and SPARTAN HIPS comparability is assumed.`}
      provenance={provenance(data, ['bridge', 'bridge_summary'], 'ftir_16 §5 and ftir_31 §3.')}>
      <SweepPlot series={[{ id: 'bridge', label: 'implied MAC', color: MAC10, interval: 'whisker',
        points: rows(data, 'bridge').map((r) => ({ x: String(r['OC/EC decile']), y: num(r.MAC_median), lo: num(r.MAC_p25), hi: num(r.MAC_p75),
          tip: [`OC/EC decile ${r['OC/EC decile']}`, `median ${f2(r.MAC_median)} (IQR ${f2(r.MAC_p25)}–${f2(r.MAC_p75)})`, `n = ${Number(r.n).toLocaleString()}`] })) }]}
        xScale="point" xLabel="TOR OC/EC decile (range)" yLabel="Implied MAC (m²/g)" include={[0]}
        yMarkers={[{ at: 6, label: 'MAC 6', color: MAC6 }, { at: 10, label: 'MAC 10', color: MAC10 }]} height={300} />
    </ChartFrame>

    <Figure kind="intervals" id="mac-adama" data={rows(data, 'adama_scenarios')} row="scenario"
      panels={[{ value: 'implied_MAC_median', lo: 'implied_MAC_p25', hi: 'implied_MAC_p75', label: 'MAC that reconciles Addis HIPS with FTIR OC (m²/g; median, IQR)', reference: 10 }]} labelWidth={170}
      title="If Addis had Adama's OC/EC — the reconciling MAC would be unphysical"
      subtitle="Per Addis filter, MAC = Fabs × r / OC_ftir for Adama's TOR OC/EC r (5 quartz filters, Jul–Aug 2024). Values near 47 m²/g are far outside any EC MAC (~4–13), so at least one holds: Addis OC/EC is much lower than Adama's, much of Addis Fabs is not EC, or FTIR OC is low at Addis."
      provenance={provenance(data, ['adama_scenarios'])} />

    <ImpliedMac rows={concRows} scenarios={rows(data, 'adama_scenarios')} provenance={provenance(data, ['adama_scenarios'], 'Per-filter MACs computed here from the gallery rows (OC (FTIR), HIPS Fabs) as ftir_16 §3 does.')} />

    <Figure kind="table" id="mac-chemspec" data={rows(data, 'chemspec')}
      columns={[{ key: 'n base filters', format: 'int' }, { key: 'r (ChemSpec EC vs Fabs/10)', format: 'f3' }, { key: 'median ChemSpec EC / Fabs', format: 'f3' }, { key: 'implied MAC', label: 'Implied MAC (m²/g)', format: 'f2' }]}
      title="SPARTAN's public ChemSpec EC at Addis is HIPS Fabs ÷ 10 — not a tie-breaker"
      subtitle="Base-filter join of ChemSpec_EC_PM2.5 to HIPS Fabs at ETAD. It is the MAC 10 convention by construction."
      provenance={provenance(data, ['chemspec'])} />

    <Figure kind="table" id="mac-power" data={rows(data, 'power')}
      columns={[{ key: 'season', label: 'Season' }, { key: 'n_days_available', label: 'Filters', format: 'int' }, { key: 'median_Fabs', label: 'Median Fabs (Mm⁻¹)', format: 'f2' },
        { key: 'median_EC_at_MAC6', label: 'EC at MAC 6', format: 'f2' }, { key: 'median_EC_at_MAC10', label: 'EC at MAC 10', format: 'f2' },
        { key: 'median_z_per_day', label: 'σ per quartz day', format: 'f2' }, { key: 'days_for_5sigma_at_half_signal', label: 'Days for 5σ at half signal', format: 'int' }]}
      title="A quartz TOR campaign decides it — about 12 days per season"
      subtitle="TOR uncertainty 10 % + 0.3 µg/m³. One co-located day separates MAC 6 from MAC 10 by about 3σ; systematics, not statistics, set the size, so spread days across seasons."
      provenance={provenance(data, ['power'])} />
  </>
}

/** what the Filters panel's choices leave out, said once under each chart */
function viewNote(view: MacView, extra = '') {
  const parts = [
    view.fallback ? 'None of the sites picked in the Samples panel has explorer calibrations (Addis, Beijing, Bishoftu, Delhi, Pasadena), so Addis is shown.' : '',
    view.unsaved.length ? `${view.unsaved.map((m) => FIT_STYLE[m].label).join(', ')}: the explorer saved Deming and OLS fits only.` : '',
    !view.fits.length ? 'No saved fit is selected under Regression fits (Deming or OLS), so no lines are drawn.' : '',
    extra,
  ].filter(Boolean)
  return parts.length ? <Note>{parts.join(' ')}</Note> : null
}
/** the set a site is scored on: the chosen one, or all pairs where the explorer has no fixed cohort for it (only Addis has one) */
const setFor = (run: CalibRun | undefined, set: EvalSet): EvalSet => (run?.metrics.some((m) => m.evaluation_set === set) ? set : 'all')
/** a run's saved fit for exactly this set and MAC (metricRow falls back to another row, which would mislabel it) */
const fitAt = (run: CalibRun | undefined, set: EvalSet, mac: number) => run?.metrics.find((m) => m.evaluation_set === set && m.MAC === mac) ?? null
const setLabel = (set: EvalSet, run: CalibRun | undefined) => {
  const n = fitAt(run, set, 10)?.n
  return set === 'fixed' ? `the fixed ${n ?? ''}-filter cohort`.replace('  ', ' ') : `all pairs${n ? ` (n = ${n})` : ''}`
}

/**
 * Every calibration on the 1:1 plane at one MAC: FTIR EC against HIPS Fabs/MAC, one saved fit line per calibration and
 * fit, one panel per site picked in the Filters panel. A calibration whose line follows the diagonal is self-consistent
 * at that MAC; another MAC rotates every line about its own intercept. The legend hides a calibration, and pointing at
 * one fades the rest and draws its filters.
 */
function MacOneToOne({ runs, view }: { runs: CalibRunsFile; view: MacView }) {
  const gallery = useMac()
  const [macText, setMac] = useSearchParam<(typeof MACS)[number]>('mac-sweep.mac', 'gallery', MACS)
  const mac = macText === 'gallery' ? gallery.mac : Number(macText)
  const lg = useLegend()
  const colours = useColourScheme('mac-sweep', 'categorical', 'Tableau 10')
  const presets = runs.presets.filter((p) => runs.runs.some((r) => r.preset === p.key && view.targets.includes(r.target)))
  const presetColour = categoricalScale(colours.name, presets.map((p) => p.key), colours.reverse)
  const colour = new Map(presets.map((p) => [p.key, presetColour(p.key)]))
  const fade = (c: string, label: string) => { const k = lg.dim(label); return k === 1 ? c : d3.color(c)?.copy({ opacity: k }).formatRgb() ?? c }
  const panels = view.targets.map((t) => {
    const lines: CrossLine[] = [], points: CrossPoint[] = [], extent: number[] = [0]
    const set = setFor(runs.runs.find((r) => r.target === t), view.set)
    for (const p of presets) {
      const run = runs.runs.find((r) => r.preset === p.key && r.target === t)
      // the saved MAC 10 fit, re-expressed at this MAC: slope × MAC/10, intercept unchanged (the explorer's 6 and 17 rows agree)
      const m = fitAt(run, set, 10)
      if (!run || !m) continue
      const name = presetName(p)
      run.eval.ref.forEach((fabs, i) => {
        const y = run.eval.pred[i]
        if ((set === 'fixed' && !run.eval.fixed[i]) || !Number.isFinite(fabs) || !Number.isFinite(y)) return
        extent.push(fabs / mac, y)
        if (lg.hover === name) points.push({ x: fabs / mac, y, color: colour.get(p.key)!, id: run.eval.id[i] ?? undefined, tip: [run.eval.id[i] ?? '', name, `Fabs/${mac} ${f2(fabs / mac)}`, `FTIR EC ${f2(y)}`] })
      })
      if (!lg.show(name)) continue
      for (const est of view.fits) {
        const slope = gallery.slope(num(m[`${est}_slope`] as number), 10), intercept = num(m[`${est}_intercept`] as number)
        if (slope == null || intercept == null) continue
        extent.push(intercept)
        lines.push({ slope, intercept, color: fade(colour.get(p.key)!, name), dash: FIT_STYLE[est].dash, method: est, stats: null, label: `${name} · ${FIT_STYLE[est].label}` })
      }
    }
    const hi = Math.max(...extent) * 1.05, lo = Math.min(0, ...extent) - (Math.min(0, ...extent) < 0 ? 0.5 : 0)
    return { t, set, lines, points, domain: [lo, hi] as [number, number] }
  })
  const first = runs.runs.find((r) => r.target === view.targets[0])
  const allOnly = panels.filter((pn) => pn.set !== view.set).map((pn) => siteName(runs, pn.t))
  return (
    <ChartFrame id="mac-sweep" title={`Every calibration at MAC ${mac} — on the 1:1 plane`}
      subtitle={`FTIR EC against HIPS Fabs ÷ ${mac}, with each calibration's saved fit on ${setLabel(setFor(first, view.set), first)}, for the sites picked in the Filters panel. A calibration self-consistent at this MAC follows the dashed 1:1 line. Switch the MAC (Filters follows the MAC set under Filters → units, now ${gallery.mac}): x rescales, so every line rotates about its own intercept, which never moves. The MAC decides which calibration is self-consistent (slope 1), never the offset. Fits follow Regression fits in the Filters panel (Deming dashed long, OLS solid); point at a calibration in the legend to see its filters.`}
      provenance={<>Stands in for ftir_19 (MAC fix across every setup) on the gallery's explorer presets and sites. Fits: <code>calibration_runs.json</code> runs[].metrics at MAC 10, <code>gallery/data/export_calibration_runs.py</code>, slopes re-expressed at the chosen MAC (lib/mac); nothing refitted.</>}
      controls={<><Segmented label="MAC (m²/g)" value={macText} options={MACS} onChange={setMac} optionLabel={(o) => (o === 'gallery' ? `Filters (${gallery.mac})` : o)} />{colours.picker}</>}>
      <FacetGrid items={panels} minCell={300} maxCols={3} render={(pn) => <div>
        {(panels.length > 1 || pn.set !== view.set) && <p className="facet-title">{siteName(runs, pn.t)}{pn.set !== view.set ? ' · all pairs' : ''}</p>}
        <Crossplot title="" points={pn.points} xLabel={`HIPS Fabs / ${mac} (µg/m³)`} yLabel="FTIR EC (µg/m³)" fit={false} grouping="none" extraLines={pn.lines} domain={pn.domain} maxSide={460} />
      </div>} />
      <Legend {...lg.props} items={presets.map((p) => ({ label: presetName(p), color: colour.get(p.key)!, shape: 'line' }))} />
      {viewNote(view, allOnly.length ? `The explorer has a fixed cohort for Addis only: ${allOnly.join(', ')} ${allOnly.length > 1 ? 'are' : 'is'} drawn on all pairs.` : '')}
    </ChartFrame>
  )
}

/** ftir_19's pivot figure for one calibration: the same filters at Fabs/10 and Fabs/6, both saved fits meeting at x = 0. */
function MacPivot({ runs, view }: { runs: CalibRunsFile; view: MacView }) {
  const target = view.targets[0]
  const presets = runs.presets.filter((p) => runs.runs.some((r) => r.preset === p.key && r.target === target))
  const o = useChartOptions('mac-pivot', [
    { key: 'preset', label: 'Calibration', style: 'select', default: 'ocec800_airspec', choices: presets.map((p) => ({ value: p.key, label: presetName(p) })) },
  ])
  const run = runs.runs.find((r) => r.preset === o.values.preset && r.target === target)
  const set = setFor(run, view.set)
  // MAC 10 against the reader's MAC, or against 6 while the reader is at 10
  const gallery = useMac()
  const alt = gallery.mac !== 10 ? gallery.mac : 6
  const m10 = fitAt(run, set, 10)
  const points: CrossPoint[] = []
  if (run) run.eval.pred.forEach((y, i) => {
    const fabs = run.eval.ref[i]
    if ((set === 'fixed' && !run.eval.fixed[i]) || !Number.isFinite(y) || !Number.isFinite(fabs)) return
    const id = run.eval.id[i] ?? undefined
    points.push({ x: fabs / 10, y, color: MAC10, id, tip: [id ?? '', `Fabs/10 ${f2(fabs / 10)}`, `FTIR EC ${f2(y)}`] })
    points.push({ x: fabs / alt, y, color: MAC6, hollow: true, id, tip: [id ?? '', `Fabs/${alt} ${f2(fabs / alt)}`, `FTIR EC ${f2(y)}`] })
  })
  const line = (m: typeof m10, mac: number, color: string, est: Estimator): CrossLine[] => {
    // the saved MAC 10 fit at `mac`: slope × mac/10, intercept unchanged
    const slope = slopeAt(num(m?.[`${est}_slope`] as number), mac, 10), intercept = num(m?.[`${est}_intercept`] as number)
    return slope == null || intercept == null ? [] : [{ slope, intercept, color, dash: FIT_STYLE[est].dash, method: est, stats: null, label: `MAC ${mac} ${FIT_STYLE[est].label}: ${f2(slope)}x ${signed(intercept)}` }]
  }
  const lines = view.fits.flatMap((est) => [...line(m10, 10, MAC10, est), ...line(m10, alt, MAC6, est)])
  const ints = lines.map((l) => l.intercept)
  const hi = Math.max(0, ...points.map((p) => Math.max(p.x, p.y))) * 1.05
  const lo = Math.min(0, ...ints) - 0.5
  return (
    <ChartFrame id="mac-pivot" title="One calibration, two MACs — the line pivots on its intercept"
      subtitle={`${siteName(runs, target)}, ${setLabel(set, run)}. Filled points: x = Fabs/10; hollow: the same filters at Fabs/${alt} (${gallery.mac !== 10 ? 'the MAC set under Filters' : 'MAC 6, until another MAC is set under Filters'}; the cloud stretches by 10/${alt}). Both fits are drawn back to x = 0, where they meet: the axes run below zero so the shared intercept is visible. Dashed grey is 1:1, where a self-consistent calibration and MAC pairing would lie. Site, filters and fits follow the Filters panel.`}
      provenance={<>Stands in for ftir_19 §2 and ftir_31 §1.11 (the pivot redone with Deming). Points and fits: <code>calibration_runs.json</code> (explorer eval and saved metrics at MAC 6 and 10).</>}
      controls={o.controls()}>
      {run ? <>
        <Crossplot title="" points={points} xLabel="HIPS Fabs / MAC (µg/m³)" yLabel="Predicted FTIR EC (µg/m³)" fit={false} grouping="none" extraLines={lines} domain={[lo, hi]} maxSide={520} />
        <Key items={[{ label: 'Fabs/10 (filled), fit at MAC 10', color: MAC10 }, { label: `Fabs/${alt} (hollow), fit at MAC ${alt}`, color: MAC6, kind: 'hollow' }, { label: '1:1', color: INK.identity, kind: 'dash' }]} />
        {lines.length > 0 && <p className="legend-note">{lines.map((l) => l.label).join(' · ')}; R² {f2(m10?.R2)} at both.</p>}
      </> : <Empty>No {siteName(runs, target)} run for this calibration.</Empty>}
      {viewNote(view, [view.targets.length > 1 ? `With several sites picked, this shows the first: ${siteName(runs, target)}.` : '',
        set !== view.set ? `${siteName(runs, target)} has no fixed cohort in the explorer, so it is drawn on all pairs.` : ''].filter(Boolean).join(' '))}
    </ChartFrame>
  )
}

/** ftir_19's per-filter columns: which setup each prediction column is (the notebook's order and names) */
const SETUP_COLUMNS: [string, string][] = [
  ['EC_deployed_ugm3', 'Deployed SPARTAN'], ['EC_smoke_906_ugm3', 'Biomass-smoke (906)'], ['EC_smoke_shape_300_ugm3', 'Ethiopia-shaped smoke (300)'],
  ['EC_analog_raw_first_ugm3', 'Spectral analogs (400)'], ['EC_lowest_ocec_800_ugm3', 'Lowest-OC/EC (800)'], ['lowest-OCEC 800, AIRSpec df1=6', 'Lowest-OC/EC + AIRSpec'],
]
const SWEEP_MACS = d3.range(4, 20.01, 0.25)

/**
 * ftir_19's six setups refitted in the browser from its per-filter table (addis_predictions_all_setups.csv):
 * OLS of each setup's EC on Fabs/MAC over the fixed 190-filter cohort, as the notebook did, at MAC 10 and at
 * the MAC set under Filters. At MAC 10 and 6 the refit reproduces the notebook's table to its printed digits.
 * The sweep shows how far each setup's RMSE against Fabs/MAC moves as the MAC does, and where its slope is 1.
 */
function SetupsAtMac({ rows: filters, saved, provenance: prov }: { rows: Row[]; saved: Row[]; provenance: React.ReactNode }) {
  const { mac } = useMac()
  const o = useChartOptions('mac-ftir19', [
    { key: 'cohort', label: 'Filters', default: 'fixed', choices: [{ value: 'fixed', label: 'Fixed 190-filter cohort' }, { value: 'all', label: 'All with Fabs' }] },
    { key: 'view', label: 'Show', default: 'table', choices: [{ value: 'table', label: 'Table' }, { value: 'rmse', label: 'RMSE by MAC' }, { value: 'slope', label: 'Slope by MAC' }] },
  ])
  const colours = useColourScheme('mac-ftir19', 'categorical', 'Tableau 10')
  // each setup keeps its colour whichever columns the table has
  const setupColour = useMemo(() => categoricalScale(colours.name, SETUP_COLUMNS.map(([, name]) => name), colours.reverse), [colours.name, colours.reverse])
  const setups = useMemo(() => {
    const use = filters.filter((r) => (o.values.cohort === 'all' || r.in_fixed_phase2_cohort === true) && num(r.Fabs) != null)
    return SETUP_COLUMNS.filter(([col]) => use.some((r) => num(r[col]) != null)).map(([col, name]) => {
      const pairs = use.flatMap((r) => { const y = num(r[col]); return y == null ? [] : [[num(r.Fabs)!, y] as const] })
      const at = (m: number) => {
        const xs = pairs.map(([f]) => f / m), ys = pairs.map(([, y]) => y)
        const st = regression(xs, ys)
        return { slope: st?.slope ?? null, intercept: st?.intercept ?? null, r2: st?.r2 ?? null,
          rmse: xs.length ? Math.sqrt(d3.mean(xs, (x, k) => (ys[k] - x) ** 2)!) : null }
      }
      const s10 = at(10)
      return { name, color: setupColour(name), n: pairs.length, s10, here: at(mac), at,
        // the MAC at which this setup's slope against Fabs/MAC is 1 (slope scales with MAC)
        unity: s10.slope ? 10 / s10.slope : null }
    })
  }, [filters, o.values.cohort, mac, setupColour])
  const savedBy = new Map(saved.map((r) => [String(r.setup), r]))
  const table: Row[] = setups.map((s) => ({ setup: s.name, n: s.n, intercept: s.s10.intercept, slope10: s.s10.slope, slopeHere: s.here.slope, r2: s.s10.r2,
    rmse10: s.s10.rmse, rmseHere: s.here.rmse, unity: s.unity, notebook: num(savedBy.get(s.name)?.['slope @ MAC 10']) }))
  // the reader's-MAC columns only when the reader is off MAC 10
  const here = mac !== 10
  const columns: [string, (r: Row) => string][] = [
    ['Setup', (r) => String(r.setup)], ['n', (r) => fmt(r.n, 0)], ['Intercept', (r) => f2(r.intercept)], ['Slope @ MAC 10', (r) => f2(r.slope10)],
    ...(here ? [[`Slope @ MAC ${mac}`, (r: Row) => f2(r.slopeHere)] as [string, (r: Row) => string]] : []),
    ['R²', (r) => fmt(r.r2, 3)], ['RMSE @ MAC 10', (r) => f2(r.rmse10)],
    ...(here ? [[`RMSE @ MAC ${mac}`, (r: Row) => f2(r.rmseHere)] as [string, (r: Row) => string]] : []),
    ['MAC for slope 1', (r) => f2(r.unity)],
  ]
  const sweep = (key: 'rmse' | 'slope') => setups.map((s) => ({ id: s.name, label: s.name, color: s.color, marks: false,
    points: SWEEP_MACS.map((m) => { const v = key === 'rmse' ? s.at(m).rmse : s.s10.slope != null ? s.s10.slope * (m / 10) : null
      return { x: m, y: v, tip: [s.name, `MAC ${fmt(m, 2)}`, `${key === 'rmse' ? 'RMSE' : 'slope'} ${f2(v)}`] } }) }))
  return (
    <ChartFrame id="mac-ftir19" title={`ftir_19's six setups at MAC ${mac} — refitted from its per-filter predictions`}
      subtitle={`OLS of each setup's Addis EC on HIPS Fabs ÷ MAC (ftir_19's fit), on ${o.values.cohort === 'fixed' ? 'the fixed 190-filter cohort' : 'every filter with Fabs'}. The MAC is the one set under Filters → units; at MAC 10 and 6 this reproduces the notebook's table. Intercept and R² do not depend on the MAC; the slope scales with it, so each setup has one MAC at which it is self-consistent (slope 1, last column). RMSE against Fabs/MAC moves only as a re-expression of that slope, so R² cannot arbitrate the MAC. Includes the deployed SPARTAN calibration, which the explorer presets do not.`}
      provenance={prov} controls={<>{o.controls()}{o.values.view !== 'table' && colours.picker}</>}>
      {o.values.view === 'table'
        ? <div style={{ overflowX: 'auto' }}><table className="census-table">
            <thead><tr>{columns.map(([h]) => <th key={h} style={{ textTransform: 'none', letterSpacing: 0 }}>{h}</th>)}</tr></thead>
            <tbody>{table.map((r) => <tr key={String(r.setup)}>
              <td>{String(r.setup)}</td>
              {columns.slice(1).map(([h, v]) => <td key={h} style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{v(r)}</td>)}
            </tr>)}</tbody>
          </table></div>
        : <SweepPlot series={sweep(o.values.view === 'rmse' ? 'rmse' : 'slope')} xLabel="MAC (m²/g)" yLabel={o.values.view === 'rmse' ? 'RMSE against Fabs/MAC (µg/m³)' : 'Slope against Fabs/MAC'}
            include={[0]} reference={o.values.view === 'slope' ? 1 : undefined} height={300}
            xMarkers={[{ at: 6, label: 'MAC 6', color: MAC6 }, { at: 10, label: 'MAC 10', color: MAC10 }, ...(mac !== 6 && mac !== 10 ? [{ at: mac, label: `Filters: ${mac}`, color: INK.text }] : [])]} />}
      {o.values.view !== 'table' && <Key items={setups.map((s) => ({ label: s.name, color: s.color, kind: 'line' as const }))} />}
    </ChartFrame>
  )
}

/** ftir_16's population: the Addis filters its ChemSpec cross-check joined (ChemSpec EC, HIPS Fabs), with FTIR OC */
type ImpliedSet = 'ftir16' | 'all'
const plural = (n: number) => `${n} filter${n === 1 ? '' : 's'}`

/**
 * ftir_16 §3 (cell 7) per filter: if Addis had Adama's TOR OC/EC r, TOR-consistent EC = OC_ftir / r and the MAC that
 * reconciles HIPS with it is Fabs × r / OC_ftir. Computed here from the gallery's Addis rows (so it follows the
 * filters, seasons and exclusions in view), for each of the notebook's three Adama ratios, against the reader's MAC.
 */
function ImpliedMac({ rows: all, scenarios, provenance: prov }: { rows: FilterRow[]; scenarios: Row[]; provenance: React.ReactNode }) {
  const { mac } = useMac()
  const o = useChartOptions('mac-implied', [
    { key: 'scenario', label: 'Adama OC/EC', default: 'Adama median', choices: scenarios.map((r) => ({ value: String(r.scenario), label: `${String(r.scenario).replace(/ \(.*\)$/, '')} (${f2(r.adama_OC_EC)})` })) },
    { key: 'set', label: 'Filters', default: 'ftir16', choices: [{ value: 'ftir16', label: 'As ftir_16 (with ChemSpec EC)' }, { value: 'all', label: 'All with OC and Fabs' }] },
  ])
  const set = o.values.set as ImpliedSet
  // ftir_16 joined on its ChemSpec cross-check, so its 175 filters are those with ChemSpec EC too
  const addis = useMemo(() => all.filter((r) => r.code === 'ETAD' && (rowNum(r, 'OC (FTIR)') ?? 0) > 0 && rowNum(r, 'HIPS Fabs') !== null
    && (set === 'all' || rowNum(r, 'EC (ChemSpec FTIR)') !== null)), [all, set])
  const summary = scenarios.map((s) => {
    const ratio = num(s.adama_OC_EC) ?? NaN
    const v = addis.map((r) => (rowNum(r, 'HIPS Fabs')! * ratio) / rowNum(r, 'OC (FTIR)')!)
    return { s, ratio, v, p25: quantile(v, 0.25), median: quantile(v, 0.5), p75: quantile(v, 0.75) }
  })
  const chosen = summary.find((x) => String(x.s.scenario) === o.values.scenario) ?? summary[1] ?? summary[0]
  if (!chosen || addis.length < 5) return (
    <ChartFrame id="mac-implied" title="Implied MAC per Addis filter" provenance={prov} controls={o.controls()}>
      <Empty>Fewer than 5 Addis filters in view carry FTIR OC and HIPS Fabs{set === 'ftir16' ? ' (and ChemSpec EC)' : ''}.</Empty>
    </ChartFrame>
  )
  // the notebook's histogram: 40 bins with the values clipped at 80 m²/g
  const clipped = chosen.v.map((x) => Math.min(x, 80))
  const lo = Math.min(0, d3.min(clipped)!), edges = d3.range(41).map((i) => lo + ((80 - lo) * i) / 40)
  const counts = edges.slice(0, -1).map((e, i) => clipped.filter((x) => x >= e && (i === 39 ? x <= edges[40] : x < edges[i + 1])).length)
  const over = chosen.v.filter((x) => x > 80).length
  const reader = mac !== 6 && mac !== 10 ? [{ at: mac, label: `Filters: MAC ${mac}`, color: INK.text }] : []
  const agree = summary.map((x) => `${String(x.s.scenario).replace(/ \(.*\)$/, '')} ${f2(x.median)} (IQR ${f2(x.p25)}–${f2(x.p75)}; ftir_16 ${f2(x.s.implied_MAC_median)}, ${f2(x.s.implied_MAC_p25)}–${f2(x.s.implied_MAC_p75)})`).join('; ')
  return (
    <ChartFrame id="mac-implied" title="Implied MAC per Addis filter — under Adama's OC/EC, every filter needs a MAC far above 10"
      subtitle={`Per Addis filter, MAC = HIPS Fabs × r / FTIR OC for Adama's TOR OC/EC r = ${f2(chosen.ratio)} (ftir_16 §3), ${addis.length} filters in view (${set === 'ftir16' ? 'those ftir_16 joined: with ChemSpec EC, FTIR OC and Fabs' : 'every Addis filter with FTIR OC and Fabs'}); values above 80 m²/g pile into the last bin, as in the notebook. Median (IQR) here, against ftir_16's table on its 175: ${agree}. ${set === 'ftir16' && addis.length === 175 ? 'Same filters, so the same numbers to rounding.' : 'A different set of filters, so the numbers differ.'} The MAC set under Filters (${mac}) is drawn with MAC 6 and 10; the implied MAC does not depend on it.`}
      provenance={prov} controls={o.controls()}>
      <SweepPlot series={[histogramSeries({ id: 'implied', label: 'Addis filters', color: MAC10, edges, counts,
        tip: (i, c) => [`MAC ${fmt(edges[i], 1)}–${fmt(edges[i + 1], 1)}${i === 39 ? ' (and above)' : ''}`, `${c} filters`] })]}
        xLabel={`Implied MAC (m²/g) under Adama OC/EC ${f2(chosen.ratio)}`} yLabel="Addis filters" include={[0]} height={280}
        xMarkers={[{ at: 6, label: 'MAC 6', color: MAC6 }, { at: 10, label: 'MAC 10', color: MAC10 }, ...reader]} />
      <Note>{BASIS_NOTE} {plural(over)} above 80 m²/g {over === 1 ? 'sits' : 'sit'} in the last bin.</Note>
    </ChartFrame>
  )
}
