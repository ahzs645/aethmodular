import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Figure } from '@/components/Figure'
import { Key } from '@/components/Key'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot } from '@/components/SweepPlot'
import { useChartOptions } from '@/lib/chartOptions'
import { f2, signed } from '@/lib/format'
import { SidebarControls } from '@/lib/pageSidebar'
import { FIT_STYLE, useRegressions } from '@/lib/regressionPrefs'
import type { Estimator } from '@/lib/regressionMethods'
import { num } from '@/lib/table'
import { INK } from '@/lib/theme'
import type { CalibRun, CalibRunsFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { PageIntro, presetName, provenance, rows, useCalibRuns, useSpectral, Waiting } from './spectralCommon'

const MAC10 = '#2b6cb0', MAC6 = '#c05621'
const siteName = (runs: CalibRunsFile, t: string) => runs.targets[t]?.label.split(':')[0] ?? t
/** the explorer's calibration targets by SPARTAN site code, so the Filters panel's site pick chooses them */
const TARGET_OF: Record<string, string> = { ETAD: 'addis', CHTS: 'chts', ETBI: 'etbi', INDH: 'indh', USPA: 'uspa' }
/** the explorer saved Deming and OLS fits only: the Regression fits setting draws those of its choices */
const SAVED_FITS: readonly Estimator[] = ['deming', 'ols']
type EvalSet = 'fixed' | 'all'
const MACS = ['6', '10', '17'] as const

/** the page's choices from the Filters panel: its calibration sites (Addis when none is picked), its filters, its fits */
interface MacView { targets: string[]; fallback: boolean; set: EvalSet; fits: Estimator[]; unsaved: Estimator[] }

/**
 * ftir_16 and ftir_19: MAC 6 or 10. HIPS enters every crossplot as x = Fabs/MAC, so the MAC rescales
 * x: each calibration keeps its intercept and R² and its slope scales by 6/10. The explorer saved
 * every preset's fits at MAC 6, 10 and 17 (calibration_runs.json), so the gallery draws that across
 * every calibration and site; ftir_16's IMPROVE bridge says which MAC the network supports.
 */
export function MacChoice({ picked }: SectionPageProps) {
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

    <Figure kind="table" id="mac-ftir19" data={rows(data, 'setups')}
      columns={[{ key: 'setup', label: 'Setup (ftir_19)' }, { key: 'intercept (both MACs)', label: 'Intercept, both MACs', format: 'f2' },
        { key: 'slope @ MAC 10', label: 'Slope @ MAC 10', format: 'f2' }, { key: 'slope @ MAC 6', label: 'Slope @ MAC 6', format: 'f2' },
        { key: 'R2 (both MACs)', label: 'R², both MACs', format: 'f3' }, { key: 'RMSE @ MAC 10', label: 'RMSE @ MAC 10', format: 'f2' }, { key: 'RMSE @ MAC 6', label: 'RMSE @ MAC 6', format: 'f2' }]}
      title="ftir_19's six setups — OLS on the fixed 190-filter cohort"
      subtitle="As committed in the notebook (OLS, which ftir_19 used; the deck's setup matrix). Includes the deployed SPARTAN calibration, which the explorer presets do not. The spectral-analogs (400) setup fails the held-out TOR test. RMSE moves with MAC only as a re-expression of the slope; R² cannot arbitrate the MAC."
      provenance={provenance(data, ['setups'])} />

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
    view.fallback ? 'None of the sites picked in the Filters panel has explorer calibrations (Addis, Beijing, Bishoftu, Delhi, Pasadena), so Addis is shown.' : '',
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
  const [macText, setMac] = useSearchParam<(typeof MACS)[number]>('mac-sweep.mac', '10', MACS)
  const mac = Number(macText)
  const lg = useLegend()
  const presets = runs.presets.filter((p) => runs.runs.some((r) => r.preset === p.key && view.targets.includes(r.target)))
  const colour = new Map(presets.map((p, i) => [p.key, d3.schemeTableau10[i % 10]]))
  const fade = (c: string, label: string) => { const k = lg.dim(label); return k === 1 ? c : d3.color(c)?.copy({ opacity: k }).formatRgb() ?? c }
  const panels = view.targets.map((t) => {
    const lines: CrossLine[] = [], points: CrossPoint[] = [], extent: number[] = [0]
    const set = setFor(runs.runs.find((r) => r.target === t), view.set)
    for (const p of presets) {
      const run = runs.runs.find((r) => r.preset === p.key && r.target === t)
      const m = fitAt(run, set, mac)
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
        const slope = num(m[`${est}_slope`] as number), intercept = num(m[`${est}_intercept`] as number)
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
      subtitle={`FTIR EC against HIPS Fabs ÷ ${mac}, with each calibration's saved fit on ${setLabel(setFor(first, view.set), first)}, for the sites picked in the Filters panel. A calibration self-consistent at this MAC follows the dashed 1:1 line. Switch the MAC: x rescales, so every line rotates about its own intercept, which never moves. The MAC decides which calibration is self-consistent (slope 1), never the offset. Fits follow Regression fits in the Filters panel (Deming dashed long, OLS solid); point at a calibration in the legend to see its filters.`}
      provenance={<>Stands in for ftir_19 (MAC fix across every setup) on the gallery's explorer presets and sites. Fits: <code>calibration_runs.json</code> runs[].metrics, <code>gallery/data/export_calibration_runs.py</code>; nothing refitted.</>}
      controls={<Segmented label="MAC (m²/g)" value={macText} options={MACS} onChange={setMac} />}>
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
  const at = (mac: number) => fitAt(run, set, mac)
  const m10 = at(10), m6 = at(6)
  const points: CrossPoint[] = []
  if (run) run.eval.pred.forEach((y, i) => {
    const fabs = run.eval.ref[i]
    if ((set === 'fixed' && !run.eval.fixed[i]) || !Number.isFinite(y) || !Number.isFinite(fabs)) return
    const id = run.eval.id[i] ?? undefined
    points.push({ x: fabs / 10, y, color: MAC10, id, tip: [id ?? '', `Fabs/10 ${f2(fabs / 10)}`, `FTIR EC ${f2(y)}`] })
    points.push({ x: fabs / 6, y, color: MAC6, hollow: true, id, tip: [id ?? '', `Fabs/6 ${f2(fabs / 6)}`, `FTIR EC ${f2(y)}`] })
  })
  const line = (m: typeof m10, mac: number, color: string, est: Estimator): CrossLine[] => {
    const slope = num(m?.[`${est}_slope`] as number), intercept = num(m?.[`${est}_intercept`] as number)
    return slope == null || intercept == null ? [] : [{ slope, intercept, color, dash: FIT_STYLE[est].dash, method: est, stats: null, label: `MAC ${mac} ${FIT_STYLE[est].label}: ${f2(slope)}x ${signed(intercept)}` }]
  }
  const lines = view.fits.flatMap((est) => [...line(m10, 10, MAC10, est), ...line(m6, 6, MAC6, est)])
  const ints = lines.map((l) => l.intercept)
  const hi = Math.max(0, ...points.map((p) => Math.max(p.x, p.y))) * 1.05
  const lo = Math.min(0, ...ints) - 0.5
  return (
    <ChartFrame id="mac-pivot" title="One calibration, both MACs — the line pivots on its intercept"
      subtitle={`${siteName(runs, target)}, ${setLabel(set, run)}. Filled points: x = Fabs/10; hollow: the same filters at Fabs/6 (the cloud stretches right by 10/6). Both saved fits are drawn back to x = 0, where they meet: the axes run below zero so the shared intercept is visible. Dashed grey is 1:1, where a self-consistent calibration and MAC pairing would lie. Site, filters and fits follow the Filters panel.`}
      provenance={<>Stands in for ftir_19 §2 and ftir_31 §1.11 (the pivot redone with Deming). Points and fits: <code>calibration_runs.json</code> (explorer eval and saved metrics at MAC 6 and 10).</>}
      controls={o.controls()}>
      {run ? <>
        <Crossplot title="" points={points} xLabel="HIPS Fabs / MAC (µg/m³)" yLabel="Predicted FTIR EC (µg/m³)" fit={false} grouping="none" extraLines={lines} domain={[lo, hi]} maxSide={520} />
        <Key items={[{ label: 'Fabs/10 (filled), fit at MAC 10', color: MAC10 }, { label: 'Fabs/6 (hollow), fit at MAC 6', color: MAC6, kind: 'hollow' }, { label: '1:1', color: INK.identity, kind: 'dash' }]} />
        {lines.length > 0 && <p className="legend-note">{lines.map((l) => l.label).join(' · ')}; R² {f2(m10?.R2)} at both.</p>}
      </> : <Empty>No {siteName(runs, target)} run for this calibration.</Empty>}
      {viewNote(view, [view.targets.length > 1 ? `With several sites picked, this shows the first: ${siteName(runs, target)}.` : '',
        set !== view.set ? `${siteName(runs, target)} has no fixed cohort in the explorer, so it is drawn on all pairs.` : ''].filter(Boolean).join(' '))}
    </ChartFrame>
  )
}
