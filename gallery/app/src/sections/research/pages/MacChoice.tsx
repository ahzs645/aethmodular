import { useMemo } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Crossplot, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { Figure } from '@/components/Figure'
import { Key } from '@/components/Key'
import { SweepPlot } from '@/components/SweepPlot'
import { metricRow } from '@/charts/calibration/common'
import { useChartOptions } from '@/lib/chartOptions'
import { f2, signed } from '@/lib/format'
import type { Estimator } from '@/lib/regressionMethods'
import { num, type Row } from '@/lib/table'
import { INK } from '@/lib/theme'
import type { CalibRunsFile } from '@/lib/types'
import type { SectionPageProps } from '../../types'
import { PageIntro, presetName, provenance, rows, useCalibRuns, useSpectral, Waiting } from './spectralCommon'

const MAC10 = '#2b6cb0', MAC6 = '#c05621'
const siteName = (runs: CalibRunsFile, t: string) => runs.targets[t]?.label.split(':')[0] ?? t

/**
 * ftir_16 and ftir_19: MAC 6 or 10. HIPS enters every crossplot as x = Fabs/MAC, so the MAC rescales
 * x: each calibration keeps its intercept and R² and its slope scales by 6/10. The explorer saved
 * every preset's fits at MAC 6, 10 and 17 (calibration_runs.json), so the gallery draws that across
 * every calibration and site; ftir_16's IMPROVE bridge says which MAC the network supports.
 */
export function MacChoice(_props: SectionPageProps) {
  const { data, error } = useSpectral('mac_choice')
  const { runs, error: runsError } = useCalibRuns()

  // every preset × site × evaluation set × estimator × MAC, from the explorer's saved fits
  const fits = useMemo(() => {
    if (!runs) return []
    const label = new Map(runs.presets.map((p) => [p.key, presetName(p)]))
    const out: Row[] = []
    for (const run of runs.runs) for (const m of run.metrics) for (const est of ['deming', 'ols'] as const) {
      if (m.MAC == null) continue
      out.push({ calibration: label.get(run.preset) ?? run.preset, site: siteName(runs, run.target),
        set: m.evaluation_set === 'fixed' ? `fixed ${m.n}-filter cohort` : `all pairs (n = ${m.n})`,
        estimator: est === 'deming' ? 'Deming' : 'OLS', MAC: m.MAC,
        slope: num(m[`${est}_slope`] as number), intercept: num(m[`${est}_intercept`] as number) })
    }
    return out
  }, [runs])

  if (!data || !runs) return <Waiting error={error || runsError} />
  const summary = rows(data, 'bridge_summary')
  return <>
    <PageIntro id="mac-intro" title="MAC 6 or 10 — the choice moves every slope and no intercept" data={data}
      notebooks="research/ftir_ec_phase3/ftir_16_mac_decision_prep, ftir_19_mac_effect_on_calibrations; ftir_31 §1.11 (the pivot with Deming)">
      <p>The 28 Oct meeting page §4 has the MAC by season. This page applies MAC 6 and 10 to every calibration the explorer exported, and sets them against the IMPROVE evidence.</p>
    </PageIntro>

    <Figure kind="sweep" id="mac-sweep" data={fits} x="MAC" y="slope" series="calibration" xLabel="MAC used to convert HIPS Fabs (m²/g)" yLabel="Slope, FTIR EC vs Fabs/MAC"
      choose={[{ column: 'site', label: 'Site', default: 'Addis (ETAD)' }, { column: 'set', label: 'Filters' }, { column: 'estimator', label: 'Fit', default: 'Deming' }]}
      variants={[{ key: 'measure', label: 'Show', choices: [
        { value: 'slope', label: 'Slope', patch: { y: 'slope', yLabel: 'Slope, FTIR EC vs Fabs/MAC', reference: 1 } },
        { value: 'intercept', label: 'Intercept', patch: { y: 'intercept', yLabel: 'Intercept (µg/m³)', reference: 0 } }] }]}
      reference={1} xMarkers={[{ at: 6, label: 'MAC 6' }, { at: 10, label: 'MAC 10' }]} height={320}
      title="Every calibration at MAC 6, 10 and 17 — slopes scale with the MAC, intercepts stay flat"
      subtitle="The explorer's saved fits for each calibration preset at each site. Switch Show to Intercept: every line is flat, because rescaling x cannot move where a line crosses x = 0. The MAC decides which calibration is self-consistent (slope 1), never the offset."
      provenance={<>Stands in for ftir_19 (MAC fix across every setup) on the gallery's nine explorer presets and five sites. Fits: <code>calibration_runs.json</code> runs[].metrics, <code>gallery/data/export_calibration_runs.py</code>; nothing refitted.</>} />

    <MacPivot runs={runs} />

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

/** ftir_19's pivot figure for one calibration at Addis: the same filters at Fabs/10 and Fabs/6, both saved fits meeting at x = 0. */
function MacPivot({ runs }: { runs: CalibRunsFile }) {
  const presets = runs.presets.filter((p) => runs.runs.some((r) => r.preset === p.key && r.target === 'addis'))
  const o = useChartOptions('mac-pivot', [
    { key: 'preset', label: 'Calibration', style: 'select', default: 'ocec800_airspec', choices: presets.map((p) => ({ value: p.key, label: presetName(p) })) },
    { key: 'fit', label: 'Fit', default: 'deming', choices: [{ value: 'deming', label: 'Deming' }, { value: 'ols', label: 'OLS' }] },
  ])
  const run = runs.runs.find((r) => r.preset === o.values.preset && r.target === 'addis')
  const est = o.values.fit as Estimator
  const at = (mac: number) => metricRow(run?.metrics, 'fixed', mac)
  const m10 = at(10), m6 = at(6)
  const points: CrossPoint[] = []
  if (run) run.eval.pred.forEach((y, i) => {
    const fabs = run.eval.ref[i]
    if (!run.eval.fixed[i] || !Number.isFinite(y) || !Number.isFinite(fabs)) return
    const id = run.eval.id[i] ?? undefined
    points.push({ x: fabs / 10, y, color: MAC10, id, tip: [id ?? '', `Fabs/10 ${f2(fabs / 10)}`, `FTIR EC ${f2(y)}`] })
    points.push({ x: fabs / 6, y, color: MAC6, hollow: true, id, tip: [id ?? '', `Fabs/6 ${f2(fabs / 6)}`, `FTIR EC ${f2(y)}`] })
  })
  const line = (m: typeof m10, mac: number, color: string, dash?: string): CrossLine[] => {
    const slope = num(m?.[`${est}_slope`] as number), intercept = num(m?.[`${est}_intercept`] as number)
    return slope == null || intercept == null ? [] : [{ slope, intercept, color, dash, method: est, stats: null, label: `MAC ${mac}: ${f2(slope)}x ${signed(intercept)}` }]
  }
  const lines = [...line(m10, 10, MAC10), ...line(m6, 6, MAC6, '6 4')]
  const ints = lines.map((l) => l.intercept)
  const hi = Math.max(0, ...points.map((p) => Math.max(p.x, p.y))) * 1.05
  const lo = Math.min(0, ...ints) - 0.5
  return (
    <ChartFrame id="mac-pivot" title="One calibration, both MACs — the line pivots on its intercept"
      subtitle="Addis, the fixed 190-filter cohort. Filled points: x = Fabs/10; hollow: the same filters at Fabs/6 (the cloud stretches right by 10/6). Both saved fits are drawn back to x = 0, where they meet: the axes run below zero so the shared intercept is visible. Dashed grey is 1:1, where a self-consistent calibration and MAC pairing would lie."
      provenance={<>Stands in for ftir_19 §2 and ftir_31 §1.11 (the pivot redone with Deming). Points and fits: <code>calibration_runs.json</code> (explorer eval and saved metrics at MAC 6 and 10).</>}
      controls={o.controls()}>
      {run ? <>
        <Crossplot title="" points={points} xLabel="HIPS Fabs / MAC (µg/m³)" yLabel="Predicted FTIR EC (µg/m³)" fit={false} grouping="none" extraLines={lines} domain={[lo, hi]} maxSide={520} />
        <Key items={[{ label: 'Fabs/10 (filled), fit at MAC 10', color: MAC10 }, { label: 'Fabs/6 (hollow), fit at MAC 6', color: MAC6, kind: 'hollow' }, { label: '1:1', color: INK.identity, kind: 'dash' }]} />
        <p className="legend-note">{lines.map((l) => l.label).join(' · ')}; R² {f2(m10?.R2)} at both.</p>
      </> : <Empty>No Addis run for this calibration.</Empty>}
    </ChartFrame>
  )
}
