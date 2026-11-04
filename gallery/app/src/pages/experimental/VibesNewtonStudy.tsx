import { useMemo } from 'react'
import { ChartFrame, ExplainOnly, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { FindingsList } from '@/components/Findings'
import { useSearchParam } from '@/lib/url'
import { baseFilterId } from '@/lib/highlight'
import { f2, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import { Forest, rowsOf, Scatter, sourceOf, useCounted, useXp, type Cols, type XpBase } from './shared'

/**
 * VIBES with the ELBO solved to convergence (the v2 run) next to the 19 Oct run: what the 155 premature stops change.
 * (research/ftir_hips_chem/vibes_newton_refit_20261103.ipynb, docs/vibes-newton-adoption-tests-2026-11-03.md)
 */

interface Checks { n: number; closer: number; closer_p: number; flatter: number; flatter_p: number; res_saved: number; res_newton: number; flat_saved: number; flat_newton: number }
interface VibesNewtonData extends XpBase { checks: Checks; addis: Cols; port: Cols; calibration: Cols; bands: Cols; featureless: Cols; ec: Cols }
interface AddisRow { filter_key: string; saved_iterations: number; flat_saved: number; flat_newton: number; res_saved: number; res_newton: number; dEC: number; es: number; closer: string }
interface PortRow { run: string; s10: number; b5: number; short_shared: number }
interface BandRow { group: string; band: string; n: number; gt5pct: number; gt20pct: number; r: number }
interface EcRow { filter_id: string; filter_key: string; code: string; saved_iterations: number; premature_stop: boolean; baseline_change_frac_rms: number; ec_saved: number; ec_newton: number }

const source = sourceOf('VibesNewtonStudy')
const runLine = (d: VibesNewtonData) => `vibes_newton_refit_20261103.ipynb · v2 run, ${d.settings.analyses} analyses`
const PREMATURE = 'stopped early'
const OTHER = 'baseline moved > 5 %'
const GROUPS = ['addis', 'network', 'improve']
const GROUP_LABEL: Record<string, string> = { addis: 'Addis', network: 'SPARTAN network', improve: 'IMPROVE' }

export function VibesNewtonStudy() {
  const { data, state } = useXp<VibesNewtonData>('vibes_newton', 'VIBES solved to convergence')
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <AddisChecks d={data} />
    <EcCross d={data} />
    <Bands d={data} />
    <Port d={data} />
  </>
}

function Findings({ d }: { d: VibesNewtonData }) {
  const c = d.checks
  return (
    <ExplainOnly>
      <ChartFrame id="xn-findings" title="VIBES solved to convergence — short answers" exportable={false}>
        <FindingsList note={<>{runLine(d)}. This is the same objective, background and tau as the 19 Oct run. Only the ELBO solver changes, which is a departure from the published procedure.</>}>
          <li><strong>{d.settings.premature} of {d.settings.analyses} analyses stop early.</strong> SciPy's relative-reduction test ends them within 30 iterations, far from the optimum. All {d.settings.certified} refit and certified.</li>
          <li><strong>The calibration doesn't depend on the solver.</strong> Rebuilt on Newton baselines, it selects the same k, and test RMSE and coefficients match.</li>
          <li><strong>At Addis the early stops left a real hump.</strong> All {c.flatter} of {c.n} early-stop filters are flatter in the featureless 2000–2500 cm⁻¹ window (p {c.flatter_p}), and EC falls on all of them.</li>
          <li><strong>Whether the lower EC is more correct isn't shown.</strong> Against the site's own EC–HIPS line, {c.closer} of {c.n} land closer (p {c.closer_p}).</li>
          <li><strong>Band areas are where it matters.</strong> About 97 % of spectra don't move, but the Addis carbonyl and 1600 cm⁻¹ bands shift by more than 20 % on a few filters.</li>
        </FindingsList>
      </ChartFrame>
    </ExplainOnly>
  )
}

function AddisChecks({ d }: { d: VibesNewtonData }) {
  const rows = useMemo(() => rowsOf<AddisRow>(d.addis).sort((a, b) => a.dEC - b.dEC), [d])
  const c = d.checks
  const flat = (v: number) => (v * 1000).toFixed(2)
  const forest = rows.map((r) => ({
    key: r.filter_key, label: r.filter_key,
    values: { res_saved: { v: r.res_saved }, res_newton: { v: r.res_newton }, flat_saved: { v: r.flat_saved * 1000 }, flat_newton: { v: r.flat_newton * 1000 } },
    tip: [`${r.filter_key}: stopped at iteration ${r.saved_iterations}`, `EC ${signed(r.dEC, 2)} µg/m³ (${f2(r.es)} before)`, `closer to the site line: ${r.closer === 'newton' ? 'converged run' : '19 Oct run'}`,
      `2000–2500 cm⁻¹ |absorbance| ×1000: ${flat(r.flat_saved)} → ${flat(r.flat_newton)}`],
  }))
  const line = [Math.min(...rows.flatMap((r) => [r.res_saved, r.res_newton])) - 0.05, Math.max(...rows.flatMap((r) => [r.res_saved, r.res_newton])) + 0.05] as [number, number]
  const top = Math.max(...rows.map((r) => r.flat_saved * 1000)) * 1.08
  return (
    <ChartFrame id="xn-addis" source={source('AddisChecks')} provenance={runLine(d)}
      title="Every Addis early stop against two independent checks"
      subtitle={`All ${c.n} PM2.5 filters that stopped within 30 iterations, largest EC drop first. The site line is fitted on Addis filters whose baseline did not change: ${c.closer} of ${c.n} land closer, mean |residual| ${f2(c.res_saved)} → ${f2(c.res_newton)} µg/m³. The featureless window should read near zero: ${c.flatter} of ${c.n} are flatter.`}>
      <Forest rows={forest} labelWidth={110} panels={[
        { key: 'res_saved', label: 'EC − site line, 19 Oct', ref: 0, fmt: f2, domain: line },
        { key: 'res_newton', label: 'EC − site line, converged', ref: 0, fmt: f2, domain: line },
        { key: 'flat_saved', label: 'Window ×1000, 19 Oct', ref: 0, fmt: f2, domain: [0, top] },
        { key: 'flat_newton', label: 'Window ×1000, converged', ref: 0, fmt: f2, domain: [0, top] },
      ]} />
    </ChartFrame>
  )
}

function EcCross({ d }: { d: VibesNewtonData }) {
  const rows = useMemo(() => rowsOf<EcRow>(d.ec).filter((r) => r.ec_saved > 0 && r.ec_newton > 0), [d])
  const lg = useLegend()
  const { nOut } = useCounted(rows, (r) => r.filter_id)
  const kind = (r: EcRow) => r.premature_stop ? PREMATURE : OTHER
  const color = (k: string) => k === PREMATURE ? INK.accent : INK.identity
  const points = rows.filter((r) => lg.show(kind(r))).map((r) => ({
    x: r.ec_saved, y: r.ec_newton, color: color(kind(r)), id: r.filter_id, dim: lg.dim(kind(r)),
    tip: [`${r.filter_id} · ${r.code}`, `EC ${f2(r.ec_saved)} → ${f2(r.ec_newton)} µg`, `19 Oct run stopped at iteration ${r.saved_iterations}`],
    rec: { id: baseFilterId(r.filter_id), source: 'VIBES solved to convergence', site: r.code, date: '',
      fields: [['EC, 19 Oct baseline (µg)', f2(r.ec_saved)], ['EC, converged (µg)', f2(r.ec_newton)], ['Stopped at iteration', String(r.saved_iterations)]] as [string, string][] },
  }))
  return (
    <ChartFrame id="xn-ec" source={source('EcCross')} provenance={runLine(d)}
      title="FTIR EC per filter: 19 Oct run against the converged run"
      subtitle={`PM2.5 and PM10 analyses on the shared 87-blank background whose baseline moved or that stopped early${nOut ? `; ${nOut} left out by Filters → excluded` : ''}. The lot-241 analyses are left out because they changed bank in v2, not only solver. Most sit on the line.`}>
      <Scatter points={points} xLabel="FTIR EC, 19 Oct baseline (µg)" yLabel="FTIR EC, converged baseline (µg)" identity maxSide={420} />
      <Legend {...lg.props} items={[PREMATURE, OTHER].map((k) => ({ label: k, color: color(k) }))} />
    </ChartFrame>
  )
}

function Bands({ d }: { d: VibesNewtonData }) {
  const [group, setGroup] = useSearchParam<string>('xn-bands.g', 'addis', GROUPS)
  const rows = rowsOf<BandRow>(d.bands).filter((r) => r.group === group)
  const forest = rows.map((r) => ({
    key: r.band, label: r.band,
    values: { gt5: { v: 100 * r.gt5pct / r.n }, gt20: { v: 100 * r.gt20pct / r.n }, r: { v: r.r } },
    tip: [r.band, `${r.gt5pct} of ${r.n} spectra move > 5 %`, `${r.gt20pct} move > 20 %`, `r ${r.r.toFixed(4)}`],
  }))
  return (
    <ChartFrame id="xn-bands" source={source('Bands')} provenance={runLine(d)}
      title="Functional-group band areas, 19 Oct baselines against converged"
      subtitle="Share of spectra whose band area moves, and the correlation between the two runs. This is where the solver matters: a few filters shift a lot, and the rest don't move."
      controls={<Segmented label="Set" value={group} options={GROUPS} optionLabel={(o) => GROUP_LABEL[o]} onChange={setGroup} />}>
      <Forest rows={forest} labelWidth={190} panels={[
        { key: 'gt5', label: 'Spectra moved > 5 % (%)', ref: 0, fmt: (v) => v.toFixed(1) },
        { key: 'gt20', label: 'Spectra moved > 20 % (%)', ref: 0, fmt: (v) => v.toFixed(1) },
        { key: 'r', label: 'Correlation r', ref: 1, fmt: (v) => v.toFixed(3) },
      ]} />
    </ChartFrame>
  )
}

function Port({ d }: { d: VibesNewtonData }) {
  const rows = rowsOf<PortRow>(d.port).map((r) => ({
    key: r.run, label: r.run, bold: r.run === 'L-BFGS-B port',
    values: { s10: { v: r.s10 }, b5: { v: r.b5 }, shared: { v: r.short_shared } },
  }))
  return (
    <ChartFrame id="xn-port" source={source('Port')} provenance={runLine(d)}
      title="How close a compiled L-BFGS-B port gets to the saved run"
      subtitle="Whole network against the 19 Oct run. Fused SciPy is the noise floor, since last-bit arithmetic differences alone move fits this much. The port stops at the same early points as SciPy, so it reproduces the problem without fixing it.">
      <Forest rows={rows} labelWidth={150} panels={[
        { key: 's10', label: 'Sigma off by > 10 % (%)', ref: 0, fmt: (v) => v.toFixed(1) },
        { key: 'b5', label: 'Baselines moved > 5 %', ref: 0, fmt: (v) => v.toFixed(0) },
        { key: 'shared', label: 'Early stops shared (of 155)', ref: 155, fmt: (v) => v.toFixed(0) },
      ]} />
    </ChartFrame>
  )
}
