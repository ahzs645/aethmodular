import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'
import { AddisK } from './ksweep/AddisK'
import { ImproveBlind } from './ksweep/ImproveBlind'
import { SmokeCheck } from './ksweep/SmokeCheck'
import { at, baselinesOf } from './ksweep/common'

const NB = 'research/ftir_hips_chem/ksweep_20261014 (built by workflows/run_ksweep_20261014.py and build_ksweep_notebook.py)'
const MEETING = '#tab=meeting&meeting=ann-20261009'

/**
 * ksweep_20261014 (Test 1 of the 14 Oct intercept plan): does the Addis intercept follow the number of PLS
 * components k, and can IMPROVE validation see it? Static: every number is a model fit (100 site-disjoint
 * splits × 5 baselines × 5 cohorts, k = 1–25), precomputed by gallery/data/ml_pages/ksweep.py from the
 * workflow's CSVs, or from the notebook's displayed tables where those are absent. At the reader's MAC:
 * Addis slopes and anything divided by a mean HIPS BC (× MAC/10); intercepts in µg/m³, R² and everything
 * against TOR do not move.
 */
export function Ksweep(_props: SectionPageProps) {
  const { data, error } = useMl('ksweep')
  const m = useMac()
  if (!data) return <MlWaiting error={error} what="the k sweep against IMPROVE" />

  const addis = rows(data, 'addis_k')
  const std = addis.filter((r) => r.cohort === 'standard'), an = addis.filter((r) => r.cohort === 'analog')
  const ks = [...new Set(std.map((r) => num(r.k)!))].sort((a, b) => a - b)
  const k2 = ks.includes(2) ? 2 : ks[0], k20 = ks.includes(20) ? 20 : ks[ks.length - 1]
  const baselines = baselinesOf(data)
  const drop = (key: string, g = 'All Addis') => `${f2(at(std, key, g, k2, 'intercept'))} → ${f2(at(std, key, g, k20, 'intercept'))}`
  const fullCurves = std.some((r) => num(r.intercept_q25) != null)
  // the analog cohorts at small k: VIBES already far off
  const anSmall = baselines.filter((b) => b.key !== 'AIRSpec').map((b) => at(an, b.key, 'All Addis', ks.includes(4) ? 4 : ks[1], 'intercept')).filter((v): v is number => v != null)

  const gap = rows(data, 'reference_gap').map((r): Row => {
    const i = num(r.intercept), x = num(r.x_mean)
    return { ...r, label: `${r.set} · ${rows(data, 'subsets').find((s) => s.set === r.set)?.label ?? ''}`, slope: m.slope(num(r.slope)),
      hips_mean: m.bc(x), rel_hips: i != null && x ? m.slope(i / x) : null }
  })
  const checks = rows(data, 'checks')

  return <>
    <PageIntro id="ksweep-intro" title="k sweep vs IMPROVE: is IMPROVE validation blind to k?" data={data} notebooks={NB}>
      <p>The workflow refits the PLS calibration on 100 site-disjoint 80/20 splits of each cohort and reads every k = 1–25 from one k = 25 fit per split (the first k NIPALS components are the k-component model). Five baselines: spline (AIRSpec), VIBES with 87 blanks over 4000–500 cm⁻¹ and cut to 4000–1425, and the two minus-LABE1 banks (287 lot-251 blanks; 365 labelled blanks). Cohorts: the standard all-IMPROVE lot-251 model, and the 500 seasonal analogs (1 Oct region) for All Addis, Dry, Belg and Kiremt. Addis readout: Deming against HIPS Fabs/MAC (σx from HIPS_Uncertainty, σy proxy 0.531); IMPROVE readout: OLS on held-out sites against TOR EC and against IMPROVE HIPS. Nothing is tuned on Addis.</p>
      <p>Static: these are model fits, so the page draws the notebook's numbers and refits nothing. {fullCurves ? 'Read from the workflow’s CSVs (every k, with the IQR over the splits).' : 'This checkout lacks the workflow’s CSVs, so the charts show only the seven k the notebook displayed (2, 4, 8, 11, 15, 20, 25), without IQRs.'} The MAC set under Filters → units (now {m.mac}) moves the Addis Deming slope and everything divided by a mean HIPS BC (× {m.mac}/10); intercepts in µg/m³, R² and everything against TOR stay as fitted.</p>
      <p>The 9 Oct meeting's <a href={MEETING}>“Addis intercept against the number of PLS factors”</a> (§2) is a different cut of a different run: each split there chose its own k, so each point averages the splits that happened to pick that k, and only spline, VIBES and VIBES cut were run. The last chart here holds k fixed on every split and adds the minus-LABE1 banks.</p>
    </PageIntro>

    <AddisK data={data} cohort="standard" id="ksweep-addis-standard"
      title={({ show, mac }) => show === 'slope'
        ? `The Addis slope moves with k too — Deming slope against Fabs/MAC per k, standard model, ${mac}`
        : `The Addis intercept falls with k for every baseline and every season — from k = ${k2} to ${k20}: spline ${drop('AIRSpec')}, VIBES 87 ${drop('VIBES-full-r70')} µg/m³`}
      subtitle={`Notebook [3]. The standard all-IMPROVE lot-251 model applied to Addis, Deming line against HIPS Fabs/MAC per Addis group; median${fullCurves ? ' and IQR' : ''} over 100 site-disjoint splits. Ringed: the k the frozen protocol picks for that baseline (1-SE rule on site-grouped CV). The minus-LABE1 banks fall like VIBES 87 (All Addis ${drop('VIBES-lot251-x1')} and ${drop('VIBES-allblanks-x1')}), so a clean blank bank does not remove the drift; the 4000–1425 cut sits between. Kiremt falls furthest (VIBES 87 ${drop('VIBES-full-r70', 'Kiremt (Jun-Sep)')}), Dry least (${drop('VIBES-full-r70', 'Dry (Oct-Feb)')}). The intercept does not depend on the MAC; the slope view is at ${m.label}.`}
      provenance={provenance(data, ['addis_k', 'protocol_k'])} />

    <ImproveBlind data={data} provenance={provenance(data, ['improve_k', 'addis_k', 'protocol_k'])} />

    <SmokeCheck data={data} provenance={provenance(data, ['improve_k', 'addis_rel', 'pooled', 'reference_gap', 'subsets'])} />

    <Figure kind="table" id="ksweep-reference-gap" data={gap}
      columns={[{ key: 'label', label: 'IMPROVE subset' }, { key: 'n', label: 'n', format: 'int' }, { key: 'slope', label: `TOR EC per HIPS BC (${m.label})`, format: 'coef' },
        { key: 'intercept', label: 'Intercept (µg/m³)', format: 'coef' }, { key: 'intercept_rel', label: '÷ mean TOR (notebook)', format: 'f2' },
        { key: 'rel_hips', label: `÷ mean HIPS BC (${m.label})`, format: 'f2' }, { key: 'hips_mean', label: `Mean HIPS BC (µg/m³, ${m.label})`, format: 'f3' }, { key: 'median_mac', label: 'Median Fabs / TOR EC (m²/g)', format: 'f2' }]}
      title={`The reference gap — in smoky IMPROVE TOR reads low against HIPS (median Fabs/TOR EC ${f2(gap.find((r) => r.set === 'smoke_cohort')?.median_mac)} in the smoke cohort, ${f2(gap.find((r) => r.set === 'all')?.median_mac)} network-wide)`}
      subtitle={`TOR EC regressed (OLS) on IMPROVE HIPS Fabs/MAC in each subset, no FTIR involved: the line a model that reproduced TOR exactly would inherit when judged against HIPS, the dashed lines of the chart above. The slope follows the MAC (× ${m.mac}/10 of the notebook's) and so does the intercept divided by the mean HIPS BC; the intercept, its ratio to mean TOR and the measured Fabs/TOR EC ratio do not.`}
      provenance={provenance(data, ['reference_gap', 'subsets'])} />

    <AddisK data={data} cohort="analog" id="ksweep-addis-analogs"
      title={({ show, mac }) => show === 'slope'
        ? `Seasonal analogs: the Addis slope per k — Deming slope against Fabs/MAC, ${mac}`
        : `For the seasonal analogs k is not the lever — the VIBES models are already ${f2(Math.max(...anSmall))} to ${f2(Math.min(...anSmall))} µg/m³ off at k = ${ks.includes(4) ? 4 : ks[1]} and the curves are not monotone`}
      subtitle={`Notebook [11]. Each season's 500 IMPROVE analogs (1 Oct region: 600–3600 cm⁻¹ with the CO₂ and 1050–1250 PTFE windows left out) as its own training cohort, judged on that season's Addis filters; median${fullCurves ? ' and IQR' : ''} over 100 site-disjoint splits, every k forced on every split. Ringed: the protocol k for that cohort. The analog failure is there from the first components, so for these cohorts the problem is what the analogs and the VIBES spectra carry, not how many components the model uses. Not the same figure as the 9 Oct meeting's a1009-k (linked in the intro), where each split chose its own k.`}
      provenance={provenance(data, ['addis_k', 'protocol_k'])} />

    {checks.length > 0 && <Figure kind="table" id="ksweep-checks" data={checks}
      columns={[{ key: 'table', label: 'Exported table · column (notebook cell)' }, { key: 'compared', label: 'Values compared', format: 'int' }, { key: 'displayed', label: 'Displayed', format: 'int' }, { key: 'decimals', label: 'Decimals', format: 'int' }, { key: 'max_abs_diff', label: 'Largest difference', format: 'f3' }]}
      title={`Agreement with the notebook — ${checks.reduce((s, r) => s + (num(r.compared) ?? 0), 0)} displayed numbers, largest difference ${f2(Math.max(...checks.map((r) => num(r.max_abs_diff) ?? 0)))}`}
      subtitle="The exporter reads the workflow's CSVs and compares each number the executed notebook displayed (rounded as it displayed them) with its own, at MAC 10. The pooled table is parsed from the notebook's two-level header by cell order."
      provenance={provenance(data, ['checks'])} />}
  </>
}
