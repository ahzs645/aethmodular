import { useMemo } from 'react'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { FindingsList } from '@/components/Findings'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { f2, fmt } from '@/lib/format'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { Forest, sourceOf, useXp, type XpBase } from './shared'

/**
 * SPARTAN's own HIPS curve (24 Nov 2026): what SPARTAN's data already say about the loading effect, and the paired
 * light/dark experiment that would settle it. gallery/data/export_spartan_curve.py; notebooks
 * research/improve_hips_offset/delhi_run_length_test_20261124.ipynb and spartan_own_curve_design_20261124.ipynb.
 */

type Num = number | null
interface Fit { key: string; label: string; n: number; a: number; tau_inf: number; lo: number; hi: number; loglog: number }
interface Reg { ratio: string; kind: 'test' | 'placebo'; spec: string; spec_label: string; n: number; b_V: number; se_V: number; b_C: number; se_C: number; expected: number; tau_median: number }
interface PowerRow { tau_inf_true: number; ratio: number; pairs: number; lo: number; mid: number; hi: number; mac_lo: number; mac_mid: number; mac_hi: number }
interface CurveData extends XpBase {
  settings: { tau_inf_published: number; k_tape: number; lambda: number }
  addis: { volume: { p5: number; p50: number; p95: number; cv: number; share_of_tau_variance: number }; fits: Fit[]; points: { id: string; x: number; tau: number }[] }
  mac: { tau_inf: number[]; mac: Num[]; intercept: Num[] }
  delhi: { filters: { id: string[]; date: string[]; hours: Num[]; volume: Num[]; tau: Num[] }; regressions: Reg[]; flow_l_min: number }
  power: { error_per_filter: number; rows: PowerRow[] }
}

const source = sourceOf('SpartanCurveStudy')
const runLine = (d: CurveData) => `export_spartan_curve.py · ${d.generated.slice(0, 10)} · delhi_run_length_test_20261124.ipynb, spartan_own_curve_design_20261124.ipynb`
const macAt = (d: CurveData, t: number) => {
  const g = d.mac.tau_inf, m = d.mac.mac
  for (let i = 1; i < g.length; i++) if (t <= g[i]) { const a = m[i - 1], b = m[i]; if (a == null || b == null) return null; return a + (b - a) * (t - g[i - 1]) / (g[i] - g[i - 1]) }
  return null
}

export function SpartanCurveStudy() {
  const { data, state } = useXp<CurveData>('spartan_curve', 'SPARTAN’s own HIPS curve')
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <Tests />
    <DelhiRecord d={data} />
    <DelhiTest d={data} />
    <AddisAeth d={data} />
    <MacCurve d={data} />
    <Power d={data} />
  </>
}

function Findings({ d }: { d: CurveData }) {
  const month = d.delhi.regressions.filter((r) => r.spec === 'month')
  const pm = month.find((r) => r.ratio.includes('PM2.5') && r.kind === 'test'), ec = month.find((r) => r.ratio.includes('FTIR EC') && r.kind === 'test')
  const placebo = month.find((r) => r.ratio.startsWith('FTIR EC'))
  const tape = d.addis.fits.find((f) => f.key === 'tape')
  const p20 = d.power.rows.filter((r) => r.ratio === 4 && r.pairs === 20)
  return (
    <ExplainOnly>
      <ChartFrame id="sc-findings" title="SPARTAN’s own HIPS curve — short answers" exportable={false}>
        <FindingsList note={<>{runLine(d)}. The question from the 24 Nov meeting: is the HIPS compression a loading effect (how much is on the filter) or the air (composition on dirty days)?</>}>
          <li><strong>At Addis the two can’t be told apart.</strong> Sampled volume is {d.addis.volume.p5} to {d.addis.volume.p95} m³ (CV {(100 * d.addis.volume.cv).toFixed(1)} %), so {(100 * (1 - d.addis.volume.share_of_tau_variance)).toFixed(1)} % of how dark a filter is comes from the air.</li>
          <li><strong>Delhi changed its run length at the same flow</strong> ({d.delhi.flow_l_min} L/min everywhere): 24 h, then 12 h, then 6 h of sampling inside the same 9am to 9am days. In the same month, HIPS reads lower per unit of reference when more air goes through: b<sub>V</sub> {pm ? `${f2(pm.b_V)} ± ${f2(pm.se_V)}` : '?'} against PM2.5 mass, {ec ? `${f2(ec.b_V)} ± ${f2(ec.se_V)}` : '?'} against FTIR EC; the IMPROVE curve predicts {pm ? f2(pm.expected) : '?'}. A loading effect, at least as strong as IMPROVE’s.</li>
          <li><strong>One reference moves when it shouldn’t.</strong> FTIR EC ÷ PM2.5 rises with volume (placebo b<sub>V</sub> {placebo ? `+${f2(placebo.b_V)} ± ${f2(placebo.se_V)}` : '?'}): either FTIR reads relatively high on heavier filters or PM2.5 mass is inflated on light ones. That is why the two estimates differ.</li>
          <li><strong>Against the self-calibrated aethalometer, Addis bends hard</strong> (log-log slope {tape ? f2(tape.loglog) : '?'}; 1 is linear), a ceiling near {tape ? f2(tape.tau_inf) : '?'}. That is a lower bound: errors in x steepen the apparent bend.</li>
          <li><strong>The Addis MAC is the ceiling.</strong> MAC for slope 1 is {fmt(macAt(d, 2.26), 1)} at τ∞ 2.26 and {fmt(macAt(d, 3.26), 1)} at 3.26: the 10 to 20 range is entirely which ceiling.</li>
          <li><strong>The paired experiment would settle it cheaply.</strong> Twenty light/dark pairs at a 4× run-length ratio give the Addis MAC to about {p20.map((r) => `±${fmt((r.mac_hi - r.mac_lo) / 2, 1)}`).join(', ')} for true ceilings {p20.map((r) => r.tau_inf_true).join(', ')}. Delhi shows SPARTAN already spreads short runs through the day.</li>
        </FindingsList>
      </ChartFrame>
    </ExplainOnly>
  )
}

function Tests() {
  const rows: [string, string, string, string, string][] = [
    ['IMPROVE groups against TOR', 'no, different days', 'yes', 'no (IMPROVE)', 'τ∞ 2.26 [1.82, 2.70]; TOR error and composition ride along'],
    ['IMPROVE against CSN, four cities', 'yes', 'yes, about 10×', 'no', 'our shape (τ∞ 2.29) with IMPROVE 10 % above CSN, or α 10 with no offset'],
    ['MA350 tape advances (aethalometer page)', 'yes', 'yes', 'aethalometer tape', 'the aethalometer’s own k, 0.0115 at Addis IR'],
    ['HIPS against the aethalometer, Addis', 'same days', 'yes', 'partly', 'τ∞ about 1.9, a lower bound; darkness = dirty air'],
    ['Delhi run lengths', 'same months', 'yes, 2 to 4×', 'yes', 'a loading effect, b_V −0.28 to −0.44'],
    ['Paired SPARTAN filters (proposed)', 'yes', 'yes, set ratio', 'yes', 'SPARTAN’s own curve'],
  ]
  return (
    <ChartFrame id="sc-tests" source={source('Tests')} exportable={false}
      title="Every test of the bend, and what each can and can’t separate"
      subtitle="Same air, two loadings, SPARTAN’s own sampler and filter holder: only the paired experiment has all three. Delhi’s run lengths come closest in the existing data.">
      <div className="closeout-table-scroll" role="region" aria-label="Tests of the HIPS bend" tabIndex={0}>
        <table className="census-table" data-export="text">
          <thead><tr><th>Test</th><th>Same air?</th><th>Different darkness?</th><th>SPARTAN hardware?</th><th>What it gave</th></tr></thead>
          <tbody>{rows.map((r) => <tr key={r[0]} style={r[0].startsWith('Paired') ? { fontWeight: 600 } : undefined}>{r.map((c, i) => <td key={i}>{c}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </ChartFrame>
  )
}

function DelhiRecord({ d }: { d: CurveData }) {
  const f = d.delhi.filters
  const groups: [string, (h: number) => boolean, string][] = [
    ['24 h', (h) => Math.round(h) === 24, INK.identity], ['12 h', (h) => Math.round(h) === 12, RESEARCH_SERIES[0]],
    ['6 h', (h) => Math.round(h) === 6, INK.negative], ['other', (h) => ![6, 12, 24].includes(Math.round(h)), RESEARCH_SERIES[1]]]
  const series: SweepSeries[] = groups.map(([label, test, color]) => ({
    id: label, label: `${label} of sampling`, color, line: false, marks: true, r: 3,
    points: f.id.flatMap((id, i) => { const h = f.hours[i]; return h != null && test(h) ? [{ x: new Date(f.date[i]), y: f.tau[i], id, tip: [id, f.date[i], `${fmt(h, 1)} h, ${fmt(f.volume[i], 2)} m³`, `τ ${fmt(f.tau[i], 3)}`] }] : [] }),
  }))
  return (
    <ChartFrame id="sc-delhi-record" source={source('DelhiRecord')} provenance={runLine(d)}
      title="Delhi sampled fewer hours inside the same days, at the same flow"
      subtitle={`Each filter spans a 9am to 9am day; the sampler ran 24 h of it (2022 to 2023), then 12 h (2024 to Oct 2025), then 6 h (from Nov 2025), all at ${d.delhi.flow_l_min} L/min. Same days, a half or a quarter of the loading: the “vary run time, not flow” design, between filters.`}>
      <SweepPlot series={series} xScale="time" xLabel="filter start" yLabel="HIPS optical depth τ" height={240} />
    </ChartFrame>
  )
}

function DelhiTest({ d }: { d: CurveData }) {
  const specs = [...new Set(d.delhi.regressions.map((r) => r.spec))]
  const [spec, setSpec] = useSearchParam<string>('sc-delhi.spec', 'month', specs)
  const rows = d.delhi.regressions.filter((r) => r.spec === spec)
  const label = rows[0]?.spec_label ?? spec
  const forest = rows.map((r) => ({
    key: r.ratio, label: r.ratio, color: r.kind === 'placebo' ? INK.muted : INK.accent, hollow: r.kind === 'placebo',
    values: { b: { v: r.b_V, lo: r.b_V - 1.645 * r.se_V, hi: r.b_V + 1.645 * r.se_V }, c: { v: r.b_C, lo: r.b_C - 1.645 * r.se_C, hi: r.b_C + 1.645 * r.se_C } },
    tip: [r.ratio, `${r.n} filters, ${r.spec_label}`, `b_V ${f2(r.b_V)} ± ${f2(r.se_V)} (expected ${f2(r.expected)})`, `b_C ${f2(r.b_C)} ± ${f2(r.se_C)}`],
  }))
  const expected = rows.find((r) => r.kind === 'test')?.expected ?? -0.24
  return (
    <ChartFrame id="sc-delhi-test" source={source('DelhiTest')} provenance={runLine(d)}
      title="Delhi: HIPS reads lower per unit of reference when more air goes through the filter"
      subtitle={`log(ratio) = b_V·log(volume) + b_C·log(reference) + ${label}; 90 % intervals, standard errors clustered by month. A loading effect makes b_V negative (the IMPROVE curve expects ${f2(expected)} at Delhi’s median τ); a pure air effect leaves it at 0. Hollow rows are placebos, which should sit at 0: FTIR EC ÷ PM2.5 does not, so one of those two references also moves with loading.`}
      controls={<Segmented label="Comparison" value={spec} options={specs} optionLabel={(s) => d.delhi.regressions.find((r) => r.spec === s)?.spec_label ?? s} onChange={setSpec} />}>
      <Forest rows={forest} labelWidth={220} panels={[
        { key: 'b', label: 'b_V: per log volume', ref: 0, fmt: f2 },
        { key: 'c', label: 'b_C: per log concentration', ref: 0, fmt: f2 },
      ]} />
      <Note>Identification: with month effects the comparison rests on November 2025 (nine 6 h and two 12 h filters), the 9 h and 11 h runs, and small volume differences within months. It is a strong hint, not the paired experiment.</Note>
    </ChartFrame>
  )
}

function AddisAeth({ d }: { d: CurveData }) {
  const tape = d.addis.fits.find((f) => f.key === 'tape')!
  const pts = d.addis.points
  const xmax = Math.max(...pts.map((p) => p.x)) * 1.05
  const xs = Array.from({ length: 80 }, (_, i) => (i / 79) * xmax)
  const series: SweepSeries[] = [
    { id: 'pts', label: 'Addis filters', color: INK.negative, line: false, marks: true, r: 2.5, points: pts.map((p) => ({ x: p.x, y: p.tau, id: p.id, tip: [p.id, `aethalometer loading ${fmt(p.x, 1)}`, `τ ${fmt(p.tau, 3)}`] })) },
    { id: 'lin', label: 'linear (no bend)', color: INK.identity, dash: '4 3', marks: false, points: xs.map((x) => ({ x, y: tape.a * x })) },
    { id: 'fit', label: `fitted, τ∞ ${f2(tape.tau_inf)}`, color: RESEARCH_SERIES[4], width: 2, marks: false, points: xs.map((x) => ({ x, y: (tape.a * x) / (1 + (tape.a * x) / tape.tau_inf) })) },
  ]
  return (
    <ChartFrame id="sc-addis-aeth" source={source('AddisAeth')} provenance={runLine(d)}
      title="Addis HIPS against the self-calibrated aethalometer: the bend is in HIPS"
      subtitle={`Aethalometer IR BC with the tape-step k (${d.settings.k_tape}) over each filter’s window, times its volume, against HIPS optical depth. The scale a is free, so only the aethalometer’s linearity matters. ${d.addis.fits.map((f) => `${f.label}: τ∞ ${f2(f.tau_inf)} [${f2(f.lo)}, ${f2(f.hi)}], log-log ${f2(f.loglog)}`).join('; ')}. Errors in x steepen the apparent bend, as grouping IMPROVE by its own TOR did, so these are lower bounds; and with Addis’s constant volume a dark filter is a dirty day.`}>
      <SweepPlot series={series} xLabel="aethalometer IR BC × sampled volume (µg, arbitrary scale)" yLabel="HIPS optical depth τ" height={280} />
    </ChartFrame>
  )
}

function MacCurve({ d }: { d: CurveData }) {
  const series: SweepSeries[] = [
    { id: 'mac', label: 'MAC for slope 1', color: RESEARCH_SERIES[0], width: 2, marks: false, points: d.mac.tau_inf.map((t, i) => ({ x: t, y: d.mac.mac[i], tip: [`τ∞ ${fmt(t, 2)}`, `MAC ${fmt(d.mac.mac[i], 1)}`, `intercept ${fmt(d.mac.intercept[i], 2)} µg/m³`] })) },
  ]
  const markers = [
    { at: d.settings.tau_inf_published, label: 'IMPROVE groups 2.26' },
    { at: 3.26, label: 'IMPROVE Nov–Jun' },
    { at: d.addis.fits.find((f) => f.key === 'tape')?.tau_inf ?? 1.9, label: 'aethalometer (lower bound)' },
  ]
  return (
    <ChartFrame id="sc-mac" source={source('MacCurve')} provenance={runLine(d)}
      title="The Addis MAC is decided by the ceiling"
      subtitle={`FTIR EC on HIPS BC linearised at each ceiling (Deming λ ${d.settings.lambda} at MAC 10), and the MAC at which the slope is 1 (MAC ÷ slope). Between τ∞ 2.26 and 3.3 the MAC runs from 18 to 10; below 2 it climbs past 30 and the intercept turns positive.`}>
      <SweepPlot series={series} xLabel="ceiling τ∞" yLabel="Addis MAC for slope 1 (m²/g)" yDomain={[0, 40]} xMarkers={markers} height={240} />
    </ChartFrame>
  )
}

function Power({ d }: { d: CurveData }) {
  const ratios = [...new Set(d.power.rows.map((r) => String(r.ratio)))]
  const [ratio, setRatio] = useSearchParam<string>('sc-power.r', '4', ratios)
  const truths = [...new Set(d.power.rows.map((r) => r.tau_inf_true))]
  const colors = [INK.negative, RESEARCH_SERIES[4], RESEARCH_SERIES[0]]
  const series: SweepSeries[] = useMemo(() => truths.map((t, i) => ({
    id: String(t), label: `true τ∞ ${t}`, color: colors[i % colors.length], interval: 'whisker' as const, dodge: (i - 1) * 6,
    points: d.power.rows.filter((r) => r.tau_inf_true === t && String(r.ratio) === ratio).map((r) => ({ x: String(r.pairs), y: r.mac_mid, lo: r.mac_lo, hi: r.mac_hi,
      tip: [`${r.pairs} pairs, ratio ${r.ratio}`, `fitted τ∞ ${f2(r.mid)} [${f2(r.lo)}, ${f2(r.hi)}]`, `MAC ${fmt(r.mac_mid, 1)} [${fmt(r.mac_lo, 1)}, ${fmt(r.mac_hi, 1)}]`] })),
  })), [d, ratio]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <ChartFrame id="sc-power" source={source('Power')} provenance={runLine(d)}
      title="How many light/dark pairs the experiment needs"
      subtitle={`Simulated: dark filters drawn from the Addis optical depths (24 h), each paired with a light one seeing the same air for 1/${ratio} of the time; ${(100 * d.power.error_per_filter).toFixed(0)} % reading error per filter (IMPROVE’s collocated HIPS precision); the ceiling fitted from the pairs alone and turned into the Addis MAC. Bars are 90 % of 300 runs. Ten pairs already separate MAC 10 from 18.`}
      controls={<Segmented label="Run-length ratio" value={ratio} options={ratios} optionLabel={(r) => `${r}×`} onChange={setRatio} />}>
      <SweepPlot series={series} xScale="point" xLabel="light/dark pairs" yLabel="Addis MAC from the pairs (m²/g)" yDomain={[0, 50]} height={260} />
      <Note>Design rules from the tests above: vary run time, not flow (face velocity sets the deposit pattern); spread the light run through the same days, as Delhi’s 6 h runs already are; cover Addis’s τ 0.6 to 1.5 and both charcoal and non-charcoal seasons; log both flows and add field blanks.</Note>
    </ChartFrame>
  )
}
