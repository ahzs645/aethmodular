import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { ExportState } from '@/components/ExportState'
import { IntervalStrip, type IntervalBlock, type IntervalRow } from '@/components/IntervalStrip'
import { Legend, useLegend } from '@/components/Legend'
import { SpectrumPlot } from '@/components/SpectrumPlot'
import { SweepPlot } from '@/components/SweepPlot'
import { IdentityLine } from '@/components/RefLines'
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { baseFilterId } from '@/lib/highlight'
import { f2 } from '@/lib/format'
import { rowsOf, Scatter, sourceOf, type Cols, type XpBase } from './shared'

/**
 * The 26 Oct lot-aware VIBES blank test (research/ftir_hips_chem/workflows/vibes_lot241_20261026.py,
 * gallery/data/export_vibes_lots.py): do the lot 241/241a filters, whose field blanks the 87-blank
 * background corrected 10-15x worse than every other lot's, correct cleanly with their own lot's blanks?
 */

interface LotData extends XpBase {
  series: Record<string, string>
  by_lot: Cols; per241: Cols; spike: Cols; pm25: Cols; backgrounds: Cols
  wn: number[]
  spectra: { filter_id: string; code: string; lot: string; raw: number[]; corrected: Record<string, number[]> }[]
  rank_curves: { bank: string; label: string; rank: number; components: number[]; mean_mse: number[]; se_mse: number[] }[]
}
const source = sourceOf('LotStudy')
const LOTS = ['241', '241a', '248', '250', '251', '253']
const SERIES = ['b87', 'ctrl251_r70', 'lot241_spartan', 'lot241_spartan_r70'] as const
type Series = (typeof SERIES)[number]
const COLOR: Record<Series, string> = { b87: '#8a8f98', ctrl251_r70: '#2171b5', lot241_spartan: '#fdae6b', lot241_spartan_r70: '#d94801' }
const LABEL: Record<Series, string> = {
  b87: 'Current background (87 blanks)', ctrl251_r70: '+9 lot-251 blanks (control)',
  lot241_spartan: '+9 lot-241/241a blanks, one-SE rank', lot241_spartan_r70: '+9 lot-241/241a blanks, rank 70',
}
// residue spans ~1e-5 to 1e-2, so the strips plot log10 and label ticks with the residue itself
const lg10 = (v: unknown) => (v == null || !(+v! > 0) ? NaN : Math.log10(+v!))
const tick = (v: number) => String(+(10 ** v).toPrecision(1))
const res = (v: unknown) => (v == null ? '—' : String(+(+v!).toPrecision(2)))
const PEAKS = ['1610', '1720', '2920', '3400'] as const

export function LotStudy() {
  const { data, error } = useExport<LotData>('experimental/vibes_lots.json', (d) => (d.schema_version === 1 ? null : 'unsupported data version'))
  if (!data) return <ExportState title="Lot-aware VIBES blanks" what="the lot-aware VIBES blank test" error={error} command="uv run python gallery/data/export_vibes_lots.py" />
  const byLot = rowsOf<{ series: string; lot: string; n: number; rms_all: number; rms_ptfe: number; rms_ch: number; q25_ptfe: number; q75_ptfe: number }>(data.by_lot)
  return <>
    <Findings d={data} byLot={byLot} />
    <ByLot d={data} byLot={byLot} />
    <Each d={data} />
    <Spectra d={data} />
    <Spikes d={data} />
    <Ranks d={data} />
    <Pm25 d={data} />
    <Backgrounds d={data} />
  </>
}

const runLine = (d: LotData) => `vibes_lot241_20261026.py · ${d.settings.n_jobs.toLocaleString()} fits (${d.settings.n_failed} failed) · bundle ${d.settings.bundle}`

function Findings({ d, byLot }: { d: LotData; byLot: { series: string; lot: string; rms_ptfe: number; n: number }[] }) {
  const v = (s: string, lot: string) => byLot.find((r) => r.series === s && r.lot === lot)?.rms_ptfe
  const others = LOTS.slice(2)
  return (
    <ExplainOnly>
      <ChartFrame id="xl-findings" title="Lot-aware VIBES blanks — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          <li>Every SPARTAN filter has a manufacturing lot on record (241, 241a, 248, 250, 251, 253). The gallery's VIBES background was learned from 87 blanks of lots {d.settings.applied.lots87.join(', ')}.</li>
          <li>Held-out field blanks of lots {others.join(', ')} already correct cleanly with it (median PTFE-band residue {others.map((l) => res(v('b87', l))).join(' / ')}), including lot 250, which it never saw. Lots 241/241a do not ({res(v('b87', '241'))} / {res(v('b87', '241a'))}).</li>
          <li>With the other lot-241/241a blanks in the bank, each held-out lot-241/241a blank drops to {res(v('lot241_spartan_r70', '241'))} / {res(v('lot241_spartan_r70', '241a'))}, as clean as the other lots. Nine lot-251 blanks instead (the control) do not help ({res(v('ctrl251_r70', '241'))} / {res(v('ctrl251_r70', '241a'))}): the gain is the lot, not the count.</li>
          <li>The rank must stay at 70. One lot-241 blank (AUMN-0007-7) has a shape no other blank shares; it drops the one-SE rank to ~32, which halves the gain and doubles the residue on the other lots.</li>
          <li>Small added peaks come back as well on lots 241/241a as on the rest once their blanks are in the bank. The {d.settings.applied.by_type['PM2.5']} lot-241/241a PM2.5 spectra change a lot (typically ~70 % of their corrected signal, mostly the PTFE band), so their VIBES traces now use this background; nothing else changed.</li>
        </ol>
        <Note>{runLine(d)}. Reproduces the 19 Oct network run with the current background (median difference {d.settings.reproduces_network_run ? res(d.settings.reproduces_network_run.median_rms_diff) : '—'}).</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

function legendItems(keys: readonly Series[]) {
  return keys.map((s) => ({ label: LABEL[s], color: COLOR[s] }))
}

function ByLot({ d, byLot }: { d: LotData; byLot: { series: string; lot: string; n: number; rms_all: number; rms_ptfe: number; rms_ch: number; q25_ptfe: number; q75_ptfe: number }[] }) {
  const lg = useLegend(['lot241_spartan'])
  const shown = SERIES.filter((s) => lg.show(s))
  const rows: IntervalRow[] = LOTS.map((lot) => {
    const at = (s: Series) => byLot.find((r) => r.series === s && r.lot === lot)
    const present = shown.filter((s) => at(s))
    return {
      key: lot, label: `Lot ${lot}`, strong: lot.startsWith('241'), note: `n ${at('b87')?.n ?? ''}`,
      tip: [`Lot ${lot}: held-out field blanks, median residue`, ...present.map((s) => { const r = at(s)!; return `${LABEL[s]}: PTFE ${res(r.rms_ptfe)} (IQR ${res(r.q25_ptfe)}–${res(r.q75_ptfe)}), full range ${res(r.rms_all)}, C–H ${res(r.rms_ch)}` })],
      marks: (['rms_ptfe', 'rms_all', 'rms_ch'] as const).map((m) => present.map((s, i) => {
        const r = at(s)!
        return { mid: lg10(r[m]), lo: m === 'rms_ptfe' ? lg10(r.q25_ptfe) : undefined, hi: m === 'rms_ptfe' ? lg10(r.q75_ptfe) : undefined, color: COLOR[s], dy: (i - (present.length - 1) / 2) * 5 }
      })),
    }
  })
  return (
    <ChartFrame id="xl-bylot" source={source('ByLot')} provenance={runLine(d)}
      title="Held-out field blanks by lot — what a correction leaves on a blank"
      subtitle="A corrected blank should be zero. Median RMS absorbance left after correction (log scale; whisker = interquartile range in the PTFE panel). Lots 241/241a: each blank held out of a bank holding the other eight. Other lots: every SPARTAN field blank not in the 87-blank bank.">
      <IntervalStrip blocks={[{ key: 'lots', rows }]} labelWidth={90} rowHeight={30} minPanelWidth={200} noteWidth={50}
        panels={[{ label: 'PTFE band 1050–1250 cm⁻¹', format: tick }, { label: 'full range 4000–500 cm⁻¹', format: tick }, { label: 'C–H 2800–3000 cm⁻¹', format: tick }]} />
      <Legend {...lg.props} items={legendItems(SERIES)} />
    </ChartFrame>
  )
}

function Each({ d }: { d: LotData }) {
  const lg = useLegend(['lot241_spartan'])
  const rows = rowsOf<{ series: string; filter_id: string; code: string; lot: string; rms_all: number; rms_ptfe: number }>(d.per241)
  const blocks: IntervalBlock[] = ['241', '241a'].map((lot) => ({
    key: lot, title: `Lot ${lot}`,
    rows: d3.groups(rows.filter((r) => r.lot === lot), (r) => r.filter_id).sort((a, b) => d3.ascending(a[0], b[0])).map(([fid, rs]) => {
      const at = (s: Series) => rs.find((r) => r.series === s)
      const present = SERIES.filter((s) => lg.show(s) && at(s))
      return {
        key: fid, label: fid, strong: fid === 'AUMN-0007-7',
        tip: [`${fid} (lot ${lot}), held out`, ...present.map((s) => `${LABEL[s]}: PTFE ${res(at(s)!.rms_ptfe)}, full range ${res(at(s)!.rms_all)}`), d.settings.lobo_ranks[fid] ? `one-SE rank without it: ${d.settings.lobo_ranks[fid]}` : ''].filter(Boolean),
        marks: (['rms_ptfe', 'rms_all'] as const).map((m) => present.map((s, i) => ({ mid: lg10(at(s)![m]), color: COLOR[s], dy: (i - (present.length - 1) / 2) * 4 }))),
      }
    }),
  }))
  return (
    <ChartFrame id="xl-each" source={source('Each')} provenance={runLine(d)}
      title="The nine lot-241/241a blanks, one at a time"
      subtitle="Each blank corrected by a bank that does not contain it. AUMN-0007-7 is the one blank its lot-mates cannot explain; it is also the blank that pulls the one-SE rank down.">
      <IntervalStrip blocks={blocks} labelWidth={110} rowHeight={24} minPanelWidth={220}
        panels={[{ label: 'PTFE-band residue (RMS)', format: tick }, { label: 'full-range residue (RMS)', format: tick }]} />
      <Legend {...lg.props} items={legendItems(SERIES)} />
    </ChartFrame>
  )
}

function Spectra({ d }: { d: LotData }) {
  const ids = d.spectra.map((s) => s.filter_id).sort()
  const [fid, setFid] = useSearchParam<string>('xl-spec.blank', 'CLST-0039-7', ids)
  const lg = useLegend()
  const s = d.spectra.find((x) => x.filter_id === fid) ?? d.spectra[0]
  const keys = (['b87', 'ctrl251_r70', 'lot241_spartan_r70'] as const).filter((k) => s.corrected[k])
  const series = keys.filter((k) => lg.show(k)).map((k) => ({ id: k, label: LABEL[k], color: COLOR[k], wn: d.wn, y: s.corrected[k] }))
  return (
    <ChartFrame id="xl-spectra" source={source('Spectra')} provenance={runLine(d)}
      title="What is left on a held-out blank — the corrected spectrum itself"
      subtitle={`${s.filter_id} (${s.code}, lot ${s.lot}) after correction by each bank. Flat at zero is a clean correction; the PTFE doublet near 1150–1210 cm⁻¹ is the filter material the current background does not cover for this lot.`}
      controls={<Segmented label="Blank" value={s.filter_id} options={ids} onChange={setFid} optionLabel={(o) => o} />}>
      <SpectrumPlot series={series} height={300} yLabel="corrected absorbance" zeroLine />
      <Legend {...lg.props} items={keys.map((k) => ({ label: LABEL[k], color: COLOR[k], shape: 'line' as const }))} />
    </ChartFrame>
  )
}

function Spikes({ d }: { d: LotData }) {
  const amps = d.settings.amplitudes.filter((a: number) => a > 0).map(String) as string[]
  const [amp, setAmp] = useSearchParam<string>('xl-spike.amp', '0.01', amps)
  const lg = useLegend(['lot241_spartan'])
  const rows = rowsOf<Record<string, any>>(d.spike).filter((r) => String(r.amplitude) === amp)
  const strip: IntervalRow[] = LOTS.map((lot) => {
    const at = (s: Series) => rows.find((r) => r.series === s && r.lot === lot)
    const present = SERIES.filter((s) => lg.show(s) && at(s))
    return {
      key: lot, label: `Lot ${lot}`, strong: lot.startsWith('241'),
      tip: [`Lot ${lot}, added peaks at amplitude ${amp}`, ...present.map((s) => { const r = at(s)!; return `${LABEL[s]}: error ${f2(r.rel_rmse)} of the added shape; recovered ${PEAKS.map((p) => `${p} ${f2(r[`recovered_${p}`])}`).join(', ')}` })],
      marks: [['rel_rmse'], ...PEAKS.map((p) => [`recovered_${p}`])].map(([m]) => present.map((s, i) => ({ mid: +at(s)![m], color: COLOR[s], dy: (i - (present.length - 1) / 2) * 5 }))),
    }
  })
  return (
    <ChartFrame id="xl-spikes" source={source('Spikes')} provenance={runLine(d)}
      title="Added peaks — does the correction keep real signal?"
      subtitle="Four Gaussian peaks (1610, 1720, 2920 and a broad 3400 cm⁻¹) added to every held-out blank at three heights. Error: RMS of what came back minus what was added, relative to the added shape (0 is perfect). Recovered: the share of each peak's area that survives (1 is perfect). Medians over blanks.">
      <div className="frame-controls"><Segmented label="Peak height (absorbance)" value={amp} options={amps} onChange={setAmp} optionLabel={(o) => o} /></div>
      <IntervalStrip blocks={[{ key: 'lots', rows: strip }]} labelWidth={90} rowHeight={30} minPanelWidth={150}
        panels={[{ label: 'error (relative)', reference: 0, format: (v) => f2(v) }, ...PEAKS.map((p) => ({ label: `recovered ${p} cm⁻¹`, reference: 1, format: (v: number) => f2(v) }))]} />
      <Legend {...lg.props} items={legendItems(SERIES)} />
      <Note>The broad 3400 cm⁻¹ peak loses ~15 % at the smallest height with every bank and lot: a VIBES property, not a lot effect.</Note>
    </ChartFrame>
  )
}

function Ranks({ d }: { d: LotData }) {
  const colors = ['#8a8f98', '#fdae6b', '#d94801']
  const series = d.rank_curves.map((c, i) => ({ id: c.bank, label: `${c.label} (one-SE rank ${c.rank})`, color: colors[i], marks: false,
    points: c.components.map((k, j) => ({ x: k, y: c.mean_mse[j], lo: c.mean_mse[j] - c.se_mse[j], hi: c.mean_mse[j] + c.se_mse[j] })) }))
  return (
    <ChartFrame id="xl-ranks" source={source('Ranks')} provenance={runLine(d)}
      title="Why rank 70 — one blank moves the one-SE rule"
      subtitle="Leave-one-blank-out reconstruction error of the blank PCA by number of components (band ± 1 SE). The one-SE rule takes the smallest rank within one SE of the minimum. AUMN-0007-7's error is ~1,700× the median blank's, so its SE swamps the curve and the rule stops at 33; without it the rule picks 74, close to the current 70.">
      <SweepPlot series={series} xLabel="components" yLabel="mean LOO error" yScale="log" height={300}
        xMarkers={d.rank_curves.map((c, i) => ({ at: c.rank, label: `${c.rank}`, color: colors[i] }))} yFormat={(v) => v.toExponential(0)} />
      <Legend items={series.map((s) => ({ label: s.label, color: s.color, shape: 'line' as const }))} />
    </ChartFrame>
  )
}

function Pm25({ d }: { d: LotData }) {
  const rows = rowsOf<{ filter_id: string; code: string; lot: string; date: string; analysis_id: string; change_all: number; change_ptfe: number; signal_all: number; change_rel_all: number }>(d.pm25)
  const lotColor: Record<string, string> = { '241': '#d94801', '241a': '#6a51a3' }
  const points = rows.map((r) => ({
    x: r.signal_all, y: r.change_all, color: lotColor[r.lot] ?? '#8a8f98', id: r.filter_id, key: r.analysis_id,
    tip: [`${r.filter_id} (lot ${r.lot}, ${r.date})`, `corrected-signal RMS (current) ${res(r.signal_all)}`, `change ${res(r.change_all)} (PTFE ${res(r.change_ptfe)}), ${Math.round(100 * r.change_rel_all)} % of the signal`],
    rec: { id: baseFilterId(r.filter_id), source: 'Lot-aware VIBES · lot 241/241a PM2.5', site: r.code, date: r.date, fields: [['lot', r.lot], ['analysis', r.analysis_id], ['VIBES change (RMS)', res(r.change_all)], ['change in PTFE band', res(r.change_ptfe)]] as [string, string][] },
  }))
  const med = d3.median(rows, (r) => r.change_rel_all) ?? NaN
  // one square log domain for both axes (the shared Scatter's identity mode clamps log axes at 1e-3)
  const all = rows.flatMap((r) => [r.signal_all, r.change_all]).filter((v) => v > 0)
  const dom: [number, number] = [d3.min(all)! / 1.5, d3.max(all)! * 1.5]
  return (
    <ChartFrame id="xl-pm25" source={source('Pm25')} provenance={runLine(d)}
      title="Lot 241/241a PM2.5 filters — how far the new background moves each spectrum"
      subtitle={`RMS difference between the corrections (new minus current) against the size of the current corrected spectrum, ${rows.length} PM2.5 analyses (log axes). Median change ${Math.round(100 * med)} % of the signal, mostly the PTFE band. The diagonal marks a change as large as the signal. Click a filter to open its record; its drawer now shows the new VIBES trace.`}>
      <Scatter points={points} xLabel="current corrected spectrum, RMS" yLabel="change in the correction, RMS" xLog yLog maxSide={380} xDomain={dom} yDomain={dom}
        under={({ x, y }) => <IdentityLine x1={x(dom[0])} y1={y(dom[0])} x2={x(dom[1])} y2={y(dom[1])} />} />
      <Legend items={Object.entries(lotColor).map(([l, c]) => ({ label: `lot ${l}`, color: c }))} />
    </ChartFrame>
  )
}

function Backgrounds({ d }: { d: LotData }) {
  const rows = rowsOf<{ bank: string; n_blanks: number; rank: number }>(d.backgrounds)
  const what: Record<string, string> = {
    b87: 'the gallery background: 59 IMPROVE (lots 248/251) + 28 Addis blanks', lot241_spartan: '+ the 9 lot-241/241a blanks, rank by the one-SE rule',
    lot241_spartan_r70: '+ the 9 lot-241/241a blanks, rank fixed at 70 (now used for lot 241/241a filters)', ctrl251: '+ 9 lot-251 SPARTAN blanks from other sites, one-SE rank',
    ctrl251_r70: '+ 9 lot-251 SPARTAN blanks, rank fixed at 70',
  }
  return (
    <ChartFrame id="xl-backgrounds" source={source('Backgrounds')} title="The backgrounds compared" exportable={false}>
      <table className="census-table">
        <thead><tr><th>Bank</th><th>Blanks</th><th>Rank</th><th>What it is</th></tr></thead>
        <tbody>{rows.map((r) => <tr key={r.bank}><td>{r.bank}</td><td>{r.n_blanks}</td><td>{r.rank}</td><td>{what[r.bank] ?? ''}</td></tr>)}</tbody>
      </table>
      <Note>Plus nine leave-one-blank-out banks per rank rule (95 blanks each; one-SE ranks {Object.entries(d.settings.lobo_ranks).map(([k, v]) => `${k} ${v}`).join(', ')}). The gallery's network VIBES traces for the {d.settings.applied.n} lot-241/241a analyses ({Object.entries(d.settings.applied.by_type).map(([k, v]) => `${v} ${k}`).join(', ')}) now come from the rank-70 bank; a lot-241/241a field blank uses the bank that leaves it out.</Note>
    </ChartFrame>
  )
}
