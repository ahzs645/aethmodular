import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import type { Row } from '@/lib/table'
import { median } from '../../common/stats'
import { provenance, rows as tableRows, type SpectralExport } from '../spectralCommon'
import { adamaPoints } from './data'

const TABLES = ['adama_variants', 'adama_tor', 'adama_han', 'adama', 'adama_calibration', 'adama_faithful', 'adama_optimized', 'adama_stabilized']
const BASIS = {
  key: 'basis', label: 'Basis', default: 'conc',
  choices: [
    { value: 'conc', label: 'µg/m³ (June 2026)', patch: { x: 'torConc', y: 'ftirConc', xLabel: 'Quartz TOR EC (reflectance, µg/m³)', yLabel: 'PTFE FTIR EC (µg/m³)' } },
    { value: 'mass', label: 'µg on each filter (spartan_ec)', patch: { x: 'torMass', y: 'ftirMass', xLabel: 'Quartz TOR EC (reflectance, µg on the filter)', yLabel: 'PTFE FTIR EC (µg on the filter)' } },
  ],
}

/**
 * The Adama TOR set: five PTFE filters with FTIR spectra, each paired by sample date with a co-located quartz
 * filter that has thermal-optical (TOR) EC — the only thermal EC in Ethiopia. June 2026 02/03/08/15 and
 * spartan_ec 03/05/07 (with the comparison scripts beside them) read every calibration they tried against it,
 * and classify the quartz filters by Han et al.'s char-EC / soot-EC. Five points: read them, do not fit them.
 */
export function AdamaFigures({ data }: { data: SpectralExport }) {
  const points = useMemo(() => adamaPoints({
    variants: tableRows(data, 'adama_variants'), tor: tableRows(data, 'adama_tor'), han: tableRows(data, 'adama_han'), june15: tableRows(data, 'adama'),
    calibration: tableRows(data, 'adama_calibration'), faithful: tableRows(data, 'adama_faithful'), optimized: tableRows(data, 'adama_optimized'), stabilized: tableRows(data, 'adama_stabilized'),
  }), [data])

  // FTIR / TOR per filter and calibration, plus each calibration's median, for the heat map
  const heat = useMemo(() => {
    const out: Row[] = points.map((p) => ({ calibration: p.calibration, filter: `${p.ptfe} · ${p.date.slice(5)}`, ratioConc: p.ratioConc, ratioMass: p.ratioMass }))
    for (const c of [...new Set(points.map((p) => p.calibration))]) {
      const ps = points.filter((p) => p.calibration === c)
      out.push({ calibration: c, filter: 'median', ratioConc: median(ps.map((p) => p.ratioConc)), ratioMass: median(ps.map((p) => p.ratioMass)) })
    }
    return out
  }, [points])
  const general = points.filter((p) => p.calibration.startsWith('Deployed general'))
  const torVolume = general.map((p) => p.torMass / p.torConc)

  // char-EC and soot-EC per quartz filter under both pyrolysis conventions (June 03), OC/EC under OPR
  const pairOf = new Map(points.map((p) => [p.quartz, p]))
  const charSoot = tableRows(data, 'adama_char_soot').flatMap((r): Row[] => {
    const p = pairOf.get(String(r.FilterId))
    const label = `${r.FilterId} (PTFE ${p?.ptfe ?? '?'}, ${p?.date ?? ''})`
    return [
      { filter: label, op: 'OP = OPTR (reflectance, TOR default)', char: r.char_EC_opr, soot: r.soot_EC_opr, ratio: r.char_soot_opr, ocec: r.OC_EC_opr },
      { filter: label, op: 'OP = OPTT (transmittance)', char: r.char_EC_opt, soot: r.soot_EC_opt, ratio: r.char_soot_opt, ocec: null },
    ]
  })
  const han = tableRows(data, 'adama_han')
  const classes = [...new Set(han.map((r) => String(r.source_class)))]

  return <>
    <Figure kind="crossplot" id="ethiopia-adama-tor" data={points as unknown as Row[]} x="torConc" y="ftirConc" idColumn="pair" tipColumns={['date', 'source']}
      xLabel="Quartz TOR EC (reflectance, µg/m³)" yLabel="PTFE FTIR EC (µg/m³)" fit={false} maxSide={420}
      choose={[{ column: 'calibration', label: 'Calibration', default: 'Deployed general (lot 241a)' }]} variants={[BASIS]}
      title="Adama — every calibration against the quartz TOR EC: the deployed general EC reads low, the refits and biomass calibrations mostly read high"
      subtitle={`Each point is a PTFE filter (FTIR) and the quartz filter sampled the same day (TOR), labelled PTFE / quartz. June 2026 02 and 08 plotted the deployed general EC and the CAL-0/1/3 refits in µg/m³, notebook 15 its lot-248+251 calibration; the spartan_ec scripts and notebooks 05 and 07 compared SPARTAN's lot-251 biomass tool (July07's "smoke" calibration), a local lot-251 biomass rebuild, a faithful replica of the tool on its own training data (each with and without σ-scaled outlier removal), and 05's baseline (K = 31) and cleaned fit and 07's cleaned K = 15–25 ensemble, in µg on each filter. Every calibration is on both footings here: µg ÷ the PTFE volume June 08 saved is µg/m³ (the exporter checks this on the general EC). The quartz and PTFE filters sampled different volumes (quartz ${f2(Math.min(...torVolume))}–${f2(Math.max(...torVolume))} m³; J1270's PTFE only 1.16 m³), so the two footings do not agree. CAL-2 and CAL-4 predict −60 to −190 µg/m³ here and are left out.`}
      provenance={provenance(data, TABLES)} />

    <Figure kind="heatmap" id="ethiopia-adama-ratio" data={heat} row="calibration" column="filter" value="ratioConc" domain={[0, 2]} ramp="diverging" format="f2" valueLabel="FTIR EC ÷ TOR EC"
      variants={[{ ...BASIS, choices: [{ value: 'conc', label: 'µg/m³ (June 2026)', patch: { value: 'ratioConc' } }, { value: 'mass', label: 'µg on each filter (spartan_ec)', patch: { value: 'ratioMass' } }] }]}
      title="FTIR ÷ TOR for each Adama filter — no calibration lands all five near 1"
      subtitle="1 is agreement; the colour saturates below 0 and above 2. Columns are the PTFE filters by sample date (month-day). June 02's median general/TOR is 0.69 (µg/m³); spartan_ec 07's stabilized biomass / TOR median is about 1.6 on µg per filter."
      provenance={provenance(data, TABLES)} />

    <Figure kind="intervals" id="ethiopia-adama-char-soot" data={charSoot} row="filter" series="op" aggregate={{ stat: 'median', interval: 'none' }} labelWidth={250}
      panels={[{ value: 'char', label: 'char-EC = EC1 − OP (µg C)', reference: 0 }, { value: 'soot', label: 'soot-EC = EC2 + EC3 (µg C)' }, { value: 'ratio', label: 'char-EC / soot-EC', reference: 1 }, { value: 'ocec', label: 'OC / EC (OPR)' }]}
      colors={{ 'OP = OPTR (reflectance, TOR default)': '#1f77b4', 'OP = OPTT (transmittance)': '#ff7f0e' }}
      title="Adama quartz filters by Han et al. — soot-leaning EC in organic-rich air; the transmittance OP makes char negative"
      subtitle={`June 03 (and spartan_ec 03, reflectance only): char-EC = EC1 − OP, soot-EC = EC2 + EC3, µg C on the filter. Han's discriminator reads ≲ 1 as diesel/soot and ≳ 2 as biomass; every filter sits below 1 under either convention (spartan_ec 03 labels all five "${classes.join('; ')}"), while OC/EC of 4.6–7.2 and the large pyrolysis correction say organic-rich, biomass-influenced aerosol. With OP from transmittance (OPTT) four of five char-EC values go negative, so the label leans on the OP convention.`}
      provenance={provenance(data, ['adama_char_soot', 'adama_han'])} />
  </>
}
