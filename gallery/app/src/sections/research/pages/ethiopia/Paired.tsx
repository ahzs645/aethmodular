import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import type { Row } from '@/lib/table'
import { provenance, rows as tableRows, type SpectralExport } from '../spectralCommon'
import { ecByCalibration, kSummary, medianRatio, ratioOfMedians } from './variants'

/**
 * Three June 2026 figures read straight off the per-spectrum ETAD EC the page already loads (variants.ts holds
 * the maths, node-tested against the notebooks' saved numbers): 07 [11] each variant against CAL-0, 09 [12]
 * panel C the median against the component count, and 10 [12] panel B lot-251-only against both lots.
 * FTIR EC against FTIR EC, so none of them depends on the MAC.
 */
const CAL0 = 'CAL-0 all-nofilter'
const LOT251 = '906 lot 251 only, k30', BOTH = '935 both lots, k30 (first major min)'

export function useEtadEc(data: SpectralExport) {
  return useMemo(() => ecByCalibration(tableRows(data, 'per_filter'), tableRows(data, 'calibrations')), [data])
}

/** 07 [11]: each variant against CAL-0, spectrum by spectrum */
export function VariantsVsCal0({ data }: { data: SpectralExport }) {
  const ec = useEtadEc(data)
  const ids = useMemo(() => tableRows(data, 'per_filter').map((f) => String(f.ExternalFilterId)), [data])
  const variants = tableRows(data, 'calibrations').filter((c) => String(c.family).startsWith('07') && c.calibration !== CAL0).map((c) => String(c.calibration))
  const ref = ec.get(CAL0) ?? []
  const ratios = variants.map((v) => ({ v, r: medianRatio(ec.get(v) ?? [], ref) }))
  const points = useMemo(() => variants.flatMap((v) => (ec.get(v) ?? []).map((y, i): Row => ({ variant: v, CAL0: ref[i], EC: y, id: ids[i], ratio: y / ref[i] }))), [ec, ids]) // eslint-disable-line react-hooks/exhaustive-deps
  const r = (v: string) => f2(ratios.find((x) => x.v.startsWith(v))?.r)
  return <Figure kind="crossplot" id="ethiopia-variant-vs-cal0" data={points} x="CAL0" y="EC" idColumn="id" tipColumns={['ratio']} fit={false} maxSide={440}
    choose={[{ column: 'variant', label: 'Variant', default: 'CAL-1 cleaned' }]}
    xLabel="CAL-0 (keep everything) ETAD EC (µg/m³)" yLabel="Variant ETAD EC (µg/m³)"
    title={`Paired with CAL-0, the 3σ trim reads every spectrum lower (median ×${r('CAL-1')}) and the below-1:1 set higher (×${r('CAL-3')}); the inverse sets go negative`}
    subtitle={`June 07 [11]: each variant's EC for an ETAD spectrum against CAL-0's for the same spectrum (k = 20, the 906-filter IMPROVE set), with the 1:1. Median variant / CAL-0, per spectrum: ${ratios.map((x) => `${x.v.split(' ')[0]} ${f2(x.r)}`).join(', ')} (07's tracked etad_variant_ratio_vs_cal0.csv to 3 decimals). CAL-2 (the 46 removed filters) and CAL-4 (EC ≥ 70 µg, 22 filters) are diagnostics: too few, too extreme filters to calibrate on, and they predict negative EC. Spectra without a sampling volume have no µg/m³ and are left out.`}
    provenance={provenance(data, ['per_filter', 'calibrations'], 'research/ftir_ec_calibration_2026_06_25/07_ethiopia_variant_ec.ipynb fig07_etad_variant_vs_cal0.')} />
}

/** 09 [12] panel C (with [11]'s mean and share negative): the 906-filter set at six component counts */
export function ComponentSweep({ data }: { data: SpectralExport }) {
  const ec = useEtadEc(data)
  const labels = tableRows(data, 'calibrations').filter((c) => String(c.family).startsWith('09')).map((c) => String(c.calibration))
  const sweep = kSummary(ec, labels)
  const rows = sweep.map((s): Row => ({ k: s.k, 'median EC': s.median, 'mean EC': s.mean, '% negative': s.pctNegative, set: '906 filters, lot 251 (09)' }))
  const lo = sweep.reduce((a, s) => (s.median < a.median ? s : a), sweep[0]), hi = sweep.reduce((a, s) => (s.median > a.median ? s : a), sweep[0])
  return <Figure kind="sweep" id="ethiopia-component-sweep" data={rows} x="k" y="median EC" series="set" xLabel="PLS components (k)" yLabel="Median ETAD EC (µg/m³)" height={250}
    chosen={[{ series: '906 filters, lot 251 (09)', x: 21 }]} reference={0}
    variants={[{ key: 'stat', label: 'Show', choices: [{ value: 'median', label: 'Median' }, { value: 'mean', label: 'Mean', patch: { y: 'mean EC', yLabel: 'Mean ETAD EC (µg/m³)' } },
      { value: 'neg', label: '% negative', patch: { y: '% negative', yLabel: 'ETAD spectra predicted negative (%)' } }] }]}
    title={`The component count alone moves the median ETAD EC ${f2(hi && lo ? hi.median / lo.median : NaN)}×, from ${f2(lo?.median)} µg/m³ at k = ${lo?.k} to ${f2(hi?.median)} at k = ${hi?.k}`}
    subtitle="June 09 [12] panel C: the 906-filter IMPROVE set (no removal) at k = 3, 10, 18, 21, 25 and 40, applied to the ETAD spectra; ringed is k = 21, the first major minimum of its 5-fold RMSECV curve (that curve, 09's panel A, is the one on Machine learning → June 2026 variants). Too few components under-read; past the minimum the median falls back as the model starts fitting noise. The medians, means and shares negative equal 09's saved summary (tables/etad_ec_component_summary.csv); the share negative counts undated spectra as not negative, as 09 did."
    provenance={provenance(data, ['per_filter', 'component_summary'], 'research/ftir_ec_calibration_2026_06_25/09_ethiopia_component_ec.ipynb fig09_ethiopia_ec_by_component (panel C).')} />
}

/** 10 [12] panel B: per spectrum, lot-251-only against both lots, at k = 30 */
export function LotsPaired({ data }: { data: SpectralExport }) {
  const ec = useEtadEc(data)
  const ids = tableRows(data, 'per_filter').map((f) => String(f.ExternalFilterId))
  const a = ec.get(LOT251) ?? [], b = ec.get(BOTH) ?? []
  const effect = ratioOfMedians(b, a)
  const points = a.map((x, i): Row => ({ lot251: x, both: b[i], id: ids[i], ratio: b[i] / x }))
  return <Figure kind="crossplot" id="ethiopia-lots-paired" data={points} x="lot251" y="both" idColumn="id" tipColumns={['ratio']} fit={false} maxSide={400}
    xLabel="Lot 251 only, 906 filters, k = 30 (µg/m³)" yLabel="Both lots, 935 filters, k = 30 (µg/m³)"
    title={`Per spectrum, adding the 29 lot-248 filters barely moves the ETAD EC: the median goes ×${f2(effect)} (${effect >= 1 ? '+' : '−'}${Math.abs(100 * (effect - 1)).toFixed(1)} %)`}
    subtitle="June 10 [12] panel B: each ETAD spectrum's EC from the 2026-07-07 export's lot-251 filters alone (which reproduce the old 906 set exactly) against the same export with the 29 lot-248 filters added, both at k = 30 (the both-lots curve's first major minimum), with the 1:1. Ratio of the medians as 10 printed it (1.030). ETAD spans filter lots 248 and 251; the lot-248+251 mirror of notebooks 12–15 is the bigger step (see the progression above)."
    provenance={provenance(data, ['per_filter'], 'research/ftir_ec_calibration_2026_06_25/10_ethiopia_ec_both_lots.ipynb fig10_ethiopia_both_lots (panel B).')} />
}
