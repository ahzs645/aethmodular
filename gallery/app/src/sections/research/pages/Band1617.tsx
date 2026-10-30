import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { bandCenter } from '@/lib/bandCenter'
import { f2 } from '@/lib/format'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { PageIntro, provenance, rows, useCalibRuns, useSpectral, Waiting } from './spectralCommon'

const PARTNER: Record<string, string> = {
  carbonyl_peak: 'C=O carbonyl', CH_peak: 'aliphatic C–H', NH_OH_3100_3400: 'N–H / O–H 3100–3400 (amine companion)',
  sym1400_partner: 'sym. COO⁻ 1380–1420 (sub-1500, indicative)', nitro_1520_1560: 'nitro 1520–1560 (sub-1500, indicative)',
}
const BAND: Record<string, string> = { band1600_height: '1600 band', carbonyl_peak: 'C=O carbonyl', CH_peak: 'aliphatic C–H' }
const TARGET: Record<string, string> = { HIPS_tau: 'HIPS τ (per filter)', Fabs: 'HIPS Fabs', EC_deployed_ugm3: 'deployed FTIR EC' }
const WINDOW_NAME: Record<string, string> = { whole_spectrum: 'whole spectrum', OH_broad: 'O–H broad', aliphatic_CH: 'aliphatic C–H', carbonyl: 'C=O carbonyl', band_1617: '1560–1680 (1617)' }
const DUST_PREDICTOR: Record<string, string> = { log_soil: 'canonical IMPROVE soil (log)', dust_index: 'dust index', log_Kion: 'potassium K⁺ (log)' }
const DUST_TARGET: Record<string, string> = { addis_augmented: 'Addis (188 filters)', indh_augmented: 'Delhi (27 filters)' }
const DUST_CONFIG = (c: unknown) => String(c).replace(/_k(\d+)$/, ' · k $1').replace(/_/g, ' ')
const WINDOWS = [
  { value: 'band', label: '1450–1800 cm⁻¹', patch: { window: [1800, 1450] } },
  { value: 'ch', label: '2600–3600 cm⁻¹ (C–H, N–H/O–H)', patch: { window: [3600, 2600] } },
]

/**
 * ftir_12 and ftir_38: what the ~1600/1617 cm⁻¹ band in the Addis spectra is. The notebooks'
 * per-filter band features need the Drive spectra; the gallery has the site and cohort median
 * spectra (calibration_runs.json), on which the same band-centre rule is recomputed here.
 */
export function Band1617(_props: SectionPageProps) {
  const { data, error } = useSpectral('band_1617')
  const { runs } = useCalibRuns()

  // median ± IQR spectra the gallery already exports: SPARTAN sites, and the calibration cohorts against Addis
  const spectra = useMemo(() => {
    if (!runs) return []
    const out: Row[] = []
    const push = (population: string, series: string, s: { wn: number[]; median: number[]; q25: number[]; q75: number[] }) =>
      s.wn.forEach((wn, i) => out.push({ population, series, wn, median: s.median[i], q25: s.q25[i], q75: s.q75[i] }))
    for (const s of runs.site_spectra.airspec?.series ?? []) if (['addis', 'etbi', 'indh', 'chts', 'uspa'].includes(s.name)) push('SPARTAN sites', s.label, s)
    const coh = runs.cohort_spectra.airspec
    if (coh) {
      push('Calibration cohorts', coh.reference_label, { wn: coh.wn, ...coh.reference })
      for (const s of coh.series) push('Calibration cohorts', s.label, { wn: coh.wn, ...s })
    }
    return out
  }, [runs])

  const centers = useMemo(() => {
    if (!data) return []
    const out: Row[] = rows(data, 'centers').map((r) => ({
      block: 'ftir_12 · per filter, offset-corrected raw spectra (median and IQR)', row: `${r.group} (n = ${r.n_used})`,
      mid: num(r.center_median), lo: num(r.center_p25), hi: num(r.center_p75) }))
    for (const r of rows(data, 'site_bands')) out.push({
      block: 'ftir_38 · site median spectrum, neutral baseline', row: `${r.site} (n = ${r.n})${r.median_has_interior_peak ? '' : ' · edge, no interior peak'}`,
      mid: num(r.median_peak_cm1), lo: null, hi: null })
    for (const s of runs?.site_spectra.airspec?.series ?? []) {
      if (!['addis', 'etbi', 'indh', 'chts', 'uspa'].includes(s.name)) continue
      const c = bandCenter(s.wn, s.median)
      if (c) out.push({ block: 'Recomputed here · site median spectrum, spline baseline (calibration_runs.json)',
        row: `${s.label.split(':')[0]} (n = ${s.n})${c.edge ? ' · edge, no interior peak' : ''}`, mid: c.center, lo: null, hi: null })
    }
    return out
  }, [data, runs])

  if (!data) return <Waiting error={error} />
  const covariation = rows(data, 'covariation').map((r): Row => ({ ...r, partner: PARTNER[String(r.partner_band)] ?? r.partner_band }))
  const tracking = rows(data, 'tracking').map((r): Row => ({ ...r, band: BAND[String(r.band)] ?? r.band, target: TARGET[String(r.target)] ?? r.target }))
  const pmf = rows(data, 'pmf_tests').map((r): Row => ({ ...r, predictor: `${String(r.predictor).replace('_frac', '')} · q = ${num(r.fdr_q)?.toFixed(3)}` }))
  const dust = rows(data, 'dust_tests').map((r): Row => ({ ...r, predictor: DUST_PREDICTOR[String(r.predictor)] ?? r.predictor,
    target: DUST_TARGET[String(r.target)] ?? r.target, config: DUST_CONFIG(r.config) }))
  const dustAddis = dust.filter((r) => String(r.target).startsWith('Addis'))
  const kAddis = dustAddis.filter((r) => String(r.predictor).startsWith('potassium')).map((r) => num(r.partial_r)).filter((v): v is number => v != null)
  const soilQ = dustAddis.filter((r) => String(r.predictor).startsWith('canonical')).map((r) => num(r.fdr_q)).filter((v): v is number => v != null)
  const nh = covariation.find((r) => r.group === 'Addis' && r.partner_band === 'NH_OH_3100_3400')
  const trackR = (target: string) => rows(data, 'tracking').find((r) => r.target === target && r.band === 'band1600_height')
  const ec = trackR('EC_deployed_ugm3'), tau = trackR('HIPS_tau')

  return <>
    <PageIntro id="band-1617-intro" title="The 1600/1617 cm⁻¹ band — what it is, and what it is not" data={data}
      notebooks="research/ftir_ec_phase3/ftir_12_band_1600_identity, ftir_38_1617_and_dust_attribution; ftir_50 §4 (band-resolved similarity)">
      <p>Read with the meeting pages: Oct 28, 2026 §3 (dust) and Oct 9, 2026 §4 (the Si–O band) show the neighbouring evidence. The sub-1500 cm⁻¹ partner tests sit outside the baselined region and stay indicative only.</p>
    </PageIntro>

    {spectra.length > 0 && <Figure kind="spectrum" id="band-1617-spectra" data={spectra} wn="wn" y="median" lo="q25" hi="q75" series="series"
      title="Median spectra around the band — SPARTAN sites and the calibration cohorts, spline baseline"
      subtitle="Median and interquartile range of the spline-baseline (AIRSpec) spectra the Calibration tab already exports. ftir_12 §7 drew the same two windows on offset-corrected raw spectra divided by each group's median C–H height; these are not normalised, so compare band shapes and positions, not heights."
      choose={[{ column: 'population', label: 'Spectra' }]} variants={[{ key: 'window', label: 'Window', choices: WINDOWS }]}
      window={[1800, 1450]} annotations={[{ lo: 1550, hi: 1680, label: 'band window 1550–1680' }]} height={340}
      provenance={<>Stands in for ftir_12 §7 (median diagnostic-window spectra). Spectra: <code>calibration_runs.json</code> (site_spectra / cohort_spectra, spline baseline), <code>gallery/data/export_calibration_runs.py</code>.</>} />}

    <Figure kind="intervals" id="band-1617-centres" data={centers} row="row" block="block"
      panels={[{ value: 'mid', lo: 'lo', hi: 'hi', label: 'Peak centre inside 1550–1680 cm⁻¹ (cm⁻¹)', reference: 1617 }]} labelWidth={330}
      title="Where the band peaks — Addis and Bishoftu near 1617, IMPROVE and Delhi/Beijing higher"
      subtitle="Wavenumber of the largest absorbance above a straight continuum (anchors 1750–1800 and 1500–1530 cm⁻¹). ftir_12 per filter (filters whose band is above noise and not an edge hit); ftir_38 on each site's median under a neutral baseline; the last block applies ftir_12's rule to the gallery's median spline-baseline spectra (lib/bandCenter). A peak on the window edge is the carbonyl flank, not a resolved band. On the spline-baseline medians only Addis keeps an interior peak near 1617; Bishoftu's median, interior at 1620 under the neutral baseline, becomes an edge hit, so the site-level position depends on the baseline. Reference line: 1617."
      provenance={provenance(data, ['centers', 'site_bands'], 'Recomputed block: lib/bandCenter.ts on calibration_runs.json site medians.')} />

    <Figure kind="intervals" id="band-1617-companions" data={covariation} row="partner" series={covariation.some((r) => r.group !== 'Addis') ? 'group' : undefined}
      panels={[{ value: 'pearson_r', label: 'Pearson r with the 1600-band height', reference: 0 }, { value: 'spearman_r', label: 'Spearman ρ', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={300}
      title="Companion bands — the band rises with carbonyl and C–H, not with the amine N–H stretch"
      subtitle={`Covariation of the 1600-band height with candidate partner bands over the Addis spectra (n = ${nh?.n ?? '—'}). An amine would carry an N–H stretch at 3100–3400 cm⁻¹; its r is ${f2(nh?.pearson_r)}. The two sub-1500 rows lie in un-baselined PTFE territory.`}
      provenance={provenance(data, ['covariation'])} />

    <Figure kind="table" id="band-1617-ratios" data={rows(data, 'ratios')}
      columns={[{ key: 'group', label: 'Group' }, { key: 'band1600_to_CH_median', label: '1600 / C–H (median)', format: 'f2' },
        { key: 'band1600_to_carbonyl_median', label: '1600 / C=O (median)', format: 'f2' }, { key: 'NH_OH_to_band1600_median', label: 'N–H,O–H / 1600 (median)', format: 'f2' }]}
      title="Band ratios — the Addis band is large against C=O and small against N–H/O–H"
      provenance={provenance(data, ['ratios'])} />

    <Figure kind="heatmap" id="band-1617-tracking" data={tracking} row="band" column="target" value="pearson_r" domain={[-1, 1]} format="f2" valueLabel="Pearson r"
      title={`Does the band track the absorbing material? — at Addis it follows deployed FTIR EC (r ${f2(ec?.pearson_r)}) more than HIPS τ (r ${f2(tau?.pearson_r)})`}
      subtitle={`Pearson r across the ${tau?.n ?? '—'} HIPS-paired Addis filters (${ec?.n ?? '—'} with deployed EC). The band shares the oxygenated-organic covariates the FTIR EC calibration leans on.`}
      provenance={provenance(data, ['tracking'])} />

    <Figure kind="table" id="band-1617-sites" data={rows(data, 'site_bands')}
      columns={[{ key: 'site', label: 'Site' }, { key: 'n', label: 'n', format: 'int' }, { key: 'median_peak_cm1', label: 'Median-spectrum peak (cm⁻¹)', format: 'f2' },
        { key: 'median_band_height', label: 'Band height', format: 'f3' }, { key: 'median_band_to_CH', label: 'A(1617) / A(2920)', format: 'f2' },
        { key: 'median_has_interior_peak', label: 'Interior peak?', format: 'text' }, { key: 'per_filter_interior_peak_pct', label: 'Filters with an interior peak (%)', format: 'f2' }]}
      title="Neutral baseline, five SPARTAN sites — the feature is Ethiopian, not general"
      subtitle="ftir_38 on the calibration explorer's neutral (pspline_arpls) target spectra."
      provenance={provenance(data, ['site_bands'])} />

    <Figure kind="intervals" id="band-1617-pmf" data={pmf} row="predictor" panels={[{ value: 'partial_r', label: 'Partial r with 1617/C–H, given loading and month', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={260}
      title="PMF attribution — no charcoal or wood signal in the band's strength"
      subtitle="102 dated Addis filters joined to the normalised PMF factors; 5,000-permutation tests with within-family BH-FDR (q). The band is a regional Ethiopian marker; the PMF has no eucalyptus-resolved factor, so do not call it that."
      provenance={provenance(data, ['pmf_tests'])} />

    {dust.length > 0 && <Figure kind="intervals" id="band-1617-dust" data={dust} row="predictor" series="config" block="target"
      panels={[{ value: 'partial_r', label: 'Partial r with the residual, given HIPS BC and month', reference: 0 }, { value: 'spearman_r', label: 'Spearman ρ (unadjusted)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title={`Dust or potassium? — the locked calibrations' Addis residual follows K⁺ (partial r ${f2(Math.min(...kAddis))}–${f2(Math.max(...kAddis))}), soil only weakly`}
      subtitle={`ftir_38: each locked PLS configuration's residual (prediction − HIPS Fabs/10, µg/m³) against the canonical IMPROVE soil (2.20 Al + 2.49 Si + 1.63 Ca + 2.42 Fe + 1.94 Ti), a dust index and soluble potassium, adjusted for HIPS BC and month (sin/cos); 5,000-permutation tests with BH-FDR. Soil has the absorption-consistent negative sign at Addis but its significance depends on the configuration (q ${f2(Math.min(...soilQ))}–${f2(Math.max(...soilQ))}); potassium is strong under every one (q < 0.001). Delhi's 27 filters cannot settle either. The residual is at MAC 10, as the notebook ran it.`}
      provenance={provenance(data, ['dust_tests'])} />}

    <Figure kind="heatmap" id="band-1617-bandr" data={rows(data, 'band_r').map((r): Row => ({ ...r, band: WINDOW_NAME[String(r.band)] ?? r.band }))} row="target" column="band" value="r" ramp="sequential" domain={[0.6, 1]} format="f3" valueLabel="median r vs library median"
      title="Band-resolved similarity — Addis separates from the library in aliphatic C–H and at 1617, not overall"
      subtitle="ftir_50 §4: each target's median Pearson r against the IMPROVE library median, whole spectrum and per window, on spline-baseline spectra (1425–3998 cm⁻¹). A whole-spectrum r of 0.95 hides the two windows where Addis differs."
      provenance={provenance(data, ['band_r'])} />
  </>
}
