import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { FixedSetNote } from '@/lib/markedRows'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { PageIntro, provenance, rows, useSpectral, Waiting } from './spectralCommon'
import { AdamaFigures } from './ethiopia/Adama'
import { ComponentSweep, LotsPaired, VariantsVsCal0 } from './ethiopia/Paired'
import { CleaningResiduals } from './ethiopia/Residuals'
import { SmokeFigures } from './ethiopia/Smoke'
import { AdamaSpectra, BandProfile, DerivativeSpectra, EtadSpectra, SpectraMissing } from './ethiopia/Spectra'

/**
 * The June 2026 Ethiopia calibration notebooks (research/ftir_ec_calibration_2026_06_25 07, 09, 10, 12,
 * 15) and the cleaning explainer they rest on (research/spartan_ec_2026_06_16/08), plus the June notebooks that
 * draw spectra rather than models (01, 04, 06 [21], 13 [9]; ethiopia/Spectra, from the Drive's spectra) and the
 * paired views of the per-spectrum EC (07 [11], 09 [12] C, 10 [12] B; ethiopia/Paired). Their per-spectrum
 * ETAD EC tables are tracked, so every calibration's Addis EC is drawn per filter and against HIPS.
 * The notebooks used HIPS Fabs/10; the HIPS axis and the saved slopes follow the MAC set under
 * Filters → units (lib/mac: x rescales, slopes scale by MAC/10, intercepts and R² stay).
 * Also: SPARTAN's lot-251 biomass tool calibration at Addis (July07, spartan_ec 02; ethiopia/Smoke, joined to
 * the gallery's Addis rows), the Adama TOR set (June 02/03/08/15, spartan_ec 03/05/07; ethiopia/Adama) and
 * 08's residual diagnostics (ethiopia/Residuals).
 */
export function EthiopiaCalibrations({ rows: galleryRows }: SectionPageProps) {
  const { data, error } = useSpectral('ethiopia_calibrations')
  const m = useMac()

  // the export keeps one row per spectrum and one column per calibration; the figures want one row per point
  const points = useMemo(() => {
    if (!data) return []
    const index = rows(data, 'calibrations')
    return rows(data, 'per_filter').flatMap((f) => index.map((c): Row => ({
      family: c.family, calibration: c.calibration, which: `${c.family} — ${c.calibration}`,
      EC: num(f[String(c.column)]), HIPS: m.bc(num(f['HIPS Fabs/10']), 10), id: f.ExternalFilterId,
    })))
  }, [data, m])

  if (!data) return <Waiting error={error} />
  const known = points.filter((p) => p.EC != null)
  // the notebooks' OLS slopes against Fabs/10, re-expressed at the reader's MAC
  const atMac = (rs: Row[]) => rs.map((r): Row => ({ ...r, slope: m.slope(num(r.slope), 10) }))
  const spectra = rows(data, 'spectra_means').length > 0
  // 06 [12]: CAL-3 trains on the filters the baseline under-predicts; the export's full fit is k = 20 (06 used k = 21)
  const calAll = rows(data, 'calibration_all').map((r): Row => ({ ...r, side: (num(r.predicted) ?? 0) < (num(r.measured) ?? 0) ? 'below 1:1: CAL-3 trains on these' : 'on or above 1:1' }))
  const below = calAll.filter((r) => String(r.side).startsWith('below')).length
  const negative = (fam: string) => known.filter((p) => String(p.family).startsWith(fam) && (p.EC as number) < 0).length

  return <>
    <PageIntro id="ethiopia-intro" title="Ethiopia (ETAD) EC under the June 2026 calibrations — which samples, how many components, which lots" data={data}
      notebooks="research/ftir_ec_calibration_2026_06_25/01_adama_spectra, 02_adama_tor_vs_ftir, 03_adama_char_soot, 04_ethiopia_etad_spectra, 06_calibration_variants_components [12], [21], 07_ethiopia_variant_ec, 08_adama_hips_crossplots, 09_ethiopia_component_ec, 10_ethiopia_ec_both_lots, 12_dedicated_ethiopia_calibration, 13_chase_origin_and_biomass [9], 15_downstream_recommended_calibration; research/spartan_ec_2026_06_16/02_biomass_calibration_comparison, 03_adama_han_char_soot, 05_calibration_optimization, 07_calibration_optimization_story, 08_outlier_removal_explainer; research/July07/regular_before_cal0_vs_smoke_hips, regular_vs_smoke_ec_vs_hips">
      <p>All of these apply PLS calibrations trained on IMPROVE to the 319 ETAD spectra (19 filters scanned twice, so one point per spectrum, as the notebooks do). None has an Ethiopian thermal EC reference: they are judged by agreement with HIPS Fabs/MAC (the notebooks: MAC 10; here the MAC set under Filters → units, now {m.mac}) and by physical sense. Notebook 07 left the HIPS comparison as a to-do; the crossplots here add it from the HIPS Fabs carried in notebook 15's table.</p>
      <FixedSetNote subject="this page" what="the notebooks' 319 ETAD spectra and saved tables (so does every chart on this page but the July07 smoke comparison, which follows the gallery's Addis rows; the per-spectrum crossplots still leave out, or under Mark cross, a filter excluded by hand)" />
      <p>A naming hazard runs through the July07 and spartan_ec notebooks: their "regular CAL-0" is the 906-filter IMPROVE set (the set the gallery's "Biomass-smoke (906)" calibration preset trains on), while their "smoke" or "biomass tool" calibration is a different model, the coefficients SPARTAN's calibration tool exported for filter lot 251. The page names them that way. The only thermal EC in Ethiopia is the five Adama quartz filters, read against every calibration below.</p>
    </PageIntro>

    <Figure kind="intervals" id="ethiopia-dist" data={points} row="calibration" block="family"
      panels={[{ value: 'EC', label: 'Predicted ETAD EC (µg/m³; median and IQR over spectra)', reference: 0 }]} labelWidth={300}
      title="How far the calibration choice moves the Ethiopia EC — about 2× by sample choice, up to 5× by component count"
      subtitle={`07: the 906-filter IMPROVE set at k = 20, kept whole (CAL-0), 3σ-cleaned (CAL-1), or cut to sub-populations; CAL-2 and CAL-4 are tiny-sample fits that predict negative EC (${negative('07')} negative spectra in 07) and are diagnostics, not candidates. 09: the whole set at six component counts. 10: the 935-filter export spanning both ETAD filter lots. 15: the old calibration against the validated lot-248+251 mirror.`}
      provenance={provenance(data, ['per_filter', 'calibrations'])} />

    <VariantsVsCal0 data={data} />
    <ComponentSweep data={data} />

    <Figure kind="crossplot" id="ethiopia-hips" data={points} x="HIPS" y="EC" idColumn="id" choose={[{ column: 'which', label: 'Calibration', default: '15 · recommended — NEW lot-248+251, flag-filtered, k10' }]}
      xLabel={`HIPS Fabs / ${m.mac} (µg/m³)`} yLabel="Predicted FTIR EC (µg/m³)" maxSide={460} errorRatio={m.lambda}
      title="Each calibration against HIPS — the old set is steep and offset, the lot-248+251 mirror moves toward 1:1"
      subtitle={`ETAD spectra with a HIPS Fabs, against Fabs ÷ ${m.mac} (the MAC set under Filters → units; the notebooks used 10). The notebooks fitted OLS (08's table below; 15: 2.27x − 5.25 old, about 1.2x − 2.2 new, at MAC 10); the line here is the gallery's estimator on the same points, Deming with the explorer's λ at this MAC (${f2(m.lambda)}).`}
      provenance={provenance(data, ['per_filter'])} />

    <Figure kind="table" id="ethiopia-variant-fits" data={atMac(rows(data, 'variant_hips_fits'))}
      columns={[{ key: 'variant', label: 'Variant (07)' }, { key: 'n', label: 'n', format: 'int' }, { key: 'slope', label: `OLS slope (MAC ${m.mac})`, format: 'f2' }, { key: 'intercept', label: 'OLS intercept', format: 'f2' }, { key: 'r2', label: 'R²', format: 'f3' }]}
      title="The variants' HIPS fits as notebook 08 saved them"
      subtitle={m.changed ? `Notebook 08 fitted against Fabs/10; the slopes are re-expressed at MAC ${m.mac} (× ${m.mac}/10). Intercepts and R² do not depend on the MAC.` : undefined}
      provenance={provenance(data, ['variant_hips_fits'])} />

    <Figure kind="sweep" id="ethiopia-trainset" data={rows(data, 'trainset_medians')} x="ncomp" y="median EC" series="training set" xLabel="PLS components (k)" yLabel="Median ETAD EC (µg/m³)" height={260}
      chosen={[{ series: 'fresh_both_lots', x: 30 }]}
      title="Spanning both filter lots changes the median by a few per cent"
      subtitle="The old 906 set is all lot 251; ETAD uses lots 248 and 251. The 2026-07-07 export adds 29 lot-248 filters (935): at the first-major-minimum k = 30 the median moves 4.81 → 4.96 µg/m³; the re-pulled lot-251 filters reproduce the old set exactly."
      provenance={provenance(data, ['trainset_medians'])} />

    <LotsPaired data={data} />

    <Figure kind="intervals" id="ethiopia-progression" data={atMac(rows(data, 'dedicated_progression'))} row="calibration" block="step"
      panels={[{ value: 'slope', label: `Slope vs HIPS Fabs/${m.mac} (OLS)`, reference: 1 }, { value: 'intercept', label: 'Intercept (µg/m³)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={330}
      title="Notebook 12 — the dedicated data is the win, not the fancy selection"
      subtitle="From the 906-filter set to the full 13,012-filter lot-248+251 mirror the intercept goes from −5.25 to about −2.3 and the slope from 2.27 to about 1.2 (at MAC 10; the slopes drawn follow the MAC set under Filters → units); a PCA Addis neighbourhood adds little; Weakley's second derivative with band selection pulls the intercept to about −1 but flattens the slope, so the remaining offset is method or MAC, not spectral."
      provenance={provenance(data, ['dedicated_progression'])} />

    {spectra ? <><EtadSpectra data={data} /><BandProfile data={data} /></> : <SpectraMissing data={data} />}

    <SmokeFigures data={data} rows={galleryRows} />

    <Figure kind="table" id="ethiopia-adama" data={rows(data, 'adama')}
      columns={[{ key: 'barcode', label: 'Adama filter' }, { key: 'date', label: 'Date' }, { key: 'TOR EC', label: 'TOR EC (µg/m³)', format: 'f2' }, { key: 'OLD rds_EC 906', label: 'Old calibration', format: 'f2' }, { key: 'NEW lot-248+251', label: 'Lot-248+251 calibration', format: 'f2' }]}
      title="Notebook 15 at Adama — five quartz TOR filters, read against the 1:1"
      subtitle="Adama PTFE spectra interpolated to the training grid; the barcode-to-filter map is notebook 01's assumed sorted-to-sorted pairing. Five points: read them, do not fit them."
      provenance={provenance(data, ['adama'])} />

    <AdamaFigures data={data} />
    {spectra && <AdamaSpectra data={data} />}

    <Figure kind="table" id="ethiopia-cleaning" data={rows(data, 'cleaning_rounds')}
      title="spartan_ec 08 — how the 3σ cleaning runs: nine rounds remove 61 of 906 calibration filters"
      subtitle={`Fit at k = 20, drop every filter whose |residual| exceeds 3 × the residual spread, refit, repeat. ${rows(data, 'cleaning_fit').map((r) => `${r.stage}: n = ${r.n}, R² ${Number(r.R2).toFixed(3)}, RMSE ${r['RMSE (µg)']} µg`).join('; ')}. (Notebook 07's CAL-1 stops after three rounds, at 862 filters.)`}
      provenance={provenance(data, ['cleaning_rounds', 'cleaning_fit'])} />

    {calAll.length > 0
      ? <Figure kind="crossplot" id="ethiopia-removed" data={calAll} x="measured" y="predicted" color="status" idColumn="AnalysisId" tipColumns={['Site', 'round', 'status']}
          colors={{ kept: '#9ecae1', 'removed by 3σ cleaning': '#e45756', 'below 1:1: CAL-3 trains on these': '#d62728', 'on or above 1:1': '#cccccc' }}
          variants={[{ key: 'by', label: 'Colour', choices: [{ value: 'status', label: '3σ cleaning (spartan_ec 08; CAL-2 is its removals)' },
            { value: 'side', label: "Below 1:1 (06's CAL-3)", patch: { color: 'side', tipColumns: ['Site', 'side'],
              title: `June 06's inverse set CAL-3: the ${below} filters the full fit under-predicts`,
              subtitle: `June 06 [12]: CAL-3 trains only on the filters below the 1:1 line (predicted < measured), the samples a missing-char hypothesis would point to. 06 drew it on its CAL-0 baseline at k = 21 (469 filters); this export's full fit is spartan_ec 08's at k = 20, which puts ${below} below the line, so the set here is 06's to within a few filters. CAL-2, its other inverse set, trains on the 3σ removals (the other colouring). TOR EC per filter; no MAC.` } }] }]}
          xLabel="Measured TOR EC (µg per filter)" yLabel="Predicted EC, full 906-filter fit at k = 20 (µg)" fit={false} maxSide={480}
          title="All 906 calibration filters, before cleaning — the removed ones are mostly heavily loaded and far off the line"
          subtitle="Recomputed from the Drive's 906-filter EC set (pls_transfer.load_current_pls_model) with notebook 08's rule; the export is refused unless the nine rounds and the 61 removed filters (measured, predicted, site, round) equal 08's printed rounds and tracked list. Hover a point for its AnalysisId and round."
          provenance={provenance(data, ['calibration_all', 'removed'])} />
      : <Figure kind="crossplot" id="ethiopia-removed" data={rows(data, 'removed')} x="measured" y="predicted" color="reason" idColumn="FilterId" tipColumns={['Site', 'date', 'round', 'reason']}
          xLabel="Measured TOR EC (µg per filter)" yLabel="Predicted EC, full 906 fit (µg)" fit={false} maxSide={440}
          title="The removed filters — mostly heavily loaded, far off the line"
          subtitle="The 61 IMPROVE filters the cleaning drops, at their full-fit predictions. The 845 kept filters need the Drive (see the intro)."
          provenance={provenance(data, ['removed'])} />}

    <CleaningResiduals data={data} />

    {spectra && <DerivativeSpectra data={data} />}

    <Figure kind="table" id="ethiopia-removed-list" data={rows(data, 'removed')}
      columns={[{ key: 'FilterId', label: 'Filter' }, { key: 'Site', label: 'Site' }, { key: 'date', label: 'Date' }, { key: 'measured', label: 'Measured (µg)', format: 'f2' },
        { key: 'predicted', label: 'Predicted (µg)', format: 'f2' }, { key: 'residual', label: 'Residual (µg)', format: 'f2' }, { key: 'round', label: 'Round', format: 'int' }, { key: 'reason', label: 'Why' }]}
      title="Every removed filter, by |residual|"
      provenance={provenance(data, ['removed'])} />
  </>
}
