import { Figure } from '@/components/Figure'
import { meanBcFromResidual, meanResidualAtMac, useMac } from '@/lib/mac'
import { FixedSetNote } from '@/lib/markedRows'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { PageIntro, provenance, rows, useSpectral, Waiting } from './spectralCommon'

const ci = (r: Row, v: string, se: string): Row => {
  const m = num(r[v]), s = num(r[se])
  return { [`${v}_lo`]: m != null && s != null ? m - 1.96 * s : null, [`${v}_hi`]: m != null && s != null ? m + 1.96 * s : null }
}

/**
 * ftir_53, the committee backup figures: does the Addis diagnosis generalise? The same locked
 * calibrations applied unchanged to five SPARTAN sites, Bishoftu's replication, and the two
 * deck lessons restated precisely. Every number is the executed notebook's table, at HIPS Fabs/10;
 * slopes (and their SEs) follow the MAC set under Filters → units. That is exact for OLS, and for York
 * too, since σx = σFabs/MAC rescales with x (lib/mac); intercepts do not move.
 */
export function Generalise(_props: SectionPageProps) {
  const { data, error } = useSpectral('committee')
  const m = useMac()
  if (!data) return <Waiting error={error} />
  const five = rows(data, 'five_site').map((r): Row => {
    const at = { ...r, slope: m.slope(num(r.slope), 10), slope_se: m.slope(num(r.slope_se), 10) }
    return { ...at, ...ci(at, 'intercept', 'intercept_se'), ...ci(at, 'slope', 'slope_se'), site: `${r.site} (n = ${r.n_used})` }
  })
  // the mean residual (prediction − Fabs/MAC) moves through each group's mean HIPS BC, which OLS pins down: it passes
  // through the means, so the notebook's residual = a + (b − 1)·x̄ at MAC 10 (lib/macMath meanBcFromResidual)
  const bishoftu = rows(data, 'bishoftu').map((r): Row => ({ ...r, ols_slope: m.slope(num(r.ols_slope), 10),
    mean_residual: m.changed ? meanResidualAtMac(num(r.ols_slope), num(r.ols_intercept), meanBcFromResidual(num(r.ols_slope), num(r.ols_intercept), num(r.mean_residual)), m.mac, 10) : num(r.mean_residual) }))

  return <>
    <PageIntro id="generalise-intro" title="Does the diagnosis generalise? — one calibration, five sites, and the deck's lessons stated precisely" data={data}
      notebooks="research/ftir_ec_phase3/ftir_53_committee_backup_figures (from ftir_47, ftir_51, ftir_46 and the locked reconstruction confirmation); ftir_31 and ftir_51 for the deck panels the gallery already draws">
      <p>The deck's Addis slides argue from one site. These are its backup: the calibrations applied unchanged to five SPARTAN sites fail in a site-specific way, so the work is diagnosis, not tuning. Reference axis HIPS Fabs/10 in the notebook; the slopes here are re-expressed at the MAC set under Filters → units (now {m.mac}). The other deck figures (ftir_31, ftir_51: setup crossplots, k sweep, analog distributions, lot crossplots) are already on the Calibration tab and the meeting pages.</p>
      <FixedSetNote subject="this page" what="ftir_53's saved tables (every figure on this page)" />
    </PageIntro>

    <Figure kind="intervals" id="generalise-five" data={five} row="site" series="config"
      panels={[{ value: 'intercept', lo: 'intercept_lo', hi: 'intercept_hi', label: 'Intercept (µg/m³; ±1.96 SE)', reference: 0 },
        { value: 'slope', lo: 'slope_lo', hi: 'slope_hi', label: `Slope vs Fabs/${m.mac} (±1.96 SE)`, reference: 1 }]} labelWidth={170}
      title="Five sites, one calibration — the intercept anomaly is Addis (and Delhi), the slope anomalies are Delhi and JPL"
      subtitle="York (errors-in-variables) fits of each locked configuration's predictions against HIPS with the per-deployed-line quadratic blank line (ftir_47), all matched filters per site. Near-zero intercepts at Bishoftu, Beijing and JPL; Delhi's intercept is poorly determined, so do not call it Addis-sized without its SE."
      provenance={provenance(data, ['five_site'])} />

    <Figure kind="table" id="generalise-bishoftu" data={bishoftu}
      columns={[{ key: 'group', label: 'Filters' }, { key: 'n', label: 'n', format: 'int' }, { key: 'ols_slope', label: `OLS slope (MAC ${m.mac})`, format: 'f2' }, { key: 'ols_intercept', label: 'OLS intercept', format: 'f2' }, { key: 'mean_residual', label: `Mean residual (µg/m³, MAC ${m.mac})`, format: 'f2' }]}
      title="Bishoftu, 45 km away — the offset is gone, on filters the calibration had never seen"
      subtitle={`The 26 accepted filters and 14 January–February filters scored after the configurations were locked (their Fabs reconstructed from raw HIPS with the dated-schedule blank line). Read the holdout as “near-zero intercept replicates”, not as a slope estimate. The mean residual is prediction − Fabs/${m.mac}: at another MAC than the notebook's 10 it is rebuilt from each group's mean HIPS BC, which the OLS fit and its MAC-10 residual fix (OLS passes through the means).`}
      provenance={provenance(data, ['bishoftu'])} />

    <Figure kind="intervals" id="generalise-swing" data={rows(data, 'protocol_swing').map((r): Row => ({ ...r, variant: `${r.variant}${r.corrected ? ' (baseline-corrected)' : ''}` }))} row="variant"
      panels={[{ value: 'intercept_swing', label: 'Spread of the Deming intercept across CV protocols (µg/m³)', reference: 0 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={260}
      title="Lesson 2, precisely — protocol swing is a raw-spectra problem"
      subtitle="Max − min Addis intercept across ftir_51's cross-validation options A, B and B2, per variant. Under 0.25 µg/m³ for the baseline-corrected cohort, up to 2.4 for raw ones: “protocol is not the driver” holds for corrected calibrations."
      provenance={provenance(data, ['protocol_swing'])} />

    <Figure kind="intervals" id="generalise-kiremt" data={rows(data, 'kiremt_dry').map((r): Row => ({ ...r, variant: `${r.variant} (k = ${r.k})` }))} row="variant"
      panels={[{ value: 'Kiremt/Dry', label: 'Kiremt / Dry ratio of median predicted EC (239 Addis filters)', reference: 1 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title="Lesson 3, precisely — rainy-season EC is 1.8 to 2.6 times the dry season across every variant"
      subtitle="ftir_51's seven variants (Option A, MAC 10). The Provisional Addis EC page draws the same ratio for the gallery's nine explorer presets."
      provenance={provenance(data, ['kiremt_dry'])} />

    <Figure kind="intervals" id="generalise-mass" data={rows(data, 'mass_term')} row="site"
      panels={[{ value: 'mass_coef_addis_range', lo: 'mass_coef_addis_range_lo', hi: 'mass_coef_addis_range_hi', label: 'HIPS Fabs per unit filter mass at fixed FTIR EC (bootstrap 95 %)', reference: 0 }]} labelWidth={220}
      title="Speaker-notes backup — Ethiopian filters carry two to three times the mass term of the other sites"
      subtitle="ftir_46: the coefficient on gravimetric mass in Fabs ~ f(X) + mass, with the locked Lowest-OC/EC 800 × spline-baseline f(X), inside the Addis loading range. A within-project result; keep it in notes, not on a slide."
      provenance={provenance(data, ['mass_term'])} />
  </>
}
