import { fmt } from '@/lib/format'
import type { SectionPageProps } from '../../types'
import { PageIntro, Waiting } from './spectralCommon'
import { EXPORTER, REPORT, useArchived, useCase, usePythonPmf } from './pythonPmf/data'
import { Contributions, Floor, Profiles, RankScreen, Residuals } from './pythonPmf/Fits'
import { Archived, HeldOut, Inventory, Measurements, Sensitivity } from './pythonPmf/Checks'

/**
 * Python PMF (Oct 2026): an independent uncertainty-weighted PMF of the Addis (ETAD) and Bishoftu (ETBI) FTIR spectra,
 * delivered as a package of code, inputs and saved results (gallery/data/export_python_pmf.py reads its CSV/JSON/NPZ,
 * never its code). The fits are the package's; the checks against held-out OM, the archived KF solution and the
 * gallery's measurements are recomputed here on the filters that count (common/pythonPmf.ts, node-tested against the
 * package's saved values). Factors are "Factor 1…5" only: the report forbids source names.
 */
export function PythonPmf({ rows, meta }: SectionPageProps) {
  const { data, error } = usePythonPmf()
  const archived = useArchived()
  const [c, setCase] = useCase(data)
  const [ce] = useCase(data, true)
  if (!data || !c) return <Waiting error={error} what="the Python PMF export" exporter={EXPORTER} />
  const p = data.heldout_pooled
  const primary = data.cases.find((x) => x.key === 'etad246_huber')!
  return <>
    <PageIntro id="ppmf-intro" title="Python PMF (Oct 2026) — an independent PMF of the Addis and Bishoftu spectra" data={data}
      notebooks={`the python_pmf package (30 Oct 2026): ${REPORT}, its saved results under results/ and analysis/`}>
      <p>An independent Python implementation of uncertainty-weighted PMF (ordinary and Huber robust losses, nonnegative G and H, signed X), run on
        the corrected spectra of {data.cases.find((x) => x.key === 'etad246_huber')?.n} ETAD filters (the original 102 as a comparison subset) and{' '}
        {data.cases.find((x) => x.key === 'etbi53_huber')?.n} ETBI filters, fitted separately: {data.package.fits} completed fits, {data.package.converged_fits} meeting
        the local convergence criteria. It is <strong>not EPA PMF5/ME-2</strong> (no numerical parity is established) and its factors are{' '}
        <strong>not validated named sources or source mass</strong>: they are Factor 1…5 throughout, with the closest archived KF factor given only as descriptive context.
        Five factors is the predeclared comparison with the historical run, not a selected source count.</p>
      <p>Headline numbers: when profile learning and mass calibration both exclude each held-out date block, the model predicts lab FTIR OM for 158 filters with RMSE{' '}
        <strong>{fmt(p.RMSE_ug_m3, 3)}</strong>, MAE <strong>{fmt(p.MAE_ug_m3, 3)}</strong> µg/m³ and pooled R² <strong>{fmt(p.R_squared, 3)}</strong> (same-method consistency, not source validation).
        The primary ETAD 246 robust fit reaches a scaled RMSE of {fmt(primary.rmse, 2)} with {(100 * primary.frac_gt3).toFixed(1)} % of residuals outside ±3U, and negative inputs alone
        force at least {fmt(primary.negative_rmse_lower_bound, 2)}: a baseline and weighting problem more factors cannot fix.</p>
      <p>What each chart answers: <strong>1</strong> what the five profiles look like and how much the bootstrap moves them; <strong>2</strong> how each factor varies filter by filter
        over time; <strong>3</strong> whether any factor count stands out; <strong>4</strong> why the misfit has a floor and where in the spectrum it sits; <strong>5</strong> how well
        the factors predict held-out total OM; <strong>6</strong> how each new factor tracks the archived KF factor it resembles; <strong>7</strong> how the factors correlate with
        the gallery's HIPS BC, FTIR EC/OC/OM and PM2.5 of the same filters; <strong>8</strong> whether the factors survive changes of loss, blank treatment, baseline and cohort;
        <strong> 9</strong> which filters the PMF holds, and where that disagrees with the gallery's exclusions. Charts 2, 5, 6 and 7 follow Filters → excluded and your hand
        exclusions (their statistics are recomputed on the filters that count, and equal the package's when every filter counts); charts 1, 3, 4 and 8 are the package's saved fits.
        One case picker (cohort × loss/preparation) drives charts 1, 2, 4, 6 and 7.</p>
    </PageIntro>
    <Profiles data={data} c={c} setCase={setCase} />
    <Contributions data={data} c={c} setCase={setCase} meta={meta} />
    <RankScreen data={data} />
    <Floor data={data} />
    <Residuals data={data} c={c} setCase={setCase} />
    <HeldOut data={data} />
    <Archived data={data} c={ce} setCase={setCase} archived={archived} />
    <Measurements data={data} c={ce} setCase={setCase} rows={rows} meta={meta} />
    <Sensitivity data={data} />
    <Inventory data={data} />
  </>
}
