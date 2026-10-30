/**
 * The browser refit against the release's own tables: every block, summary, ID-11 fold, omission comparison and
 * denominator row, recomputed at the release's settings (every diagnostic filter, minimum training n 10) and
 * compared. The charts above follow the reader's settings; this panel does not.
 */
import { useMemo } from 'react'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { fmt } from '@/lib/format'
import { MIN_TRAIN, POPULATIONS, denominatorSensitivity, evaluate, fitTraining, id11Sensitivity, inPopulation, pairedInfluence, summarize, type Block, type FPoint, type Population, type Scheme } from '../../common/filterOnly'
import { DataTable } from '../../common/ui'
import { NB_DIAG, NB_PROP, NB_STAB, SITES, SOURCE, rowsOf, useRelease } from './data'

interface Check { table: string; rows: number; mismatched: number; worst: number; what: string }

function compare(points: FPoint[], release: NonNullable<ReturnType<typeof useRelease>['data']>): Check[] {
  const cache = new Map<string, Block[]>()
  const cohort = (site: string, pop: Population, variant: string, omit = 'ETAD-0037') =>
    points.filter((p) => p.site === site && inPopulation(p, pop) && !(variant === 'without_ETAD_0037' && p.id === omit))
  const run = (site: string, pop: Population, variant: string, scheme: Scheme) => {
    const key = `${site}|${pop}|${variant}|${scheme}`
    if (!cache.has(key)) cache.set(key, evaluate(points.filter((p) => p.site === site), cohort(site, pop, variant), scheme, MIN_TRAIN))
    return cache.get(key)!
  }
  const diff = (a: number, b: unknown) => (typeof b === 'number' ? Math.abs(a - b) : Number.isFinite(a) ? Infinity : 0)
  const checks: Check[] = []

  let bad = 0, worst = 0
  const blocks = rowsOf(release.blocks)
  for (const r of blocks) {
    const b = run(r.site, r.population, r.variant, r.scheme).find((x) => x.block === r.block)
    if (!b || b.status !== r.status || b.trainN !== r.train_n || b.testN !== r.test_n) { bad++; continue }
    const ds = [diff(b.fit.slope, r.ols_slope), diff(b.fit.intercept, r.ols_intercept), diff(b.fit.median, r.training_median), diff(b.mae.ols, r.ols_mae), diff(b.mae.median, r.median_mae), diff(b.mse.ols, r.ols_mean_signed_error)]
    if (r.variant === 'baseline' && r.status === 'evaluated') ds.push(diff(b.fit.k, r.proportional_k), diff(b.mae.proportional, r.proportional_mae))
    worst = Math.max(worst, ...ds)
  }
  checks.push({ table: 'Quarter blocks (both schemes, 6 populations, ETAD-0037 omitted)', rows: blocks.length, mismatched: bad, worst, what: 'slope, intercept, median, k, MAE ×3, signed error' })

  bad = 0; worst = 0
  const prop = rowsOf(release.proportionality)
  for (const r of prop) {
    const s = summarize(run(r.site, r.population, 'baseline', r.scheme)), m = r.weighting === 'equal_filter' ? s.filters : s.quarters
    if (s.filters.n !== r.evaluated_filters || s.interceptBetterBlocks !== r.intercept_better_blocks) bad++
    worst = Math.max(worst, diff(m.mae.ols, r.ols_mae), diff(m.mae.proportional, r.proportional_mae), diff(m.mae.median, r.median_mae), diff(m.mse.ols, r.ols_mean_signed_error))
  }
  checks.push({ table: 'Proportionality summaries (equal filters, equal quarters)', rows: prop.length, mismatched: bad, worst, what: 'MAE ×3, OLS signed error' })

  bad = 0; worst = 0
  const perf = rowsOf(release.performance)
  for (const r of perf) {
    const s = summarize(run(r.site, r.population, r.variant, r.scheme))
    if (s.improvedBlocks !== r.improved_blocks || s.filters.n !== r.evaluated_filters) bad++
    worst = Math.max(worst, diff(s.improvementPct, r.mae_improvement_pct) / 100)
  }
  checks.push({ table: 'Stability summaries', rows: perf.length, mismatched: bad, worst, what: 'improvement over the median (fraction)' })

  bad = 0; worst = 0
  const coef = rowsOf(release.coefficients)
  for (const r of coef) { const f = fitTraining(cohort(r.site, r.population, r.variant)); worst = Math.max(worst, diff(f.slope, r.slope), diff(f.intercept, r.intercept)) }
  checks.push({ table: 'Full-cohort coefficients', rows: coef.length, mismatched: 0, worst, what: 'slope, intercept' })

  bad = 0; worst = 0
  const id11 = rowsOf(release.id11)
  for (const pop of POPULATIONS.map((p) => p.key)) {
    const rows = id11Sensitivity(run('Addis Ababa', pop, 'baseline', 'later_period'), cohort('Addis Ababa', pop, 'baseline'), MIN_TRAIN)
    for (const r of id11.filter((x) => x.population === pop)) {
      const m = rows.find((x) => x.block === r.block)
      if (!m || m.status !== r.status || m.testN !== r.test_n) { bad++; continue }
      const x = r.training_choice === 'all_earlier' ? m.all : m.id11
      worst = Math.max(worst, diff(x.mae.ols, r.ols_mae), diff(x.mse.ols, r.ols_mean_signed_error))
    }
  }
  checks.push({ table: 'ID-11 training folds (Addis)', rows: id11.length, mismatched: bad, worst, what: 'OLS MAE and signed error' })

  bad = 0; worst = 0
  const inf = rowsOf(release.influence)
  for (const pop of POPULATIONS.map((p) => p.key)) for (const scheme of ['leave_quarter_out', 'later_period'] as Scheme[]) {
    const mine = pairedInfluence(run('Addis Ababa', pop, 'baseline', scheme), run('Addis Ababa', pop, 'without_ETAD_0037', scheme))
    inf.filter((r) => r.population === pop && r.scheme === scheme).forEach((r, i) => {
      if (!mine[i] || mine[i].n !== r.common_n) { bad++; return }
      worst = Math.max(worst, diff(mine[i].change, r.mae_change_without_minus_baseline))
    })
  }
  checks.push({ table: 'ETAD-0037 common-filter MAE change', rows: inf.length, mismatched: bad, worst, what: 'MAE change' })

  bad = 0; worst = 0
  const den = rowsOf(release.denominator)
  for (const r of den) {
    const d = denominatorSensitivity(points.filter((p) => p.site === r.site)).find((x) => x.multiple === r.minimum_ec_mdl_multiple)
    if (!d || d.n !== r.n) { bad++; continue }
    if (d.n) worst = Math.max(worst, diff(d.median, r.ratio_median), diff(d.q25, r.ratio_q25), diff(d.q75, r.ratio_q75))
  }
  checks.push({ table: 'Stricter-denominator ratios (diagnostics 05)', rows: den.length, mismatched: bad, worst, what: 'median and quartiles (m²/g)' })
  return checks
}

export function Agreement({ all }: { all: FPoint[] }) {
  const { data: release, error } = useRelease()
  const checks = useMemo(() => (release ? compare(all, release) : []), [all, release])
  return (
    <ChartFrame id="fo-agreement" title="Against the release — the browser's refit compared with the release's own result tables" exportable={false}
      subtitle="At the release's settings: all 545 diagnostic filters, minimum training n 10, no hand exclusions · largest absolute difference in each table's own units"
      provenance={<>{NB_STAB}, {NB_PROP}, {NB_DIAG}; deliverables/filter_only_scientific_release_2026-09-11/data</>} source={`${SOURCE}/Agreement.tsx`}>
      {error ? <Empty>Could not load the release tables: {error}</Empty> : !release ? <Empty>Loading the release tables…</Empty> : <>
        <DataTable rows={checks} rowKey={(c) => c.table} highlight={(c) => c.mismatched > 0}
          columns={[
            { key: 't', label: 'Release table', value: (c) => c.table },
            { key: 'n', label: 'Rows', num: true, value: (c) => c.rows },
            { key: 'm', label: 'Status or n differs', num: true, value: (c) => c.mismatched },
            { key: 'w', label: 'Largest |Δ|', num: true, value: (c) => (c.worst === 0 ? '0' : c.worst < 1e-3 ? c.worst.toExponential(1) : fmt(c.worst, 4)) },
            { key: 'q', label: 'Compared', value: (c) => c.what },
          ]} />
        <Note>The release tables are stored to 7 significant digits, so differences of about 10⁻⁶ of a value are rounding. Every chart on this page is the same code run on the reader's population, minimum training n and hand exclusions; at the defaults they reproduce the notebooks' figures. Sites: {SITES.join(', ')}.</Note>
      </>}
    </ChartFrame>
  )
}
