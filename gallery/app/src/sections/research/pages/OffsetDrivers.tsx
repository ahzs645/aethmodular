import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { FieldSelect } from '@/components/FieldSelect'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { twoSample } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow } from '@/lib/types'
import type { SectionPageProps } from '../../types'
import { correlationP, influence, multiOls, ols, originFit, pearsonR, spearman, splitScan, type ScanRow } from '../lib/stats'
import { Bars, DataTable, Grid, HIGH, LOW, SitePick, fitText, line, num, ofSite, point, seasonsOf, shortSeason, useSite } from '../lib/ui'
import { defaultRh } from './RhSweep'

const RECURSIVE = 'notebooks/analysis/meteorology/hips_ftir_intercept_rh_recursive_diagnostics.ipynb'
const INVESTIGATION = 'notebooks/analysis/meteorology/hips_intercept_investigation.ipynb'
const ENVELOPE = 'research/ftir_hips_chem/addis_06_hips_ptfe_operating_envelope.ipynb (§3 residual screen)'

type Residual = 'fit' | 'relative'

/**
 * What tracks the HIPS–FTIR offset at a site: the baseline fit against the through-origin one, the residual
 * against humidity, nested models with a covariate, the recursive split scan over every measurement in view,
 * a residual correlation screen, Bland–Altman, and the influential filters with what they are made of.
 */
export function OffsetDrivers({ rows, meta }: SectionPageProps) {
  const [site, setSite, sites] = useSite(rows, 'od')
  const siteRows = useMemo(() => ofSite(rows, site), [rows, site])
  const pick = (key: string, fallback: string) => (meta.fields.includes(key) ? key : fallback)
  const [xParam, setX] = useSearchParam<string>('od.x', 'EC (FTIR)')
  const [yParam, setY] = useSearchParam<string>('od.y', 'HIPS BC')
  const [covParam, setCov] = useSearchParam<string>('od.cov', '')
  const [resid, setResid] = useSearchParam<Residual>('od.resid', 'fit', ['fit', 'relative'])
  const [minParam, setMin] = useSearchParam<string>('od.min', '30', ['15', '30'])
  const xF = pick(xParam, meta.fields[0]), yF = pick(yParam, meta.fields[1] ?? meta.fields[0])
  const cov = meta.fields.includes(covParam) ? covParam : defaultRh(meta.fields, siteRows)
  const seasons = useMemo(() => seasonsOf(siteRows, meta), [siteRows, meta])
  const colorOf = (r: FilterRow) => seasons.find((s) => s.name === r.season)?.color ?? INK.muted

  const pairs = useMemo(() => siteRows.flatMap((r) => { const x = num(r, xF), y = num(r, yF); return x !== null && y !== null ? [{ r, x, y }] : [] }), [siteRows, xF, yF])
  const base = useMemo(() => ols(pairs.map((p) => p.x), pairs.map((p) => p.y)), [pairs])
  const origin = useMemo(() => originFit(pairs.map((p) => p.x), pairs.map((p) => p.y)), [pairs])
  /** each pair's residual: from the OLS line, or (y − x) / x as the operating-envelope notebook defines it */
  const residuals = useMemo(() => pairs.map((p) => ({ ...p, e: resid === 'fit' ? (base ? p.y - base.intercept - base.slope * p.x : NaN) : p.x !== 0 ? (p.y - p.x) / p.x : NaN })), [pairs, base, resid])
  const residLabel = resid === 'fit' ? `${yF} residual (obs − OLS)` : `(${yF} − ${xF}) / ${xF}`

  // the residual against the covariate (Panel C of the recursive notebook)
  const covPts = residuals.flatMap((p) => { const c = num(p.r, cov); return c !== null && Number.isFinite(p.e) ? [{ ...p, c }] : [] })
  const covFit = ols(covPts.map((p) => p.c), covPts.map((p) => p.e))
  const covRho = spearman(covPts.map((p) => p.c), covPts.map((p) => p.e))

  // nested models (does the covariate help?)
  const nested = useMemo(() => {
    const ok = pairs.flatMap((p) => { const c = num(p.r, cov); return c !== null ? [{ ...p, c }] : [] })
    const xs = ok.map((p) => p.x), cs = ok.map((p) => p.c), ys = ok.map((p) => p.y)
    return [
      { model: `${yF} ~ ${xF}`, fit: multiOls([xs], ys) },
      { model: `${yF} ~ ${xF} + ${cov}`, fit: multiOls([xs, cs], ys) },
      { model: `${yF} ~ ${xF} + ${cov} + ${xF}:${cov}`, fit: multiOls([xs, cs, xs.map((v, i) => v * cs[i])], ys) },
    ]
  }, [pairs, cov, xF, yF])

  // candidates: every measurement in view other than the pair and its own rescalings (HIPS BC = Fabs / MAC …)
  // and the reader's ratios built from either half of the pair (HIPS BC / EC (FTIR) is the residual restated)
  const candidates = useMemo(() => meta.fields.filter((f) => f !== xF && f !== yF && !/uncertainty|MDL/.test(f) && !f.split(' / ').some((part) => part === xF || part === yF)).filter((f) => {
    const vals = pairs.map((p) => num(p.r, f))
    const keep = pairs.filter((_, i) => vals[i] !== null)
    if (keep.length < 10) return false
    const rx = pearsonR(keep.map((p) => p.x), keep.map((p) => num(p.r, f)!)), ry = pearsonR(keep.map((p) => p.y), keep.map((p) => num(p.r, f)!))
    return !(Math.abs(rx ?? 0) > 0.999 || Math.abs(ry ?? 0) > 0.999)
  }), [meta.fields, pairs, xF, yF])

  const minN = Number(minParam)
  const scan = useMemo(() => {
    const all: ScanRow[] = []
    for (const f of candidates) all.push(...splitScan(f, pairs.flatMap((p) => { const s = num(p.r, f); return s !== null ? [{ x: p.x, y: p.y, s }] : [] }), pairs.length, minN))
    const best = new Map<string, ScanRow>()
    for (const r of all) if (!best.has(r.variable) || r.score > best.get(r.variable)!.score) best.set(r.variable, r)
    return [...best.values()].sort((a, b) => b.score - a.score)
  }, [candidates, pairs, minN])

  const screen = useMemo(() => candidates.map((f) => {
    const ok = residuals.flatMap((p) => { const v = num(p.r, f); return v !== null && Number.isFinite(p.e) ? [{ v, e: p.e }] : [] })
    const rho = ok.length >= 10 ? spearman(ok.map((o) => o.v), ok.map((o) => o.e)) : null
    const r = ok.length >= 10 ? pearsonR(ok.map((o) => o.v), ok.map((o) => o.e)) : null
    return { f, n: ok.length, rho, r }
  }).filter((s) => s.rho !== null).sort((a, b) => b.rho! ** 2 - a.rho! ** 2), [candidates, residuals])

  // Bland–Altman (is the bias fixed or proportional?)
  const ba = pairs.map((p) => ({ ...p, m: (p.x + p.y) / 2, d: p.y - p.x }))
  const bias = ba.length ? ba.reduce((a, p) => a + p.d, 0) / ba.length : NaN
  const sd = ba.length > 2 ? Math.sqrt(ba.reduce((a, p) => a + (p.d - bias) ** 2, 0) / (ba.length - 1)) : NaN
  const prop = ols(ba.map((p) => p.m), ba.map((p) => p.d))

  // influential filters (Cook's distance > 4 / n) and their chemistry
  const infl = useMemo(() => influence(pairs.map((p) => ({ x: p.x, y: p.y }))), [pairs])
  const cut = 4 / Math.max(1, pairs.length)
  const flagged = infl ? pairs.filter((_, i) => infl[i].cook > cut) : []
  const kept = infl ? pairs.filter((_, i) => infl[i].cook <= cut) : []
  const without = ols(kept.map((p) => p.x), kept.map((p) => p.y))
  const enrichment = useMemo(() => candidates.flatMap((f) => {
    const a = flagged.map((p) => num(p.r, f)).filter((v): v is number => v !== null), b = kept.map((p) => num(p.r, f)).filter((v): v is number => v !== null)
    const t = twoSample(a, b)
    if (!t || a.length < 3) return []
    const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length
    return mb !== 0 ? [{ f, ratio: ma / mb - 1, p: t.p, nA: a.length }] : []
  }).filter((e) => e.p < 0.1).sort((a, b) => Math.abs(b.ratio) - Math.abs(a.ratio)), [candidates, flagged, kept])

  const unitY = meta.field_units[yF] ?? '', unitX = meta.field_units[xF] ?? ''
  const controls = <>
    <SitePick value={site} options={sites} onChange={setSite} />
    <FieldSelect label="x" value={xF} meta={meta} onChange={setX} />
    <FieldSelect label="y" value={yF} meta={meta} onChange={setY} />
    <FieldSelect label="Covariate" value={cov} meta={meta} onChange={setCov} ratios={false} />
  </>
  if (pairs.length < 10) return <ChartFrame id="od-empty" title="What tracks the offset" provenance={RECURSIVE} controls={controls}><Empty>Fewer than 10 filters at {site} carry {xF} and {yF}.</Empty></ChartFrame>
  const ptsXY = pairs.map((p) => point(p.r, p.x, p.y, colorOf(p.r), xF, yF, 'Research · what tracks the offset'))
  return (
    <>
      <ChartFrame id="od-baseline" title={`Baseline fit and the residual — ${yF} against ${xF} at ${site}`}
        subtitle={`Left: OLS with a free intercept (black) and through the origin (dashed), points coloured by season. Right: the residual against ${cov}, with its OLS line. The through-origin R² is uncentred, as the notebook computes it.`}
        provenance={`${RECURSIVE} (Panels A and C)`} controls={controls}>
        <Grid>
          <Crossplot points={ptsXY} title={`${yF} vs ${xF}`} xLabel={`${xF}${unitX ? ` (${unitX})` : ''}`} yLabel={`${yF}${unitY ? ` (${unitY})` : ''}`} fit={false}
            extraLines={[...line(base, INK.fit, 'OLS'), ...(origin ? line({ slope: origin.slope, intercept: 0 }, INK.fit, 'through origin', '6 4') : [])]} />
          {covPts.length >= 10 ? <Crossplot points={covPts.map((p) => point(p.r, p.c, p.e, colorOf(p.r), cov, 'residual', 'Research · what tracks the offset'))} sameScale={false}
            title={`Residual vs ${cov}`} xLabel={cov} yLabel={residLabel} fit={false} extraLines={line(covFit, INK.fit, 'OLS')} />
            : <Empty>Fewer than 10 filters carry {cov}.</Empty>}
        </Grid>
        <Key items={[...seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: 'OLS', color: INK.fit, kind: 'line' }, { label: 'through the origin', color: INK.fit, kind: 'dash' }]} />
        <Note>OLS: {fitText(base)}. Through the origin: slope {fmt(origin?.slope, 3)}, R² {fmt(origin?.r2, 3)}. Residual against {cov}: Spearman ρ {fmt(covRho, 3)} (n {covPts.length}){covFit ? `, OLS slope ${fmt(covFit.slope, 4)} per unit, R² ${fmt(covFit.r2, 3)}` : ''}.</Note>
      </ChartFrame>

      <ChartFrame id="od-nested" title={`Does ${cov} help? — nested models for ${yF}`} subtitle="Least squares with an intercept on the filters that carry all three measurements. A covariate that matters lifts adjusted R² and lowers the RMSE."
        provenance={`${RECURSIVE} (nested model comparison)`} exportable={false}>
        <DataTable rows={nested} rowKey={(r) => r.model} columns={[
          { key: 'm', label: 'Model', value: (r) => r.model },
          { key: 'n', label: 'n', num: true, value: (r) => r.fit?.n ?? '—' },
          { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.fit?.r2, 3) },
          { key: 'a', label: 'Adjusted R²', num: true, value: (r) => fmt(r.fit?.adjR2, 3) },
          { key: 'e', label: 'RMSE', num: true, value: (r) => fmt(r.fit?.rmse, 3) },
          { key: 'c', label: 'Coefficients (intercept, then in order)', value: (r) => r.fit?.coef.map((c) => fmt(c, 4)).join(', ') ?? '—' },
        ]} />
      </ChartFrame>

      <ChartFrame id="od-scan" title={`Which split changes the line most? — the recursive scan over ${candidates.length} measurements`}
        subtitle={`Every measurement in view is split at its 20th … 80th percentile; each side gets its own OLS of ${yF} on ${xF}; the score is (|Δslope| + 0.2 |Δintercept|) × the share of filters both sides use. Each measurement's best split is shown. Both sides need ${minN} filters.`}
        provenance={`${RECURSIVE} (variable_split_scan)`} controls={<Segmented label="Min filters per side" value={minParam} options={['15', '30']} onChange={setMin} />}>
        {!scan.length ? <Empty>No measurement has {minN} filters on both sides of a split.</Empty> : <>
          <Bars label="split score" bars={scan.slice(0, 15).map((s) => ({ key: s.variable, label: s.variable, value: s.score,
            tip: [s.variable, `split at p${Math.round(s.quantile * 100)} = ${fmt(s.threshold, 3)}`, `low n ${s.nLow}: slope ${fmt(s.low.slope, 3)}, intercept ${fmt(s.low.intercept, 3)}`, `high n ${s.nHigh}: slope ${fmt(s.high.slope, 3)}, intercept ${fmt(s.high.intercept, 3)}`] }))} />
          <DataTable rows={scan.slice(0, 10)} rowKey={(r) => r.variable} columns={[
            { key: 'v', label: 'Measurement', value: (r) => r.variable },
            { key: 'q', label: 'Split', num: true, value: (r) => `p${Math.round(r.quantile * 100)} = ${fmt(r.threshold, 3)}` },
            { key: 'lo', label: <span style={{ color: LOW }}>Low side: slope / intercept (n)</span>, num: true, value: (r) => `${fmt(r.low.slope, 3)} / ${fmt(r.low.intercept, 3)} (${r.nLow})` },
            { key: 'hi', label: <span style={{ color: HIGH }}>High side: slope / intercept (n)</span>, num: true, value: (r) => `${fmt(r.high.slope, 3)} / ${fmt(r.high.intercept, 3)} (${r.nHigh})` },
            { key: 'c1', label: 'Closest to slope 1', num: true, value: (r) => fmt(r.closestTo1, 3) },
            { key: 's', label: 'Score', num: true, value: (r) => fmt(r.score, 3) },
          ]} />
          <Note>A top score says the line differs most across that split, not why. Measurements that move with the season (humidity, temperature, iron, source shares) can top the list together.</Note>
        </>}
      </ChartFrame>

      <ChartFrame id="od-screen" title={`Residual screen — Spearman ρ of the residual with every measurement`}
        subtitle={`${residLabel} against each measurement in view with at least 10 filters, ranked by ρ²; bars blue where the residual rises with the measurement, red where it falls.`}
        provenance={<>{INVESTIGATION} (§2 residual decomposition) · {ENVELOPE}</>}
        controls={<Segmented label="Residual" value={resid} options={['fit', 'relative'] as const} onChange={setResid} optionLabel={(o) => (o === 'fit' ? 'From the OLS line' : `(${yF} − ${xF}) / ${xF}`)} />}>
        <Bars label="Spearman ρ²" max={Math.max(0.1, ...screen.slice(0, 20).map((s) => s.rho! ** 2))} bars={screen.slice(0, 20).map((s) => ({ key: s.f, label: s.f, value: s.rho! ** 2, signed: s.rho,
          note: `ρ ${fmt(s.rho, 2)}`, tip: [s.f, `Spearman ρ ${fmt(s.rho, 3)} · Pearson r ${fmt(s.r, 3)} (p ${fmt(correlationP(s.r ?? NaN, s.n), 4)})`, `n ${s.n}`] }))} />
      </ChartFrame>

      <ChartFrame id="od-ba" title={`Bland–Altman — is the bias fixed or proportional?`}
        subtitle={`${yF} − ${xF} against their mean. Solid: the mean difference; dashed: ± 1.96 SD. A trend in the difference with the mean is a proportional bias (a slope), a level shift is a fixed one (an intercept).`}
        provenance={`${INVESTIGATION} (§5 Bland–Altman)`}>
        <Crossplot points={ba.map((p) => point(p.r, p.m, p.d, colorOf(p.r), 'mean', 'difference', 'Research · Bland–Altman'))} sameScale={false} title={`${yF} − ${xF}`} xLabel={`Mean of the two${unitX ? ` (${unitX})` : ''}`} yLabel="Difference" fit={false}
          extraLines={[...line({ slope: 0, intercept: bias }, INK.fit, 'bias'), ...line({ slope: 0, intercept: bias + 1.96 * sd }, INK.muted, '+1.96 SD', '5 4'), ...line({ slope: 0, intercept: bias - 1.96 * sd }, INK.muted, '−1.96 SD', '5 4'), ...line(prop, INK.deming, 'trend', '2 3')]} maxSide={420} />
        <Note>Mean difference {fmt(bias, 3)}, limits of agreement {fmt(bias - 1.96 * sd, 2)} to {fmt(bias + 1.96 * sd, 2)}. Trend of the difference with the mean (pink dotted): {fitText(prop)}.</Note>
      </ChartFrame>

      <ChartFrame id="od-influence" title={`Influential filters — Cook's distance above 4 / n, and what they are made of`}
        subtitle={`Leverage and Cook's distance of the OLS fit of ${yF} on ${xF}. Below: the measurements whose mean differs between the influential filters and the rest (Welch p < 0.10), as (mean in the influential filters / mean in the rest) − 1.`}
        provenance={`${INVESTIGATION} (§6 leverage and influential samples)`}>
        <Note>{flagged.length} of {pairs.length} filters have Cook's distance above {fmt(cut, 3)}. With them: {fitText(base)}. Without them: {fitText(without)}.</Note>
        <Grid>
          <DataTable rows={flagged.map((p) => ({ p, c: infl![pairs.indexOf(p)] })).sort((a, b) => b.c.cook - a.c.cook).slice(0, 15)} rowKey={(r) => r.p.r.id} columns={[
            { key: 'id', label: 'Filter', value: (r) => r.p.r.id },
            { key: 'd', label: 'Date', value: (r) => r.p.r.date },
            { key: 'x', label: xF, num: true, value: (r) => fmt(r.p.x, 2) },
            { key: 'y', label: yF, num: true, value: (r) => fmt(r.p.y, 2) },
            { key: 'h', label: 'Leverage', num: true, value: (r) => fmt(r.c.h, 3) },
            { key: 'c', label: "Cook's D", num: true, value: (r) => fmt(r.c.cook, 3) },
          ]} />
          {enrichment.length ? <Bars label="(mean influential / mean rest) − 1" max={Math.max(...enrichment.slice(0, 12).map((e) => Math.abs(e.ratio)))} bars={enrichment.slice(0, 12).map((e) => ({ key: e.f, label: e.f, value: Math.abs(e.ratio), signed: e.ratio,
            note: `${e.ratio >= 0 ? '+' : '−'}${fmt(Math.abs(e.ratio) * 100, 0)} %`, tip: [e.f, `Welch p ${fmt(e.p, 3)}`, `${e.nA} influential filters carry it`] }))} />
            : <Empty>No measurement differs at p &lt; 0.10 between the influential filters and the rest.</Empty>}
        </Grid>
      </ChartFrame>
    </>
  )
}
