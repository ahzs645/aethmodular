/**
 * Research → Python PMF, the fits themselves (charts 1–4): the five normalised profiles with the bootstrap band, the
 * per-filter contributions over time, the rank screen, and why more factors cannot fix the residuals. Profiles,
 * contributions and fit statistics are the package's saved results; only which filters the contribution chart draws
 * follows Filters → excluded (the fits were made on every modelled filter).
 */
import { useMemo } from 'react'
import type { ScaleTime } from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { Figure } from '@/components/Figure'
import { Key } from '@/components/Key'
import { timeSeasonBands } from '@/components/SeasonBands'
import { SpectrumPlot, type SpectrumBand, type SpectrumSeries } from '@/components/SpectrumPlot'
import { SweepPlot, type SweepMark, type SweepPoint, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { markTip } from '@/lib/exclusionMode'
import { ExcludedNote, FixedSetNote, useExclusionSplit, useKnownFilter, useMarkReason } from '@/lib/markedRows'
import { num, type Row } from '@/lib/table'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { DataTable, shortSeason } from '../../common/ui'
import { gapAfter } from '../../common/pythonPmf'
import {
  CasePick, FACTOR_COLOR, FACTORS, REPORT, addisCalendar, archivedFor, caseFilters, caseLabel, closest, factorName, tableRows, useProfiles,
  type CaseFilter, type PmfCase, type PythonPmfExport,
} from './data'

/** the modelled windows; 2500–2300 cm⁻¹ (CO₂) is left out of every fit */
const GAP = { lo: 2300, hi: 2500, label: 'not modelled 2300–2500 cm⁻¹', fill: INK.shade }
const WINDOW: [number, number] = [4000, 1520]
const COHORT_COLOR: Record<string, string> = { etad246: '#2c7fb8', etad102: '#d08c2c', etbi53: '#4f9d69' }
const COHORT_LABEL: Record<string, string> = { etad246: 'ETAD 246', etad102: 'ETAD 102', etbi53: 'ETBI 53' }
/** a share as a percentage with one decimal, as the report quotes them */
const pct1 = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? `${(100 * v).toFixed(1)} %` : '—')
const FIT_NOTE = 'the package\'s saved fits, each made on every filter its cohort modelled'

/** a NaN between channels more than `step` cm⁻¹ apart, so a line breaks across the unmodelled window */
function withGap(wn: readonly number[], ys: readonly (readonly number[])[], step = 10): { wn: number[]; ys: number[][] } {
  const w: number[] = [], out: number[][] = ys.map(() => [])
  wn.forEach((v, i) => {
    if (i > 0 && Math.abs(v - wn[i - 1]) > step) { w.push((v + wn[i - 1]) / 2); out.forEach((o) => o.push(NaN)) }
    w.push(v); ys.forEach((y, k) => out[k].push(y[i]))
  })
  return { wn: w, ys: out }
}

// --------------------------------------------------------------------------------------------- 1. profiles

export function Profiles({ data, c, setCase }: { data: PythonPmfExport; c: PmfCase; setCase: (k: string) => void }) {
  const { data: prof, error } = useProfiles()
  const [scale, setScale] = useSearchParam<'sum' | 'peak'>('ppmf-profiles.scale', 'sum', ['sum', 'peak'])
  const archived = archivedFor(data, c.key)
  const boot = prof?.bootstrap[c.key]
  const accepted = data.bootstrap[c.key]
  const panels = useMemo(() => {
    if (!prof) return []
    const H = prof.profiles[c.key], grid = prof.grids[c.grid]
    return FACTORS.map((k) => {
      const ref = H[k - 1]
      const div = scale === 'peak' ? Math.max(...ref) || 1 : 1 / 1000
      const sc = (v: readonly number[]) => v.map((x) => x / div)
      const g = withGap(grid, boot ? [sc(ref), sc(boot.p05[k - 1]), sc(boot.p95[k - 1])] : [sc(ref)])
      const series: SpectrumSeries[] = [{ id: `f${k}`, wn: g.wn, y: g.ys[0], color: FACTOR_COLOR[k - 1], role: 'line', width: 1.4, label: factorName(k) }]
      const bands: SpectrumBand[] = boot ? [{ id: `b${k}`, wn: g.wn, lo: g.ys[1], hi: g.ys[2], color: FACTOR_COLOR[k - 1], opacity: 0.25 }] : []
      return { k, series, bands, m: archived.get(k) }
    })
  }, [prof, c, scale, boot, archived])
  return (
    <ChartFrame id="ppmf-profiles" title={`Five factor profiles — ${caseLabel(c)}`}
      subtitle={`Each factor's normalised spectral shape (H, rows summing to one${scale === 'peak' ? '; here each scaled to its own peak' : '; drawn as ‰ of the profile per channel'}) over the two modelled windows, 4000–2500 and 2300–1520 cm⁻¹ (the CO₂ window between is not fitted, shaded). ${boot ? `The band is the segmented block bootstrap's 5–95 % per channel (${accepted?.resamples ?? 20} resamples, only accepted factor mappings: ${accepted?.accepted.map((a, i) => `F${i + 1} ${a}`).join(', ')} of ${accepted?.resamples ?? 20}); a central 90 % empirical spread, not a confidence interval.` : 'No bootstrap was run for this case (the package resampled ETAD 246 and ETAD 102, robust).'} Factor numbers are display coordinates, not sources${archived.size ? '; the closest archived KF factor is given as descriptive context only' : ''}.`}
      provenance={<>Reproduces the profile panels of {REPORT} §3 (analysis/figures/{c.key}_profiles_and_contributions.png where saved) and §7. Data: python_pmf/results/{c.folder}/{c.key}/k05/profiles.csv{boot ? `, results/uncertainty/${c.key.replace('_huber', '')}_bootstrap/ensemble.npz` : ''} → <code>research/python_pmf/profiles.json</code>.</>}
      controls={<>
        <CasePick data={data} value={c} onChange={setCase} />
        <Segmented label="Scale" value={scale} options={['sum', 'peak'] as const} onChange={setScale} optionLabel={(o) => (o === 'sum' ? '‰ of profile' : 'Own peak = 1')} />
      </>}>
      {!prof ? <Empty>{error ? `Could not load the profiles: ${error}` : 'Loading the profiles…'}</Empty> : <>
        <FacetGrid items={panels} minCell={320} maxCols={2} title={(p) => factorName(p.k)} note={(p) => closest(p.m)}
          render={(p, w) => <SpectrumPlot series={p.series} bands={p.bands} window={WINDOW} width={w} height={190} regions={[GAP]} hover="readout"
            yLabel={scale === 'peak' ? 'Relative to peak' : '‰ per channel'} includeZero readoutFormat={(v) => fmt(v, 3)} />} />
        <Key items={[...FACTORS.map((k) => ({ label: factorName(k), color: FACTOR_COLOR[k - 1], kind: 'line' as const })), ...(boot ? [{ label: 'bootstrap 5–95 %', color: INK.muted, kind: 'square' as const }] : [])]} />
      </>}
      <FixedSetNote what={FIT_NOTE} />
    </ChartFrame>
  )
}

// --------------------------------------------------------------------------------------------- 2. contributions

export function Contributions({ data, c, setCase, meta }: { data: PythonPmfExport; c: PmfCase; setCase: (k: string) => void; meta: MetaFile }) {
  const calendar = useMemo(() => addisCalendar(meta), [meta])
  const items = useMemo(() => caseFilters(data, c.key, calendar), [data, c.key, calendar])
  const split = useExclusionSplit(items, (f) => f.base)
  const reasonOf = useMarkReason()
  const known = useKnownFilter()
  const archived = archivedFor(data, c.key)
  const seasons = useMemo(() => calendar.filter((s) => items.some((f) => f.season === s.name)), [calendar, items])
  const xDomain = useMemo((): [Date, Date] | undefined => {
    if (!items.length) return undefined
    const t = items.map((f) => Date.parse(f.date))
    return [new Date(Math.min(...t) - 5 * 864e5), new Date(Math.max(...t) + 5 * 864e5)]
  }, [items])
  const panels = useMemo(() => {
    const counted = [...split.counted].sort((a, b) => a.date.localeCompare(b.date))
    const breaks = gapAfter(counted.map((f) => f.date), 14)
    const tip = (f: CaseFilter, k: number) => [f.id, `${f.date}${f.season ? ` · ${shortSeason(f.season)}` : ''}`, `${factorName(k)} G ${fmt(f.g[k - 1], 3)}`, `scaled RMSE of this filter ${fmt(f.rmse, 2)}`]
    return FACTORS.map((k) => {
      const line: SweepPoint[] = []
      counted.forEach((f, i) => {
        line.push({ x: new Date(f.date), y: f.g[k - 1] })
        if (breaks.has(i)) line.push({ x: new Date((Date.parse(f.date) + Date.parse(counted[i + 1].date)) / 2), y: null })
      })
      const series: SweepSeries[] = [
        { id: 'line', label: factorName(k), color: FACTOR_COLOR[k - 1], points: line, marks: false, gaps: true, width: 1.1 },
        ...seasons.map((s) => ({ id: s.name, label: shortSeason(s.name), color: s.color, marks: true, width: 0, r: 2.4,
          points: counted.filter((f) => f.season === s.name).map((f) => ({ x: new Date(f.date), y: f.g[k - 1], id: f.base, tip: tip(f, k) })) })),
      ]
      const marks: SweepMark[] = split.marked.map((f) => ({ x: new Date(f.date), y: f.g[k - 1], tip: markTip({ id: f.id, date: f.date }, reasonOf(f.base) ?? 'excluded', [`${factorName(k)} G ${fmt(f.g[k - 1], 3)}`]) }))
      return { k, series, marks, m: archived.get(k) }
    })
  }, [split, seasons, archived, reasonOf])
  const noGallery = items.filter((f) => !known(f.base)).length
  return (
    <ChartFrame id="ppmf-contributions" title={`Contributions over time — every modelled filter, ${caseLabel(c)}`}
      subtitle={`Each filter's contribution G to each factor at its sampling date (${split.counted.length} of ${items.length} filters drawn; dots coloured by the season of the date under your Addis calendar${c.cohort === 'etbi53' ? ', which Bishoftu, 45 km away, takes too' : ''}). The line breaks across sampling gaps longer than 14 days, the package's own segment rule. G is summed-channel spectral amplitude per sampled m³, not µg/m³: no factor here has a mass calibration.${noGallery ? ` ${noGallery} of these filters have no gallery row (${c.cohort === 'etbi53' ? 'Bishoftu is not a gallery site' : 'sampled after the gallery\'s Addis series ends'}); they are drawn, and count, but cannot be excluded except by hand.` : ''}`}
      provenance={<>Reproduces the contribution panels of {REPORT} §3. Data: python_pmf/results/{c.folder}/{c.key}/k05/contributions.csv → <code>research/python_pmf/python_pmf.json</code>.</>}
      controls={<CasePick data={data} value={c} onChange={setCase} />}>
      {!items.length ? <Empty>No contributions for this case.</Empty> : <>
        <FacetGrid items={panels} minCell={300} maxCols={1} title={(p) => factorName(p.k)} note={(p) => closest(p.m)}
          render={(p, w) => <SweepPlot series={p.series} marks={p.marks} xScale="time" xDomain={xDomain} yLabel="G" height={130} width={w} include={[0]}
            background={({ x, iw, ih }) => timeSeasonBands(x as ScaleTime<number, number>, seasons, iw).map((b, i) => <rect key={i} x={b.x} y={0} width={b.w} height={ih} fill={b.color} opacity={0.08} />)} />} />
        <Key items={seasons.map((s) => ({ label: shortSeason(s.name), color: s.color }))} />
      </>}
      <ExcludedNote count={split.nOut} drawn />
    </ChartFrame>
  )
}

// --------------------------------------------------------------------------------------------- 3. rank screen

export function RankScreen({ data }: { data: PythonPmfExport }) {
  const rank = tableRows(data, 'rank'), starts = tableRows(data, 'starts')
  const cohorts = ['etad246', 'etad102', 'etbi53']
  const bound = (co: string) => num(rank.find((r) => r.cohort === co)?.negative_rmse_lower_bound ?? null)
  const plot = (metric: 'loss_per_cell' | 'rmse'): SweepSeries[] => cohorts.flatMap((co) => {
    const sel = rank.filter((r) => r.cohort === co)
    const st = starts.filter((r) => r.cohort === co)
    const tipOf = (r: Row) => [`${COHORT_LABEL[co]}, k = ${r.k}`, `selected fit: ${metric === 'rmse' ? 'Q(true) RMSE' : 'Huber loss / cell'} ${fmt(r[metric], 3)}`,
      `${r.converged_starts} of ${r.total_starts} starts converged${r.converged ? '' : '; none did, so this is the lowest capped start'}`,
      `near-best minimum profile cosine ${r.near_best_min_cosine == null ? '— (no qualifying start)' : fmt(r.near_best_min_cosine, 3)}`]
    const startPts = (conv: boolean) => st.filter((s) => !!s.converged === conv).map((s) => ({ x: Number(s.k), y: num(s[metric]), tip: [`${COHORT_LABEL[co]}, k = ${s.k}, seed ${s.seed}`, `${conv ? 'converged' : 'capped (iteration limit)'}`, `${metric === 'rmse' ? 'RMSE' : 'loss / cell'} ${fmt(s[metric], 3)}`] }))
    return [
      { id: `${co}-sel`, label: COHORT_LABEL[co], color: COHORT_COLOR[co], r: 3.2, marks: true, points: sel.map((r) => ({ x: Number(r.k), y: num(r[metric]), tip: tipOf(r) })) },
      { id: `${co}-ok`, label: `${COHORT_LABEL[co]} starts`, color: COHORT_COLOR[co], r: 1.6, marks: true, width: 0, points: startPts(true) },
      { id: `${co}-cap`, label: `${COHORT_LABEL[co]} capped`, color: INK.muted, r: 1.8, marks: true, width: 0, points: startPts(false) },
    ]
  })
  const ks = [...new Set(rank.map((r) => Number(r.k)))].sort((a, b) => a - b)
  const cell = (co: string, k: number) => rank.find((r) => r.cohort === co && Number(r.k) === k)
  return (
    <ChartFrame id="ppmf-rank" title="Rank screening — the loss falls with every factor added, and no rank stands out"
      subtitle={`The selected fit at each factor count (line), every start behind it (small dots; grey: stopped at the iteration cap) for the three cohorts: 2–13 factors for ETAD, 2–8 for ETBI, five starts per rank and 20 at five factors (ringed: the predeclared comparison with the historical five-factor run, not a selected rank). Left: the Huber objective per cell, what the fits minimise. Right: Q(true) as a scaled RMSE, √(Q/cells), with each cohort's negative-cell floor dashed (chart below). A lower training loss is expected as flexibility grows and does not identify a number of sources; convergence thins out above 7–8 factors.`}
      provenance={<>Reproduces {REPORT} §5 (rank tables and analysis/figures/rank_screening.png). Data: python_pmf/analysis/rank_diagnostics.csv, results/rank_screen/*/all_starts.csv and results/five_factor/*_huber/all_starts.csv.</>}>
      <FacetGrid items={['loss_per_cell', 'rmse'] as const} minCell={300} maxCols={2}
        title={(m) => (m === 'rmse' ? 'Q(true) RMSE (scaled by U)' : 'Huber loss per cell')}
        render={(m, w) => <SweepPlot series={plot(m)} xScale="linear" xLabel="Factors" yLabel={m === 'rmse' ? 'scaled RMSE' : 'loss / cell'} height={250} width={w} include={[0]}
          chosen={cohorts.map((co) => ({ series: `${co}-sel`, x: 5 }))}
          yMarkers={m === 'rmse' ? cohorts.map((co) => ({ at: bound(co) ?? 0, color: COHORT_COLOR[co] })) : []} />} />
      <Key items={[...cohorts.map((co) => ({ label: COHORT_LABEL[co], color: COHORT_COLOR[co], kind: 'line' as const })), { label: 'capped start', color: INK.muted },
        ...cohorts.map((co) => ({ label: `${COHORT_LABEL[co]} negative-cell floor ${fmt(bound(co), 2)}`, color: COHORT_COLOR[co], kind: 'dash' as const }))]} />
      <DataTable rows={ks} rowKey={(k) => String(k)} highlight={(k) => k === 5} columns={[
        { key: 'k', label: 'Factors', num: true, value: (k) => k },
        ...cohorts.flatMap((co) => [
          { key: `${co}-c`, label: `${COHORT_LABEL[co]} converged / starts`, num: true, value: (k: number) => { const r = cell(co, k); return r ? `${r.converged_starts}/${r.total_starts}` : '—' } },
          { key: `${co}-cos`, label: 'min cosine', num: true, title: 'near-best minimum matched profile cosine across starts', value: (k: number) => { const r = cell(co, k); return r ? (r.near_best_min_cosine == null ? 'none' : fmt(r.near_best_min_cosine, 3)) : '—' } },
        ]),
      ]} />
      <FixedSetNote what={FIT_NOTE} />
    </ChartFrame>
  )
}

// --------------------------------------------------------------------------------------------- 4. the residual floor

export function Floor({ data }: { data: PythonPmfExport }) {
  const rows: Row[] = data.cases.map((c) => ({ cohort: c.cohort_label, case: c.prep_label, rmse: c.rmse, bound: c.negative_rmse_lower_bound, share: 100 * c.lower_bound_share_of_q, gt3: c.frac_gt3, neg: 100 * c.frac_negative }))
  const primary = data.cases.find((c) => c.key === 'etad246_huber')
  return (
    <Figure kind="intervals" id="ppmf-floor" data={rows} block="cohort" row="case" labelWidth={210}
      panels={[{ value: 'rmse', lo: 'bound', hi: 'rmse', label: 'Scaled RMSE: the negative-cell floor (bar start) to the fit (dot)', reference: 0 },
        { value: 'share', label: 'Floor as % of the fit\'s Q(true)' }, { value: 'neg', label: '% of input cells below zero' }]}
      title="Why more factors cannot fix it — negative observations set a floor no nonnegative fit can go below"
      subtitle={`G H is nonnegative, so every negative observation X is missed by at least |X|: Q(true) ≥ Σ (X/U)² over the negative cells, whatever the factor count. ${primary ? `For ETAD 246, robust: ${pct1(primary.frac_negative)} of cells are negative, the floor is a scaled RMSE of ${fmt(primary.negative_rmse_lower_bound, 3)} against the fit's ${fmt(primary.rmse, 3)}, ${pct1(primary.lower_bound_share_of_q)} of its Q(true), and ${pct1(primary.frac_gt3)} of its residuals lie outside ±3U.` : ''} The floor is a mathematical bound, not the measured share of Q from negative cells; it points at the baseline and the U proxy, and the package does not clip X or enlarge U to beat it.`}
      provenance={<>Reproduces {REPORT} §4 (the negative-cell table). Data: python_pmf/analysis/rank_diagnostics.csv (negative_observation_Q_lower_bound, Q_true, n_observed_cells).</>}
      aside={<FixedSetNote what={FIT_NOTE} />} />
  )
}

export function Residuals({ data, c, setCase }: { data: PythonPmfExport; c: PmfCase; setCase: (k: string) => void }) {
  const { data: prof, error } = useProfiles()
  const regions = tableRows(data, 'regions').filter((r) => r.case === c.key)
  const plots = useMemo(() => {
    if (!prof) return null
    const res = prof.residuals[c.key]
    const g = withGap(prof.grids[c.grid], [res.mean, res.negative.map((v) => v * 100)])
    return {
      mean: [{ id: 'mean', wn: g.wn, y: g.ys[0], color: INK.text, role: 'line' as const, width: 1.2, label: 'mean scaled residual' }],
      neg: [{ id: 'neg', wn: g.wn, y: g.ys[1], color: INK.negative, role: 'line' as const, width: 1.2, label: '% of filters with X < 0' }],
    }
  }, [prof, c])
  return (
    <ChartFrame id="ppmf-residuals" title={`Where the misfit sits — residuals by wavenumber and spectral region, ${caseLabel(c)}`}
      subtitle="Top: the mean scaled residual (X − GH)/U at each channel over the case's filters. Bottom: the share of filters whose input is negative at that channel. The fit undershoots most where the input goes negative (above 3500 cm⁻¹ and 1700–2300 cm⁻¹): the model cannot follow a signed baseline. The table is the package's summary by broad region."
      provenance={<>Reproduces {REPORT} §4 (spectral-region diagnostics). Data: python_pmf/results/{c.folder}/{c.key}/k05/feature_residuals.csv and analysis/spectral_region_residuals.csv.</>}
      controls={<CasePick data={data} value={c} onChange={setCase} />}>
      {!plots ? <Empty>{error ? `Could not load the residuals: ${error}` : 'Loading the residuals…'}</Empty> : <>
        <SpectrumPlot series={plots.mean} window={WINDOW} height={200} regions={[GAP]} zeroLine hover="readout" yLabel="mean (X − GH)/U" readoutFormat={(v) => fmt(v, 2)} />
        <SpectrumPlot series={plots.neg} window={WINDOW} height={150} regions={[GAP]} hover="readout" yLabel="% filters X < 0" includeZero readoutFormat={(v) => `${fmt(v, 0)} %`} />
      </>}
      <DataTable rows={regions} rowKey={(r) => `${r.wavenumber_min}`} columns={[
        { key: 'w', label: 'Region (cm⁻¹)', value: (r) => `${r.wavenumber_max}–${r.wavenumber_min}` },
        { key: 'f', label: 'Channels', num: true, value: (r) => r.features },
        { key: 'm', label: 'Mean scaled residual', num: true, value: (r) => fmt(r.mean_scaled_residual, 2) },
        { key: 'r', label: 'Scaled RMSE', num: true, value: (r) => fmt(r.scaled_rmse, 2) },
        { key: 'g', label: '|r| > 3', num: true, value: (r) => pct1(r.fraction_abs_gt3) },
        { key: 'n', label: 'X < 0', num: true, value: (r) => pct1(r.fraction_negative_X) },
        { key: 'u', label: 'X < −3U', num: true, value: (r) => pct1(r.fraction_X_below_minus3U) },
      ]} />
      <Note>Correlated neighbouring channels and the proxy U make a chi-square or parameter-count reading of these numbers inappropriate (report §4).</Note>
      <FixedSetNote what={FIT_NOTE} />
    </ChartFrame>
  )
}
