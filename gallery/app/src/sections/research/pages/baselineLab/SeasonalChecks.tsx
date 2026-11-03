import * as d3 from 'd3'
import { ChartFrame, Empty, Select, Segmented } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { BarRows, type BarRow } from '@/components/BarRows'
import { HeatMatrix } from '@/components/HeatMatrix'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { useFitHeight } from '@/lib/chartFit'
import { useColourScheme } from '@/lib/colourSchemes'
import { FixedSetNote } from '@/lib/markedRows'
import { fmt, pct, signed } from '@/lib/format'
import { shortSeason } from '@/lib/seasons'
import { baselineColor, INK } from '@/lib/theme'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { finite, LAB_SOURCE } from '../../common/baselineLab'
import { CHECK_TARGET_LABEL, CHECK_TARGETS, ecLineFor, pairResidualScores, selectResidualImportance, validateSeasonalChecks,
  SEASONAL_CHECKS_FILE, type CheckRegionSet, type CheckTarget, type ResidualImportance, type SeasonalChecksData } from '../../common/seasonalChecks'
import type { LabChartProps } from './data'

export const useSeasonalChecks = () => useExport<SeasonalChecksData>(SEASONAL_CHECKS_FILE, validateSeasonalChecks)

const VIEWS = ['ec', 'regions', 'lot'] as const
const VIEW_LABEL: Record<typeof VIEWS[number], string> = { ec: 'Against the EC line', regions: 'Regions beyond EC', lot: 'Lot control' }
const FOLD_LABEL: Record<string, string> = { forward_2024: '2023–24 (97 filters, all months)', forward_2025: '2024–25 (30 filters, sparse)' }
const FAMILY_COLOR: Record<string, string> = { rf: RESEARCH_SERIES[2], linear: RESEARCH_SERIES[0], pls: RESEARCH_SERIES[4] }
const unique = <T,>(values: T[]) => [...new Set(values)]
const famLabel = (data: SeasonalChecksData | null, f: string) => data?.labels.family[f] ?? f
const repLabel = (data: SeasonalChecksData | null, r: string) => data?.labels.representation[r] ?? r.replace(/_/g, ' ')
const repColor = (r: string) => /removed/.test(r) ? INK.muted : /spline/.test(r) ? baselineColor('AIRSpec') : INK.text
const regionName = (row: ResidualImportance) => row.region_id === 'whole_spectrum' ? 'Whole spectrum'
  : /^bin_/.test(row.region_id) ? `${row.region_id.replace(/^bin_/, '').replace('_', '–')} cm⁻¹` : row.region_label
const cycleLabel = (cycle: number) => `${cycle - 1}–${String(cycle).slice(2)}`

function CheckNotes({ data }: { data: SeasonalChecksData | null }) {
  const corr = data?.disagreement_ec_correlation ?? {}
  return <>
    <p><strong>Why these checks.</strong> D = FTIR EC − HIPS BC, and FTIR EC is computed from the same spectrum.
      {Object.keys(corr).length ? ` D tracks FTIR EC closely (r = ${Object.entries(corr).map(([c, r]) => `${fmt(r, 2)} in ${cycleLabel(Number(c))}`).join(', ')}).` : ''}
      {' '}A spectral model of D can score well just by re-learning the EC calibration. The EC line (D ≈ a + b·FTIR EC, fitted on the training filters, no spectral channel) is the comparator every D score should beat.</p>
    <p>The residual models fit the spectrum to what the training-fold EC line leaves over; their prediction is the line plus that residual.
      Region tests on the residual show spectral information about the disagreement beyond FTIR EC itself.</p>
    <p>Lot: in 2022–23, the first training year, every Bega filter is lot 248 and every Kiremt filter lot 251; both test years are lot 251.
      The lot-controlled tests train on 2023–24 (all lot 251) and test other years by lot.</p>
    {data?.lots.note && <p>{data.lots.note}</p>}
    {data?.limitations.map((text, i) => <p key={i}>{text}</p>)}
    {data && <p>Spectra: {data.provenance.spectra?.kind ?? '—'} · {data.reproduction.compared_models} saved D scores re-derived, largest RMSE difference {fmt(data.reproduction.max_abs_rmse_difference, 3)} µg/m³ · {data.generated_at}.</p>}
  </>
}

/** Lot, EC-line and residual-region controls for the saved seasonal ML. */
export function SeasonalMLChecks(_props: LabChartProps) {
  const { data, error } = useSeasonalChecks()
  const height = useFitHeight() ?? 380
  const [view, setView] = useSearchParam<string>('blab.chk.view', 'ec', [...VIEWS])
  const [wantedFold, setFold] = useSearchParam<string>('blab.chk.fold', 'forward_2024')
  const [wantedRep, setRep] = useSearchParam<string>('blab.chk.rep', 'all')
  const [wantedTarget, setTarget] = useSearchParam<CheckTarget>('blab.chk.target', 'residual', [...CHECK_TARGETS])
  const [wantedImpRep, setImpRep] = useSearchParam<string>('blab.chk.irep', 'spline_common')
  const [wantedSeason, setSeason] = useSearchParam<string>('blab.chk.season', 'all')
  const [wantedSet, setSet] = useSearchParam<CheckRegionSet>('blab.chk.set', 'named', ['bins', 'named'])
  const [wantedFamily, setFamily] = useSearchParam<string>('blab.chk.family', 'linear', ['rf', 'linear'])
  const colours = useColourScheme('blab.chk.lots', 'sequential')

  const folds = unique(data?.residual_scores.map(row => row.fold_id) ?? [])
  const fold = folds.includes(wantedFold) ? wantedFold : folds[0] ?? wantedFold
  const line = ecLineFor(data?.ec_line, { calendar: 'belg_feb', cohort: 'direct232', fold_id: fold })

  // --- view 1: every D model next to the EC line, and the same model on top of it
  const reps = unique(data?.residual_scores.map(row => row.representation) ?? [])
  const rep = wantedRep === 'all' || reps.includes(wantedRep) ? wantedRep : 'all'
  const pairs = data ? pairResidualScores(data.residual_scores, fold, rep) : []
  const ecRows: BarRow[] = !line ? [] : [{ key: 'ec_line', label: 'FTIR EC line only · no spectrum', value: line.rmse, color: INK.muted, group: 'Comparator',
    labelWeight: 700, tip: ['D ≈ a + b·FTIR EC, fitted on the training filters', `slope ${fmt(line.slope, 3)} · intercept ${fmt(line.intercept, 3)} µg/m³`,
      `RMSE ${fmt(line.rmse, 3)} · R² ${fmt(line.r2, 3)} · ${line.n_train} training / ${line.n_test} test filters`] }]
  for (const p of pairs) {
    const group = repLabel(data, p.representation)
    for (const [kind, row] of [['direct', p.direct], ['residual', p.residual]] as const) {
      if (!row) continue
      const better = line ? line.rmse - row.rmse : null
      ecRows.push({ key: `${p.model_id}-${kind}`, group, color: repColor(p.representation), opacity: kind === 'direct' ? .45 : 1,
        label: `${famLabel(data, p.family)} · ${kind === 'direct' ? 'D directly' : 'EC line + spectral residual'}`, value: row.rmse,
        note: finite(better) ? `${better >= 0 ? '−' : '+'}${fmt(Math.abs(better), 2)} vs line` : undefined,
        tip: [`${famLabel(data, p.family)} · ${repLabel(data, p.representation)}`, kind === 'direct' ? 'Fitted to D directly (as in the saved analysis)' : 'EC line + a spectral model of what the line misses',
          `RMSE ${fmt(row.rmse, 3)} µg/m³ · R² ${fmt(row.r2, 3)} · bias ${signed(row.bias, 3)}`,
          ...(finite(row.saved_rmse) ? [`Saved analysis RMSE: ${fmt(row.saved_rmse, 3)}`] : []),
          ...(finite(better) ? [`${better >= 0 ? 'Better' : 'Worse'} than the EC line by ${fmt(Math.abs(better), 3)} µg/m³`] : [])] })
    }
  }

  // --- view 2: region tests, D vs what the EC line misses
  const target = wantedTarget
  const impReps = unique(data?.residual_importance.map(row => row.representation) ?? [])
  const impRep = impReps.includes(wantedImpRep) ? wantedImpRep : impReps[0] ?? wantedImpRep
  const seasons = unique(data?.residual_importance.filter(row => row.fold_id === fold && row.season).map(row => row.season!) ?? []).sort()
  const season = seasons.includes(wantedSeason) ? wantedSeason : 'all'
  const regionSet = wantedSet
  const imp = data ? selectResidualImportance(data.residual_importance, { fold_id: fold, target, representation: impRep,
    season: season === 'all' ? null : season, set: regionSet }) : []
  const families = unique(imp.map(row => row.family))
  const impTip = (row: ResidualImportance) => [regionName(row), `${famLabel(data, row.family)} · ${repLabel(data, row.representation)}`,
    `${signed(row.delta_mean, 3)} µg/m³ RMSE when shuffled · positive = worse`,
    ...(finite(row.low) && finite(row.high) ? [`${fmt(row.low, 0)}–${fmt(row.high, 0)} cm⁻¹ · ${row.n_channels} channels`] : []),
    ...(finite(row.q05) && finite(row.q95) ? [`5th–95th of ${row.repeats} shuffles: ${signed(row.q05, 3)} to ${signed(row.q95, 3)} (not a confidence interval)`] : []),
    `Unshuffled RMSE ${fmt(row.baseline_loss, 3)} µg/m³ · ${row.n_test} held-out filters${row.season ? ` (${shortSeason(row.season)})` : ''}`,
    CHECK_TARGET_LABEL[row.target]]
  const sweep: SweepSeries[] = families.map(f => ({ id: f, label: famLabel(data, f), color: FAMILY_COLOR[f] ?? INK.accent, line: false, marks: true,
    interval: 'whisker', r: 4, points: imp.filter(row => row.family === f && finite(row.low) && finite(row.high)).map(row => ({
      x: (row.low! + row.high!) / 2, y: row.delta_mean, lo: row.q05 ?? null, hi: row.q95 ?? null, tip: impTip(row) })) }))
  const named = unique(imp.map(row => row.region_id))
  const namedRows: BarRow[] = named.map(id => {
    const items = imp.filter(row => row.region_id === id)
    return { key: id, label: regionName(items[0]), valueLabel: null,
      bars: items.map(row => ({ key: row.family, value: row.delta_mean ?? 0, color: FAMILY_COLOR[row.family] ?? INK.accent })),
      tip: items.flatMap(row => [`${famLabel(data, row.family)}: ${signed(row.delta_mean, 3)} µg/m³`]).concat(
        [finite(items[0].low) ? `${fmt(items[0].low, 0)}–${fmt(items[0].high, 0)} cm⁻¹` : '', CHECK_TARGET_LABEL[target]]) }
  })
  const limits = d3.extent(imp.flatMap(row => finite(row.low) && finite(row.high) ? [row.low!, row.high!] : []))
  const xDomain: [number, number] | undefined = finite(limits[0]) && finite(limits[1]) ? [limits[1]!, limits[0]!] : undefined

  // --- view 3: lot
  const family = wantedFamily
  const testOrder = unique((data?.lot_control ?? []).map(row => row.test_id))
  const repOrder = ['raw_common', 'spline_common', 'removed_common']
  const lotRows: BarRow[] = (data?.lot_control ?? []).filter(row => row.family === family)
    .sort((a, b) => testOrder.indexOf(a.test_id) - testOrder.indexOf(b.test_id) || repOrder.indexOf(a.representation) - repOrder.indexOf(b.representation)).map(row => ({
    key: `${row.test_id}-${row.representation}`, group: row.test_label, label: repLabel(data, row.representation),
    value: row.accuracy, color: repColor(row.representation),
    tip: [row.test_label, `${famLabel(data, row.family)} · ${repLabel(data, row.representation)}`,
      `Accuracy ${pct(row.accuracy, 0)}${finite(row.balanced_accuracy) ? ` · balanced ${pct(row.balanced_accuracy, 0)}` : ''}`,
      `${row.n_train} training (lot ${row.lots_train}) · ${row.n_test} test (lot ${row.lots_test})`,
      ...Object.entries(row).filter(([k]) => k.startsWith('recall__')).map(([k, v]) => `${k.slice(8)} recall ${pct(v as number, 0)} (n=${(row as unknown as Record<string, number>)[`n__${k.slice(8)}`]})`)] }))
  const counts = data?.lots.counts ?? []
  const lotIds = unique(counts.map(row => row.lot_group)).sort()
  const cycleSeason = unique(counts.map(row => `${row.cycle_end}|${row.season}`))
  const maxN = Math.max(1, ...counts.map(row => row.n))
  const countColor = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([0, maxN])
  const sep = (data?.lot_separability ?? []).filter(row => row.family === family)

  const fold2024 = fold === 'forward_2024'
  return <ChartFrame id="baseline-lab-ml-checks" title="Seasonal ML: checks for filter lot and FTIR EC"
    source={`${LAB_SOURCE}/SeasonalChecks.tsx › SeasonalMLChecks`}
    subtitle={view === 'ec' ? 'Every FTIR − HIPS model against a straight line on FTIR EC (no spectrum). Lower is better.'
      : view === 'regions' ? 'Grouped held-out permutation on corrected spectra. Positive = prediction worsens when the region is shuffled.'
        : 'Season classification trained on one lot (251) and tested on other years by lot.'}
    tip={<CheckNotes data={data} />}
    controls={<>
      <Segmented label="Check" value={view} options={[...VIEWS]} onChange={setView} optionLabel={v => VIEW_LABEL[v as typeof VIEWS[number]]} />
      {view !== 'lot' && <Select label="Test cycle" value={fold} options={folds} onChange={setFold} optionLabel={v => FOLD_LABEL[v] ?? v} />}
      {view === 'ec' && <Select label="Representation" value={rep} options={['all', ...reps]} onChange={setRep} optionLabel={v => v === 'all' ? 'All representations' : repLabel(data, v)} />}
      {view === 'regions' && <>
        <Segmented label="Target" value={target} options={[...CHECK_TARGETS]} onChange={v => setTarget(v as CheckTarget)} optionLabel={v => CHECK_TARGET_LABEL[v as CheckTarget]} />
        <Select label="Spectra" value={impRep} options={impReps} onChange={setImpRep} optionLabel={v => repLabel(data, v)} />
        {fold2024 && <Select label="Held-out season" value={season} options={['all', ...seasons]} onChange={setSeason} optionLabel={v => v === 'all' ? 'All held-out seasons' : v} />}
        <Segmented label="Regions" value={regionSet} options={['named', 'bins']} onChange={v => setSet(v as CheckRegionSet)} optionLabel={v => v === 'named' ? 'Named regions' : '100 cm⁻¹ bins'} />
      </>}
      {view === 'lot' && <Segmented label="Model" value={family} options={['linear', 'rf']} onChange={setFamily} optionLabel={v => v === 'linear' ? 'PCA + logistic' : 'Random forest'} />}
    </>}>
    {!data ? <LoadState what="seasonal ML checks" error={error} /> : view === 'ec' ? !ecRows.length ? <Empty>No saved EC-line comparison for this cycle.</Empty> : <>
      <div className="chart-note">{line ? `${line.n_train} training · ${line.n_test} held-out filters · EC line RMSE ${fmt(line.rmse, 3)} µg/m³ (dashed)` : ''} · pale bars fit D directly, solid bars add a spectral residual to the line</div>
      <div className="chart-wrap">
        <BarRows rows={ecRows} groupLabels reference={line?.rmse} axis={{ label: 'RMSE (µg/m³) · lower is better' }} format={v => fmt(v, 2)}
          rowHeight={Math.max(20, Math.min(30, (height - 60) / Math.max(1, ecRows.length + 4)))} />
      </div>
      <FixedSetNote what="the saved seasonal ML training and held-out physical-filter cohorts" />
    </> : view === 'regions' ? !imp.length ? <Empty>No saved region test for this selection{season !== 'all' ? ' (seasons are tested in the complete 2023–24 cycle only)' : ''}.</Empty> : <>
      <div className="chart-note">{CHECK_TARGET_LABEL[target]} · {repLabel(data, impRep)} · {imp[0].n_test} held-out filters{season !== 'all' ? ` (${shortSeason(season)})` : ''} · unshuffled RMSE {families.map(f => `${famLabel(data, f)} ${fmt(imp.find(r => r.family === f)?.baseline_loss, 2)}`).join(' / ')}</div>
      <div className="chart-wrap">
        {regionSet === 'bins' ? <SweepPlot series={sweep} xScale="linear" xDomain={xDomain} xLabel="Region midpoint (cm⁻¹)" yLabel="Change in RMSE when shuffled (µg/m³)"
          reference={0} height={height} yFormat={v => fmt(v, 2)} />
          : <BarRows rows={namedRows} reference={0} axis={{ label: 'Change in RMSE when shuffled (µg/m³)' }} format={v => fmt(v, 2)}
            rowHeight={Math.max(24, Math.min(40, (height - 45) / Math.max(1, namedRows.length)))} />}
        <div className="row" style={{ gap: 14, fontSize: 12 }}>{families.map(f => <span key={f} style={{ marginRight: 16 }}><span style={{ display: 'inline-block', width: 10, height: 10, background: FAMILY_COLOR[f], marginRight: 5 }} />{famLabel(data, f)}</span>)}</div>
      </div>
      <FixedSetNote what="the saved held-out region tests" />
    </> : <>
      <div className="chart-note">Lot by sampling cycle and season · filters on lot 248 exist only in the first training year</div>
      <div className="chart-wrap">
        <HeatMatrix rows={cycleSeason.map(k => { const [c, s] = k.split('|'); return { key: k, label: `${cycleLabel(Number(c))} · ${shortSeason(s)}`, title: s } })}
          columns={lotIds.map(l => ({ key: l, label: `Lot ${l}` }))} rowHeader="Cycle · season" colHeader="Filter lot"
          cellHeight={18} cellLabels="always" color={countColor}
          cell={(i, j) => { const [c, s] = cycleSeason[i].split('|'); const n = counts.find(r => `${r.cycle_end}|${r.season}` === cycleSeason[i] && r.lot_group === lotIds[j])?.n ?? 0
            return { value: n || null, label: n ? String(n) : '', tip: [`${cycleLabel(Number(c))} · ${s}`, `Lot ${lotIds[j]}: ${n} physical filters`] } }}
          legend={{ scale: countColor, scheme: colours.scheme, label: 'Physical filters', format: v => fmt(v, 0) }} />
        <BarRows rows={lotRows} groupLabels domain={[0, 1]} reference={1 / 3} axis={{ label: 'Season accuracy · higher is better' }} format={v => pct(v, 0)}
          rowHeight={22} />
        {!!sep.length && <div className="chart-note">Inside 2022–23 Belg, lot 248 vs 251 is told apart from the spectrum alone ({sep.map(r => `${repLabel(data, r.representation)} ${pct(r.balanced_accuracy, 0)}`).join(' · ')}, leave-one-out). Lot and date overlap there, so this is indicative only.</div>}
      </div>
      <FixedSetNote what="the saved lot-controlled season tests" />
    </>}
  </ChartFrame>
}
