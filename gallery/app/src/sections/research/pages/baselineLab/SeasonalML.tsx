import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select, Segmented } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { BarRows, type BarRow } from '@/components/BarRows'
import { HeatMatrix } from '@/components/HeatMatrix'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { SpectrumPlot, type SpectrumBand, type SpectrumSeries } from '@/components/SpectrumPlot'
import { Legend, useLegend } from '@/components/Legend'
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { useFitHeight } from '@/lib/chartFit'
import { useColourScheme } from '@/lib/colourSchemes'
import { FixedSetNote } from '@/lib/markedRows'
import { fmt, pct, signed } from '@/lib/format'
import { CONVENTION_LABEL, shortSeason } from '@/lib/seasons'
import { seasonColor } from '@/lib/siteColours'
import { baselineColor, INK } from '@/lib/theme'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { finite, labSpectrumValues, LAB_SOURCE } from '../../common/baselineLab'
import { chooseMLFold, mlConfusionMatrix, mlMetricChoices, mlRegionSet, mlScoreValue, selectMLImportance, selectMLProfiles, selectMLRows,
  validateSeasonalMLData, ML_CALENDARS, ML_COHORT_LABEL, ML_FAMILY_LABEL, ML_METRIC_LABEL, ML_TASK_LABEL, ML_TASKS,
  SEASONAL_ML_FILE, type MLCalendar, type MLFamily, type MLImportance, type MLImportanceKind, type MLKey,
  type MLMetric, type MLModel, type MLPermutationScheme, type MLRegionSet, type MLTask, type SeasonalMLData } from '../../common/seasonalML'
import type { LabChartProps } from './data'

const unique = <T,>(values: T[]) => [...new Set(values)]
const calendarLabel = (calendar: string) => CONVENTION_LABEL[calendar]?.split(' · ')[1] ?? calendar
const useSeasonalML = () => useExport<SeasonalMLData>(SEASONAL_ML_FILE, validateSeasonalMLData)
const representationLabel = (data: SeasonalMLData | null, id: string) => data?.representations.find(row => row.id === id)?.label
  ?? (id === 'none' ? 'Metadata / reference' : id.replace(/_/g, ' '))
const covariateLabel = (model: MLModel) => {
  if (Array.isArray(model.covariates)) return model.covariates.map(value => ({
    log_optical_bc_ugm3: 'log HIPS optical BC (part of the disagreement target)', log_pm25_ugm3: 'log PM2.5',
    log_volume_m3: 'log sampled volume', lot_group: 'recorded filter lot', rh: 'station RH', log_rain: 'log(1 + station rain)',
    season: 'calendar season',
  }[value] ?? value)).join(', ') || 'None'
  if (!model.covariates || model.covariates === 'none') return 'None'
  if (model.covariates.startsWith('loading_lot')) return ['HIPS optical BC (part of the disagreement target)', 'PM2.5', 'sampled volume', 'recorded filter lot',
    ...(model.covariates.includes('weather') ? ['station RH and rainfall'] : []), ...(model.covariates.includes('season') ? ['calendar season'] : [])].join(', ')
  return model.covariates
}
const modelLabel = (model: MLModel) => model.label.replace(/loading(?:\s*\+\s*|\s+and\s+|_)lot/g, 'HIPS + PM2.5 + volume + lot')
const hasRepeatInterval = (row: MLImportance) => row.repeats > 1 && finite(row.q05) && finite(row.q95)
const modelColor = (model: MLModel) => /removed/.test(model.representation) ? INK.muted
  : /spline|corrected/.test(model.representation) ? baselineColor('AIRSpec')
    : /deriv/.test(model.representation) ? RESEARCH_SERIES[2]
      : /raw/.test(model.representation) ? INK.text : RESEARCH_SERIES[4]

/** Shared selectors name one saved evaluation. Samples never silently refits or changes these scores. */
function useMLEvaluation(data: SeasonalMLData | null) {
  const [task, setTask] = useSearchParam<MLTask>('blab.ml.task', 'season', ML_TASKS)
  const [calendar, setCalendar] = useSearchParam<MLCalendar>('blab.ml.calendar', 'belg_feb', ML_CALENDARS)
  const [wantedCohort, setCohort] = useSearchParam<string>('blab.ml.cohort', 'direct232')
  const [wantedFold, setFold] = useSearchParam<string>('blab.ml.fold', '')
  const byTask = selectMLRows(data?.scores ?? [], { task, calendar })
  const cohorts = unique(byTask.map(row => row.cohort))
  const cohort = cohorts.includes(wantedCohort) ? wantedCohort : cohorts[0] ?? wantedCohort
  const candidates = selectMLRows(byTask, { cohort })
  const availableFolds = unique(candidates.map(row => row.fold_id))
  const fold = chooseMLFold(data?.folds ?? [], availableFolds, wantedFold)
  const selection: Partial<MLKey> = { task, calendar, cohort, fold_id: fold?.id ?? '' }
  const scores = selectMLRows(candidates, selection)
  const controls = <>
    <Select label="Task" value={task} options={[...ML_TASKS]} onChange={v => setTask(v as MLTask)} optionLabel={v => ML_TASK_LABEL[v as MLTask]} />
    <Select label="Calendar" value={calendar} options={[...ML_CALENDARS]} onChange={v => setCalendar(v as MLCalendar)} optionLabel={calendarLabel} />
    <Select label="Cohort" value={cohort} options={cohorts} onChange={setCohort} optionLabel={v => ML_COHORT_LABEL[v] ?? v} />
    <Select label="Test cycle" value={fold?.id ?? ''} options={availableFolds} onChange={setFold}
      optionLabel={v => data?.folds.find(f => f.id === v)?.label ?? v} />
  </>
  const coverage = fold ? `${scores[0]?.n_train ?? fold.n_train} training · ${scores[0]?.n_test ?? fold.n_test} held-out physical filters · ${fold.complete_test
    ? 'all 12 test months sampled' : `incomplete test months${fold.missing_test_months.length ? ` (${fold.missing_test_months.join(', ')} missing)` : ''}`}` : ''
  return { task, calendar, cohort, fold, selection, scores, controls, coverage }
}

function MLNotes({ data }: { data: SeasonalMLData | null }) {
  return <>
    <p>These are saved predictions from models evaluated on later physical filters. Individual repeat scans never become separate train/test observations. Only one test cycle has all twelve collection months represented; the later sparse cycle stays separate.</p>
    <p>Season classifiers receive neither month, date nor cycle as predictors. Their inputs are spectra or the explicitly labelled loading and lot metadata. FTIR − optical prediction measures disagreement: its target already contains FTIR EC, which is derived from spectra. Predicting that target does not validate EC accuracy.</p>
    <p>The loading/lot models use HIPS optical BC, PM2.5, sampled volume and recorded filter lot. HIPS is itself part of FTIR − HIPS: these are HIPS-conditioned diagnostics. Weather variants add station RH and rainfall. Cohort names describe the available pool; each selected fold reports the filters that actually participate.</p>
    <p>Raw full-spectrum and raw common-window models distinguish spectral coverage from a baseline-method change. Corrected and removed spectra use supported channels; padded spline zeros are not measurements.</p>
    {data?.limitations.map((text, i) => <p key={i}>{text}</p>)}
    {data?.generated_at && <p>Saved analysis: {data.generated_at}{data.source_revision ? ` · ${data.source_revision}` : ''}.</p>}
  </>
}

/** Held-out model scores and the classifier's actual × predicted season counts. */
export function SeasonalMLPerformance(_props: LabChartProps) {
  const { data, error } = useSeasonalML()
  const state = useMLEvaluation(data)
  const height = useFitHeight() ?? 360
  const [wantedView, setView] = useSearchParam<string>('blab.ml.performance.view', 'scores', ['scores', 'confusion'])
  const [wantedFamily, setFamily] = useSearchParam<string>('blab.ml.performance.family', 'rf')
  const [wantedRepresentation, setRepresentation] = useSearchParam<string>('blab.ml.performance.representation', 'all')
  const [wantedMetric, setMetric] = useSearchParam<string>('blab.ml.performance.metric', 'balanced_accuracy')
  const [wantedModel, setModel] = useSearchParam<string>('blab.ml.model', 'rf_raw_full')
  const colours = useColourScheme('blab.ml.confusion', 'sequential')
  const view = state.task === 'season' ? wantedView : 'scores'
  const availableModels = data?.models.filter(model => state.scores.some(score => score.model_id === model.id)) ?? []
  const families = unique(availableModels.map(model => model.family))
  const family = families.includes(wantedFamily as MLFamily) || wantedFamily === 'all' ? wantedFamily : families[0] ?? wantedFamily
  const byFamily = availableModels.filter(model => family === 'all' || model.family === family)
  const representations = unique(byFamily.map(model => model.representation))
  const representation = wantedRepresentation === 'all' || representations.includes(wantedRepresentation) ? wantedRepresentation : 'all'
  const models = byFamily.filter(model => representation === 'all' || model.representation === representation)
  const model = models.find(item => item.id === wantedModel) ?? models.find(item => item.id === 'rf_raw_full') ?? models[0]
  const metrics = mlMetricChoices(state.task)
  const metric = metrics.includes(wantedMetric as MLMetric) ? wantedMetric as MLMetric : metrics[0]
  const scores = state.scores.filter(score => models.some(model => model.id === score.model_id))
  const bars: BarRow[] = scores.flatMap(score => {
    const value = mlScoreValue(score, metric)
    const model = models.find(item => item.id === score.model_id)
    if (!model || value === null) return []
    const rep = data?.representations.find(item => item.id === model.representation)
    return [{ key: model.id, label: modelLabel(model), value, color: modelColor(model), valueLabel: metric === 'r2' ? null : fmt(value, 3),
      tip: [modelLabel(model), `${ML_METRIC_LABEL[metric]}: ${fmt(value, 3)}`, `${score.n_train} training · ${score.n_test} test filters`,
        state.fold?.label ?? score.fold_id, `${representationLabel(data, model.representation)}${rep ? ` · ${rep.n_channels} channels` : ''}`,
        `Additional predictors: ${covariateLabel(model)}`, ...(model.label_note ? [model.label_note] : []), ...(state.fold?.note ? [state.fold.note] : [])] }]
  })
  const confusionRows = selectMLRows(data?.confusion ?? [], { ...state.selection, model_id: model?.id ?? '' })
  const confusion = useMemo(() => mlConfusionMatrix(confusionRows), [confusionRows])
  const actualClasses = new Set(confusionRows.filter(row => row.n > 0).map(row => row.actual)).size
  const color = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([0, 1])
  return <ChartFrame id="baseline-lab-ml-performance" title="Seasonal ML: held-out performance"
    source={`${LAB_SOURCE}/SeasonalML.tsx › SeasonalMLPerformance`}
    subtitle="Later-cycle tests on physical filters. Classification identifies seasonal structure; FTIR − optical prediction measures disagreement."
    tip={<><MLNotes data={data} /><p>Balanced accuracy gives each observed season equal weight. Confusion cells show the share of each actual season assigned to each predicted season, with the filter count below. The one-third reference applies when all three seasons are represented; it is not a permutation-test result.</p></>}
    controls={<>{state.controls}
      {state.task === 'season' && <Segmented label="View" value={view} options={['scores', 'confusion']} onChange={setView} />}
      <Select label="Model family" value={family} options={[...families, 'all']} onChange={setFamily}
        optionLabel={v => v === 'all' ? 'All families' : ML_FAMILY_LABEL[v as MLFamily] ?? v} />
      <Select label="Representation" value={representation} options={['all', ...representations]} onChange={setRepresentation}
        optionLabel={v => v === 'all' ? 'All representations' : representationLabel(data, v)} />
      {view === 'scores' ? <Select label="Metric" value={metric} options={metrics} onChange={setMetric} optionLabel={v => ML_METRIC_LABEL[v as MLMetric]} />
        : <Select label="Model" value={model?.id ?? ''} options={models.map(item => item.id)} onChange={setModel} optionLabel={v => { const item = models.find(item => item.id === v); return item ? modelLabel(item) : v }} />}
    </>}>
    {!data ? <LoadState what="seasonal ML results" error={error} /> : !state.fold || !scores.length ? <Empty>No saved model evaluation for this task, calendar and cohort.</Empty> : <>
      <div className="chart-note">{state.coverage}</div>
      {view === 'scores' ? !bars.length ? <Empty>This metric is unavailable for the selected models.</Empty> : <div className="chart-wrap">
        <BarRows rows={bars} domain={state.task === 'season' ? [0, 1] : undefined}
          reference={metric === 'balanced_accuracy' && actualClasses === 3 ? 1 / 3 : undefined}
          axis={{ label: `${ML_METRIC_LABEL[metric]} · ${state.task === 'season' || metric === 'r2' ? 'higher is better' : 'lower is better'}` }}
          format={v => fmt(v, 2)} rowHeight={Math.max(23, Math.min(44, (height - 45) / Math.max(1, bars.length)))} />
      </div> : !confusionRows.length ? <Empty>No saved held-out confusion counts for this model.</Empty> : <div className="chart-wrap">
        <HeatMatrix rows={confusion.classes.map(name => ({ key: name, label: shortSeason(name), title: name }))}
          columns={confusion.classes.map(name => ({ key: name, label: shortSeason(name), title: name }))}
          rowHeader="Actual season" colHeader="Predicted season" cellHeight={Math.max(42, Math.min(85, (height - 95) / 3))}
          fitRows={[34, 95]} cellLabels="always" center color={color}
          cell={(i, j) => { const cell = confusion.cells[i][j]; return { value: cell.fraction,
            label: cell.fraction == null ? '—' : pct(cell.fraction, 0), sublabel: `n=${cell.n}`,
            tip: [`Actual: ${confusion.classes[i]}`, `Predicted: ${confusion.classes[j]}`, `${cell.n} of ${cell.total} actual-class filters`,
              cell.fraction == null ? 'No observations of this actual season' : `${pct(cell.fraction, 1)} of the actual class`, model ? modelLabel(model) : '', state.fold?.label ?? ''] } }}
          legend={{ scale: color, scheme: colours.scheme, label: 'Share of the actual season', format: v => pct(v, 0) }} />
      </div>}
      <FixedSetNote what="the saved seasonal ML training and held-out physical-filter cohorts" />
    </>}
  </ChartFrame>
}

const KIND_LABEL: Record<MLImportanceKind, string> = { permutation: 'Grouped permutation', ablation: 'Leave region out · refit' }
const SCHEME_LABEL: Record<MLPermutationScheme, string> = { across_filters: 'Across test filters', within_season: 'Within test season' }
const REGION_SET_LABEL: Record<MLRegionSet, string> = { bins: '100 cm⁻¹ intervals', named: 'Named broad regions', whole: 'Whole included spectrum' }
const regionLabel = (row: MLImportance) => row.region_id === 'whole_spectrum' ? REGION_SET_LABEL.whole
  : /^bin_[\d.]+_[\d.]+$/.test(row.region_label) ? `${row.region_label.replace(/^bin_/, '').replace('_', '–')} cm⁻¹` : row.region_label

/** Importance is a held-out loss change, not a composition fraction or a causal source attribution. */
export function SeasonalMLRegions({ meta }: LabChartProps) {
  const { data, error } = useSeasonalML()
  const state = useMLEvaluation(data)
  const height = useFitHeight() ?? 360
  const [content, setContent] = useSearchParam<string>('blab.ml.regions.content', 'tests', ['tests', 'spectra'])
  const [wantedProfile, setProfile] = useSearchParam<string>('blab.ml.profiles.representation', 'raw_full')
  const profileLegend = useLegend()
  const [wantedModel, setModel] = useSearchParam<string>('blab.ml.model', 'rf_raw_full')
  const [wantedKind, setKind] = useSearchParam<MLImportanceKind>('blab.ml.regions.kind', 'permutation', ['permutation', 'ablation'])
  const [wantedScheme, setScheme] = useSearchParam<MLPermutationScheme>('blab.ml.regions.scheme', 'across_filters', ['across_filters', 'within_season'])
  const [wantedSeason, setSeason] = useSearchParam<string>('blab.ml.regions.season', 'all')
  const [wantedSet, setRegionSet] = useSearchParam<MLRegionSet>('blab.ml.regions.set', 'bins', ['bins', 'named', 'whole'])
  const [wantedView, setView] = useSearchParam<string>('blab.ml.regions.view', 'spectrum', ['spectrum', 'rows'])
  const candidates = selectMLRows(data?.importance ?? [], state.selection)
  const modelIds = unique(candidates.map(row => row.model_id))
  const models = data?.models.filter(model => modelIds.includes(model.id)) ?? []
  const model = models.find(item => item.id === wantedModel) ?? models.find(item => item.id === 'rf_raw_full') ?? models[0]
  const byModel = candidates.filter(row => row.model_id === model?.id)
  const seasons = unique(byModel.map(row => row.season).filter((value): value is string => typeof value === 'string' && !!value)).sort()
  const season = state.task === 'disagreement' && seasons.includes(wantedSeason) ? wantedSeason : 'all'
  const bySeason = byModel.filter(row => season === 'all' ? !row.season : row.season === season)
  const kinds = unique(bySeason.map(row => row.kind))
  const kind = kinds.includes(wantedKind) ? wantedKind : kinds[0] ?? wantedKind
  const schemes = unique(bySeason.filter(row => row.kind === kind).map(row => row.scheme))
  const scheme = schemes.includes(wantedScheme) ? wantedScheme : schemes[0] ?? wantedScheme
  const allSelected = selectMLImportance(candidates, { model_id: model?.id ?? '' }, kind, scheme, season === 'all' ? null : season)
  const regionSets = unique(allSelected.map(mlRegionSet))
  const regionSet = regionSets.includes(wantedSet) ? wantedSet : regionSets.find(value => value !== 'whole') ?? regionSets[0] ?? wantedSet
  const selected = allSelected.filter(row => mlRegionSet(row) === regionSet)
  const trainN = state.scores.find(row => row.model_id === model?.id)?.n_train ?? state.fold?.n_train
  const coverage = season === 'all' ? state.coverage
    : `${trainN ?? '—'} pooled-training physical filters · ${selected[0]?.n_test ?? '—'} ${shortSeason(season)} held-out filters · ${state.fold?.label ?? ''}`
  const spectral = selected.filter(row => mlRegionSet(row) !== 'whole' && finite(row.low) && finite(row.high))
  const view = wantedView === 'spectrum' && !spectral.length && selected.length ? 'rows' : wantedView
  const scale = state.task === 'season' ? 100 : 1
  const units = state.task === 'season' ? 'percentage points' : 'µg/m³'
  const axis = state.task === 'season' ? 'Change in balanced error (percentage points)' : 'Change in RMSE (µg/m³)'
  const scaled = (value: number | null | undefined) => finite(value) ? value * scale : null
  const tip = (row: MLImportance) => [regionLabel(row),
    `${signed(scaled(row.delta_mean), 3)} ${units} · positive means worse prediction`,
    ...(finite(row.low) && finite(row.high) ? [`${fmt(row.low, 0)}–${fmt(row.high, 0)} cm⁻¹ · ${row.n_channels} channels`] : [`${row.n_channels} predictors`]),
    `${KIND_LABEL[row.kind]} · ${SCHEME_LABEL[row.scheme]}`,
    ...(row.season ? [`${row.season} · ${row.n_test ?? '—'} held-out filters`, `Same pooled model trained on ${trainN ?? '—'} filters; no season-specific tuning.`] : []),
    ...(row.repeats > 1 ? [`${row.repeats} perturbations`,
      ...(hasRepeatInterval(row) ? [`5th–95th repeat percentiles: ${signed(scaled(row.q05), 3)} to ${signed(scaled(row.q95), 3)} ${units}`] : []),
      ...(finite(row.delta_sd) ? [`Repeat SD: ${fmt(scaled(row.delta_sd), 3)} ${units}`] : []), 'Repeat spread is not a confidence interval.'] : []),
    `Unperturbed ${row.metric === 'rmse' ? 'RMSE' : 'balanced error'}: ${row.metric === 'rmse' ? `${fmt(row.baseline_loss, 3)} µg/m³` : pct(row.baseline_loss, 1)}`,
    ...(model ? [modelLabel(model), `Additional predictors: ${covariateLabel(model)}`] : []), state.fold?.label ?? row.fold_id]
  const bars: BarRow[] = selected.map(row => ({ key: row.region_id, label: regionLabel(row), value: scaled(row.delta_mean),
    color: model ? modelColor(model) : INK.accent, valueLabel: null, tip: tip(row),
    marks: hasRepeatInterval(row) ? [scaled(row.q05), scaled(row.q95)].filter(finite).map(value => ({ value, color: INK.muted })) : undefined }))
  const series: SweepSeries[] = model ? [{ id: model.id, label: modelLabel(model), color: modelColor(model), line: false, marks: true,
    interval: 'whisker', r: 4, points: spectral.map(row => ({ x: (row.low! + row.high!) / 2, y: scaled(row.delta_mean),
      lo: hasRepeatInterval(row) ? scaled(row.q05) : null, hi: hasRepeatInterval(row) ? scaled(row.q95) : null, tip: tip(row) })) }] : []
  const limits = d3.extent(spectral.flatMap(row => [row.low!, row.high!]))
  const domain: [number, number] | undefined = finite(limits[0]) && finite(limits[1]) && limits[0] < limits[1] ? [limits[1], limits[0]] : undefined
  const profileChoices = unique(data?.seasonal_spectra?.map(row => row.representation) ?? [])
  const profileRepresentation = profileChoices.includes(wantedProfile) ? wantedProfile : profileChoices[0] ?? wantedProfile
  const profiles = selectMLProfiles(data?.seasonal_spectra ?? [], profileRepresentation)
  const storedCalendar = (data?.provenance.settings as { season_primary?: MLCalendar } | undefined)?.season_primary
  const profileCalendar = storedCalendar && ML_CALENDARS.includes(storedCalendar) ? storedCalendar : 'belg_feb'
  const seasonMeta = { seasons: meta.season_conventions[profileCalendar] ?? meta.seasons }
  const profileN = profiles.reduce((sum, row) => sum + row.n, 0)
  const descriptiveOnly = data?.counts.descriptive_only_filters
  const visibleProfiles = profiles.filter(row => profileLegend.show(row.season))
  const profileSeries: SpectrumSeries[] = visibleProfiles.map(row => ({ id: `${row.representation}-${row.season}`, label: row.season,
    group: row.season, wn: row.wavenumbers, y: labSpectrumValues(row.median), color: seasonColor(seasonMeta, row.season), role: 'line', width: 2,
    tip: [row.season, `${row.n} physical filters · full descriptive cohort`, representationLabel(data, row.representation), 'Signed channel median; shading is the interquartile range.'] }))
  const profileBands: SpectrumBand[] = visibleProfiles.map(row => ({ id: `${row.representation}-${row.season}-iqr`, group: row.season,
    wn: row.wavenumbers, lo: labSpectrumValues(row.q25), hi: labSpectrumValues(row.q75), color: seasonColor(seasonMeta, row.season), opacity: .13 }))
  return <ChartFrame id="baseline-lab-ml-regions" title="Seasonal ML: spectral regions and profiles"
    source={`${LAB_SOURCE}/SeasonalML.tsx › SeasonalMLRegions`}
    subtitle={content === 'spectra' ? 'Signed seasonal medians and interquartile ranges from the full descriptive cohort. Saved profiles retain their original calendar.'
      : 'Positive change means prediction worsens when a region is perturbed or removed. Negative results remain visible.'}
    tip={<><p>Seasonal spectra use the full available direct-HIPS descriptive cohort, including the latest Bega filters outside model evaluation. Their counts differ from the held-out test folds. They retain their signed values and the primary calendar recorded with the analysis; changing an ML fold does not change these saved profiles.</p><MLNotes data={data} /><p>Grouped permutation shuffles a region's channels together in the held-out filters. Within-season shuffling, when available, preserves the broad seasonal grouping. Leave-region-out comparisons refit the model using its training filters before testing again.</p>
      <p>Whiskers or end ticks are the 5th–95th percentiles across repeated perturbations, not confidence intervals. A large value identifies information used by this model. Correlated or overlapping regions can substitute for one another; these values cannot be added into chemical source fractions.</p>
      <p>The interval, broad-region and whole-spectrum views are separate tests. Whole-spectrum importance is an overall control and is shown as a row, never as a localized effect at the midpoint of the spectrum.</p>
      <p>The disagreement Season selector, where available, partitions the held-out filters while retaining the same model trained on all training seasons. It shuffles within the selected test season; no separate seasonal model is tuned. Its displayed count is the season subset, not the full test cycle.</p>
      <p>For disagreement, the target contains FTIR EC(X). Importance can therefore recover information already used by that product. It does not establish a causal baseline error or validate a corrected EC estimate. Regions outside a representation's support are unavailable, never assigned zero importance. Hover for exact window boundaries, channel counts and the unperturbed loss.</p></>}
    controls={<><Segmented label="View" value={content} options={['tests', 'spectra']} onChange={setContent}
      optionLabel={v => v === 'tests' ? 'Region tests' : 'Seasonal spectra'} />
      {content === 'spectra' ? <Select label="Representation" value={profileRepresentation} options={profileChoices} onChange={setProfile}
        optionLabel={v => representationLabel(data, v)} /> : <>{state.controls}
      <Select label="Model" value={model?.id ?? ''} options={models.map(item => item.id)} onChange={setModel} optionLabel={v => { const item = models.find(item => item.id === v); return item ? modelLabel(item) : v }} />
      {state.task === 'disagreement' && <Select label="Held-out season" value={season} options={['all', ...seasons]} onChange={setSeason}
        optionLabel={v => v === 'all' ? 'All held-out seasons' : v} />}
      <Select label="Test" value={kind} options={kinds} onChange={v => setKind(v as MLImportanceKind)} optionLabel={v => KIND_LABEL[v as MLImportanceKind]} />
      {kind === 'permutation' && <Select label="Shuffle" value={scheme} options={schemes} onChange={v => setScheme(v as MLPermutationScheme)} optionLabel={v => SCHEME_LABEL[v as MLPermutationScheme]} />}
      <Select label="Region set" value={regionSet} options={regionSets} onChange={v => setRegionSet(v as MLRegionSet)} optionLabel={v => REGION_SET_LABEL[v as MLRegionSet]} />
      {regionSet !== 'whole' && <Segmented label="Layout" value={view} options={['spectrum', 'rows']} onChange={setView} optionLabel={v => v === 'spectrum' ? 'Spectral position' : 'Region rows'} />}
      </>}
    </>}>
    {!data ? <LoadState what="spectral ML results" error={error} /> : content === 'spectra' ? !profiles.length ? <Empty>No saved seasonal spectra for this representation.</Empty> : <>
      <div className="chart-note">Descriptive cohort · {profileN} physical filters · {profileCalendar} · {profiles.map(row => `${shortSeason(row.season)} ${row.n}`).join(' / ')}
        {finite(descriptiveOnly) ? ` · includes ${descriptiveOnly} filters outside model folds` : ''}</div>
      <div className="chart-wrap">
        {visibleProfiles.length ? <SpectrumPlot series={profileSeries} bands={profileBands} height={height} hover="readout" zeroLine includeZero
          dim={profileLegend.dim} yLabel={profileRepresentation.includes('derivative') ? 'Second derivative [absorbance / (cm⁻¹)²]' : 'Signed absorbance'} />
          : <Empty>Choose a season in the legend to show its profile.</Empty>}
        <Legend items={profiles.map(row => ({ label: row.season, color: seasonColor(seasonMeta, row.season) }))} {...profileLegend.props} />
      </div>
      <FixedSetNote what="the full descriptive seasonal spectra, whose membership is separate from the selected ML test fold" />
    </> : !selected.length ? <Empty>No saved held-out region test for this model, task and cycle.</Empty> : <>
      <div className="chart-note">{coverage} · {selected.length} region tests{selected.some(hasRepeatInterval) ? ' · marks show repeat spread' : ''}</div>
      <div className="chart-wrap">
        {view === 'rows' ? <BarRows rows={bars} axis={{ label: axis }} reference={0} format={v => fmt(v, 2)}
          rowHeight={Math.max(22, Math.min(40, (height - 45) / Math.max(1, bars.length)))} />
          : <SweepPlot series={series} xScale="linear" xDomain={domain} xLabel="Region midpoint (cm⁻¹)" yLabel={axis}
            reference={0} height={height} yFormat={v => fmt(v, 2)} />}
      </div>
      {view === 'spectrum' && spectral.length < selected.length && <div className="chart-note">Non-spectral predictors are available in Region rows.</div>}
      <FixedSetNote what="the saved ML region tests on the stated held-out filters" />
    </>}
  </ChartFrame>
}
