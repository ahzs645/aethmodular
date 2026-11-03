import { useMemo, useRef } from 'react'
import { ChartFrame, Empty, Select, Segmented } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { ScatterPlot, type ScatterPoint } from '@/components/ScatterPlot'
import { SpectrumPlot, type SpectrumBand, type SpectrumSeries } from '@/components/SpectrumPlot'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { Legend, useLegend } from '@/components/Legend'
import { BarRows, type BarRow } from '@/components/BarRows'
import { IntervalStrip } from '@/components/IntervalStrip'
import { useSearchParam } from '@/lib/url'
import { useFitHeight } from '@/lib/chartFit'
import { FixedSetNote, useExclusionSplit } from '@/lib/markedRows'
import { fmt, pct, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { clipText, labelGutter } from '@/lib/fitLabel'
import { useDimensions } from '@/hooks/useGalleryData'
import { finite, featureLabel, labCalibrationRows, labChronologyRows, labInjectionRows, labRepeatValues, labSpectrumValues, LAB_SOURCE, type LabBlank, type LabRepeat } from '../../common/baselineLab'
import { labMethodColor, labMethodLabel, labSiteName, LabSiteSelect, useLabExport, useLabSelection, useLabSite, type LabChartProps } from './data'

const CONTROL_VIEWS = ['blanks', 'injections', 'repeats', 'blank spectra'] as const
const BLANK_METRICS = ['corrected_rms', 'corrected_1617', 'negative_point_fraction'] as const
const blankLabel = (key: string) => ({ corrected_rms: 'Corrected RMS (absorbance)', corrected_1617: 'Signed absorbance at 1617',
  negative_point_fraction: 'Fraction of negative channels' }[key] ?? key)
const NO_BLANKS: LabBlank[] = []
const NO_REPEATS: LabRepeat[] = []

/** Existing-data controls: methods are compared only where the export contains a completed run. */
export function BaselineControls(_props: LabChartProps) {
  const { data, error } = useLabExport()
  const [view, setView] = useSearchParam<typeof CONTROL_VIEWS[number]>('blab.controls.view', 'injections', CONTROL_VIEWS)
  const [metric, setMetric] = useSearchParam<typeof BLANK_METRICS[number]>('blab.controls.metric', 'corrected_rms', BLANK_METRICS)
  const [wantedControl, setControl] = useSearchParam<string>('blab.controls.injection', 'broad')
  const [curveKind, setCurveKind] = useSearchParam<string>('blab.controls.signal', 'corrected', ['raw', 'corrected', 'removed'])
  const [wantedSplit, setSplit] = useSearchParam<string>('blab.controls.split', '')
  const [repeatType, setRepeatType] = useSearchParam<string>('blab.controls.repeatType', 'PM2.5', ['PM2.5', 'FB'])
  const [repeatMetric, setRepeatMetric] = useSearchParam<string>('blab.controls.repeatMetric', 'relative', ['relative', 'absolute'])
  const relativeRepeat = repeatType !== 'FB' && repeatMetric === 'relative'
  const [, selectFilter] = useLabSelection()
  const height = useFitHeight() ?? 345
  const lg = useLegend()
  const blanks = useExclusionSplit(data?.blanks ?? NO_BLANKS, r => r.id)
  const repeats = useExclusionSplit(data?.repeats ?? NO_REPEATS, r => r.id)
  const splits = [...new Set(data?.blanks.map(r => r.split ?? 'Recorded controls') ?? [])]
  const split = splits.includes(wantedSplit) ? wantedSplit : splits.find(s => /held|leave|oof/i.test(s)) ?? splits[0] ?? wantedSplit
  const controls = [...new Set(data?.injections.map(r => r.control) ?? [])]
  const control = controls.includes(wantedControl) ? wantedControl
    : controls.find(s => /held-out blank.*broad/i.test(s)) ?? controls.find(s => /broad/i.test(s)) ?? controls[0] ?? wantedControl
  const methods = [...new Set((view === 'blanks' ? data?.blanks.filter(r => (r.split ?? 'Recorded controls') === split)
    : view === 'repeats' ? data?.repeats?.filter(r => (r.filter_type ?? 'PM2.5') === repeatType)
      : data?.injections.filter(r => r.control === control))?.map(r => r.method) ?? [])]
  const showMethods = methods.filter(m => data && lg.show(labMethodLabel(data, m)))
  const points = useMemo<ScatterPoint[]>(() => {
    if (!data) return []
    if (view === 'blanks') return [...blanks.counted, ...blanks.marked].filter(r => finite(r[metric]) && showMethods.includes(r.method) && (r.split ?? 'Recorded controls') === split).map((r, i) => ({
      id: r.id, key: `${r.id}-${r.method}`, x: showMethods.indexOf(r.method) + .5, dx: ((i % 9) - 4) * 2.2, y: r[metric]!,
      color: labMethodColor(data, r.method), dim: lg.dim(labMethodLabel(data, r.method)),
      tip: [r.id, `${labMethodLabel(data, r.method)} · lot ${r.lot ?? 'unknown'}`, `${blankLabel(metric)}: ${fmt(r[metric], 6)}`, ...(r.split ? [r.split] : []),
        ...(r.converged === false ? ['Solver did not converge; result retained'] : []), ...(r.convergence_note ? [r.convergence_note] : [])],
    }))
    if (view === 'repeats') return [...repeats.counted, ...repeats.marked]
      .filter(r => (r.filter_type ?? 'PM2.5') === repeatType && labRepeatValues(r, relativeRepeat) !== null && showMethods.includes(r.method))
      .map(r => ({ id: r.id, key: `${r.id}-${r.method}-${r.analysis_a}-${r.analysis_b}`, ...labRepeatValues(r, relativeRepeat)!,
        color: labMethodColor(data, r.method),
        dim: lg.dim(labMethodLabel(data, r.method)),
        tip: [r.id, `${labMethodLabel(data, r.method)} · scans ${r.analysis_a ?? '—'} / ${r.analysis_b ?? '—'}`,
          `Raw ${relativeRepeat ? 'relative ' : ''}difference RMS: ${fmt(labRepeatValues(r, relativeRepeat)!.x, 6)}`,
          `Corrected ${relativeRepeat ? 'relative ' : ''}difference RMS: ${fmt(labRepeatValues(r, relativeRepeat)!.y, 6)}`,
          ...(r.n_pairs ? [`${r.n_pairs} acquisition pair${r.n_pairs === 1 ? '' : 's'}`] : [])],
      }))
    return []
  }, [data, view, blanks, repeats, metric, split, repeatType, relativeRepeat, showMethods.join('|'), lg.dim]) // eslint-disable-line react-hooks/exhaustive-deps
  const injectionSeries = useMemo<SweepSeries[]>(() => !data ? [] : showMethods.map(method => ({
    id: method, label: labMethodLabel(data, method), group: labMethodLabel(data, method), color: labMethodColor(data, method), marks: true,
    points: labInjectionRows(data.injections, control, method).map(r => ({
      x: r.amplitude, y: r.recovery_fraction, lo: r.recovery_q25, hi: r.recovery_q75,
      tip: [labMethodLabel(data, method), control, `Injected amplitude: ${fmt(r.amplitude, 5)} absorbance`,
        `Recovery: ${pct(r.recovery_fraction, 1)}`, `${r.n_parents} independent parent spectra`,
        ...(finite(r.n_finite) ? [`${r.n_finite}/${r.n_parents} finite parent results`] : []),
        ...(finite(r.n_converged) ? [`${r.n_converged}/${r.n_parents} injection pairs converged`] : []),
        ...(finite(r.recovery_converged) ? [`Converged-pair median recovery: ${pct(r.recovery_converged, 1)}`] : []),
        ...(finite(r.recovery_q25) && finite(r.recovery_q75) ? [`Parent IQR: ${pct(r.recovery_q25, 1)}–${pct(r.recovery_q75, 1)}`] : [])],
    })),
  })).filter(s => s.points.length), [data, control, showMethods.join('|')]) // eslint-disable-line react-hooks/exhaustive-deps
  const aggregate = data?.curves?.filter(c => c.kind === curveKind) ?? []
  const spectrumSeries: SpectrumSeries[] = data ? aggregate.filter(c => lg.show(c.label)).map(c => ({
    id: c.id, label: c.label, group: c.label, wn: c.wn, y: labSpectrumValues(c.y), color: labMethodColor(data, c.method), role: 'line',
    dash: c.group === 'FB' ? '5 3' : undefined, tip: [c.label, `${c.n ?? '—'} physical filters`, 'Median signed absorbance'],
  })) : []
  const spectrumBands: SpectrumBand[] = data ? aggregate.filter(c => c.lo && c.hi && lg.show(c.label)).map(c => ({
    id: `${c.id}-iqr`, group: c.label, wn: c.wn, lo: labSpectrumValues(c.lo!), hi: labSpectrumValues(c.hi!), color: labMethodColor(data, c.method), opacity: .1,
  })) : []
  const hasChart = view === 'injections' ? injectionSeries.length > 0 : view === 'blank spectra' ? spectrumSeries.length > 0 : points.length > 0
  return <ChartFrame id="baseline-lab-controls" title="Baseline controls"
    source={`${LAB_SOURCE}/Experiments.tsx › BaselineControls`}
    subtitle="Blank residuals, known-signal recovery and original repeat acquisitions. Lower blank RMS alone does not establish analyte preservation."
    tip={<><p>All values are from completed existing-data experiments. Blank selection and model fitting must be assessed separately from their held-out controls. A field blank can contain handling or environmental material.</p><p>Injection recovery compares the correction of a parent spectrum with the correction of the same parent plus a known signal. Recovery of one injected shape does not establish ambient EC accuracy. Reused amplitudes do not increase the number of independent parents.</p><p>Repeat differences are grouped by physical filter; a near-flat acquisition remains visible. Inspect scan identity before changing a canonical mean. A larger relative RMS after correction can reflect a smaller normalization denominator; compare absolute RMS before concluding that repeatability worsened.</p></>}
    controls={<><Select label="Control" value={view} options={[...CONTROL_VIEWS]} onChange={v => setView(v as typeof CONTROL_VIEWS[number])} />
      {view === 'blanks' && <Select label="Metric" value={metric} options={[...BLANK_METRICS]} onChange={v => setMetric(v as typeof BLANK_METRICS[number])} optionLabel={blankLabel} />}
      {view === 'blanks' && splits.length > 1 && <Select label="Blank origin / validation" value={split} options={splits} onChange={setSplit} optionLabel={v => v.replace(/_/g, ' ')} />}
      {view === 'injections' && <Select label="Injected shape" value={control} options={controls} onChange={setControl} optionLabel={v => v.replace(/_/g, ' ')} />}
      {view === 'blank spectra' && <Segmented label="Signal" value={curveKind} options={['raw', 'corrected', 'removed']} onChange={setCurveKind} />}
      {view === 'repeats' && <><Segmented label="Filter type" value={repeatType} options={['PM2.5', 'FB']} onChange={setRepeatType}
        optionLabel={v => v === 'FB' ? 'Field blanks' : v} />
        <Select label="Repeat metric" value={relativeRepeat ? 'relative' : 'absolute'} options={repeatType === 'FB' ? ['absolute'] : ['relative', 'absolute']}
          onChange={setRepeatMetric} optionLabel={v => v === 'absolute' ? 'Absolute RMS' : 'Relative RMS'} /></>}</>}>
    {!data ? <LoadState what="baseline controls" error={error} /> : !hasChart ? <Empty>No completed {view} results for this selection.</Empty> : <div className="chart-wrap">
      {view === 'blanks' && <div className="chart-note">{new Set(blanks.counted.filter(r => (r.split ?? 'Recorded controls') === split).map(r => r.id)).size} physical blanks · {split.replace(/_/g, ' ')}</div>}
      {view === 'repeats' && <div className="chart-note">{repeatType === 'FB' ? 'Field blanks · absolute absorbance differences' : 'PM2.5 · original acquisition pairs'}</div>}
      {view === 'injections' && data.injections.some(r => r.control === control && finite(r.n_converged) && r.n_converged < r.n_parents) && <div className="chart-note">Finite nonconverged results retained · hover for convergence counts</div>}
      {view === 'injections' ? <SweepPlot series={injectionSeries} xLabel="Injected amplitude (absorbance)" yLabel="Recovered / injected signal" reference={1}
        height={height} dim={lg.dim} />
        : view === 'blank spectra' ? <SpectrumPlot series={spectrumSeries} bands={spectrumBands} height={height} window={[1800, 1450]}
          hover="readout" zeroLine includeZero dim={lg.dim} yLabel="Signed absorbance" />
          : <ScatterPlot points={points} height={height} note="hand"
            xBands={view === 'blanks' ? showMethods.map(m => ({ key: m, label: labMethodLabel(data, m) })) : undefined}
            xLabel={view === 'blanks' ? 'Correction method' : relativeRepeat ? 'Raw relative repeat RMS difference' : 'Raw repeat RMS difference (absorbance)'}
            yLabel={view === 'blanks' ? blankLabel(metric) : relativeRepeat ? 'Corrected relative repeat RMS difference' : 'Corrected repeat RMS difference (absorbance)'}
            refY={view === 'blanks' ? [0] : []} xInclude={[0]} yInclude={[0]}
            onPick={p => { if (p.id && data.spectra_ids.includes(p.id)) selectFilter(p.id) }} />}
      <Legend items={view === 'blank spectra' ? aggregate.map(c => ({ label: c.label, color: labMethodColor(data, c.method), shape: c.group === 'FB' ? 'dashed' as const : 'line' as const }))
        : methods.map(m => ({ label: labMethodLabel(data, m), color: labMethodColor(data, m), shape: view === 'injections' ? 'line' as const : 'dot' as const,
          detail: view === 'injections' ? `${[...new Set(data.injections.filter(r => r.control === control && r.method === m).map(r => r.n_parents))].join('/')} parents` : undefined }))} {...lg.props} />
    </div>}
    {(view === 'injections' || view === 'blank spectra') && <FixedSetNote what="the exported control runs and their recorded parent spectra" />}
  </ChartFrame>
}

/** Saved partial associations are shown as saved results, never refitted after a visual filter switch. */
export function BaselineAssociations(_props: LabChartProps) {
  const { data, error } = useLabExport()
  const [wantedCohort, setCohort] = useSearchParam<string>('blab.association.cohort', 'portable')
  const [site, setSite] = useLabSite('blab.association.site', 'ETAD')
  const height = useFitHeight() ?? 350
  const cohorts = [...new Set(data?.associations.map(r => r.cohort) ?? [])]
  const cohort = cohorts.includes(wantedCohort) ? wantedCohort : cohorts[0] ?? wantedCohort
  const saved = data?.associations.filter(r => r.cohort === cohort && r.code === site && finite(r.partial_rank_r_squared)) ?? []
  const bars: BarRow[] = [...saved].sort((a, b) => b.partial_rank_r_squared! - a.partial_rank_r_squared!).map(r => ({
    key: r.feature, label: featureLabel(r.feature), value: r.partial_rank_r_squared, signed: r.partial_rank_r,
    note: `n=${r.n}`, tip: [featureLabel(r.feature), `${data ? labSiteName(data, r.code) : r.code} · ${r.cohort}`,
      `Partial rank r²: ${fmt(r.partial_rank_r_squared, 4)}`, `Signed partial r: ${signed(r.partial_rank_r, 4)}`,
      `${r.n} physical filters`, ...(r.controls ? [`Adjusted for: ${r.controls}`] : [])],
  }))
  return <ChartFrame id="baseline-lab-associations" title="Which spectral features track disagreement?"
    source={`${LAB_SOURCE}/Experiments.tsx › BaselineAssociations`}
    subtitle="Saved squared partial rank correlations. The response contains FTIR-derived EC, so association is a diagnostic clue, not causal validation."
    tip={<p>These results retain their exported cohorts and nuisance adjustments. Changing Samples does not refit these statistics. Native segment sensitivity, native removed absorbance and portable smoother sensitivity are different features. Hover exposes signed correlation, population size and adjustment variables.</p>}
    controls={<><LabSiteSelect data={data} value={site} onChange={setSite} all={false} />
      <Select label="Analysis population" value={cohort} options={cohorts} onChange={setCohort} optionLabel={v => v.replace(/_/g, ' ')} /></>}>
    {!data ? <LoadState what="spectral associations" error={error} /> : !bars.length ? <Empty>No saved associations for this site and population.</Empty> : <div className="chart-wrap">
      <BarRows rows={bars} domain={[0, 1]} format={v => fmt(v, 3)} axis={{ label: 'Squared partial rank correlation (r²)' }}
        rowHeight={Math.max(15, Math.min(27, (height - 45) / Math.max(1, bars.length)))} />
      <Legend items={[{ label: 'Positive partial r', color: INK.positive, shape: 'bar' }, { label: 'Negative partial r', color: INK.negative, shape: 'bar' }]} />
    </div>}
    <FixedSetNote what="the saved physical-filter analysis populations" />
  </ChartFrame>
}

const MODEL_SERIES = [
  { key: 'metadata_rmse', label: 'Metadata', color: RESEARCH_SERIES[3] },
  { key: 'baseline_indicators_rmse', label: '+ baseline indicators', color: RESEARCH_SERIES[1] },
  { key: 'band_contrasts_rmse', label: '+ band contrasts', color: RESEARCH_SERIES[2] },
  { key: 'spectra_rmse', label: '+ all spectral indicators', color: RESEARCH_SERIES[0] },
] as const

export function BaselineChronology(_props: LabChartProps) {
  const { data, error } = useLabExport()
  const [wanted, setPopulation] = useSearchParam<string>('blab.chronology.population', 'all_hips')
  const [scope, setScope] = useSearchParam<string>('blab.chronology.scope', 'quarters', ['quarters', 'pooled'])
  const height = useFitHeight() ?? 345
  const lg = useLegend()
  const populations = [...new Set(data?.chronology.map(r => r.population) ?? [])]
  const population = populations.includes(wanted) ? wanted : populations[0] ?? wanted
  const entries = labChronologyRows(data?.chronology ?? [], population, scope === 'pooled')
  const series: SweepSeries[] = MODEL_SERIES.filter(m => lg.show(m.label)).map(m => ({
    id: m.key, label: m.label, group: m.label, color: m.color, marks: true, bar: scope === 'pooled',
    points: entries.map(r => ({ x: r.quarter, y: r[m.key], tip: [r.quarter, m.label, `RMSE: ${fmt(r[m.key], 3)} µg/m³`, `${r.n} held-out physical filters`, population] })),
  }))
  return <ChartFrame id="baseline-lab-chronology" title="Does the diagnostic carry forward?"
    source={`${LAB_SOURCE}/Experiments.tsx › BaselineChronology`}
    subtitle="Addis: train on earlier quarters, predict later filters. RMSE is error predicting optical disagreement, not error against independent EC."
    tip={<><p>Scaling and Ridge-penalty selection use training data only. Each physical filter appears in one outer test quarter. The original failed quarter is retained.</p><p>Models within a population are evaluated on the same filters. Direct-only and known-lot selections change the test population and cannot be compared as if they were a common held-out set. Feature design was informed by prior work; this is exploratory.</p></>}
    controls={<><Select label="Population" value={population} options={populations} onChange={setPopulation} optionLabel={v => v.replace(/_/g, ' ')} />
      <Segmented label="View" value={scope} options={['quarters', 'pooled']} onChange={setScope} /></>}>
    {!data ? <LoadState what="chronological results" error={error} /> : !entries.length ? <Empty>No saved forward predictions for this population.</Empty> : <div className="chart-wrap">
      <SweepPlot series={series} xScale="point" xLabel="Held-out quarter" yLabel="Disagreement prediction RMSE (µg/m³)" height={height} include={[0]} dim={lg.dim} />
      <Legend items={MODEL_SERIES.map(m => ({ label: m.label, color: m.color, shape: 'line' as const }))} {...lg.props} />
    </div>}
    <FixedSetNote what="the chronological training/test cohorts" />
  </ChartFrame>
}

export function BaselineRecalibration(_props: LabChartProps) {
  const { data, error } = useLabExport()
  const [wanted, setCohort] = useSearchParam<string>('blab.calibration.cohort', '')
  const [wantedSplit, setSplit] = useSearchParam<string>('blab.calibration.split', '')
  const [view, setView] = useSearchParam<string>('blab.calibration.view', 'source validation', ['source validation', 'Ethiopian blank predictions'])
  const height = useFitHeight() ?? 280
  const blankRef = useRef<HTMLDivElement>(null)
  const { width: blankWidth } = useDimensions(blankRef)
  const cohorts = [...new Set(data?.calibration?.map(r => r.cohort) ?? [])]
  const cohort = cohorts.includes(wanted) ? wanted : cohorts[0] ?? wanted
  const splits = [...new Set(data?.calibration?.filter(r => r.cohort === cohort).map(r => r.split) ?? [])]
  const split = splits.includes(wantedSplit) ? wantedSplit : splits[0] ?? wantedSplit
  const { rows: entries, units, sameUnits } = labCalibrationRows(data?.calibration ?? [], cohort, split)
  const bars: BarRow[] = !data || !sameUnits ? [] : entries.map((r, i) => ({
    key: `${r.method}-${i}`, label: labMethodLabel(data, r.method), value: r.rmse, color: labMethodColor(data, r.method), note: `n=${r.n}`,
    tip: [labMethodLabel(data, r.method), `${cohort} · ${split}`, `RMSE: ${fmt(r.rmse, 3)} ${r.units}`, `${r.n} physical filters`,
      `Reference: ${r.reference}`, ...(r.note ? [r.note] : [])],
  }))
  const blankRows = data?.calibration_blanks?.filter(r => r.n > 0 && finite(r.median)) ?? []
  const blankUnits = blankRows[0]?.units ?? 'µg/filter'
  const sameBlankUnits = new Set(blankRows.map(r => r.units)).size <= 1
  const blankLabelWidth = labelGutter(blankRows.map(r => data ? labMethodLabel(data, r.method) : r.method), { size: 11, pad: 12, min: 75, max: Math.min(175, Math.max(75, blankWidth * .4)) })
  const blanks = !data ? [] : blankRows.map((r, i) => ({
    key: `${r.method}-${i}`, label: clipText(labMethodLabel(data, r.method), blankLabelWidth - 12, { size: 11 }), note: `n=${r.n}`,
    marks: [[{ mid: r.median, lo: r.q25, hi: r.q75, color: labMethodColor(data, r.method) }]],
    tip: [labMethodLabel(data, r.method), data.calibration_blanks_cohort ?? 'Primary source models', `${r.n} Ethiopian field blanks`,
      `Median predicted EC loading: ${signed(r.median, 3)} ${r.units}`, `Field-blank IQR: ${signed(r.q25, 3)} to ${signed(r.q75, 3)} ${r.units}`,
      `Mean: ${signed(r.mean, 3)} ${r.units}`, `RMS from nominal zero: ${fmt(r.rms_from_nominal_zero, 3)} ${r.units}`,
      'Nominal-zero diagnostic; these blanks have no measured thermal EC reference'],
  }))
  return <ChartFrame id="baseline-lab-recalibration" title="Matched preprocessing and recalibration"
    source={`${LAB_SOURCE}/Experiments.tsx › BaselineRecalibration`}
    subtitle="Existing source-reference data only. Each displayed preprocessing method has its own matched calibration. Source-domain scores do not establish Ethiopian EC accuracy."
    tip={<><p>Only completed, matched calibration runs enter this chart. Reusing old coefficients after changing the baseline is not a matched pipeline.</p><p>Ethiopian field-blank predictions are a separate target diagnostic. Whiskers are the distribution's interquartile range, not uncertainty intervals for a measured EC truth. Zero is nominal; field blanks can contain handling or environmental material. A better source-reference score does not establish transfer to Ethiopian filters.</p>{data?.calibration_note && <p>{data.calibration_note}</p>}</>}
    controls={<><Select label="View" value={view} options={['source validation', 'Ethiopian blank predictions']} onChange={setView} />
      {view === 'source validation' && <><Select label="Reference population" value={cohort} options={cohorts} onChange={setCohort} optionLabel={v => v.replace(/_/g, ' ')} />
        <Select label="Validation" value={split} options={splits} onChange={setSplit} optionLabel={v => v.replace(/_/g, ' ')} /></>}</>}>
    {!data ? <LoadState what="matched recalibration results" error={error} /> : view === 'Ethiopian blank predictions' ? <>
      {!sameBlankUnits ? <Empty>Field-blank prediction units differ; these rows cannot share an axis.</Empty> : !blanks.length ? <Empty>No completed Ethiopian field-blank predictions are available.</Empty> : <div className="chart-wrap" ref={blankRef}>
        <div className="chart-note">{data.calibration_blanks_cohort ?? 'Primary source models'} · medians and field-blank IQR · nominal zero</div>
        <IntervalStrip blocks={[{ key: 'ethiopian-field-blanks', rows: blanks }]}
          panels={[{ label: `Predicted EC loading on field blanks (${blankUnits})`, reference: 0, format: v => fmt(v, 2) }]}
          labelWidth={blankLabelWidth} minPanelWidth={115} noteWidth={35} rowHeight={Math.max(18, Math.min(30, (height - 65) / Math.max(1, blanks.length)))} />
      </div>}
    </> : !sameUnits ? <Empty>These exported scores use different units; select a comparable population.</Empty>
      : !bars.length ? <Empty>{data.calibration_note ?? 'No completed matched source-reference recalibration is available in this export.'}</Empty>
        : <div className="chart-wrap"><BarRows rows={bars} axis={{ label: `Held-out reference EC RMSE (${units})` }} format={v => fmt(v, 3)}
          rowHeight={Math.max(22, Math.min(38, (height - 45) / Math.max(1, bars.length)))} /></div>}
    <FixedSetNote what={view === 'source validation' ? 'the saved source-reference training and validation folds' : 'the primary source models and saved Ethiopian field blanks'} />
  </ChartFrame>
}
