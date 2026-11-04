import { useMemo } from 'react'
import { ChartFrame, Empty, Select, Segmented } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { ScatterPlot, type ScatterPoint } from '@/components/ScatterPlot'
import { SpectrumPlot, type SpectrumSeries } from '@/components/SpectrumPlot'
import { Legend, useLegend } from '@/components/Legend'
import { useFitHeight } from '@/lib/chartFit'
import { useSearchParam } from '@/lib/url'
import { ExcludedNote } from '@/lib/markedRows'
import { fmt, signed } from '@/lib/format'
import { INK, SPECTRAL_REGION } from '@/lib/theme'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { finite, featureLabel, labSpectrumValues, selectLabCurves, LAB_RESPONSE, LAB_SOURCE, type LabFilter } from '../../common/baselineLab'
import { labMethodColor, labMethodLabel, labSiteColor, labSiteName, LabSiteSelect, LabSourceSelect, SOURCE_CHOICES,
  sourceIncludes, spectrumWindows, WINDOW_KEYS, useLabExport, useLabFilters, useLabSelection, useLabSite,
  useLabSpectrum, type LabChartProps, type LabSource, type LabWindow } from './data'

const X_FIELDS = ['baseline_sensitivity_rms', 'contrast_1580_1650', 'contrast_1600_1635', 'contrast_2850_2950',
  'escape_recomputed', 'optical_bc_ugm3', 'pm25_ugm3']
const axisLabel = (field: string) => ({ optical_bc_ugm3: 'Optical BC (µg/m³)', pm25_ugm3: 'PM2.5 (µg/m³)',
  escape_recomputed: 'Fraction outside reference envelope' }[field] ?? `${featureLabel(field)} (absorbance)`)

/** One physical filter per point; the shared ScatterPlot supplies linked hover, pin and exclusion marks. */
export function BaselineFilterScatter({ rows }: LabChartProps) {
  const { data, error } = useLabExport()
  const view = useLabFilters(data, rows)
  const [site, setSite] = useLabSite()
  const [source, setSource] = useSearchParam<LabSource>('blab.source', 'direct', SOURCE_CHOICES)
  const [field, setField] = useSearchParam<string>('blab.x', 'baseline_sensitivity_rms', X_FIELDS)
  const [, selectFilter] = useLabSelection()
  const height = useFitHeight() ?? 340
  const lg = useLegend()
  const plotted = useMemo(() => {
    const use = (r: LabFilter) => r.type === 'PM2.5' && (r.spectral_pair_eligible || r.is_excluded)
      && (site === 'all' || r.code === site) && sourceIncludes(r, source)
      && finite(r[field]) && finite(r.optical_disagreement_ugm3)
    return { counted: view.counted.filter(use), marked: view.marked.filter(use) }
  }, [view, site, source, field])
  const points = useMemo<ScatterPoint[]>(() => data ? [...plotted.counted, ...plotted.marked].map(r => ({
    id: r.id, key: r.id, x: r[field] as number, y: r.optical_disagreement_ugm3!, color: labSiteColor(data, r.code),
    site: labSiteName(data, r.code), hollow: r.optical_source !== 'direct_hips_fabs', dim: lg.dim(labSiteName(data, r.code)),
    tip: [r.id, `${labSiteName(data, r.code)} · ${r.date ?? 'date unavailable'} · lot ${r.lot_group}`,
      `${axisLabel(field)}: ${fmt(r[field] as number, 5)}`, `${LAB_RESPONSE}: ${signed(r.optical_disagreement_ugm3!, 3)}`,
      `FTIR EC: ${fmt(r.ec_ftir_ugm3, 3)} · optical BC: ${fmt(r.optical_bc_ugm3, 3)} µg/m³`,
      r.optical_source === 'direct_hips_fabs' ? 'Direct HIPS Fabs / MAC' : 'Public HIPS BC extension',
      ...(r.review_note ? [r.review_note] : []), 'Click to inspect this spectrum'],
  })).filter(p => lg.show(p.site!)) : [], [data, plotted, field, lg.dim, lg.show])
  const codes = [...new Set(plotted.counted.map(r => r.code))]
  return <ChartFrame id="baseline-lab-filters" title="Filters and optical disagreement"
    source={`${LAB_SOURCE}/Filters.tsx › BaselineFilterScatter`}
    subtitle="Physical PM2.5 filters in Samples. Click a point to open its spectrum below. Filled points use direct Fabs; hollow points use public HIPS BC."
    tip={<><p>The y axis is existing FTIR EC minus HIPS BC, not independent EC error. Spectral predictors also contribute to the existing FTIR estimator. No filter is excluded for disagreeing with HIPS.</p><p>Baseline sensitivity is an exploratory smoother comparison on the packed common grid. It is not an uncertainty interval or the native AIRSpec parameter experiment.</p></>}
    controls={<><LabSiteSelect data={data} value={site} onChange={setSite} /><LabSourceSelect value={source} onChange={setSource} />
      <Select label="Diagnostic" value={field} options={X_FIELDS} onChange={setField} optionLabel={axisLabel} /></>}>
    {!data ? <LoadState what="filter diagnostics" error={error} /> : !points.length ? <Empty>No eligible pairs in the selected Samples, site and optical source.</Empty> : <>
      <div className="chart-note">{plotted.counted.length} physical pairs · MAC {fmt(data.mac_m2_g, 1)} m²/g · click a filter to inspect</div>
      <div className="chart-wrap">
        <ScatterPlot points={points} xLabel={axisLabel(field)} yLabel={LAB_RESPONSE} height={height} refY={[0]} note={false}
          onPick={p => { if (p.id) selectFilter(p.id) }} swap="baseline-lab-scatter" />
        <Legend items={codes.map(code => ({ label: labSiteName(data, code), color: labSiteColor(data, code) }))} {...lg.props} />
      </div>
      <ExcludedNote count={plotted.marked.length} drawn />
    </>}
  </ChartFrame>
}

const VIEWS = ['corrected', 'raw', 'removed'] as const
const REPEATS = ['mean', 'scans', 'both'] as const

/**
 * Native signed traces and original acquisitions are fetched only for the chosen physical filter. The picker offers the
 * filters Samples leaves in view (useLabFilters), and the undated ones (field blanks, native samples) of the sites in view,
 * which no date or season can place. A selection Samples has left out gives way to the first filter in view.
 */
export function BaselineSelectedSpectrum({ rows }: LabChartProps) {
  const { data, error } = useLabExport()
  const inSamples = useLabFilters(data, rows)
  const [selected, setId] = useLabSelection()
  const options = useMemo(() => {
    if (!data) return null
    const codes = new Set(rows.map(r => r.code))
    const ids = new Set([...inSamples.counted, ...inSamples.marked, ...data.filters.filter(r => !r.date && codes.has(r.code))].map(r => r.id))
    return data.spectra_ids.filter(i => ids.has(i))
  }, [data, rows, inSamples])
  const id = options && !options.includes(selected) ? inSamples.counted.find(r => options.includes(r.id))?.id ?? options[0] ?? '' : selected
  const [view, setView] = useSearchParam<typeof VIEWS[number]>('blab.spectra.view', 'corrected', VIEWS)
  const [window, setWindow] = useSearchParam<LabWindow>('blab.spectra.window', 'band', WINDOW_KEYS)
  const [method, setMethod] = useSearchParam<string>('blab.spectra.method', 'all')
  const [repeats, setRepeats] = useSearchParam<typeof REPEATS[number]>('blab.spectra.repeats', 'mean', REPEATS)
  const loaded = useLabSpectrum(data, id)
  const height = useFitHeight() ?? 340
  const lg = useLegend()
  const row = data?.filters.find(r => r.id === id)
  const methodIds = [...new Set(loaded.data?.curves.filter(c => c.kind !== 'raw').map(c => c.method) ?? [])]
  const effectiveMethod = method === 'all' || methodIds.includes(method) ? method : 'all'
  // The packed fallback contains raw absorbance only. Opening a filter still shows its real data;
  // the controls visibly switch to Raw instead of implying that a signed correction was exported.
  const chosen = useMemo(() => selectLabCurves(loaded.data, view, effectiveMethod, repeats), [loaded.data, view, effectiveMethod, repeats])
  const effectiveView = chosen.view, curves = chosen.curves
  const allSeries = useMemo<SpectrumSeries[]>(() => !data ? [] : curves.map((c, i) => ({
    id: c.id, wn: c.wn, y: labSpectrumValues(c.y), label: c.label, group: c.label, role: 'line',
    color: c.scan_id ? RESEARCH_SERIES[i % RESEARCH_SERIES.length] : labMethodColor(data, c.method),
    dash: c.scan_id && i % 2 ? '5 3' : undefined, width: c.scan_id ? 1.6 : 2.1,
    tip: [id, c.label, `${loaded.data?.source} grid · ${c.wn.length} channels`],
  })), [curves, data, id, loaded.data?.source])
  const series = allSeries.filter(c => lg.show(c.label!))
  const showMissing = loaded.data && !curves.length
  return <ChartFrame id="baseline-lab-spectrum" title="Raw, removed and signed spectra"
    source={`${LAB_SOURCE}/Filters.tsx › BaselineSelectedSpectrum`}
    subtitle="Select any exported physical filter in Samples, or a field blank or undated native sample of a site in view. Original scans remain separate from the physical-filter mean."
    tip={<><p>Raw and corrected absorbance retain their sign and scale. The removed trace is raw minus the matching correction. Packed clipped/normalized AIRSpec display spectra are never used for this subtraction.</p>{loaded.data?.notes?.map((note, i) => <p key={i}>{note}</p>)}</>}
    controls={<><Select label="Filter" value={id} options={options ?? [id]} onChange={setId}
      optionLabel={key => { const f = data?.filters.find(r => r.id === key); return `${key}${f?.type && f.type !== 'PM2.5' ? ` · ${f.type}` : ''}` }} />
      <Segmented label="Signal" value={effectiveView} options={VIEWS} onChange={setView} optionLabel={v => v === 'corrected' ? 'Signed corrected' : v === 'removed' ? 'Removed' : 'Raw'} />
      <Select label="Window (cm⁻¹)" value={window} options={WINDOW_KEYS} onChange={v => setWindow(v as LabWindow)} optionLabel={v => spectrumWindows[v as LabWindow].label} />
      {methodIds.length > 1 && <Select label="Method" value={effectiveMethod} options={['all', ...methodIds]} onChange={setMethod}
        optionLabel={v => v === 'all' ? 'All methods' : data ? labMethodLabel(data, v) : v} />}
      <Select label="Acquisitions" value={repeats} options={[...REPEATS]} onChange={v => setRepeats(v as typeof REPEATS[number])}
        optionLabel={v => v === 'mean' ? 'Physical-filter mean' : v === 'scans' ? 'Original scans' : 'Mean + scans'} /></>}>
    {!data ? <LoadState what="spectrum index" error={error} /> : !id ? <Empty>No filter in Samples has an exported spectrum: pick Addis Ababa, Bishoftu, Beijing, Delhi or JPL.</Empty>
      : !data.spectra_ids.includes(id) ? <Empty>No spectrum exported for {id}. Choose another filter.</Empty>
      : !loaded.data ? <LoadState what={`spectrum ${id}`} error={loaded.error} /> : <>
        <div className="chart-note">{id} · {row?.date ?? 'date unavailable'} · lot {row?.lot_group ?? 'unknown'} · {loaded.data.source} grid
          {finite(row?.optical_disagreement_ugm3) && <> · FTIR − optical {signed(row!.optical_disagreement_ugm3!, 2)} µg/m³</>}</div>
        {showMissing ? <Empty>{view === 'corrected' || view === 'removed' ? 'No signed correction in this selection; inspect Raw or select an available method.' : 'No original-scan series in this export; choose Physical-filter mean.'}</Empty>
          : <div className="chart-wrap">
            <SpectrumPlot series={series} window={spectrumWindows[window].range} height={height} zeroLine includeZero
              hover="readout" yLabel="Signed absorbance" dim={lg.dim} regions={window === 'full' ? [SPECTRAL_REGION.ptfe] : []}
              markers={window === 'band' ? [{ at: 1617, label: '1617', color: INK.muted }] : []} />
            <Legend items={allSeries.map(c => ({ label: c.label!, color: c.color, shape: 'line' as const }))} {...lg.props} />
          </div>}
      </>}
  </ChartFrame>
}
