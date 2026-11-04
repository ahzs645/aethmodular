import { useMemo } from 'react'
import { useExport } from '@/lib/data'
import { useChartOptions } from '@/lib/chartOptions'
import { rowsOf, type Table } from '@/lib/table'
import { BASELINE_COLOR, INK } from '@/lib/theme'
import { ChartFrame } from '@/components/ChartFrame'
import { ExportState } from '@/components/ExportState'
import { Figure } from '@/components/Figure'
import { SavedFigure, type ExportedTables } from '@/components/SavedFigure'

interface Evidence extends ExportedTables {
  schema_version: number
  text: Record<string, string>
  tables: Record<string, Table>
}

const COLORS = { 'Spline baseline': BASELINE_COLOR.AIRSpec, VIBES: BASELINE_COLOR['VIBES-full-r70'], 'Known addition': INK.text }

/** Independent controls stay visible even while an older model bundle is absent. */
export function VibesValidation() {
  const { data, error } = useExport<Evidence>('baseline_validation.json',
    (d) => d.schema_version === 1 ? null : 'unsupported validation export')
  if (!data) return <ExportState title="VIBES signal-preservation check" what="the independent continuum check" error={error}
    command="uv run --extra vibes python research/ftir_hips_chem/workflows/check_vibes_continuum.py --reference-dir DIR" />
  return <ValidationFigures data={data} />
}

function ValidationFigures({ data }: { data: Evidence }) {
  const run = rowsOf(data.tables.run)[0]
  const fits = useMemo(() => rowsOf(data.tables.fits), [data])
  const passed = fits.filter((r) => r.success === true).length
  const retried = fits.filter((r) => typeof r.retry_count === 'number' && r.retry_count > 0).length
  const options = useChartOptions('vibes-validation', [
    { key: 'signal', label: 'Known addition', persist: true, default: 'Power-law continuum',
      choices: ['Power-law continuum', 'Broad hump', 'Four diagnostic bands'] },
    { key: 'window', label: 'Comparison window', persist: true, default: 'Shared spline window',
      choices: ['Shared spline window', 'Full measured grid'] },
  ])
  const signal = String(options.values.signal), window = String(options.values.window)
  const recoveryWhere = useMemo(() => ({ signal, window }), [signal, window])
  const windowWhere = useMemo(() => ({ window }), [window])
  const signalWhere = useMemo(() => ({ signal }), [signal])
  const spectrumWindow: [number, number] = window === 'Shared spline window' ? [4000, 1425] : [4000, 500]
  return <>
    <ChartFrame id="vibes-validation-scope" title="VIBES signal-preservation check" exportable={false} controls={options.controls()}
      provenance={`Frozen reference ${String(run.reference_signature)}; local check ${String(run.created_utc)}`}>
      <p>{String(run.independent_test_blanks)} independent held-out Addis blanks; background trained on {String(run.training_blanks)} other blanks,
        rank {String(run.background_rank)}. Each method is rerun after adding known spectral signals.</p>
      <p>{passed}/{fits.length} VIBES optimizer and finite-output checks succeeded; {retried} fits needed a retry.</p>
      {typeof run.map_replayed === 'number' && <p>Convex-solver replay: {String(run.map_optimal)}/{String(run.map_replayed)} reported optimal.
        Recovery metrics reproduce within 10⁻⁸.</p>}
      <p>{data.text.definition}</p>
      <p>These artificial additions probe signal preservation; they do not identify real EC absorption.</p>
      <p>{data.text.acceptance} The separately reported 32,286-fit/G4 run awaits its own results export.</p>
    </ChartFrame>
    <SavedFigure bundle={data} table="recovery" kind="sweep" id="vibes-validation-recovery" x="amplitude" y="area_recovery"
      series="method" colors={COLORS} where={recoveryWhere} reference={1}
      aggregate={{ stat: 'median', interval: 'iqr' }} xLabel="Known addition scale (AU)" yLabel="Signed area recovery (1 = retained)"
      title="Does the known signal survive?" bandLabel="Line: median across independent parent blanks · band: interquartile range"
      note="Unavailable windows remain missing." />
    <SavedFigure bundle={data} table="recovery" kind="scatter" id="vibes-validation-errors" x="amplitude" y="relative_RMSE"
      color="method" colors={COLORS} where={recoveryWhere} filterIdColumn="filter_id"
      tipColumns={['method', 'filter_id', 'amplitude', 'area_recovery']} refY={[1]}
      xLabel="Known addition scale (AU)" yLabel="Recovery RMSE / known-signal RMS"
      title="Signal recovery for each held-out blank"
      note="Zero is exact recovery. Removing the entire known addition produces a relative RMS error of 1." noteAlways />
    <SavedFigure bundle={data} table="traces" kind="spectrum" id="vibes-validation-traces" wn="wn" y="value" series="method"
      where={signalWhere} choose={[{ column: 'amplitude', label: 'Addition scale', default: '0.05' }]}
      window={spectrumWindow} colors={COLORS} yLabel="Incremental corrected absorbance (AU)" includeZero zeroLine
      title="Known addition versus recovered spectral shape" note={data.text.trace} noteAlways />
    <SavedFigure bundle={data} table="blank_rms" kind="ecdf" id="vibes-validation-blanks" value="RMS" series="method" colors={COLORS}
      where={windowWhere} filterIdColumn="filter_id" xLabel="Unspiked blank residual RMS (AU)"
      title="Blank removal on the same independent filters" />
    <Figure kind="table" id="vibes-validation-runtime" data={data.tables.runtime} title="Measured execution of this local check"
      columns={[{ key: 'method', label: 'Method' }, { key: 'attempted_fits', label: 'Attempted', format: 'int' },
        { key: 'successful_fits', label: 'Successful', format: 'int' }, { key: 'wall_seconds', label: 'Elapsed seconds', format: 'f2' },
        { key: 'successful_fits_per_minute', label: 'Successful fits/min', format: 'f2' }, { key: 'workers', label: 'Workers', format: 'int' },
        { key: 'reported_logical_cpus', label: 'Reported logical CPUs', format: 'int' }]}
      note="Timing covers correction, including VIBES worker startup. Spline success refers to its supported window. Faster fitting does not establish signal preservation." noteAlways />
    <Figure kind="table" id="vibes-validation-reference" data={data.tables.reference_fit_counts}
      title="Original rank-70 fit log, before recovery" note={data.text.reference} noteAlways
      columns={[{ key: 'kind', label: 'Spectrum role' }, { key: 'success', label: 'Fit succeeded' }, { key: 'n', label: 'Fits', format: 'int' }]} />
  </>
}
