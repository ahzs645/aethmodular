import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useChartOptions } from '@/lib/chartOptions'
import { useFitHeight } from '@/lib/chartFit'
import { Legend, useLegend } from '@/components/Legend'
import { SpectrumPlot } from '@/components/SpectrumPlot'
import { INK } from '@/lib/theme'
import { PREPROCESSING, label } from '@/lib/labels'
import { FixedSetNote } from '@/lib/markedRows'

const MARKS = [{ wn: 1600, label: '~1600 aromatic / carboxylate' }, { wn: 2920, label: 'C–H stretch' }, { wn: 1720, label: 'C=O' }]

export interface SpectrumSeries {
  label: string
  color: string
  wn: number[]
  median: number[]
  q25: number[]
  q75: number[]
  dash?: string
}

/**
 * Median spectrum with an interquartile band, one per series, wavenumber
 * decreasing to the right as FTIR convention has it. Used twice on the
 * Calibration tab: the three selection cohorts against the Addis median
 * (Ann's multi-cohort ask), and every SPARTAN evaluation site against each
 * other. The spectra space (raw or AIRSpec-baselined) is a control because
 * the two disagree about what the ~1600 cm⁻¹ Addis band even looks like.
 */
export function SpectraOverlay({
  id, title, subtitle, provenance, bySpace, defaultSpace = 'raw', fixedSet,
}: {
  id: string
  title: string
  subtitle: string
  provenance: string
  /** series per spectra space; an empty record renders an empty state */
  bySpace: Record<string, SpectrumSeries[]>
  defaultSpace?: string
  /** the saved filter set the medians describe, for the FixedSetNote (Filters → excluded does not act on it) */
  fixedSet?: string
}) {
  const spaces = Object.keys(bySpace)
  // one preprocessing is not a choice: the engine draws the control only with two or more
  const o = useChartOptions(id, [
    { key: 'space', label: 'Preprocessing', default: spaces.includes(defaultSpace) ? defaultSpace : spaces[0] ?? '', choices: spaces.map((s) => ({ value: s, label: label(PREPROCESSING, s) })) },
    { key: 'bands', label: 'IQR bands', toggle: true, default: true },
  ])
  const space = o.values.space as string
  const bands = o.values.bands as boolean
  const height = useFitHeight() ?? 380
  const { hidden, dim, props: legendProps } = useLegend()
  const series = (bySpace[space] ?? []).filter((s) => !hidden.has(s.label))

  return (
    <ChartFrame
      id={id}
      title={title}
      subtitle={subtitle}
      provenance={provenance}
      controls={o.controls()}
    >
      <div className="chart-wrap">
        {series.length === 0 ? (
          <Empty>No spectra exported for this space.</Empty>
        ) : (
          <SpectrumPlot height={height} hover="readout" dim={dim} xLabel="wavenumber (cm⁻¹)" yLabel={space === 'raw' ? 'absorbance' : 'baselined absorbance'}
            series={series.map((s) => ({ id: s.label, label: s.label, group: s.label, wn: s.wn, y: s.median, color: s.color, dash: s.dash, width: 1.8 }))}
            bands={bands ? series.map((s) => ({ id: s.label, group: s.label, wn: s.wn, lo: s.q25, hi: s.q75, color: s.color, opacity: 0.12 })) : []}
            markers={MARKS.map((m) => ({ at: m.wn, label: m.label, color: INK.muted }))} />
        )}
        <Legend items={(bySpace[space] ?? []).map((s) => ({ label: s.label, color: s.color, shape: s.dash ? ('dashed' as const) : ('line' as const) }))} {...legendProps} />
        {fixedSet && <FixedSetNote what={fixedSet} />}
      </div>
    </ChartFrame>
  )
}
