import { Segmented } from '@/components/ChartFrame'
import { UndoIcon } from '@/components/Icons'
import { ScrubField } from '@/components/ScrubField'
import { ROLL_ALIGNS, ROLL_DEFAULT, ROLL_MAX_DAYS, ROLL_STATS, clampDays, rollLabel, type RollAlign, type RollSpec, type RollStat } from '@/lib/rolling'
import { useSearchParam } from '@/lib/url'

/**
 * A chart's rolling-statistic setting (lib/rolling): the window in days (a scrub field: drag sideways, or click to
 * type any number), median or mean, trailing or centred. Kept in the link under the chart's id (`<id>.roll`,
 * `<id>.rollstat`, `<id>.rollalign`), so a shared view reopens with the same window; a Dashboard tile keeps its
 * own. `defaults` is the chart's reproducible choice (ftir_29's trailing 45-day median unless a chart says
 * otherwise). Returns the spec for `rollingByTime`, its legend label and the controls for the chart's frame.
 */
export function useRolling(chartId: string, defaults: RollSpec = ROLL_DEFAULT) {
  const [daysText, setDays] = useSearchParam<string>(`${chartId}.roll`, String(defaults.days))
  const [stat, setStat] = useSearchParam<RollStat>(`${chartId}.rollstat`, defaults.stat, ROLL_STATS)
  const [align, setAlign] = useSearchParam<RollAlign>(`${chartId}.rollalign`, defaults.align, ROLL_ALIGNS)
  const spec: RollSpec = { days: clampDays(daysText, defaults.days), stat, align, minN: defaults.minN }
  const isDefault = spec.days === defaults.days && stat === defaults.stat && align === defaults.align
  const controls = <>
    <span className="control" title={`The rolling window, in days (1–${ROLL_MAX_DAYS}). Drag sideways (a day per 3 px, Shift for finer) or click to type.`}>
      Rolling window
      {/* no max on the field: bounded, a drag would span 1–3650 across its width (~40 days a pixel); clampDays caps it */}
      {/* the bar is drawn on a log scale (1 day empty, ROLL_MAX_DAYS full): linear, 45 days would be a 1% sliver */}
      <ScrubField label="Rolling window (days)" value={spec.days} min={1} step={1} pixelsPerStep={3} unit="days" width={84}
        fill={Math.log(spec.days) / Math.log(ROLL_MAX_DAYS)}
        onChange={(d) => setDays(String(clampDays(String(d), spec.days)))} />
      {!isDefault && <button type="button" className="btn quiet x-btn" aria-label={`Reset to the ${rollLabel(defaults)}`}
        title={`Back to the ${rollLabel(defaults)}, at least ${defaults.minN} filters per window`}
        onClick={() => { setDays(String(defaults.days)); setStat(defaults.stat); setAlign(defaults.align) }}><UndoIcon /></button>}
    </span>
    <Segmented label="Rolling" value={stat} options={ROLL_STATS} onChange={setStat} />
    <Segmented label="Window" value={align} options={ROLL_ALIGNS} onChange={setAlign}
      optionLabel={(a) => (a === 'trailing' ? 'Trailing' : 'Centred')} />
  </>
  return { spec, label: rollLabel(spec), controls, isDefault }
}
