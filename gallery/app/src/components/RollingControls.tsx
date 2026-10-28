import { Segmented } from '@/components/ChartFrame'
import { CountPicker } from '@/components/Pickers'
import { ROLL_ALIGNS, ROLL_DEFAULT, ROLL_MAX_DAYS, ROLL_STATS, clampDays, rollLabel, type RollAlign, type RollSpec, type RollStat } from '@/lib/rolling'
import { useSearchParam } from '@/lib/url'

/** the window buttons; the last is the box for any other number of days */
const DAY_PRESETS = ['15', '30', '45', '90', '60'] as const

/**
 * A chart's rolling-statistic setting (lib/rolling): the window in days (preset buttons, or any number typed in
 * the box), median or mean, trailing or centred. Kept in the link under the chart's id (`<id>.roll`,
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
    <CountPicker label="Rolling window (days)" value={String(spec.days)} presets={DAY_PRESETS} max={ROLL_MAX_DAYS} onChange={setDays} noun="days" />
    <Segmented label="Rolling" value={stat} options={ROLL_STATS} onChange={setStat} />
    <Segmented label="Window" value={align} options={ROLL_ALIGNS} onChange={setAlign}
      optionLabel={(a) => (a === 'trailing' ? 'Trailing' : 'Centred')} />
    {!isDefault && <button type="button" className="btn quiet" onClick={() => { setDays(String(defaults.days)); setStat(defaults.stat); setAlign(defaults.align) }}
      title={`Back to the ${rollLabel(defaults)}, at least ${defaults.minN} filters per window`}>Reset</button>}
  </>
  return { spec, label: rollLabel(spec), controls, isDefault }
}
