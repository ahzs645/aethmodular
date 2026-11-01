import { codesOf, keyOf, useLoaded } from '@/lib/data'
import { loadDaily } from '@/lib/weather'
import { rainRuleCalendar, rainRuleParam, type RainRule } from '@/lib/rainSeasons'
import { collect, type WeatherCalendars } from '@/lib/weatherCalendars'

const NONE: WeatherCalendars = { cals: new Map(), failed: new Set() }

/**
 * The rain-rule calendar (lib/rainSeasons) for each site code in `codes`, from its daily ERA5
 * rain (weather/daily/<CODE>.json, loaded as needed). Sites still loading are in neither `cals` nor
 * `failed`; sites without daily weather are in `failed`. Callers classify no filter for either.
 */
export function useRainCalendars(codes: readonly string[], rule: RainRule): WeatherCalendars {
  const key = keyOf(codes)
  // one object per load, so callers' memos see a change only when a load lands
  return useLoaded(key ? `${key}|${rainRuleParam(rule)}` : null, () =>
    Promise.all(codesOf(key).map((c) => loadDaily(c).then((d) => [c, rainRuleCalendar(d, rule)] as const).catch(() => c))).then(collect),
  { keep: true }).data ?? NONE
}
