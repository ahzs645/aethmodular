import { useEffect, useState } from 'react'
import { loadDaily } from '@/lib/weather'
import { rainRuleCalendar, rainRuleParam, type RainRule } from '@/lib/rainSeasons'
import { collect, type WeatherCalendars } from '@/lib/weatherCalendars'

/**
 * The rain-rule calendar (lib/rainSeasons) for each site code in `codes`, from its daily ERA5
 * rain (weather/daily/<CODE>.json, loaded as needed). Sites still loading are in neither `cals` nor
 * `failed`; sites without daily weather are in `failed`. Callers classify no filter for either.
 */
export function useRainCalendars(codes: readonly string[], rule: RainRule): WeatherCalendars {
  const key = [...new Set(codes)].sort().join(',')
  const ruleKey = rainRuleParam(rule)
  const [state, setState] = useState<{ key: string } & WeatherCalendars>({ key: '', cals: new Map(), failed: new Set() })
  useEffect(() => {
    if (!key) return
    let live = true
    Promise.all(key.split(',').map((c) => loadDaily(c).then((d) => [c, rainRuleCalendar(d, rule)] as const).catch(() => c)))
      .then((pairs) => { if (live) setState({ key: `${key}|${ruleKey}`, ...collect(pairs) }) })
    return () => { live = false }
  }, [key, ruleKey]) // eslint-disable-line react-hooks/exhaustive-deps
  return state   // one object per load, so callers' memos see a change only when a load lands
}
