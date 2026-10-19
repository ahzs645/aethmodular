import { useEffect, useState } from 'react'
import { loadDaily } from '@/lib/weather'
import { rainRuleCalendar, rainRuleParam, type RainRule } from '@/lib/rainSeasons'
import type { SeasonMeta } from '@/lib/types'

/**
 * The rain-rule calendar (lib/rainSeasons) for each site code in `codes`, from its daily ERA5
 * rain (weather/daily/<CODE>.json, loaded as needed). Sites still loading, or without daily
 * weather, are absent from the map; callers fall back to the rule's usual months.
 */
export function useRainCalendars(codes: readonly string[], rule: RainRule): Map<string, SeasonMeta[]> {
  const key = [...new Set(codes)].sort().join(',')
  const ruleKey = rainRuleParam(rule)
  const [state, setState] = useState<{ key: string; cals: Map<string, SeasonMeta[]> }>({ key: '', cals: new Map() })
  useEffect(() => {
    if (!key) return
    let live = true
    Promise.all(key.split(',').map((c) => loadDaily(c).then((d) => [c, rainRuleCalendar(d, rule)] as const).catch(() => null)))
      .then((pairs) => { if (live) setState({ key: `${key}|${ruleKey}`, cals: new Map(pairs.filter((p): p is NonNullable<typeof p> => !!p)) }) })
    return () => { live = false }
  }, [key, ruleKey]) // eslint-disable-line react-hooks/exhaustive-deps
  return state.cals
}
