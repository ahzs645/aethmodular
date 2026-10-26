import { useEffect, useState } from 'react'
import { loadDaily } from '@/lib/weather'
import { dynamicCalendar, type SiteRules } from '@/lib/dynamicSeasons'
import type { SeasonMeta } from '@/lib/types'

/**
 * Each site's dynamic seasons (lib/dynamicSeasons) from its daily ERA5 weather, for the sites
 * with a rule. Sites still loading, or without daily weather, are absent from the map; callers
 * fall back to the rule's season names.
 */
export function useDynamicCalendars(rules: SiteRules): Map<string, SeasonMeta[]> {
  const key = JSON.stringify(Object.entries(rules).sort(([a], [b]) => a.localeCompare(b)))
  const [state, setState] = useState<{ key: string; cals: Map<string, SeasonMeta[]> }>({ key: '', cals: new Map() })
  useEffect(() => {
    const entries = Object.entries(rules)
    if (!entries.length) { setState({ key, cals: new Map() }); return }
    let live = true
    Promise.all(entries.map(([code, rule]) => loadDaily(code).then((d) => [code, dynamicCalendar(d, rule)] as const).catch(() => null)))
      .then((pairs) => { if (live) setState({ key, cals: new Map(pairs.filter((p): p is NonNullable<typeof p> => !!p)) }) })
    return () => { live = false }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return state.cals
}
