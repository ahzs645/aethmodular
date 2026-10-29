import type { SeasonMeta } from './types'

/**
 * Weather-computed calendars by site code (hooks/useRainCalendars, hooks/useDynamicCalendars). A code in
 * neither is still loading; `failed` holds the sites whose daily weather could not be loaded, so the
 * reader can tell "no weather" from "not yet".
 */
export interface WeatherCalendars { cals: Map<string, SeasonMeta[]>; failed: Set<string> }

/** each load's result: [code, calendar], or the code alone where it failed */
export function collect(results: readonly (readonly [string, SeasonMeta[]] | string)[]): WeatherCalendars {
  const cals = new Map<string, SeasonMeta[]>(), failed = new Set<string>()
  for (const r of results) {
    if (typeof r === 'string') failed.add(r)
    else cals.set(r[0], r[1])
  }
  return { cals, failed }
}
