/**
 * Dynamic seasons: a site's seasons from its own daily weather (ERA5, weather/daily/<CODE>.json)
 * instead of fixed months, set per site in the Filters panel. Two kinds of rule:
 *
 * - `rules`: each day takes the first season whose conditions all hold, the last season taking
 *   every other day. A condition is a centred rolling mean of one variable (rain, temperature,
 *   humidity, wind, boundary-layer height) above or below a value, or a percentile of that site's
 *   own rolling means, so one rule reads the same at a wet and a dry site. Runs shorter than
 *   `minRun` days join the season before them. This is Task 3 of addis_03_meteorology
 *   (30-day rain against two thresholds), generalised to any variable and any number of seasons.
 * - `onset`: the PMF paper's rain rule (lib/rainSeasons), seasons in order through each year.
 *
 * Days without weather take the nearest classified day's season, so every filter gets one.
 */
import type { DailyWeather } from '@/lib/weather'
import type { SeasonMeta } from '@/lib/types'
import { DEFAULT_RAIN_RULE, rainRuleCalendar, type RainRule } from '@/lib/rainSeasons'

export type DynVar = 'p' | 't' | 'tmin' | 'tmax' | 'rh' | 'ws' | 'blh'
export const DYN_VARS: { key: DynVar; label: string; unit: string }[] = [
  { key: 'p', label: 'Rain', unit: 'mm/day' },
  { key: 't', label: 'Temperature', unit: '°C' },
  { key: 'tmin', label: 'Min temperature', unit: '°C' },
  { key: 'tmax', label: 'Max temperature', unit: '°C' },
  { key: 'rh', label: 'Humidity', unit: '%' },
  { key: 'ws', label: 'Wind speed', unit: 'm/s' },
  { key: 'blh', label: 'Boundary layer', unit: 'm' },
]

/** `w`-day centred mean of `v` above (`>`) or below (`<`) `x`; with `pct`, x is a percentile of the site's own means */
export interface DynCond { v: DynVar; w: number; op: '>' | '<'; x: number; pct?: boolean }
export interface DynSeason { name: string; when: DynCond[] }
export type DynRule =
  | ({ kind: 'onset'; preset?: string } & RainRule)
  | { kind: 'rules'; preset?: string; seasons: DynSeason[]; minRun: number }

export const DYN_COLORS = ['#B8860B', '#67A96A', '#428EAF', '#D99145', '#8A6FB0', '#AA6B78', '#6F86BA', '#5A9E9A']

interface Preset { key: string; label: string; rule: DynRule; sites?: string[] }
const notebook = (lo: number, hi: number, label: string): Preset => ({
  key: `nb-${lo}-${hi}`, label, sites: ['ETAD'],
  rule: { kind: 'rules', minRun: 1, seasons: [
    { name: 'Kiremt rainy', when: [{ v: 'p', w: 30, op: '>', x: hi }] },
    { name: 'Belg rainy', when: [{ v: 'p', w: 30, op: '>', x: lo }] },
    { name: 'Dry', when: [] },
  ] },
})
/** the starting points offered in a site's section; `sites` limits one to the sites it was made for */
export const DYN_PRESETS: Preset[] = [
  { key: 'pmf-rain', label: 'PMF paper rain rule (Addis 2023)', sites: ['ETAD'], rule: { kind: 'onset', ...DEFAULT_RAIN_RULE } },
  notebook(5, 15, '30-day rain, 5 / 15 mm/day (notebook standard)'),
  notebook(3, 10, '30-day rain, 3 / 10 mm/day (notebook conservative)'),
  notebook(7, 20, '30-day rain, 7 / 20 mm/day (notebook liberal)'),
  { key: 'wet-dry', label: 'Wet and dry by own rain', rule: { kind: 'rules', minRun: 14, seasons: [
    { name: 'Wet', when: [{ v: 'p', w: 30, op: '>', x: 50, pct: true }] },
    { name: 'Dry', when: [] },
  ] } },
  { key: 'rain-temp', label: 'Rain and temperature', rule: { kind: 'rules', minRun: 14, seasons: [
    { name: 'Wet', when: [{ v: 'p', w: 30, op: '>', x: 66, pct: true }] },
    { name: 'Warm dry', when: [{ v: 't', w: 30, op: '>', x: 50, pct: true }] },
    { name: 'Cool dry', when: [] },
  ] } },
]
export const presetsFor = (code: string) => DYN_PRESETS.filter((p) => !p.sites || p.sites.includes(code))
/** a site's first offered preset: the paper's rule at Addis, wet and dry elsewhere */
export const defaultRuleFor = (code: string): DynRule => structuredClone({ ...presetsFor(code)[0].rule, preset: presetsFor(code)[0].key })

const iso = (t: number) => new Date(t).toISOString().slice(0, 10)
const DAY = 86_400_000

/** centred `w`-day mean, needing half the window present */
function rolling(x: (number | null)[], w: number): (number | null)[] {
  const half = Math.floor(w / 2), n = x.length
  const out: (number | null)[] = new Array(n).fill(null)
  for (let i = 0; i < n; i++) {
    let s = 0, k = 0
    for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half - (w % 2 ? 0 : 1)); j++) {
      const v = x[j]
      if (v != null && Number.isFinite(v)) { s += v; k++ }
    }
    out[i] = k >= w / 2 ? s / k : null
  }
  return out
}
const quantile = (xs: number[], q: number) => {
  if (!xs.length) return NaN
  const s = [...xs].sort((a, b) => a - b), i = Math.min(s.length - 1, Math.max(0, (q / 100) * (s.length - 1)))
  const lo = Math.floor(i), hi = Math.ceil(i)
  return s[lo] + (s[hi] - s[lo]) * (i - lo)
}

/** one site's seasons under a rule, each with its dated `spans` */
export function dynamicCalendar(daily: DailyWeather, rule: DynRule): SeasonMeta[] {
  if (rule.kind === 'onset') return rainRuleCalendar(daily, rule)
  const n = daily.date.length
  const series = new Map<string, { m: (number | null)[]; cut: (c: DynCond) => number }>()
  const get = (c: DynCond) => {
    const key = `${c.v}:${c.w}`
    let s = series.get(key)
    if (!s) {
      const m = rolling(daily[c.v] ?? [], Math.max(1, Math.round(c.w)))
      const vals = m.filter((v): v is number => v != null)
      s = { m, cut: (cc) => (cc.pct ? quantile(vals, cc.x) : cc.x) }
      series.set(key, s)
    }
    return s
  }
  const last = rule.seasons.length - 1
  // the first season whose conditions all hold; null where a condition has no weather that day
  const day: (number | null)[] = daily.date.map((_, i) => {
    for (let k = 0; k < last; k++) {
      const conds = rule.seasons[k].when
      if (!conds.length) return k
      let ok = true
      for (const c of conds) {
        const s = get(c), v = s.m[i]
        if (v == null) return null
        const x = s.cut(c)
        if (!(c.op === '>' ? v > x : v < x)) { ok = false; break }
      }
      if (ok) return k
    }
    return last
  })
  // fill days without weather from the nearest classified day (before, else after)
  let prev: number | null = null
  for (let i = 0; i < n; i++) { if (day[i] == null) day[i] = prev; else prev = day[i] }
  let next: number | null = null
  for (let i = n - 1; i >= 0; i--) { if (day[i] == null) day[i] = next; else next = day[i] }
  // runs shorter than minRun days join the season before them
  const min = Math.max(1, Math.round(rule.minRun))
  if (min > 1) {
    for (let i = 0; i < n;) {
      let j = i
      while (j < n && day[j] === day[i]) j++
      if (j - i < min && i > 0) for (let k = i; k < j; k++) day[k] = day[i - 1]
      i = j
    }
  }
  const spans: [string, string][][] = rule.seasons.map(() => [])
  for (let i = 0; i < n;) {
    let j = i
    while (j < n && day[j] === day[i]) j++
    const s = day[i]
    if (s != null) spans[s].push([daily.date[i], iso(Date.parse(`${daily.date[j - 1]}T00:00:00Z`) + DAY)])
    i = j
  }
  // the record's own ends stretch to the open past and future, so no filter falls outside
  const first = spans.find((sp) => sp.some(([a]) => a === daily.date[0]))
  if (first) first[0][0] = '0000-01-01'
  const end = spans.flat().find(([, b]) => b === iso(Date.parse(`${daily.date[n - 1]}T00:00:00Z`) + DAY))
  if (end) end[1] = '9999-12-31'
  return rule.seasons.map((s, i) => ({ name: s.name, color: DYN_COLORS[i % DYN_COLORS.length], months: [], spans: spans[i] }))
}

/** the rule's seasons before the weather has loaded (names and colours only) */
export const dynamicPlaceholder = (rule: DynRule): SeasonMeta[] => rule.kind === 'onset'
  ? rainRuleCalendar({ date: [], p: [] }, rule)
  : rule.seasons.map((s, i) => ({ name: s.name, color: DYN_COLORS[i % DYN_COLORS.length], months: [] }))

/** per-site rules in the link (dyn=) and in this browser: code -> rule */
export type SiteRules = Record<string, DynRule>
export function parseSiteRules(v: string | null): SiteRules | null {
  try {
    const o = v ? JSON.parse(v) : null
    if (!o || typeof o !== 'object' || Array.isArray(o)) return null
    const out: SiteRules = {}
    for (const [code, r] of Object.entries(o as Record<string, DynRule>)) {
      if (r?.kind === 'onset' && [r.window, r.onset, r.heavy].every(Number.isFinite) && r.heavy > r.onset) out[code] = r
      else if (r?.kind === 'rules' && Array.isArray(r.seasons) && r.seasons.length >= 2) {
        const keys = new Set(DYN_VARS.map((d) => d.key))
        out[code] = { kind: 'rules', preset: r.preset, minRun: Number(r.minRun) || 1, seasons: r.seasons.slice(0, 8).map((s) => ({
          name: String(s.name ?? '').slice(0, 40) || 'Season',
          when: (s.when ?? []).filter((c) => keys.has(c.v) && Number.isFinite(c.w) && Number.isFinite(c.x) && (c.op === '>' || c.op === '<')),
        })) }
      }
    }
    return out
  } catch { return null }
}
export const siteRulesParam = (r: SiteRules) => (Object.keys(r).length ? JSON.stringify(r) : null)
