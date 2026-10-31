import { createContext, createElement, useContext, useEffect, useState, type ReactNode } from 'react'

/**
 * The whole subset — tab, calendar, seasons, sites, date range, the fields on
 * the axes, a pinned filter — lives in the URL hash, so a view can be pasted
 * into a message or a notebook and reopen exactly as it was. Nothing here is
 * React-specific: App reads once on mount and writes on every change.
 *
 *   #tab=correlation&x=HIPS+BC&y=EC+(FTIR)&sites=Addis+Ababa&from=2023-01-01
 */
export function readParams(): URLSearchParams {
  const h = window.location.hash.replace(/^#/, '')
  return new URLSearchParams(h)
}

export function writeParams(values: Record<string, string | null | undefined>) {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(values)) {
    if (v !== null && v !== undefined && v !== '') p.set(k, v)
  }
  const next = p.toString()
  const cur = window.location.hash.replace(/^#/, '')
  if (next !== cur) window.history.replaceState(null, '', next ? `#${next}` : window.location.pathname)
}

export const listParam = (v: string | null): Set<string> | null =>
  v ? new Set(v.split('|').filter(Boolean)) : null

export const listToParam = (s: Set<string> | null): string | null =>
  s ? [...s].join('|') : null

/**
 * One control held in the query string rather than the hash, for tabs whose
 * state is independent of the filter-row subset above. The value survives a
 * copied URL and the back button; the fallback is never written, so a
 * default view keeps a clean URL.
 */
/**
 * Query keys are shared by every page, so two controls that mean different things
 * must not use the same key (one would silently overwrite the other). In
 * development, a key registered with two different option lists is reported.
 */
const registered = new Map<string, string>()
function registerKey(name: string, allowed?: readonly string[]) {
  if (!import.meta.env?.DEV || !allowed) return
  const sig = [...allowed].sort().join('|')
  const prev = registered.get(name)
  if (prev !== undefined && prev !== sig) console.warn(`URL key "${name}" is used by two controls with different options: [${prev}] vs [${sig}]`)
  else registered.set(name, sig)
}

/**
 * A prefix for every query key read inside it: a Dashboard tile wraps its chart in
 * <ParamScope prefix="t3."> so the chart's controls are `t3.scatter.layout`, not the
 * Correlation tab's `scatter.layout`, and two copies of one chart keep their own.
 */
const ScopeCtx = createContext('')
export const ParamScope = ({ prefix, children }: { prefix: string; children: ReactNode }) =>
  createElement(ScopeCtx.Provider, { value: useContext(ScopeCtx) + prefix }, children)
export const useParamScope = () => useContext(ScopeCtx)

/**
 * A chart's own setting kept in plain state (a statistic, a bin count, a colour field): inside a
 * Dashboard tile it lives in the URL under the tile's prefix, so a saved or shared dashboard
 * reopens each tile as it was; everywhere else it is ordinary component state, as before.
 */
export function useTileSetting<T extends string>(key: string, fallback: T, allowed?: readonly string[]): [T, (v: T) => void] {
  const prefix = useParamScope()
  const name = prefix + key
  const read = (): T => {
    if (!prefix) return fallback
    const v = new URL(window.location.href).searchParams.get(name)
    return v !== null && (!allowed || allowed.length === 0 || allowed.includes(v)) ? (v as T) : fallback
  }
  const [value, setValue] = useState<T>(read)
  useEffect(() => {
    if (!prefix) return
    const sync = () => setValue(read())
    window.addEventListener('popstate', sync)
    return () => window.removeEventListener('popstate', sync)
  }, [name]) // eslint-disable-line react-hooks/exhaustive-deps
  const update = (v: T) => {
    setValue(v)
    if (!prefix) return
    const url = new URL(window.location.href)
    if (v === fallback) url.searchParams.delete(name)
    else url.searchParams.set(name, v)
    window.history.replaceState(null, '', url)
  }
  return [value, update]
}

/** forget every query key under a prefix (a removed Dashboard tile's controls) */
export function clearScopedParams(prefix: string) {
  const url = new URL(window.location.href)
  for (const k of [...url.searchParams.keys()]) if (k.startsWith(prefix)) url.searchParams.delete(k)
  window.history.replaceState(null, '', url)
}

/** copy every query key under one prefix to another (a duplicated tile starts with the same controls) */
export function copyScopedParams(from: string, to: string) {
  const url = new URL(window.location.href)
  for (const [k, v] of [...url.searchParams]) if (k.startsWith(from)) url.searchParams.set(to + k.slice(from.length), v)
  window.history.replaceState(null, '', url)
}

export function useSearchParam<T extends string>(key: string, fallback: T, allowed?: readonly string[]): [T, (v: T) => void] {
  const name = useParamScope() + key
  registerKey(name, allowed)
  // an empty list means the options are still loading: keep a link's value until they arrive
  const known = (v: string) => !allowed || allowed.length === 0 || allowed.includes(v)
  // the raw value: checked against the options at render (below), not here. The sync listener outlives
  // the render that made it, so a list that grew since (a palette the reader just saved) would turn
  // their new pick back into the fallback
  const read = () => (new URL(window.location.href).searchParams.get(name) || fallback) as T
  const [value, setValue] = useState<T>(read)
  useEffect(() => {
    const sync = () => setValue(read())
    window.addEventListener('popstate', sync)
    window.addEventListener('gallery:querychange', sync)
    return () => { window.removeEventListener('popstate', sync); window.removeEventListener('gallery:querychange', sync) }
  }, [name, fallback])
  const update = (v: T) => {
    setValue(v)
    const url = new URL(window.location.href)
    if (v === fallback) url.searchParams.delete(name)
    else url.searchParams.set(name, v)
    window.history.replaceState(null, '', url)
    window.dispatchEvent(new Event('gallery:querychange'))
  }
  // checked on every render, not only on mount: the options can change (another export, a
  // version switch) and a fallback the data does not have (a missing bank or season) must not
  // reach a chart that looks it up; then the first allowed value stands in
  const valid = known(value) ? value : allowed && allowed.includes(fallback) ? fallback : ((allowed?.[0] ?? fallback) as T)
  return [valid, update]
}
