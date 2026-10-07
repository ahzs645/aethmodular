import { useEffect, useState } from 'react'

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
export function useSearchParam<T extends string>(name: string, fallback: T, allowed?: readonly string[]): [T, (v: T) => void] {
  const read = () => {
    const value = new URL(window.location.href).searchParams.get(name)
    return value && (!allowed || allowed.includes(value)) ? (value as T) : fallback
  }
  const [value, setValue] = useState<T>(read)
  useEffect(() => {
    const sync = () => setValue(read())
    window.addEventListener('popstate', sync)
    return () => window.removeEventListener('popstate', sync)
  }, [name, fallback])
  const update = (v: T) => {
    setValue(v)
    const url = new URL(window.location.href)
    if (v === fallback) url.searchParams.delete(name)
    else url.searchParams.set(name, v)
    window.history.replaceState(null, '', url)
  }
  return [value, update]
}
