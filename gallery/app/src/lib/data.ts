import { useEffect, useState } from 'react'

/**
 * Every exported file the gallery reads goes through here: one base path, one
 * cache, one error format. A file is fetched once per session however many
 * charts ask for it (concurrent requests share the same promise); a failed
 * fetch is forgotten so a later request can retry.
 *
 *   const { data, error } = useExport<MeetingData>('meeting-r70/followup_20260917.json', d =>
 *     d.schema_version === 1 ? null : 'unsupported data version')
 */
const DATA = `${import.meta.env?.BASE_URL ?? "/"}data/`

/** URL of an exported file, from its path under public/data. */
export const dataUrl = (path: string) => DATA + path.replace(/^\/+/, '')

const cache = new Map<string, Promise<unknown>>()

function cached<T>(key: string, load: () => Promise<T>): Promise<T> {
  let p = cache.get(key) as Promise<T> | undefined
  if (!p) {
    p = load()
    p.catch(() => cache.delete(key))
    cache.set(key, p)
  }
  return p
}

async function get(path: string): Promise<Response> {
  const r = await fetch(dataUrl(path))
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return r
}

/** Parsed JSON of an exported file (cached per session). */
export const loadJson = <T,>(path: string): Promise<T> => cached(`json:${path}`, () => get(path).then((r) => r.json() as Promise<T>))

/** Raw bytes of an exported binary (cached per session). */
export const loadBinary = (path: string): Promise<ArrayBuffer> => cached(`bin:${path}`, () => get(path).then((r) => r.arrayBuffer()))

/** Any derived value built from loaded files (an index, a decoded table), cached under its own key. */
export const loadDerived = <T,>(key: string, build: () => Promise<T>): Promise<T> => cached(`derived:${key}`, build)

/**
 * One export for a component. `data` is the file at `path` once it has loaded and
 * passed `check` (which returns an error message, or nothing when the file is
 * usable); it is null while loading and whenever `path` changes. `null` as the
 * path loads nothing.
 */
export function useExport<T>(path: string | null, check?: (data: T) => string | null | undefined | void): { data: T | null; error: string; loading: boolean } {
  const [state, setState] = useState<{ path: string; data: T | null; error: string } | null>(null)
  useEffect(() => {
    if (!path) return
    let live = true
    loadJson<T>(path)
      .then((d) => {
        const problem = check?.(d)
        if (live) setState({ path, data: problem ? null : d, error: problem ? `${path}: ${problem}` : '' })
      })
      .catch((e) => { if (live) setState({ path, data: null, error: String(e instanceof Error ? e.message : e) }) })
    return () => { live = false }
    // check is a pure validator of the same file
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path])
  const current = state && state.path === path ? state : null
  return { data: current?.data ?? null, error: current?.error ?? '', loading: !!path && !current }
}
