import { createContext, useContext, useEffect, useMemo, useState, type DependencyList } from 'react'

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
  // Static hosts may answer a missing data URL with the app's HTML page and
  // status 200. Report a missing export before JSON/gzip decoding obscures it.
  if (r.headers.get('content-type')?.toLowerCase().includes('text/html')) {
    throw new Error(`${path}: the data export is not available on this host`)
  }
  return r
}

/** Explicit gzip assets work on static hosts without Content-Encoding configuration.
 * A host may also decode gzip in transit; the magic bytes prevent double decoding. */
async function bytes(path: string): Promise<ArrayBuffer> {
  const buffer = await (await get(path)).arrayBuffer()
  const header = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength))
  if (path.endsWith('.gz') && header[0] === 0x1f && header[1] === 0x8b) {
    return new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
  }
  return buffer
}

/** Parsed JSON of an exported file (cached per session). */
export const loadJson = <T,>(path: string): Promise<T> => cached(`json:${path}`, async () =>
  path.endsWith('.gz') ? JSON.parse(new TextDecoder().decode(await bytes(path))) as T : (await get(path)).json() as Promise<T>)

/** Raw bytes of an exported binary (cached per session). */
export const loadBinary = (path: string): Promise<ArrayBuffer> => cached(`bin:${path}`, () => bytes(path))

/** Any derived value built from loaded files (an index, a decoded table), cached under its own key. */
export const loadDerived = <T,>(key: string, build: () => Promise<T>): Promise<T> => cached(`derived:${key}`, build)

/** Snapshot data is scoped to a tile; it never poisons the current-source cache. */
export const SnapshotFiles = createContext<Record<string, unknown> | null>(null)
export function useJsonLoader() {
  const files = useContext(SnapshotFiles)
  return useMemo(() => files ? <T,>(path: string): Promise<T> => Object.prototype.hasOwnProperty.call(files, path)
    ? Promise.resolve(files[path] as T) : Promise.reject(new Error(`${path}: not included in this snapshot`)) : loadJson, [files])
}

/**
 * One export for a component. `data` is the file at `path` once it has loaded and
 * passed `check` (which returns an error message, or nothing when the file is
 * usable); it is null while loading and whenever `path` changes. `null` as the
 * path loads nothing.
 */
export function useExport<T>(path: string | null, check?: (data: T) => string | null | undefined | void): { data: T | null; error: string; loading: boolean } {
  const loader = useJsonLoader()
  // check is a pure validator of the same file, so the path stands for it
  const { data, error, loading } = useLoaded<T>(path, () => loader<T>(path!).then((d) => {
    const problem = check?.(d)
    if (problem) throw new Error(`${path}: ${problem}`)
    return d
  }), { deps: [loader] })
  return { data, error, loading }
}

/** The message of a failed load, as the loading panels print it (components/ExportState). */
export const errorText = (e: unknown): string => String(e instanceof Error ? e.message : e)

/** What a loading hook hands back. */
export interface Loaded<T> {
  /** the result once it has landed for the current key (or, with `keep`, the last one that did); null otherwise */
  data: T | null
  /** why the current key failed to load ('' while loading and on success) */
  error: string
  /** a load for the current key is in flight */
  loading: boolean
  /** the key `data` was loaded for: the current key, an older one while a `keep` load is in flight, or null */
  key: string | null
}

/** One settled load: the key it was for, its result or error, and the last load that succeeded (for `keep`). */
export interface Settled<T> { key: string; data: T | null; error: string; good?: { key: string; data: T } }

/**
 * What a loading hook shows for `key` given the last load that settled (pure, so it is node-tested).
 * A load for another key is never shown, except with `keep`: then the last successful load stays up
 * while the current key loads, and beside the current key's error if it fails, so a chart does not
 * blank between two picks. A null key loads nothing.
 */
export function settle<T>(last: Settled<T> | null, key: string | null, keep = false): Loaded<T> {
  if (key === null) return { data: null, error: '', loading: false, key: null }
  const current = last && last.key === key ? last : null
  if (current && (current.data !== null || !keep || !current.good)) return { data: current.data, error: current.error, loading: false, key }
  if (keep && last?.good) return { data: last.good.data, error: current?.error ?? '', loading: !current, key: last.good.key }
  return { data: null, error: '', loading: true, key: null }
}

/** A list of codes as one request key: order and repeats do not matter. */
export const keyOf = (codes: readonly string[]): string => [...new Set(codes)].sort().join(',')
/** The codes of a `keyOf` key. */
export const codesOf = (key: string): string[] => (key ? key.split(',') : [])

export interface LoadOptions {
  /** keep the last successful load up while the next key loads, and beside its error (see `settle`) */
  keep?: boolean
  /** other inputs that should reload the same key (a snapshot's loader from `useJsonLoader`) */
  deps?: DependencyList
}

/**
 * Anything asynchronous a component needs, run once per `key` and dropped if the component has moved on
 * (another key, or unmounted) before it lands; the one place the gallery writes that "live" guard.
 * `key` stands for every input of `load` (`null` loads nothing); `load` itself may be a fresh closure on
 * every render. Put the files it reads through loadJson/loadBinary/loadDerived so they share the cache.
 *
 *   const runs = useLoaded(code, () => loadJson<Runs>(`runs/${code}.json`))
 *   const files = useLoaded(keyOf(codes), () => Promise.all(...), { keep: true })
 */
export function useLoaded<T>(key: string | null, load: () => Promise<T>, { keep = false, deps = [] }: LoadOptions = {}): Loaded<T> {
  const [last, setLast] = useState<Settled<T> | null>(null)
  useEffect(() => {
    if (key === null) return
    let live = true
    load().then(
      (data) => { if (live) setLast({ key, data, error: '', good: data === null ? undefined : { key, data } }) },
      (e) => { if (live) setLast((prev) => ({ key, data: null, error: errorText(e), good: prev?.good })) })
    return () => { live = false }
    // key stands for load's inputs; deps adds what a key cannot name
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, ...deps])
  return settle(last, key, keep)
}

/**
 * A value derived from loaded files (an index, a decoded table, a relabelled export), built once per session
 * under `key` in the shared cache (`loadDerived`) and handed to the component as `useLoaded` does.
 */
export function useDerived<T>(key: string | null, build: () => Promise<T>, options?: LoadOptions): Loaded<T> {
  return useLoaded(key, () => loadDerived(key!, build), options)
}

/**
 * One file (or value) per code, each appearing as it lands: `{ code: data }`, with null for a code whose
 * load failed and no entry while it loads. Codes dropped from the list keep their entries (the record only
 * grows), so switching back is instant. For the sites a page shows, one exported file each.
 */
export function useEach<T>(codes: readonly string[], load: (code: string) => Promise<T>, deps: DependencyList = []): Record<string, T | null> {
  const [files, setFiles] = useState<Record<string, T | null>>({})
  const key = keyOf(codes)
  useEffect(() => {
    let live = true
    for (const code of codesOf(key)) {
      load(code).then(
        (d) => { if (live) setFiles((f) => (f[code] === d ? f : { ...f, [code]: d })) },
        () => { if (live) setFiles((f) => (f[code] === null ? f : { ...f, [code]: null })) })
    }
    return () => { live = false }
    // key stands for the codes; deps adds what it cannot name
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, ...deps])
  return files
}
