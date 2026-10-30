import { useEffect, useState } from 'react'
import { loadEtadIds } from '@/lib/similarity'

/**
 * An Addis filter's SPARTAN id (ETAD-0017) by its ETAD MediaId, from similarity/etad_ids.json, so a table keyed by
 * MediaId (a meeting export) joins the gallery's filters and follows Filters → excluded. Empty where the file is
 * missing: the MediaId then stays the id, as before.
 */
let byMedia: Promise<Map<string, string>> | null = null
const load = () => (byMedia ??= loadEtadIds()
  .then((j) => new Map(Object.entries(j.ids).flatMap(([base, sids]) => sids.map((s) => [mediaKey(s.replace(/^etad:/, '')), base] as const))))
  .catch(() => new Map<string, string>()))

/** a MediaId as the map keys it: 401.0 and '401' are one filter */
export const mediaKey = (v: unknown) => { const n = Number(v); return Number.isFinite(n) ? String(n) : String(v) }

/** the MediaId → SPARTAN id map, loaded only when `on` (a table with a MediaId column); null until then */
export function useEtadByMedia(on: boolean): Map<string, string> | null {
  const [map, setMap] = useState<Map<string, string> | null>(null)
  useEffect(() => {
    if (!on) return
    let live = true
    load().then((m) => { if (live) setMap(m) })
    return () => { live = false }
  }, [on])
  return on ? map : null
}
