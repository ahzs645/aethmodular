import { useCallback, useRef } from 'react'

/**
 * Structural sharing (as Immer and React Query do it): `next` with every part that equals the same
 * part of `prev` swapped for prev's, and prev itself when nothing changed. Plain objects and arrays
 * are walked `depth` levels; anything else (Maps, functions, class instances) is compared by
 * identity. App rebuilds the charts' `meta` on every render; passed through this, an unchanged
 * meta.fields or meta.sites keeps its identity, so the charts' memos keyed on it hold.
 */
export function share<T>(prev: T, next: T, depth = 4): T {
  if (Object.is(prev, next) || depth <= 0 || prev === null || next === null || typeof prev !== 'object' || typeof next !== 'object') return next
  if (Array.isArray(prev) && Array.isArray(next)) {
    if (prev.length !== next.length) return next
    let same = true
    const out = next.map((v, i) => { const s = share(prev[i], v, depth - 1); if (s !== prev[i]) same = false; return s })
    return (same ? prev : out) as T
  }
  if (Object.getPrototypeOf(prev) !== Object.prototype || Object.getPrototypeOf(next) !== Object.prototype) return next
  const pk = Object.keys(prev), nk = Object.keys(next)
  let same = pk.length === nk.length
  const out: Record<string, unknown> = {}
  for (const k of nk) {
    const pv = (prev as Record<string, unknown>)[k]
    const s = k in (prev as object) ? share(pv, (next as Record<string, unknown>)[k], depth - 1) : (next as Record<string, unknown>)[k]
    if (s !== pv) same = false
    out[k] = s
  }
  return (same ? prev : out) as T
}

/** share() against what the same slot returned on the last render: `const stable = useShared(); stable('meta', meta)` */
export function useShared(): <T>(slot: string, next: T) => T {
  const cache = useRef(new Map<string, unknown>())
  return useCallback(<T,>(slot: string, next: T): T => {
    const v = cache.current.has(slot) ? share(cache.current.get(slot) as T, next) : next
    cache.current.set(slot, v)
    return v
  }, [])
}

/**
 * The previous array back while the rows hold the same filters with the same values in `fields`,
 * so a memo keyed on it skips work whose inputs did not change. A site pick in the sidebar
 * rebuilds every row (each site's colour follows the pick), but a chart drawn for one site, or
 * from a fixed set of fields, has the same numbers to work with and need not recompute them.
 * The check is one pass over the rows × fields, far cheaper than what it saves (a correlation
 * matrix, a force layout).
 */
export function useStableRows<T extends { id: string }>(rows: T[], fields: string[]): T[] {
  const prev = useRef<{ rows: T[]; fields: string[] } | null>(null)
  const p = prev.current
  if (p && sameRows(p.rows, rows, p.fields, fields)) return p.rows
  prev.current = { rows, fields }
  return rows
}

function sameRows<T extends { id: string }>(a: T[], b: T[], fa: string[], fb: string[]): boolean {
  if (a === b) return true
  if (a.length !== b.length || fa.length !== fb.length || fa.some((f, i) => f !== fb[i])) return false
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as Record<string, unknown>, y = b[i] as Record<string, unknown>
    if (x === y) continue
    if (x.id !== y.id) return false
    for (const f of fb) if (x[f] !== y[f] && !(Number.isNaN(x[f]) && Number.isNaN(y[f]))) return false
  }
  return true
}
