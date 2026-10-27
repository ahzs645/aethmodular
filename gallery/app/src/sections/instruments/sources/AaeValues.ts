/**
 * Every AAE value a chart on the page draws, for one site: the daily bins or the filters' sampling
 * windows, from absorption (as exported) or from BCc (as the notebooks computed it).
 */
import type { FilterRow } from '@/lib/types'
import type { AethDaily } from '../lib'
import { AAE_PAIRS, MA350_NM, aae, bccOffset, ma350Ratio, type AaePair } from './aae'

export type Basis = 'Daily bins' | 'Filter windows'
export const BASES: readonly Basis[] = ['Daily bins', 'Filter windows']
export type AaeFrom = 'absorption' | 'BCc'

/** a daily bin's AAE: the exported absorption one, or recomputed from that bin's BCc */
export function dailyAae(d: AethDaily, pair: AaePair, from: AaeFrom): (number | null)[] {
  const p = AAE_PAIRS[pair]
  if (from === 'absorption') return d[p.daily]
  const s = d[p.short], l = d[p.long]
  return s.map((v, i) => aae(v, l[i], MA350_NM[p.short], MA350_NM[p.long]))
}

/**
 * A filter's AAE over its sampling window. The export averages the daily absorption AAE over the window;
 * the BCc one is that average plus the fixed offset, which is what averaging each bin's BCc AAE gives.
 */
export function rowAae(r: FilterRow, pair: AaePair, from: AaeFrom): number | null {
  const p = AAE_PAIRS[pair]
  const v = typeof r[p.field] === 'number' ? (r[p.field] as number) : null
  return v === null || from === 'absorption' ? v : v + bccOffset(ma350Ratio(p.short, p.long), MA350_NM[p.short], MA350_NM[p.long])
}
