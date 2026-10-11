import { createContext, useContext, type ReactNode } from 'react'
import { baselineColor } from '@/lib/theme'
import { baselineName } from '@/lib/labels'

/**
 * The background rank of the VIBES run a view is showing. The Baselines exports
 * call every VIBES run "VIBES"; the rank says whether that is the archived
 * truncated fit or the completed rank-70 run, and so which colour and name it gets.
 */
const BaselineRank = createContext<number | null>(null)

export function BaselineRankProvider({ rank, children }: { rank: number | null | undefined; children: ReactNode }) {
  return <BaselineRank.Provider value={rank ?? null}>{children}</BaselineRank.Provider>
}

/** Colour and reader-facing name for a method code, under the view's VIBES run. */
export function useBaselineStyle() {
  const rank = useContext(BaselineRank)
  return {
    rank,
    color: (method: string) => baselineColor(method, rank),
    name: (method: string, short = false) => baselineName(method, { short, backgroundRank: rank }),
  }
}

/** The same, as a lookup table (`COLOR.VIBES`, `COLOR[method]`) for charts written against one. */
export function useBaselineColors(): Record<string, string> {
  const rank = useContext(BaselineRank)
  return new Proxy({} as Record<string, string>, { get: (_, key) => baselineColor(String(key), rank) })
}
