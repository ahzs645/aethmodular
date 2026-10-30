import { createContext, useContext, useMemo, type ReactNode } from 'react'
import { ChartAside } from '@/lib/singleView'
import { markTip, type ExclMode } from '@/lib/exclusionMode'
import type { FilterRow } from '@/lib/types'

/**
 * Filters → excluded: Mark (lib/exclusionMode). The charts keep receiving the counted rows for every statistic; the
 * excluded filters in view (the same sites, seasons, dates and PMF pick) come from here, for the point charts to
 * draw as × (components/ScatterLayer `marks`) and for the summary charts to count in a note (ExcludedNote).
 */
interface Marked {
  mode: ExclMode
  /** the excluded filters in view; empty unless the mode is mark */
  rows: FilterRow[]
  /** why a filter is out, for its hover; null when it counts */
  reasonOf: (id: string) => string | null
}
const Ctx = createContext<Marked>({ mode: 'hide', rows: [], reasonOf: () => null })

export function MarkedRowsProvider({ mode, rows, reasonOf, children }: Marked & { children: ReactNode }) {
  const value = useMemo(() => ({ mode, rows: mode === 'mark' ? rows : [], reasonOf }), [mode, rows, reasonOf])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useExclMode = () => useContext(Ctx).mode
/** the excluded filters a point chart draws as × (none unless Filters → excluded is Mark); never feed these to a fit */
export const useMarkedRows = () => useContext(Ctx).rows
/** why a filter is excluded (null: it counts); a crossplot fed outside the rows uses it for hand exclusions */
export const useMarkReason = () => useContext(Ctx).reasonOf

/** one excluded filter drawn as a × at plot pixels `at` (components/ScatterLayer); `tip` is its hover */
export interface MarkPoint { id: string; at: readonly [number, number]; tip: string[] }

/**
 * The marks of the excluded filters a chart can place: `at` gives a row's plot position, or null where the chart
 * would not draw it (a missing value, another panel). `extra` adds the chart's own values to the hover.
 */
export function toMarks(rows: readonly FilterRow[], at: (r: FilterRow) => readonly [number, number] | null,
  reasonOf: (id: string) => string | null, extra?: (r: FilterRow) => string[]): MarkPoint[] {
  const out: MarkPoint[] = []
  for (const r of rows) {
    const p = at(r)
    if (!p || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue
    out.push({ id: r.id, at: p, tip: markTip(r, reasonOf(r.id) ?? 'excluded', extra?.(r)) })
  }
  return out
}

/**
 * A page's marks built the way it builds its points: `build` takes the excluded filters in view (useMarkedRows) and
 * returns what the page would draw for them (e.g. CrossPoint[] for components/Crossplot `marked`). Memoised on `deps`.
 */
export function useMarkedPoints<T>(build: (rows: FilterRow[]) => T[], deps: unknown[]): T[] {
  const rows = useMarkedRows()
  return useMemo(() => (rows.length ? build(rows) : []), [rows, ...deps]) // eslint-disable-line react-hooks/exhaustive-deps
}

/**
 * Under Mark, the note on a chart that counts filters: how many excluded filters with its measurement(s) are in
 * view and left out. `drawn`: the chart draws them as × (the point charts); else it only says they are not counted.
 */
export function ExcludedNote({ fields = [], drawn = false, of }: { fields?: string[]; drawn?: boolean; of?: (r: FilterRow) => boolean }) {
  const rows = useMarkedRows()
  const n = useMemo(() => rows.filter((r) => fields.every((f) => typeof r[f] === 'number' && Number.isFinite(r[f] as number)) && (!of || of(r))).length,
    [rows, fields.join('|'), of]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!n) return null
  const what = `${n} excluded filter${n === 1 ? '' : 's'}`
  return <ChartAside><p className="chart-note excluded-note">{drawn
    ? `${what} drawn as × where they fall inside the axes (Filters → excluded: Mark); not counted in any fit, R² or summary.`
    : `${what} not counted (Filters → excluded: Mark draws them only on the point charts).`}</p></ChartAside>
}
