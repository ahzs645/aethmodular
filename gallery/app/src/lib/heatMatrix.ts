import { MONTHS } from './theme.ts'
/**
 * The arithmetic behind components/HeatMatrix and components/BarRows, kept free of React and the DOM
 * so it can be tested in Node: where each row starts when rows come in groups, which row and column a
 * pointer is over, and the month columns of a site × month matrix.
 */

/** The top of each row and the total height: rows of `h` px, with `gap` px before each new group. */
export function rowTops(groups: readonly (string | undefined)[], h: number, gap: number): { tops: number[]; height: number } {
  const tops: number[] = []
  let y = 0
  groups.forEach((g, i) => {
    if (i > 0 && g !== groups[i - 1]) y += gap
    tops.push(y)
    y += h
  })
  return { tops, height: y }
}

/** The row under a y offset (rows `h` tall from `tops`), or -1 in a gap between groups or outside. */
export function rowAt(tops: readonly number[], h: number, y: number): number {
  let lo = 0, hi = tops.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (y < tops[mid]) hi = mid - 1
    else if (y >= tops[mid] + h) lo = mid + 1
    else return mid
  }
  return -1
}

/** The column under an x offset in columns `w` wide, or -1 outside. */
export const colAt = (n: number, w: number, x: number) => {
  const j = Math.floor(x / w)
  return x < 0 || j >= n ? -1 : j
}

/** The start of each run of equal group names: index and name (a year over its months, a site over its lanes). */
export function groupStarts(groups: readonly (string | undefined)[]): { index: number; group: string }[] {
  const out: { index: number; group: string }[] = []
  groups.forEach((g, i) => { if (g !== undefined && (i === 0 || g !== groups[i - 1])) out.push({ index: i, group: g }) })
  return out
}


export interface MonthColumn { key: string; label: string; short: string; group?: string; title: string }

/**
 * The columns of a site × month matrix. 'YYYY-MM' months are consecutive year-months, grouped by
 * year (the year labels and the January rules), labelled by month number; '01'…'12' (or 1…12) are
 * calendar months pooled over the years, labelled Jan…Dec, J…D where a column is narrow.
 */
export function monthColumns(months: readonly (string | number)[]): MonthColumn[] {
  return months.map((m) => {
    const s = String(m)
    if (/^\d{4}-\d{2}$/.test(s)) {
      const k = +s.slice(5, 7)
      return { key: s, label: s.slice(5), short: '', group: s.slice(0, 4), title: `${MONTHS[k - 1] ?? s.slice(5)} ${s.slice(0, 4)}` }
    }
    const k = +s
    const name = MONTHS[k - 1] ?? s
    return { key: s, label: name, short: name[0], title: name }
  })
}
