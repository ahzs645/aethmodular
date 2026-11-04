/**
 * Which of a page's own per-filter items Samples leaves in view, before any exclusion, for the items that may have no
 * sampling date. lib/markedRows useSampleSplit leaves an undated item out, which is right for a chart of dated filters;
 * the spectra tables and the PMF inventory also hold undated records (no sampling volume, field blanks) that a saved
 * statistic counts, so they have to stay while Samples is not narrowing anything.
 */

/** an item's site code and sampling date (YYYY-MM-DD; none: it cannot be placed by date) */
export interface FilterPlace { code: string; date: string | null | undefined }

/**
 * The items Samples keeps: a dated item when `inView` says so; an undated one only while every dated item of its site
 * is kept, since nothing can say which side of a date range or season it falls on. So a Samples set that narrows
 * nothing keeps everything, and any narrowing of a site takes that site's undated items out.
 */
export function placedIn<T>(items: readonly T[], at: (t: T) => FilterPlace, inView: (code: string, date: string) => boolean): T[] {
  const cut = new Set<string>()
  const kept = items.map((t) => {
    const { code, date } = at(t)
    if (!date) return null
    const ok = inView(code, date)
    if (!ok) cut.add(code)
    return ok
  })
  return items.filter((t, i) => kept[i] ?? !cut.has(at(t).code))
}
