/** Sampling diagnostics describe records in an export, rather than operational site status. */
export interface DatedRecord { id: string; date: string; [field: string]: unknown }
export interface AvailabilityFile { rows?: DatedRecord[]; code?: string; date?: string[]; poc?: number[]; [field: string]: unknown }
/** SPARTAN exports rows; IMPROVE exports columns keyed by site/date/POC. */
export function availabilityRecords(file: AvailabilityFile): DatedRecord[] {
  if (Array.isArray(file.rows)) return file.rows
  if (!Array.isArray(file.date) || !file.code) return []
  const columns = Object.entries(file).filter(([key, values]) => key !== 'date' && key !== 'poc' && Array.isArray(values))
  return file.date.map((date, i) => ({ id: `${file.code}-${date}-${file.poc?.[i] ?? 1}`, date,
    ...Object.fromEntries(columns.map(([key, values]) => [key, (values as unknown[])[i] ?? null])) }))
}
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
export function recentCutoff(asOf: string, years: number): string {
  const date = new Date(`${asOf}T00:00:00Z`)
  if (!Number.isFinite(+date)) return ''
  const month = date.getUTCMonth()
  date.setUTCFullYear(date.getUTCFullYear() - Math.max(0, Math.round(years)))
  // Clamp leap day to February's last day.
  if (date.getUTCMonth() !== month) date.setUTCDate(0)
  return date.toISOString().slice(0, 10)
}
export function samplingStatus(last: string | null | undefined, cutoff: string, asOf: string) {
  return !last || !/^\d{4}-\d{2}-\d{2}$/.test(last) || !Number.isFinite(Date.parse(last)) ? 'No samples' : last > asOf ? 'After reference date' : last >= cutoff ? 'Recent samples' : 'No recent samples'
}
export function monthlyAvailability(records: DatedRecord[], field?: string) {
  // Repeated analyses of the same physical filter must not inflate sampling counts.
  const dated = [...new Map(records.filter(r => /^\d{4}-\d{2}-\d{2}$/.test(r.date) && Number.isFinite(Date.parse(r.date))).map(r => [r.id, r])).values()].sort((a, b) => a.date.localeCompare(b.date))
  if (!dated.length) return { months: [], gaps: [], jumps: [], records: 0, measured: 0 }
  const counts = new Map<string, { samples: number; measured: number }>()
  for (const r of dated) {
    const key = r.date.slice(0, 7), c = counts.get(key) ?? { samples: 0, measured: 0 }
    c.samples++; if (!field || finite(r[field])) c.measured++
    counts.set(key, c)
  }
  const start = new Date(`${dated[0].date.slice(0, 7)}-01T00:00:00Z`), end = dated[dated.length - 1].date.slice(0, 7)
  const months: { period: string; samples: number; measured: number }[] = []
  for (let d = start; d.toISOString().slice(0, 7) <= end; d.setUTCMonth(d.getUTCMonth() + 1)) {
    const period = d.toISOString().slice(0, 7)
    months.push({ period, ...(counts.get(period) ?? { samples: 0, measured: 0 }) })
  }
  const gaps: { start: string; end: string; months: number }[] = []
  for (const [i, m] of months.entries()) {
    if (m.samples) continue
    const last = gaps[gaps.length - 1]
    if (i > 0 && !months[i - 1].samples && last) { last.end = m.period; last.months++ }
    else gaps.push({ start: m.period, end: m.period, months: 1 })
  }
  // Flag only substantial changes between consecutive observed months; gaps are reported separately.
  const jumps = months.flatMap((m, i) => {
    const prev = months[i - 1]
    return prev && prev.samples > 0 && m.samples > 0 && Math.abs(m.samples - prev.samples) >= 3 && Math.max(m.samples, prev.samples) / Math.min(m.samples, prev.samples) >= 2
      ? [{ period: m.period, before: prev.samples, after: m.samples }] : []
  })
  return { months, gaps, jumps, records: dated.length, measured: dated.filter(r => !field || finite(r[field])).length }
}
