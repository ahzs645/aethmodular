import type { ScaleTime } from 'd3'
import type { SeasonBand } from '@/components/SeasonBands'
import type { SeasonMeta } from '@/lib/types'

/**
 * The PMF paper's own conventions (Anwar et al., Addis Ababa, Atmos. Env. draft), shared by the
 * PMF tab's charts and the drawer.
 *
 * Seasons: defined in the paper from 2023 precipitation, "following the seasonal representation
 * used in Ethiopia", by date rather than by month:
 *   First Dry   1 Jan – 25 Feb 2023   (~0 mm/day)
 *   Less Rainy 27 Feb – 27 Apr 2023
 *   More Rainy 30 Apr – 30 Sep 2023   (~8.3 mm/day)
 *   Second Dry  1 Oct – 17 Dec 2023   (~0.1 mm/day)
 * Samples are every third day, so the cut points sit between sample dates: a date is in the
 * season whose start (month-day) it has reached. On pmf.json this gives First Dry 18, Less Rainy
 * 17, More Rainy 45, Second Dry 22: Second Dry matches the paper's n=22; its First Dry n=17 is one
 * fewer than the export's 18 (2023-02-10 sits two days after 02-08, off the every-third-day
 * cadence, perhaps a filter the paper left out). Stacking order and colours are its Figure 7's.
 */
export const PAPER_SEASONS = [
  { name: 'First Dry', from: '01-01', months: [1, 2], color: '#f8cfa5', note: '1 Jan – 25 Feb' },
  { name: 'Less Rainy', from: '02-26', months: [3, 4], color: '#a8d8e8', note: '27 Feb – 27 Apr' },
  { name: 'More Rainy', from: '04-29', months: [5, 6, 7, 8, 9], color: '#5aaee8', note: '30 Apr – 30 Sep' },
  { name: 'Second Dry', from: '10-01', months: [10, 11, 12], color: '#f09a5a', note: '1 Oct – 17 Dec' },
] as const
/** the same seasons as a gallery calendar (Filters → season calendar 'PMF paper'), dated boundaries included */
export const PAPER_CALENDAR: SeasonMeta[] = PAPER_SEASONS.map(({ name, from, months, color }) => ({ name, from, months: [...months], color }))
export const PAPER_ORDER = ['Fossil Fuel Combustion', 'Polluted Marine', 'Sea Salt Mixed', 'Wood Burning', 'Charcoal']
export const PAPER_COLORS: Record<string, string> = {
  'Fossil Fuel Combustion': '#f4b6c2', 'Polluted Marine': '#e0191c', 'Sea Salt Mixed': '#0a14d0', 'Wood Burning': '#92e192', Charcoal: '#2e8b2e',
}

/** the paper season a filter's date (YYYY-MM-DD) falls in */
export const paperSeasonOf = (date: string) => {
  const md = date.slice(5, 10)
  return [...PAPER_SEASONS].reverse().find((s) => md >= s.from) ?? PAPER_SEASONS[0]
}

/** the paper's seasons as bands along a time axis (each year in view), clipped to [0, width] */
export function paperSeasonBands(x: ScaleTime<number, number>, width: number): SeasonBand[] {
  const [d0, d1] = x.domain() as [Date, Date]
  const bands: SeasonBand[] = []
  for (let year = d0.getFullYear(); year <= d1.getFullYear(); year++) {
    PAPER_SEASONS.forEach((s, i) => {
      const start = new Date(`${year}-${s.from}T00:00:00`)
      const end = i + 1 < PAPER_SEASONS.length ? new Date(`${year}-${PAPER_SEASONS[i + 1].from}T00:00:00`) : new Date(`${year + 1}-01-01T00:00:00`)
      const a = Math.max(0, x(start)), b = Math.min(width, x(end))
      if (b > a) bands.push({ x: a, w: b - a, color: s.color, name: s.name })
    })
  }
  return bands
}
