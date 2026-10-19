/**
 * The PMF paper's Figure 7 (daily contribution of sources to OM, Addis 2023): its four seasons,
 * read off the figure's strip (First Dry Jan–Feb, Less Rainy Mar–Apr, More Rainy May–Sep, Second
 * Dry Oct–Dec), its stacking order from the bottom, and its colours. Shared by the PMF tab's charts.
 */
export const PAPER_SEASONS = [
  { name: 'First Dry', months: [1, 2], color: '#f8cfa5' },
  { name: 'Less Rainy', months: [3, 4], color: '#a8d8e8' },
  { name: 'More Rainy', months: [5, 6, 7, 8, 9], color: '#5aaee8' },
  { name: 'Second Dry', months: [10, 11, 12], color: '#f09a5a' },
]
export const PAPER_ORDER = ['Fossil Fuel Combustion', 'Polluted Marine', 'Sea Salt Mixed', 'Wood Burning', 'Charcoal']
export const PAPER_COLORS: Record<string, string> = {
  'Fossil Fuel Combustion': '#f4b6c2', 'Polluted Marine': '#e0191c', 'Sea Salt Mixed': '#0a14d0', 'Wood Burning': '#92e192', Charcoal: '#2e8b2e',
}

/** the paper season a filter's date falls in */
export const paperSeasonOf = (date: string) => {
  const m = Number(date.slice(5, 7))
  return PAPER_SEASONS.find((s) => (s.months as readonly number[]).includes(m)) ?? null
}
