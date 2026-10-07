/**
 * Visual constants. Site colours are NOT invented here — they are exported
 * from research/ftir_hips_chem/scripts/config.py (SITES[site]['color']) and
 * arrive in meta.json, so the web gallery and the matplotlib figures agree.
 * Season colours likewise come from config.ETHIOPIA_SEASONS.
 */
export const MARGIN = { top: 28, right: 24, bottom: 52, left: 68 }

export const INK = {
  axis: '#5b6470',
  grid: '#e8ebef',
  text: '#1f2933',
  muted: '#7b8794',
  fit: '#111827',
  deming: '#c026d3',
  identity: '#9aa5b1',
  panel: '#ffffff',
  page: '#f6f7f9',
  border: '#dde2e8',
  /** the app accent (nav, selection, positive r) */
  accent: '#2171b5',
  positive: '#2171b5',
  negative: '#b2182b',
  /** "passes the test" green (held-out floor, built badge) */
  good: '#0f766e',
  /** a category that carries no signal: "no HIPS", "no carbon EC" */
  neutral: '#c7ccd1',
  /** an empty cell */
  empty: '#f3f5f8',
}

/** Sequential + diverging ramps used by the heatmap and correlogram. */
export const RAMP_DIVERGING = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']
export const RAMP_SEQUENTIAL = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']

export const FONT = {
  family: 'ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
  mono: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace',
}

/** "EC (FTIR)" + "ug/m3" -> "EC (FTIR) (µg/m³)". Units come from meta.json. */
export function withUnit(field: string, units: Record<string, string> | undefined): string {
  const u = units?.[field]
  if (!u) return field
  const pretty = u.replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹')
  return `${field} (${pretty})`
}
