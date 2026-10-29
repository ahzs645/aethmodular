/**
 * Small-multiple layout: how many columns fit, and each cell's width, from
 * the container width. One rule for every facet grid (docs/gallery-chart-standards.md §1
 * "Sizing"), instead of each chart picking 4, 2 or 1 columns by hand.
 */
export function facetGrid(width: number, n: number, { minCell = 260, maxCols = 4, gap = 16, maxCell = Infinity } = {}) {
  if (n <= 0) return { cols: 1, cell: Math.min(maxCell, Math.max(minCell, width)), gap }
  const fit = Math.max(1, Math.floor((width + gap) / (minCell + gap)))
  const cols = Math.max(1, Math.min(n, maxCols, fit))
  return { cols, cell: Math.min(maxCell, Math.max(minCell, Math.floor((width - gap * (cols - 1)) / cols))), gap }
}

/** One panel of a composite figure (components/Panels): its narrowest, preferred and widest width, and its share of spare room. */
export interface PanelSize { min?: number; basis?: number; max?: number; grow?: number }

/**
 * Composite layout: different charts side by side (a crossplot and its sweep, a chart and its
 * table). Side by side when every panel's `min` fits, each starting at its `basis` (clamped to
 * min..max); spare room goes to panels by `grow` up to their `max`, a shortfall comes out of
 * each panel's room above its `min`. When the mins do not fit, the panels stack, each as wide as
 * the container (up to its `max`). Widths are whole pixels.
 */
export function panelRow(width: number, panels: PanelSize[], gap = 24): { stacked: boolean; widths: number[] } {
  const spec = panels.map((p) => {
    const min = p.min ?? 240, max = Math.max(min, p.max ?? Infinity)
    return { min, max, basis: Math.min(max, Math.max(min, p.basis ?? min)), grow: p.grow ?? 1 }
  })
  const room = width - gap * Math.max(0, spec.length - 1)
  if (spec.length <= 1 || spec.reduce((s, p) => s + p.min, 0) > room)
    return { stacked: spec.length > 1, widths: spec.map((p) => Math.floor(Math.max(p.min, Math.min(p.max, width)))) }
  const w = spec.map((p) => p.basis)
  let spare = room - w.reduce((s, v) => s + v, 0)
  if (spare < 0) {
    // shrink in proportion to each panel's room above its min
    const slack = spec.map((p, i) => w[i] - p.min), total = slack.reduce((s, v) => s + v, 0)
    spec.forEach((_, i) => { w[i] -= total ? (-spare * slack[i]) / total : 0 })
  } else {
    // hand out spare room by grow, again whenever a panel hits its max
    for (let pass = 0; pass < spec.length && spare > 0.5; pass++) {
      const open = spec.map((p, i) => (p.grow > 0 && w[i] < p.max ? i : -1)).filter((i) => i >= 0)
      const g = open.reduce((s, i) => s + spec[i].grow, 0)
      if (!g) break
      let used = 0
      for (const i of open) { const add = Math.min(spec[i].max - w[i], (spare * spec[i].grow) / g); w[i] += add; used += add }
      spare -= used
    }
  }
  return { stacked: false, widths: w.map(Math.floor) }
}

/** the panel letter for index i: a, b, …, z, aa, ab, … */
export const panelLetter = (i: number): string => (i < 26 ? String.fromCharCode(97 + i) : panelLetter(Math.floor(i / 26) - 1) + panelLetter(i % 26))

/**
 * How many equal columns of boxes fit a width, from what the widest box needs (its longest line):
 * as many as fit, up to `n` and `max`, never fewer than one. Used for the fit readouts above a crossplot
 * (components/RegressionOverlay), so short readouts sit side by side and long ones stack.
 */
export function fitColumns(width: number, need: number, n: number, { gap = 12, max = 4 } = {}) {
  if (n <= 1 || need <= 0) return 1
  return Math.max(1, Math.min(n, max, Math.floor((width + gap) / (need + gap))))
}
