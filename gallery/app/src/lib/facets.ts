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

/**
 * Wrapped composite layout (components/Panels `wrap`, the research and meeting `Grid`): one row while every panel's
 * `min` fits (panelRow), else equal columns as many as the widest `min` allows, the panels flowing into rows
 * (what `repeat(auto-fit, minmax(min, 1fr))` did by hand), at most `maxCols`. One column is a stack. `cols` is the column count,
 * `rows` the row count; widths are whole pixels.
 */
export function panelWrap(width: number, panels: PanelSize[], gap = 24, maxCols = Infinity): { stacked: boolean; cols: number; rows: number; widths: number[] } {
  const n = panels.length
  const row = panelRow(width, panels, gap)
  if (!row.stacked && n <= maxCols) return { stacked: false, cols: Math.max(1, n), rows: 1, widths: row.widths }
  const need = Math.max(...panels.map((p) => p.min ?? 240))
  const cols = Math.max(1, Math.min(n, maxCols, Math.floor((width + gap) / (need + gap))))
  if (cols <= 1) return { stacked: n > 1, cols: 1, rows: n, widths: panels.map((p) => Math.floor(Math.max(p.min ?? 240, Math.min(p.max ?? Infinity, width)))) }
  const cell = Math.floor((width - gap * (cols - 1)) / cols)
  return { stacked: false, cols, rows: Math.ceil(n / cols), widths: panels.map((p) => Math.floor(Math.min(p.max ?? Infinity, cell))) }
}

/**
 * Which panels a group draws, and in what order (components/Panels and FacetGrid `show` / `order`): the indices of
 * `keys` to draw. `show` keeps only those keys (unknown keys are ignored; an empty list or none keeps every panel);
 * `order` puts the keys it names first, in its order, and the rest after them in their own order.
 */
export function pickPanels(keys: readonly string[], show?: readonly string[] | null, order?: readonly string[] | null): number[] {
  const keep = show && show.length ? new Set(show) : null
  const idx = keys.map((_, i) => i).filter((i) => !keep || keep.has(keys[i]))
  if (!order || !order.length) return idx
  const rank = new Map(order.map((k, r) => [k, r] as const))
  const at = (i: number) => rank.get(keys[i]) ?? order.length
  return idx.map((i, pos) => ({ i, pos })).sort((a, b) => at(a.i) - at(b.i) || a.pos - b.pos).map((e) => e.i)
}

/** a numeric range [lo, hi] */
export type Extent = [number, number]

/** the union of ranges, ignoring missing or non-finite ones; null when none is left */
export function unionExtent(extents: Iterable<Extent | null | undefined>): Extent | null {
  let lo = Infinity, hi = -Infinity
  for (const e of extents) {
    if (!e || !Number.isFinite(e[0]) || !Number.isFinite(e[1])) continue
    lo = Math.min(lo, e[0], e[1]); hi = Math.max(hi, e[0], e[1])
  }
  return lo <= hi ? [lo, hi] : null
}

/** the finite extent of some values, or null */
export function extentOf(values: Iterable<number | null | undefined>): Extent | null {
  let lo = Infinity, hi = -Infinity
  for (const v of values) if (v != null && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v }
  return lo <= hi ? [lo, hi] : null
}

/**
 * Shared axis ranges for a panel group (components/PanelGroup): each registration is one chart's own data range on one
 * axis, filed under its range group (lib/axes panelAxesKey: '' for Shared, the site for Per site, the panel for Own).
 * Returns the union per `${group}|${axis}`, so every panel of a range group draws the same range.
 */
export function groupExtents(regs: Iterable<{ group: string; axis: string; extent: Extent }>): Map<string, Extent> {
  const out = new Map<string, Extent>()
  for (const r of regs) {
    const k = `${r.group}|${r.axis}`, u = unionExtent([out.get(k), r.extent])
    if (u) out.set(k, u)
  }
  return out
}
