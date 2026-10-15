/**
 * Small-multiple layout: how many columns fit, and each cell's width, from
 * the container width. One rule for every facet grid (docs/gallery-chart-standards.md §1
 * "Sizing"), instead of each chart picking 4, 2 or 1 columns by hand.
 */
export function facetGrid(width: number, n: number, { minCell = 260, maxCols = 4, gap = 16 } = {}) {
  if (n <= 0) return { cols: 1, cell: Math.max(minCell, width), gap }
  const fit = Math.max(1, Math.floor((width + gap) / (minCell + gap)))
  const cols = Math.max(1, Math.min(n, maxCols, fit))
  return { cols, cell: Math.max(minCell, Math.floor((width - gap * (cols - 1)) / cols)), gap }
}
