/**
 * A legend's item order, set by hand in the export dialog (components/ExportLegendOrder). It reorders the legend
 * alone: `Legend` sorts its own copy of the items, and the chart keeps the array it draws from, so the marks, their
 * colours and what sits on top do not change. An order is a list of item labels, kept per chart.
 *
 * An item the order does not name (a site added to the subset later) keeps its place among the others after the named
 * ones, and a label no longer shown is ignored, so a saved order survives a change of subset.
 */
export interface SavedLegendOrder {
  order: string[]
  /** the legend on the page follows it too; otherwise only the file does */
  page: boolean
}

export interface LegendRow { key: string; label: string; color: string; shape: string; off: boolean }
/** one group of a legend's items: a site in a "Site · Season" legend (`name`), or the items outside any group (null) */
export interface LegendBlock { name: string | null; rows: LegendRow[] }

export function orderBy<T>(items: readonly T[], order: readonly string[] | undefined, keyOf: (item: T) => string): T[] {
  if (!order?.length) return [...items]
  const rank = new Map(order.map((key, i) => [key, i]))
  const at = (item: T) => rank.get(keyOf(item)) ?? order.length
  return items.map((item, i) => ({ item, i })).sort((a, b) => at(a.item) - at(b.item) || a.i - b.i).map(({ item }) => item)
}

/** the blocks as `Legend` draws them under an order: loose items first, then groups by their first member */
export function arrangeBlocks(blocks: readonly LegendBlock[], order: readonly string[] | undefined): LegendBlock[] {
  const sorted = blocks.map((b) => ({ ...b, rows: orderBy(b.rows, order, (r) => r.key) }))
  const rank = new Map((order ?? []).map((key, i) => [key, i]))
  const first = (b: LegendBlock) => Math.min(...b.rows.map((r) => rank.get(r.key) ?? Infinity))
  const groups = sorted.filter((b) => b.name !== null).map((b, i) => ({ b, i }))
    .sort((a, z) => { const d = first(a.b) - first(z.b); return Number.isNaN(d) || d === 0 ? a.i - z.i : d })
  return [...sorted.filter((b) => b.name === null), ...groups.map(({ b }) => b)]
}

export const flatKeys = (blocks: readonly LegendBlock[]) => blocks.flatMap((b) => b.rows.map((r) => r.key))

/** `list` with the item at `from` moved to index `to` */
export function moved<T>(list: readonly T[], from: number, to: number): T[] {
  const next = [...list]
  const [item] = next.splice(from, 1)
  next.splice(to, 0, item)
  return next
}

/** a new order for what is shown, keeping the places of labels the subset has hidden for now */
export const mergeOrder = (shown: readonly string[], saved: readonly string[] = []) =>
  [...shown, ...saved.filter((key) => !shown.includes(key))]

export function readLegendOrders(value: unknown): Map<string, SavedLegendOrder> {
  const out = new Map<string, SavedLegendOrder>()
  if (!value || typeof value !== 'object' || Array.isArray(value)) return out
  for (const [key, entry] of Object.entries(value)) {
    const raw = entry as Partial<SavedLegendOrder> | null
    if (!Array.isArray(raw?.order)) continue
    const order = raw.order.filter((label): label is string => typeof label === 'string')
    if (order.length) out.set(key, { order, page: raw.page === true })
  }
  return out
}
