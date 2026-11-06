import type * as d3 from 'd3'
import { FONT, INK } from '@/lib/theme'
import { fitsWidth } from '@/lib/fitLabel'

/** Wrap a label into lines no wider than `room` px, measured at the size and weight it is drawn at (11 px by default); long hyphenated words break after a hyphen. */
export function wrapCategory(label: string, room = 92, size = 11, weight = 400): string[] {
  const words = label.split(/\s+/).flatMap(word => !fitsWidth(word, room, size, weight) && word.includes('-') ? word.match(/[^-]+-?/g) ?? [word] : [word])
  const lines: string[] = []
  for (const word of words) {
    const last = lines.length - 1
    const join = last >= 0 && lines[last].endsWith('-') ? '' : ' '
    if (last >= 0 && fitsWidth(lines[last] + join + word, room, size, weight)) lines[last] += join + word
    else lines.push(word)
  }
  return lines
}
export function categoryLines(label: string, room = 92, size = 11, weight = 400): string[] {
  const paren = label.indexOf(' (')
  return paren < 0 ? wrapCategory(label, room, size, weight) : [
    ...wrapCategory(label.slice(0, paren), room, size, weight),
    ...wrapCategory(label.slice(paren + 2).replace(/\)$/, '').split(',')[0], room, size, weight),
  ]
}
export interface CategoryLabel { key: string; label: string; parent?: string; color?: string; parentColor?: string }
/** Explicit spans let date blocks share the category axis's two label levels. */
export interface CategorySpans { span: (key: string) => [number, number]; range: () => [number, number]; gap: number }
type CategoryScale = d3.ScaleBand<string> | CategorySpans

function layout(items: CategoryLabel[], scale: CategoryScale) {
  const band = 'bandwidth' in scale
  const span = (key: string): [number, number] => band ? [scale(key) ?? 0, (scale(key) ?? 0) + scale.bandwidth()] : scale.span(key)
  const lines = (item: CategoryLabel) => categoryLines(item.label, band ? Math.max(80, scale.step() - 10) : Math.max(1, span(item.key)[1] - span(item.key)[0] - 8))
  const maxLines = Math.max(1, ...items.map(i => lines(i).length))
  const gap = band ? scale.step() - scale.bandwidth() : scale.gap
  const parents = [...new Set(items.map(i => i.parent).filter((p): p is string => !!p))].map(parent => {
    const mine = items.filter(i => i.parent === parent)
    const left = span(mine[0].key)[0], right = span(mine[mine.length - 1].key)[1]
    // wrap within the cluster plus the padding either side, so neighbouring clusters' labels never meet
    return { parent, mine, left, right, lines: categoryLines(parent, Math.max(60, right - left + gap - 8), 12, 600) }
  })
  return { lines, maxLines, parents, span }
}
/** Height the axis takes below the plot: leaf labels, then the wrapped cluster labels. */
export function categoryAxisHeight(items: CategoryLabel[], scale: CategoryScale): number {
  const { maxLines, parents } = layout(items, scale)
  return 20 + maxLines * 14 + (parents.length ? 16 + Math.max(...parents.map(p => p.lines.length)) * 15 : 0)
}
/** Shared category hierarchy, with wrapped leaf labels and one wrapped parent label per cluster. */
export function CategoryAxis({ items, scale, y }: { items: CategoryLabel[]; scale: CategoryScale; y: number }) {
  const { lines, maxLines, parents, span } = layout(items, scale)
  return <g transform={`translate(0,${y})`} fontFamily={FONT.family}>
    <line x2={scale.range()[1]} stroke={INK.axis} />
    {items.map(i => <g key={i.key} data-category-leaf={i.key} data-category-parent={i.parent} transform={`translate(${(span(i.key)[0] + span(i.key)[1]) / 2},0)`}>
      <title>{[i.parent, i.label].filter(Boolean).join(' · ')}</title>
      <text textAnchor="middle" fill={INK.muted} fontSize={11}>{lines(i).map((line, n) => <tspan key={n} x={0} y={16 + n * 14}>{line}</tspan>)}</text>
    </g>)}
    {parents.map(({ parent, mine, left, right, lines: pl }) => {
      const py = 22 + maxLines * 14
      return <g key={parent} data-category-group={parent} fill={mine[0].parentColor ?? mine[0].color ?? INK.text} color={mine[0].parentColor ?? mine[0].color ?? INK.text}>
        <title>{parent}</title>
        <line x1={left} x2={right} y1={py - 12} y2={py - 12} stroke="currentColor" />
        <text textAnchor="middle" fontSize={12} fontWeight={600}>{pl.map((line, n) => <tspan key={n} x={(left + right) / 2} y={py + 3 + n * 15}>{line}</tspan>)}</text>
      </g>
    })}
  </g>
}
