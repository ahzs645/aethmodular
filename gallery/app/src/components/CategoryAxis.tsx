import type * as d3 from 'd3'
import { FONT, INK } from '@/lib/theme'

export function wrapCategory(label: string, room = 92): string[] {
  const words = label.split(/\s+/).flatMap(word => word.length * 6 > room && word.includes('-') ? word.match(/[^-]+-?/g) ?? [word] : [word])
  const lines: string[] = []
  for (const word of words) {
    const last = lines.length - 1
    const join = last >= 0 && lines[last].endsWith('-') ? '' : ' '
    if (last >= 0 && (lines[last] + join + word).length * 6 <= room) lines[last] += join + word
    else lines.push(word)
  }
  return lines
}
export function categoryLines(label: string, room = 92): string[] {
  const paren = label.indexOf(' (')
  return paren < 0 ? wrapCategory(label, room) : [
    ...wrapCategory(label.slice(0, paren), room),
    ...wrapCategory(label.slice(paren + 2).replace(/\)$/, '').split(',')[0], room),
  ]
}
export interface CategoryLabel { key: string; label: string; parent?: string; color?: string; parentColor?: string }
/** Shared season/site hierarchy, with wrapped leaf labels and one site label per cluster. */
export function CategoryAxis({ items, scale, y }: { items: CategoryLabel[]; scale: d3.ScaleBand<string>; y: number }) {
  const parents = [...new Set(items.map(i => i.parent).filter((p): p is string => !!p))]
  const lines = (label: string) => categoryLines(label, Math.max(80, scale.step() - 10))
  const maxLines = Math.max(1, ...items.map(i => lines(i.label).length))
  return <g transform={`translate(0,${y})`} fontFamily={FONT.family}>
    <line x2={scale.range()[1]} stroke={INK.axis} />
    {items.map(i => <g key={i.key} transform={`translate(${(scale(i.key) ?? 0) + scale.bandwidth() / 2},0)`}>
      <title>{[i.parent, i.label].filter(Boolean).join(' · ')}</title>
      <text textAnchor="middle" fill={INK.muted} fontSize={11}>{lines(i.label).map((line, n) => <tspan key={n} x={0} y={16 + n * 14}>{line}</tspan>)}</text>
    </g>)}
    {parents.map(parent => {
      const mine = items.filter(i => i.parent === parent)
      const left = scale(mine[0].key) ?? 0, right = (scale(mine[mine.length - 1].key) ?? 0) + scale.bandwidth()
      const py = 22 + maxLines * 14
      return <g key={parent} fill={mine[0].parentColor ?? mine[0].color ?? INK.text} color={mine[0].parentColor ?? mine[0].color ?? INK.text}>
        <line x1={left} x2={right} y1={py - 12} y2={py - 12} stroke="currentColor" />
        <text x={(left + right) / 2} y={py + 3} textAnchor="middle" fontSize={12} fontWeight={600}>{parent}</text>
      </g>
    })}
  </g>
}
