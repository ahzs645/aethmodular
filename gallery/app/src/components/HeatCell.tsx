import type { MouseEvent, ReactNode } from 'react'
import { FONT, INK, inkOn } from '@/lib/theme'

/**
 * One heat-map cell: the coloured rect and, when there is room, its value in
 * ink that stays readable on the fill (lib/theme inkOn). Every heat map pairs
 * its cells with a ColorLegend for the ramp (docs/gallery-chart-standards.md §1).
 */
export function HeatCell({ x, y, width, height, fill, label, outline, onEnter, onLeave, onClick, children }: {
  x: number; y: number; width: number; height: number
  /** null draws the empty-cell colour */
  fill: string | null
  /** the value printed in the cell; omit for colour only */
  label?: string | null
  /** outline the cell, e.g. the current target */
  outline?: boolean
  onEnter?: (e: MouseEvent) => void
  onLeave?: () => void
  onClick?: () => void
  children?: ReactNode
}) {
  const bg = fill ?? INK.empty
  return <g onMouseMove={onEnter} onMouseLeave={onLeave} onClick={onClick} style={onClick ? { cursor: 'pointer' } : undefined}>
    <rect x={x} y={y} width={Math.max(0, width)} height={Math.max(0, height)} fill={bg} stroke={outline ? INK.text : 'none'} strokeWidth={outline ? 1 : 0} />
    {label != null && <text x={x + width / 2} y={y + height / 2} dy="0.35em" textAnchor="middle" fontSize={10} fontFamily={FONT.mono} fill={inkOn(bg)} pointerEvents="none">{label}</text>}
    {children}
  </g>
}
