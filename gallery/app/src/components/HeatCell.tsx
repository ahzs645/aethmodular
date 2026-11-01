import type { MouseEvent, ReactNode, SVGProps } from 'react'
import { FONT, INK, inkOn } from '@/lib/theme'
import { useTheme } from '@/lib/colorTheme'

export interface HeatCellProps {
  x: number; y: number; width: number; height: number
  /** null draws the empty-cell colour */
  fill: string | null
  /** the value printed in the cell; omit for colour only */
  label?: string | null
  /** a smaller second line under the label (e.g. the k a cell used) */
  sublabel?: string | null
  /** outline the cell, e.g. the current target */
  outline?: boolean
  /** a thin border in this colour (an outline wins) */
  stroke?: string
  strokeWidth?: number
  /** fade the cell (a thin month, a failing fit) */
  opacity?: number
  /** label ink when the fill's own contrast pick is wrong for a faded cell */
  ink?: string
  radius?: number
  onEnter?: (e: MouseEvent) => void
  onLeave?: () => void
  onClick?: () => void
  children?: ReactNode
  /** size of the printed value */
  fontSize?: number
  fontWeight?: number
  /** attributes for the rect, e.g. data-fids so a right-click finds the cell's filters (lib/filterEdits) */
  rectProps?: SVGProps<SVGRectElement>
}

/**
 * One heat-map cell: the coloured rect and, when there is room, its value in
 * ink that stays readable on the fill (lib/theme inkOn). Every heat map pairs
 * its cells with a ColorLegend for the ramp (docs/gallery-chart-standards.md §1);
 * a whole matrix is components/HeatMatrix, which draws its cells with HeatCellBody.
 */
export function HeatCell(props: HeatCellProps) {
  useTheme() // inkOn picks the label for the cell as the current theme shows it
  return <HeatCellBody {...props} />
}

/**
 * HeatCell without its theme subscription: for a caller that already re-renders on a theme switch
 * (HeatMatrix calls useTheme once for all its cells, instead of once per cell).
 */
export function HeatCellBody({ x, y, width, height, fill, label, sublabel, outline, stroke, strokeWidth = 1, opacity, ink, radius, onEnter, onLeave, onClick, children, fontSize = 10, fontWeight, rectProps }: HeatCellProps) {
  const bg = fill ?? INK.empty
  const textInk = ink ?? inkOn(bg)
  const two = label != null && sublabel != null
  const handlers = onEnter || onLeave || onClick
  const body = <>
    <rect x={x} y={y} width={Math.max(0, width)} height={Math.max(0, height)} rx={radius} fill={bg} fillOpacity={opacity}
      stroke={outline ? INK.text : stroke ?? 'none'} strokeWidth={outline ? 1 : stroke ? strokeWidth : 0} {...rectProps} />
    {label != null && <text x={x + width / 2} y={y + height / 2 - (two ? fontSize * 0.45 : 0)} dy="0.35em" textAnchor="middle" fontSize={fontSize} fontWeight={fontWeight} fontFamily={FONT.mono} fill={textInk} pointerEvents="none">{label}</text>}
    {two && <text x={x + width / 2} y={y + height / 2 + fontSize * 0.75} dy="0.35em" textAnchor="middle" fontSize={fontSize * 0.82} fontFamily={FONT.family} fill={textInk} fillOpacity={0.8} pointerEvents="none">{sublabel}</text>}
    {children}
  </>
  return handlers || children
    ? <g onMouseMove={onEnter} onMouseLeave={onLeave} onClick={onClick} style={onClick ? { cursor: 'pointer' } : undefined}>{body}</g>
    : body
}
