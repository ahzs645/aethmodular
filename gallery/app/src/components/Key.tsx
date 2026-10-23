import type { ReactNode } from 'react'
import { FIT_STYLE, useEstimator } from '@/lib/regressionPrefs'
import { INK } from '@/lib/theme'

/**
 * A static key in the gallery's legend style: what each colour, mark and line
 * means, for charts whose series are fixed (the interactive `Legend` hides and
 * dims series). One set of marks for every research page.
 */
export type KeyKind = 'dot' | 'square' | 'hollow' | 'diamond' | 'line' | 'dash' | 'bar' | 'faded'
export interface KeyItem { label: string; color: string; kind?: KeyKind }

export function Key({ items, children, relabelFits = false }: {
  items: KeyItem[]
  /** extra legend content, e.g. a ColorLegend ramp */
  children?: ReactNode
  /** labels say "Deming" for a fit the browser refits: show the active estimator's name instead */
  relabelFits?: boolean
}) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label} className="legend-item">
          <svg width={20} height={12} aria-hidden="true">
            {(it.kind ?? 'dot') === 'dot' && <circle cx={10} cy={6} r={4} fill={it.color} fillOpacity={0.8} />}
            {it.kind === 'square' && <rect x={6} y={2} width={8} height={8} fill={it.color} />}
            {it.kind === 'hollow' && <circle cx={10} cy={6} r={4} fill="none" stroke={it.color} strokeWidth={1.2} />}
            {it.kind === 'diamond' && <path d="M10 1 L15 6 L10 11 L5 6 Z" fill={INK.panel} stroke={it.color} strokeWidth={1.3} />}
            {it.kind === 'line' && <line x1={1} x2={19} y1={6} y2={6} stroke={it.color} strokeWidth={2.2} />}
            {it.kind === 'dash' && <line x1={1} x2={19} y1={6} y2={6} stroke={it.color} strokeWidth={1.4} strokeDasharray="4 3" />}
            {it.kind === 'bar' && <rect x={1} y={3} width={18} height={6} fill={it.color} />}
            {it.kind === 'faded' && <rect x={1} y={1} width={18} height={10} fill={it.color} opacity={0.3} />}
          </svg>
          {relabelFits ? <FitLabel text={it.label} /> : it.label}
        </span>
      ))}
      {children}
    </div>
  )
}

function FitLabel({ text }: { text: string }) {
  return <>{text.replace(/Deming/g, FIT_STYLE[useEstimator()].label)}</>
}
