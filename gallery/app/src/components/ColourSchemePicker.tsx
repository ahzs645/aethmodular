import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { SCHEMES, SCHEME_NOTES, schemeStops, type SchemeKind } from '@/lib/colourSchemes'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'
import { useTheme } from '@/lib/colorTheme'

/** a palette as a small gradient bar */
const Swatch = ({ stops, wide = false }: { stops: string[]; wide?: boolean }) => (
  <span className={`scheme-swatch${wide ? ' wide' : ''}`} style={{ background: `linear-gradient(90deg, ${stops.join(', ')})` }} aria-hidden />
)

/** what a colour control needs: the scheme's kind, the current palette and direction, and how to change them */
export interface SchemeChoice { kind: SchemeKind; value: string; reverse: boolean; onChange: (name: string, reverse: boolean) => void }

/**
 * A colour control for a chart with no colour ramp to click (lib/colourSchemes useColourScheme):
 * a button showing the current palette, opening a list of palettes (each as a swatch, in the
 * current direction) and a Reverse switch. A chart with a ramp (components/ColorLegend) is given
 * the scheme instead, and the ramp itself opens the same list.
 */
export function ColourSchemePicker({ kind, value, reverse, onChange }: {
  kind: SchemeKind; value: string; reverse: boolean; onChange: (name: string, reverse: boolean) => void
}) {
  useTheme() // the swatch shows the ramp of the current theme
  const [open, setOpen] = useState(false)
  const btn = useRef<HTMLButtonElement>(null)
  return (
    <span className="control">
      <span className="combobox-label">Colours</span>
      <button ref={btn} type="button" className={`combobox-trigger${open ? ' open' : ''}`} aria-haspopup="dialog" aria-expanded={open}
        title={`${value}${reverse ? ', reversed' : ''}: click for other palettes`} onClick={() => setOpen(!open)}>
        <Swatch stops={schemeStops(kind, value, reverse)} />
        <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && btn.current && <SchemePopover anchor={btn.current} kind={kind} value={value} reverse={reverse} onChange={onChange} onClose={() => setOpen(false)} />}
    </span>
  )
}

/** the list of palettes and the Reverse switch, opened under `anchor` (the picker's button, or a legend's ramp) */
export function SchemePopover({ anchor, kind, value, reverse, onChange, onClose }: {
  anchor: HTMLElement; kind: SchemeKind; value: string; reverse: boolean; onChange: (name: string, reverse: boolean) => void; onClose: () => void
}) {
  useTheme()
  const box = useRef<HTMLDivElement>(null)
  const trigger = useMemo(() => ({ current: anchor }), [anchor])
  const pos = useFloatingMenu(true, trigger, box, 450)
  useEffect(() => {
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node) && !anchor.contains(e.target as Node)) onClose() }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); anchor.focus({ preventScroll: true }) } }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [anchor, onClose])
  return createPortal(
    <div ref={box} className="scheme-pop" role="dialog" aria-label="Colour scheme" style={pos ?? { left: 0, top: 0, visibility: 'hidden' }}
      // React events bubble through the portal: a pick should not reach the chart the ramp sits in
      onClick={(e) => e.stopPropagation()}>
      <div className="scheme-list" role="listbox">
        {Object.keys(SCHEMES[kind]).map((n) => (
          <button key={n} type="button" role="option" aria-selected={n === value} className={`scheme-item${n === value ? ' on' : ''}`} onClick={() => onChange(n, reverse)}>
            <Swatch stops={schemeStops(kind, n, reverse)} wide />
            <span className="scheme-name">{n}</span>
            {SCHEME_NOTES[n] && <span className="scheme-note">{SCHEME_NOTES[n]}</span>}
          </button>
        ))}
      </div>
      <label className="scheme-reverse" title={kind === 'diverging' ? 'Swap the colours of negative and positive values' : 'Swap the colours of the low and high ends'}>
        <input type="checkbox" checked={reverse} onChange={(e) => onChange(value, e.target.checked)} />
        Reverse
      </label>
    </div>,
    document.body,
  )
}
