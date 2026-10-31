import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import * as d3 from 'd3'
import { schemeStops, type SchemeKind } from '@/lib/colourSchemes'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'
import { useTheme } from '@/lib/colorTheme'
import { getPalette, hasNamedColours, isCustomPalette, paletteColours, paletteNote } from '@/lib/palettes'
import { MAX_STOPS, MIN_STOPS, customStops, deleteCustomPalette, freeName, nameProblem, paletteFromStops, saveCustomPalette, toHex, useCustomPalettes } from '@/lib/customPalettes'
import { PencilIcon } from '@/components/Icons'

/** a palette as a small gradient bar */
const Swatch = ({ stops, wide = false, discrete = false }: { stops: string[]; wide?: boolean; discrete?: boolean }) => (
  <span className={`scheme-swatch${wide ? ' wide' : ''}`} style={{ background: `linear-gradient(90deg, ${discrete
    ? stops.map((c, i) => `${c} ${i * 100 / stops.length}% ${(i + 1) * 100 / stops.length}%`).join(', ')
    : stops.join(', ')})` }} aria-hidden />
)

/** what a colour control needs: the scheme's kind, the current palette and direction, and how to change them */
export interface SchemeChoice {
  kind: SchemeKind
  /** the palettes listed, in order (lib/colourSchemes paletteChoices) */
  names: readonly string[]
  value: string; reverse: boolean; onChange: (name: string, reverse: boolean) => void
}

/**
 * A colour control for a chart with no colour ramp to click (lib/colourSchemes useColourScheme):
 * a button showing the current palette, opening a list of palettes (each as a swatch, in the
 * current direction) and a Reverse switch. A chart with a ramp (components/ColorLegend) is given
 * the scheme instead, and the ramp itself opens the same list.
 */
export function ColourSchemePicker({ kind, names, value, reverse, onChange }: SchemeChoice) {
  useTheme() // the swatch shows the ramp of the current theme
  const [open, setOpen] = useState(false)
  const btn = useRef<HTMLButtonElement>(null)
  return (
    <span className="control">
      <span className="combobox-label">Colours</span>
      <button ref={btn} type="button" className={`combobox-trigger${open ? ' open' : ''}`} aria-haspopup="dialog" aria-expanded={open}
        aria-label={`Colours: ${value}${reverse ? ', reversed' : ''}`}
        title={`${value}${reverse ? ', reversed' : ''}: click for other palettes`} onClick={() => setOpen(!open)}>
        <Swatch stops={schemeStops(kind, value, reverse)} discrete={kind === 'categorical'} />
        <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && btn.current && <SchemePopover anchor={btn.current} kind={kind} names={names} value={value} reverse={reverse} onChange={onChange} onClose={() => setOpen(false)} />}
    </span>
  )
}

/** the list of palettes and the Reverse switch, opened under `anchor` (the picker's button, or a legend's ramp);
 * "+ New palette" and a palette of the reader's own (✎) swap the list for the palette editor */
export function SchemePopover({ anchor, kind, names, value, reverse, onChange, onClose }: SchemeChoice & { anchor: HTMLElement; onClose: () => void }) {
  useTheme()
  useCustomPalettes()
  const box = useRef<HTMLDivElement>(null)
  const trigger = useMemo(() => ({ current: anchor }), [anchor])
  // null: the list; otherwise the editor, on a new palette ('') or one of the reader's own (its name)
  const [editing, setEditing] = useState<string | null>(null)
  const editingRef = useRef(editing)
  editingRef.current = editing
  const pos = useFloatingMenu(true, trigger, box, 450)
  useEffect(() => {
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node) && !anchor.contains(e.target as Node)) onClose() }
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.preventDefault()
      // Escape leaves the editor first, then the list
      if (editingRef.current !== null) { setEditing(null); return }
      onClose(); anchor.focus({ preventScroll: true })
    }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [anchor, onClose])
  return createPortal(
    <div ref={box} className="scheme-pop" role="dialog" aria-label={editing === null ? 'Colour scheme' : editing ? `Edit palette ${editing}` : 'New palette'} style={pos ?? { left: 0, top: 0, visibility: 'hidden' }}
      // React events bubble through the portal: a pick should not reach the chart the ramp sits in
      onClick={(e) => e.stopPropagation()}>
      {editing !== null
        ? <PaletteEditor kind={kind} editing={editing || undefined} startFrom={value}
          onDone={(saved) => { setEditing(null); if (saved) onChange(saved, reverse) }} />
        : <>
          <div className="scheme-list" role="listbox">
            {names.map((n) => {
              const note = paletteNote(kind, n), { source } = getPalette(kind, n), own = isCustomPalette(kind, n)
              const item = (
                <button key={n} type="button" role="option" aria-selected={n === value} className={`scheme-item${n === value ? ' on' : ''}`} title={source} onClick={() => onChange(n, reverse)}>
                  <Swatch stops={schemeStops(kind, n, reverse)} discrete={kind === 'categorical'} wide />
                  <span className="scheme-name">{n}</span>
                  {note && <span className="scheme-note">{note}</span>}
                </button>
              )
              return own
                ? <div key={n} className="scheme-own">{item}
                  <button type="button" className="btn quiet icon-btn scheme-edit" aria-label={`Edit ${n}`} title="Edit or delete this palette" onClick={() => setEditing(n)}><PencilIcon /></button>
                </div>
                : item
            })}
          </div>
          <button type="button" className="scheme-new" onClick={() => setEditing('')}
            title={kind === 'categorical' ? 'Pick your own set of colours, kept in this browser' : 'Make your own gradient from colours you pick, kept in this browser'}>
            + New {kind === 'categorical' ? 'colour set' : 'gradient'}
          </button>
          {!(kind === 'categorical' && hasNamedColours(value)) && <label className="scheme-reverse" title={kind === 'categorical' ? 'Reverse the order of the category colours' : kind === 'diverging' ? 'Swap the colours of negative and positive values' : 'Swap the colours of the low and high ends'}>
            <input type="checkbox" checked={reverse} onChange={(e) => onChange(value, e.target.checked)} />
            Reverse
          </label>}
        </>}
    </div>,
    document.body,
  )
}

const STOP_HINT: Record<SchemeKind, string> = {
  sequential: 'Low to high. The chart blends between the colours.',
  diverging: 'Negative, then zero, then positive: the middle colour is zero.',
  categorical: 'One colour per group, in order; never blended.',
}

/** where a new palette starts: a few stops of the one in use (its ends and middle), or a categorical set's colours */
function startingStops(kind: SchemeKind, from: string): string[] {
  const c = paletteColours(kind, hasNamedColours(from) ? 'Gallery' : from)
  if (kind === 'categorical') return c.slice(0, Math.min(6, MAX_STOPS))
  return [c[0], c[Math.floor((c.length - 1) / 2)], c[c.length - 1]]
}

/**
 * Make or edit a palette of the reader's own (lib/customPalettes): a name, its colours (each a colour
 * well, ✕ to remove, + to add one; or typed or pasted as hex codes), and a preview of the ramp or set as a
 * chart will use it. Saved in this browser and picked at once; Delete removes it from every chart.
 */
function PaletteEditor({ kind, editing, startFrom, onDone }: {
  kind: SchemeKind
  /** the palette being edited, or none for a new one */
  editing?: string
  /** the palette in use, for a new palette's starting colours */
  startFrom: string
  /** back to the list, with the saved palette's name to pick it */
  onDone: (saved?: string) => void
}) {
  const [name, setName] = useState(() => editing ?? freeName(kind, kind === 'categorical' ? 'My colours' : 'My gradient'))
  const [stops, setStops] = useState<string[]>(() => (editing && customStops(kind, editing)) || startingStops(kind, startFrom))
  const [text, setText] = useState(() => stops.join(' '))
  const [confirmDelete, setConfirmDelete] = useState(false)
  useEffect(() => { setText(stops.join(' ')) }, [stops])
  const problem = nameProblem(kind, name, editing)
  const preview = paletteFromStops(kind, stops)
  const setStop = (i: number, c: string) => setStops((s) => s.map((x, j) => (j === i ? c : x)))
  const removeStop = (i: number) => setStops((s) => (s.length > MIN_STOPS ? s.filter((_, j) => j !== i) : s))
  const addStop = () => setStops((s) => {
    if (s.length >= MAX_STOPS) return s
    if (kind === 'categorical') {
      const next = paletteColours('categorical', 'Gallery').find((c) => !s.includes(c)) ?? '#888888'
      return [...s, next]
    }
    // a ramp gains a stop halfway between its last two, so the gradient does not jump
    const mid = d3.interpolateLab(s[s.length - 2], s[s.length - 1])(0.5)
    return [...s.slice(0, -1), d3.color(mid)!.formatHex(), s[s.length - 1]]
  })
  const takeText = () => {
    const parsed = text.split(/[\s,;]+/).map((t) => t && toHex(t)).filter((c): c is string => !!c).slice(0, MAX_STOPS)
    if (parsed.length >= MIN_STOPS) setStops(parsed)
    else setText(stops.join(' '))
  }
  const save = () => {
    if (problem) return
    saveCustomPalette(kind, name, stops, editing)
    onDone(name.trim())
  }
  return (
    <form className="scheme-editor" onSubmit={(e) => { e.preventDefault(); save() }}>
      <div className="scheme-editor-head">
        <button type="button" className="btn quiet icon-btn" aria-label="Back to the palettes" title="Back to the palettes (Esc)" onClick={() => onDone()}>
          <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3.5 5.5 8l4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </button>
        <strong>{editing ? 'Edit palette' : kind === 'categorical' ? 'New colour set' : 'New gradient'}</strong>
      </div>
      <label className="scheme-field">
        <span>Name</span>
        <input className="select" value={name} onChange={(e) => setName(e.target.value)} maxLength={40} aria-invalid={!!problem} autoFocus />
      </label>
      {problem && <span className="scheme-problem" role="alert">{problem}</span>}
      <Swatch stops={preview} discrete={kind === 'categorical'} wide />
      <div className="scheme-stops" role="group" aria-label="Colours">
        {stops.map((c, i) => (
          <span key={i} className="scheme-stop">
            <input type="color" value={c} onChange={(e) => setStop(i, e.target.value)} aria-label={`Colour ${i + 1} of ${stops.length}`} title={c} />
            {stops.length > MIN_STOPS && <button type="button" className="scheme-stop-x" aria-label={`Remove colour ${i + 1}`} title="Remove this colour" onClick={() => removeStop(i)}>×</button>}
          </span>
        ))}
        {stops.length < MAX_STOPS && <button type="button" className="scheme-stop-add" aria-label="Add a colour" title="Add a colour" onClick={addStop}>+</button>}
      </div>
      <span className="scheme-hint">{STOP_HINT[kind]}</span>
      <label className="scheme-field">
        <span>Hex</span>
        <input className="select scheme-hex" value={text} spellCheck={false} onChange={(e) => setText(e.target.value)} onBlur={takeText}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); takeText() } }}
          title="The colours as hex codes: type or paste a list (spaces or commas between), then Enter" />
      </label>
      <div className="scheme-editor-actions">
        {editing && (confirmDelete
          ? <button type="button" className="btn danger" onClick={() => { deleteCustomPalette(kind, editing); onDone() }}>Delete it</button>
          : <button type="button" className="btn quiet danger" onClick={() => setConfirmDelete(true)}>Delete</button>)}
        <span className="scheme-editor-gap" />
        <button type="button" className="btn quiet" onClick={() => onDone()}>Cancel</button>
        <button type="submit" className="btn primary" disabled={!!problem}>Save</button>
      </div>
    </form>
  )
}
