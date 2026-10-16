import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { prettyUnit } from '@/components/FieldSelect'
import { DEFINE_RATIO, useRatios } from '@/components/RatioDialog'
import { RATIO_GROUP } from '@/lib/viewFields'
import { chemText } from '@/lib/format'
import type { MetaFile } from '@/lib/types'

/** Which shared choice an axis shows: the crossplot pair, or the one measurement of the single-field tabs. */
export type AxisRole = 'x' | 'y' | 'field'

interface AxisFields {
  meta: MetaFile
  value: Record<AxisRole, string>
  set: (role: AxisRole, field: string) => void
  /** offer "Define a ratio…" (the gallery's fields); a chart's own choices (PC1, PC2 …) leave it out */
  ratios?: boolean
  /** the search box's placeholder */
  noun?: string
}

const Ctx = createContext<AxisFields | null>(null)

/**
 * Lets a chart's axis title change the field it shows. App provides the sidebar's x, y and
 * measurement choices here; an axis given `pick` (components/Axes) turns its title into a
 * button that opens a field popover. Both edit the same state, so the sidebar, the URL and
 * every chart on the tab stay in step. Without a provider, axes stay plain text.
 */
export function AxisFieldsProvider({ meta, value, set, ratios = true, noun, children }: AxisFields & { children: ReactNode }) {
  const ctx = useMemo(() => ({ meta, value, set, ratios, noun }), [meta, value, set, ratios, noun])
  return <Ctx.Provider value={ctx}>{children}</Ctx.Provider>
}

export const useAxisFields = () => useContext(Ctx)

const ROLE_TITLE: Record<AxisRole, string> = { x: 'x axis', y: 'y axis', field: 'Measurement' }

/** A searchable, grouped field list anchored to the clicked axis title. */
export function AxisFieldPopover({ role, anchor, onClose }: { role: AxisRole; anchor: DOMRect; onClose: () => void }) {
  const ctx = useAxisFields()
  const ratioCtx = useRatios()
  const ratio = ctx?.ratios === false ? null : ratioCtx
  const box = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  useEffect(() => {
    input.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) onClose() }
    // the page scrolling away from the title closes it; scrolling the popover's own list must not
    const onScroll = (e: Event) => { if (!box.current?.contains(e.target as Node)) onClose() }
    window.addEventListener('keydown', onKey)
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('scroll', onScroll, { passive: true, capture: true })
    return () => {
      window.removeEventListener('keydown', onKey)
      document.removeEventListener('pointerdown', onDown)
      window.removeEventListener('scroll', onScroll, { capture: true })
    }
  }, [onClose])
  if (!ctx) return null
  const { meta, value, set } = ctx
  const q = query.trim().toLowerCase()
  const rows = meta.field_groups.flatMap((g) => g.fields.filter((f) => meta.fields.includes(f)).map((f) => ({ f, group: g.label })))
    .filter((r) => !q || `${r.f} ${r.group}`.toLowerCase().includes(q))
  // the ratio dialog (components/RatioDialog), always last
  if (ratio) rows.push({ f: DEFINE_RATIO, group: RATIO_GROUP })
  const pick = (f: string | undefined) => {
    if (!f) return
    onClose()
    if (f === DEFINE_RATIO) ratio?.open((field) => set(role, field))
    else set(role, f)
  }
  // below the title, kept inside the window
  const W = 300
  const left = Math.max(8, Math.min(window.innerWidth - W - 8, anchor.left + anchor.width / 2 - W / 2))
  // open on the side with more room, and fit the list into it, so the search box never leaves the window
  const roomBelow = window.innerHeight - anchor.bottom - 14, roomAbove = anchor.top - 14
  const below = roomBelow >= Math.min(320, roomAbove)
  const listMax = Math.max(120, Math.min(300, (below ? roomBelow : roomAbove) - 80))
  const style = { left, ...(below ? { top: anchor.bottom + 6 } : { bottom: window.innerHeight - anchor.top + 6 }), ['--list-max' as string]: `${listMax}px` }
  return createPortal(
    <div className="axis-popover" ref={box} style={{ ...style, width: W }} role="dialog" aria-label={`Choose the ${ROLE_TITLE[role].toLowerCase()} field`}>
      <div className="axis-popover-head">
        <strong>{ROLE_TITLE[role]}</strong>
        {role !== 'field' && <button type="button" className="btn quiet" title="Swap x and y"
          onClick={() => { set('x', value.y); onClose() }}>⇅ swap</button>}
      </div>
      <input ref={input} className="select" type="text" placeholder={`Search ${ctx.noun ?? 'measurements'}…`} value={query}
        onChange={(e) => { setQuery(e.target.value); setActive(0) }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(rows.length - 1, a + 1)) }
          else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(0, a - 1)) }
          else if (e.key === 'Enter') { e.preventDefault(); pick(rows[active]?.f) }
        }} />
      <div className="axis-popover-list" role="listbox">
        {rows.map((r, i) => (
          <div key={r.f} role="presentation">
            {r.group !== rows[i - 1]?.group && <div className="combobox-group">{r.group}</div>}
            <button type="button" role="option" aria-selected={r.f === value[role]}
              className={`combobox-option${i === active ? ' active' : ''}`}
              onMouseEnter={() => setActive(i)} onClick={() => pick(r.f)}>
              <span className="combobox-check" aria-hidden="true">{r.f === value[role] ? '✓' : ''}</span>
              <span className="combobox-text">{r.f === DEFINE_RATIO ? 'Define a ratio…' : chemText(r.f)}{prettyUnit(meta.field_units?.[r.f]) && <span className="site-sub"> · {prettyUnit(meta.field_units?.[r.f])}</span>}</span>
              {role !== 'field' && r.f === value[role === 'x' ? 'y' : 'x'] && <span className="combobox-hint" title="Choosing it swaps x and y">on {role === 'x' ? 'y' : 'x'} · swap</span>}
              {meta.coverage?.[r.f] !== undefined && <span className="combobox-hint">{meta.coverage[r.f]}</span>}
            </button>
          </div>
        ))}
        {rows.length === (ratio ? 1 : 0) && <div className="site-empty">No measurement matches “{query}”.</div>}
      </div>
    </div>,
    document.body,
  )
}
