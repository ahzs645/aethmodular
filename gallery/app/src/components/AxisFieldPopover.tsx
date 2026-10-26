import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode, useLayoutEffect } from 'react'
import { fieldHelp } from '@/lib/glossary'
import { pickerName } from '@/lib/pmfGroups'
import { createPortal } from 'react-dom'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { DEFINE_RATIO, useRatios } from '@/components/RatioDialog'
import { RATIO_GROUP } from '@/lib/viewFields'
import { chemText } from '@/lib/format'
import type { MetaFile } from '@/lib/types'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'

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
  /** a chart's own choices: what each is called, and which axes offer it (default: every axis) */
  labelOf?: (key: string) => string
  rolesOf?: (key: string) => AxisRole[]
}

const Ctx = createContext<AxisFields | null>(null)

/**
 * Lets a chart's axis title change the field it shows. App provides the sidebar's x, y and
 * measurement choices here; an axis given `pick` (components/Axes) turns its title into a
 * button that opens a field popover. Both edit the same state, so the sidebar, the URL and
 * every chart on the tab stay in step. Without a provider, axes stay plain text.
 */
export function AxisFieldsProvider({ meta, value, set, ratios = true, noun, labelOf, rolesOf, children }: AxisFields & { children: ReactNode }) {
  const ctx = useMemo(() => ({ meta, value, set, ratios, noun, labelOf, rolesOf }), [meta, value, set, ratios, noun, labelOf, rolesOf])
  return <Ctx.Provider value={ctx}>{children}</Ctx.Provider>
}

export const useAxisFields = () => useContext(Ctx)

const ROLE_TITLE: Record<AxisRole, string> = { x: 'x axis', y: 'y axis', field: 'Measurement' }

/** A searchable, grouped field list anchored to the clicked axis title. */
export function AxisFieldPopover({ role, anchor, onClose }: { role: AxisRole; anchor: Element; onClose: (restoreFocus?: boolean) => void }) {
  const ctx = useAxisFields()
  const ratioCtx = useRatios()
  const ratio = ctx?.ratios === false ? null : ratioCtx
  const box = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const trigger = useMemo(() => ({ current: anchor }), [anchor])
  const place = useFloatingMenu(true, trigger, box, 400)
  const positioned = place !== null
  useEffect(() => {
    if (!positioned) return
    if (window.matchMedia('(hover: none), (pointer: coarse)').matches) box.current?.focus({ preventScroll: true })
    else input.current?.focus({ preventScroll: true })
  }, [positioned])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose() } }
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node) && !anchor.contains(e.target as Node)) onClose(false) }
    window.addEventListener('keydown', onKey)
    document.addEventListener('pointerdown', onDown)
    return () => {
      window.removeEventListener('keydown', onKey)
      document.removeEventListener('pointerdown', onDown)
    }
  }, [onClose])
  // open at the current choice, centred in the list, so stepping through nearby fields needs no scrolling
  useLayoutEffect(() => {
    if (!positioned) return
    const list = box.current?.querySelector<HTMLElement>('.axis-popover-list')
    const cur = list?.querySelector<HTMLElement>('[aria-selected="true"]')
    if (!list || !cur) return
    const top = cur.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop
    list.scrollTop = Math.max(0, top - list.clientHeight / 2 + cur.offsetHeight / 2)
    setActive(Number(cur.dataset.i ?? 0))
  }, [positioned])
  // arrow keys keep their choice in view; opening and hovering never move the list
  const keyTo = (i: number) => {
    setActive(i)
    requestAnimationFrame(() => box.current?.querySelector<HTMLElement>(`.axis-popover-list [data-i="${i}"]`)?.scrollIntoView({ block: 'nearest' }))
  }
  if (!ctx) return null
  const { meta, value, set } = ctx
  const q = query.trim().toLowerCase()
  const shown = (f: string) => ctx.labelOf?.(f) ?? f
  const rows = meta.field_groups.flatMap((g) => g.fields.filter((f) => meta.fields.includes(f) && (!ctx.rolesOf || ctx.rolesOf(f).includes(role))).map((f) => ({ f, group: g.label })))
    .filter((r) => !q || `${shown(r.f)} ${r.group}`.toLowerCase().includes(q))
  // the ratio dialog (components/RatioDialog), always last
  if (ratio) rows.push({ f: DEFINE_RATIO, group: RATIO_GROUP })
  const pick = (f: string | undefined) => {
    if (!f || (f !== DEFINE_RATIO && meta.coverage?.[f] === 0 && f !== value[role] && meta.fields.some(key => (meta.coverage?.[key] ?? 0) > 0))) return
    onClose()
    if (f === DEFINE_RATIO) ratio?.open((field) => set(role, field))
    else set(role, f)
  }
  return createPortal(
    <div className="axis-popover" ref={box} tabIndex={-1} style={place ?? { position: 'fixed', visibility: 'hidden' }} role="dialog" aria-label={`Choose the ${ROLE_TITLE[role].toLowerCase()} field`}>
      <div className="axis-popover-head">
        <strong>{ROLE_TITLE[role]}</strong>
        {role !== 'field' && value.x && value.y && (!ctx.rolesOf || ctx.rolesOf(value[role === 'x' ? 'y' : 'x']).includes(role)) && <button type="button" className="btn quiet" title="Swap x and y"
          onClick={() => { set('x', value.y); onClose() }}>⇅ swap</button>}
      </div>
      <input ref={input} className="select" type="text" aria-label={`Search ${ctx.noun ?? 'measurements'}`} placeholder={`Search ${ctx.noun ?? 'measurements'}…`} value={query}
        onChange={(e) => { setQuery(e.target.value); setActive(0) }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') { e.preventDefault(); keyTo(Math.min(rows.length - 1, active + 1)) }
          else if (e.key === 'ArrowUp') { e.preventDefault(); keyTo(Math.max(0, active - 1)) }
          else if (e.key === 'Enter') { e.preventDefault(); pick(rows[active]?.f) }
        }} />
      <div className="axis-popover-list" role="listbox">
        {rows.map((r, i) => (
          <div key={r.f} role="presentation">
            {r.group !== rows[i - 1]?.group && <div className="combobox-group">{r.group}</div>}
            <button type="button" role="option" aria-selected={r.f === value[role]} data-i={i} disabled={meta.coverage?.[r.f] === 0 && r.f !== value[role] && meta.fields.some(key => (meta.coverage?.[key] ?? 0) > 0)}
              className={`combobox-option${i === active ? ' active' : ''}`}
              onMouseEnter={() => setActive(i)} onClick={() => pick(r.f)}>
              <span className="combobox-check" aria-hidden="true">{r.f === value[role] ? '✓' : ''}</span>
              <span className="combobox-text" title={r.f === DEFINE_RATIO ? undefined : chemText(fieldHelp(r.f, shown(r.f)))}>{r.f === DEFINE_RATIO ? 'Define a ratio…' : chemText(pickerName(shown(r.f)))}{prettyUnit(meta.field_units?.[r.f]) && <span className="site-sub"> · {prettyUnit(meta.field_units?.[r.f])}</span>}</span>
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

/** Keep measurement recovery available even when a chart has no axes to click. */
export function MeasurementControl() {
  const ctx = useAxisFields()
  return ctx ? <FieldSelect label="Measurement" value={ctx.value.field} meta={ctx.meta} onChange={(f) => ctx.set('field', f)} /> : null
}
