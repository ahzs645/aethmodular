import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'

/**
 * The header's page menu: tabs in a few groups (Charts, Sites, Calibration, Meetings), each a
 * button that opens its list on hover (or click / Enter, for touch and keyboard). The group
 * holding the open page shows that page's name. When the header is too narrow for every
 * group, the ones that do not fit fold into "More" (all of them into "Menu" on a phone);
 * the widths are measured, so it adapts to the labels and the text size. `extra` adds a section
 * to the phone's menu (the header's display toggles go to its ⋯ menu instead, components/HeaderMore).
 */
/** `meeting` picks a meeting page; `sub` picks a page on any other tab with subpages (Experimental);
 *  `heading` puts a group's items under a sub-heading (Investigations: the open area's name);
 *  `home` is the group's own landing page, which leaves the group's name as it is */
export interface NavItem { key: string; label: string; title?: string; tab: string; meeting?: string; sub?: string; heading?: string; home?: boolean }
export interface NavGroup { label: string; items: NavItem[] }

const isActive = (i: NavItem, tab: string, meeting: string, sub?: string) => i.tab === tab && (i.meeting === undefined || i.meeting === meeting) && (i.sub === undefined || i.sub === sub)

export function NavMenu({ groups, tab, meeting, sub, onPick, extra }: {
  groups: NavGroup[]; tab: string; meeting: string; onPick: (i: NavItem) => void
  /** the open subpage of a tab other than Meeting */
  sub?: string
  /** controls the header hides on a phone (styles.css), offered at the foot of More / Menu instead */
  extra?: ReactNode
}) {
  const box = useRef<HTMLDivElement>(null)
  const measure = useRef<HTMLDivElement>(null)
  const [fit, setFit] = useState(groups.length)
  // a phone (styles.css hides the header's display toggles below 560 px): More always exists to hold them
  const [phone, setPhone] = useState(() => window.matchMedia('(max-width: 560px)').matches)
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 560px)')
    const on = () => setPhone(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  // the group shows the open page's name, or its area's where the group has sub-headings
  const label = (g: NavGroup) => { const a = g.items.find((i) => isActive(i, tab, meeting, sub)); return a && !a.home ? `${g.label} · ${a.heading ?? a.label}` : g.label }
  const labels = groups.map(label).join('|')

  // how many groups fit beside the title and tools; the rest go into More
  useLayoutEffect(() => {
    const el = box.current, m = measure.current
    if (!el || !m) return
    const update = () => {
      const widths = [...m.querySelectorAll<HTMLElement>('[data-measure="group"]')].map((b) => b.offsetWidth)
      const more = m.querySelector<HTMLElement>('[data-measure="more"]')?.offsetWidth ?? 70
      const gap = 4, room = el.clientWidth
      let used = 0, n = 0
      for (let i = 0; i < widths.length; i++) {
        const last = i === widths.length - 1
        const need = used + widths[i] + (n ? gap : 0)
        if (need + (last ? 0 : gap + more) > room) break
        used = need; n++
      }
      setFit(n)
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [labels, groups.length])

  // one list open at a time: hovering another group swaps at once; leaving closes after a short grace
  // (enough to cross from a button into its own list)
  const [openId, setOpenId] = useState<string | null>(null)
  const timer = useRef<number>()
  const openedAt = useRef(0)
  const menu = (id: string) => ({
    open: openId === id,
    show: () => { window.clearTimeout(timer.current); if (openId !== id) openedAt.current = Date.now(); setOpenId(id) },
    hide: () => { window.clearTimeout(timer.current); timer.current = window.setTimeout(() => setOpenId((cur) => (cur === id ? null : cur)), 90) },
    // a tap fires a hover and a click together: a click right after the hover opened it keeps it open
    toggle: () => { if (openId === id && Date.now() - openedAt.current < 500) return; setOpenId(openId === id ? null : id) },
    close: () => { window.clearTimeout(timer.current); setOpenId(null) },
  })
  useEffect(() => () => window.clearTimeout(timer.current), [])

  const shown = groups.slice(0, fit), folded = groups.slice(fit)
  return (
    <div className="navmenu" ref={box}>
      {/* the same buttons, invisible, to measure */}
      <div className="navmenu-measure" ref={measure} aria-hidden>
        {groups.map((g) => <span key={g.label} data-measure="group" className="nav-btn nav-group">{label(g)} ▾</span>)}
        <span data-measure="more" className="nav-btn nav-group">More ▾</span>
      </div>
      {shown.map((g) => (
        <Dropdown key={g.label} {...menu(g.label)} label={label(g)} active={g.items.some((i) => isActive(i, tab, meeting, sub))}>
          {(close) => <Items items={g.items} tab={tab} meeting={meeting} sub={sub} pick={(i) => { onPick(i); close() }} />}
        </Dropdown>
      ))}
      {(folded.length > 0 || (phone && extra)) && (
        <Dropdown {...menu('__more')} label={fit === 0 ? 'Menu' : 'More'} active={folded.some((g) => g.items.some((i) => isActive(i, tab, meeting, sub)))}>
          {(close) => <>
            {folded.map((g) => <div key={g.label} className="navmenu-section">
              <div className="navmenu-heading">{g.label}</div>
              <Items items={g.items} tab={tab} meeting={meeting} sub={sub} pick={(i) => { onPick(i); close() }} />
            </div>)}
            {extra && <div className="navmenu-section navmenu-extra"><div className="navmenu-heading">Display</div><div className="navmenu-extra-row">{extra}</div></div>}
          </>}
        </Dropdown>
      )}
    </div>
  )
}

/** a group's items, those with a `heading` under it; one list, as wide as its longest label */
function Items({ items, tab, meeting, sub, pick }: {
  items: NavItem[]; tab: string; meeting: string; sub?: string; pick: (i: NavItem) => void
}) {
  const item = (i: NavItem) => <Item key={`${i.tab}-${i.meeting ?? i.sub ?? ''}`} i={i} on={isActive(i, tab, meeting, sub)} pick={() => pick(i)} />
  if (!items.some((i) => i.heading)) return <>{items.map(item)}</>
  const loose = items.filter((i) => !i.heading)
  const headed = [...new Set(items.flatMap((i) => i.heading ? [i.heading] : []))].map((h) => [h, items.filter((i) => i.heading === h)] as const)
  return <>
    {loose.length > 0 && <div className="navmenu-section">{loose.map(item)}</div>}
    {headed.map(([h, list]) => <div key={h} className="navmenu-section">
      <div className="navmenu-heading">{h}</div>
      {list.map(item)}
    </div>)}
  </>
}

function Item({ i, on, pick }: { i: NavItem; on: boolean; pick: () => void }) {
  return <button type="button" role="menuitem" className={`navmenu-item${on ? ' on' : ''}`} title={i.title} aria-current={on ? 'page' : undefined} onClick={pick}>
    {i.label}
  </button>
}

/** a group button whose list opens on hover or click; NavMenu keeps one open at a time */
function Dropdown({ label, active, open, show, hide, toggle, close, children }: {
  label: string; active: boolean; open: boolean
  show: () => void; hide: () => void; toggle: () => void; close: () => void
  children: (close: () => void) => React.ReactNode
}) {
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node)) close() }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps
  // a list opened near the right edge moves left to stay in the window
  const list = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = list.current
    if (!open || !el) return
    el.style.left = ''
    const over = el.getBoundingClientRect().right - (document.documentElement.clientWidth - 8)
    if (over > 0) el.style.left = `${-Math.min(over, el.getBoundingClientRect().left - 8)}px`
  }, [open])
  return (
    <div className="navmenu-group" ref={root} onMouseEnter={show} onMouseLeave={hide}>
      <button type="button" className={`nav-btn nav-group${active ? ' active' : ''}`} aria-haspopup="menu" aria-expanded={open} onClick={toggle}>
        {label} <span className="nav-caret" aria-hidden>▾</span>
      </button>
      {open && <div ref={list} className="navmenu-list" role="menu">{children(close)}</div>}
    </div>
  )
}
