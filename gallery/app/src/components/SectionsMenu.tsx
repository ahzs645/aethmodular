import { useEffect, useRef, useState } from 'react'
import { CheckIcon, PlusIcon, SectionsIcon } from '@/components/Icons'
import { useHeaderNav } from '@/lib/headerNav'
import { activeOf, chartCount, createSection, itemView, sameItem, setActiveSection, toggleItem, useSections } from '@/lib/sections'
import { patchHash } from '@/lib/url'
import { currentView, isEmbed } from '@/lib/views'

export const CUSTOM_TAB = 'custom'
const useOffered = () => !!useHeaderNav()?.tabs.some((t) => t.key === CUSTOM_TAB) && !isEmbed()

/** a chart's place in the section being built: whether it is there at the settings on screen, and the press that adds or removes it */
export function useSectionPick(chart: string, title: string) {
  const { sections, active } = useSections()
  const offered = useOffered()
  const s = activeOf(sections, active)
  const here = () => ({ chart, view: itemView(currentView()) })
  // read on hover as well as on a change of section: the chart's own settings move without a render here
  const [, redraw] = useState(0)
  const on = !!s?.items.some((i) => sameItem(i, here()))
  // on the Dashboard only its tiles can be picked (the one-chart view is off there)
  return { offered: offered && (chart.startsWith('tile:') || currentView().h.tab !== 'dashboard'), on, name: s?.name ?? 'a new section', toggle: () => toggleItem(chart, title), refresh: () => redraw((n) => n + 1) }
}

/** On a chart's title line: one press puts the chart, as it is now, in the section being built. */
export function SectionAddButton({ chart, title }: { chart: string; title: string }) {
  const pick = useSectionPick(chart, title)
  if (!pick.offered) return null
  return <button type="button" className={`btn quiet icon-btn section-add${pick.on ? ' on' : ''}`} aria-pressed={pick.on} onPointerEnter={pick.refresh} onFocus={pick.refresh} onClick={pick.toggle}
    aria-label={pick.on ? `Take this chart out of ${pick.name}` : `Add this chart to ${pick.name}`}
    title={pick.on ? `In “${pick.name}” with these settings. Click to take it out` : `Add to “${pick.name}” with its settings, sites and filters as they are now`}>
    {pick.on ? <CheckIcon /> : <PlusIcon />}
  </button>
}

/** The header's list of sections: which one charts are being added to, a new one, and the way to its page. */
export function SectionsMenu() {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [open])
  const { sections, active } = useSections()
  const shownSection = activeOf(sections, active)
  const count = shownSection ? chartCount(shownSection) : 0
  if (!useOffered()) return null
  return (
    <div className="views-menu" ref={box}>
      <button type="button" className={`btn quiet icon-btn section-menu-btn${open ? ' on' : ''}`} aria-haspopup="dialog" aria-expanded={open}
        aria-label={`Custom sections, ${count} chart${count === 1 ? '' : 's'} in the current one`} title="Custom sections: the charts you have picked" onClick={() => setOpen(!open)}>
        <SectionsIcon />{count > 0 && <span className="section-count" aria-hidden>{count}</span>}
      </button>
      {open && <div className="views-pop" role="dialog" aria-label="Custom sections"><SectionsPanel onClose={() => setOpen(false)} /></div>}
    </div>
  )
}

/** the menu's contents, also shown from the phone's ⋯ menu (components/HeaderMore) */
export function SectionsPanel({ onClose }: { onClose: () => void }) {
  const { sections, active } = useSections()
  const [name, setName] = useState('')
  const current = activeOf(sections, active)
  const open = (id: string) => { setActiveSection(id); patchHash({ tab: CUSTOM_TAB, sec: id }); onClose() }
  const create = () => { const s = createSection(name.trim() || undefined); setName(''); setActiveSection(s.id) }
  return <>
    <div className="views-section">Adding charts to</div>
    {sections.length === 0 && <div className="views-note views-pad">Press + on any chart to start a section. It keeps the chart’s sites, filters and options.</div>}
    {sections.map((s) => (
      <button key={s.id} type="button" className={`views-item ${s.id === current?.id ? 'on' : ''}`} aria-pressed={s.id === current?.id} onClick={() => setActiveSection(s.id)}>
        <span>{s.name}</span><span className="views-note">{chartCount(s)} chart{chartCount(s) === 1 ? '' : 's'}</span>
      </button>
    ))}
    <form className="views-save" onSubmit={(e) => { e.preventDefault(); create() }}>
      <input className="select" placeholder="Name a new section…" aria-label="Name a new section" value={name} onChange={(e) => setName(e.target.value)} />
      <button type="submit" className="btn">New</button>
    </form>
    {current && <div className="views-share"><button type="button" className="btn" onClick={() => open(current.id)}>Open “{current.name}”</button></div>}
  </>
}
