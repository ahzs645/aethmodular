import { useEffect, useRef, useState } from 'react'
import { SectionFrame } from '@/components/SectionFrame'
import { Segmented } from '@/components/Controls'
import { Combobox } from '@/components/Combobox'
import { XIcon } from '@/components/Icons'
import { Markdown } from '@/lib/markdown'
import {
  HEIGHTS, MAX_ITEMS, activeOf, addText, chartCount, createSection, decodeSection, deleteSection, encodeSection, isSectionId, isText, itemHref, itemTab, moveItem, patchItem, removeItem,
  restoreSections, sameSection, saveSection, sectionUrl, sectionsNow, setActiveSection, tooLongText, updateSection, useSections, type ItemHeight, type Section, type SectionItem,
} from '@/lib/sections'
import type { ExportProgress, SectionFormat } from '@/lib/sectionExport'
import { copyText } from '@/lib/views'

const FULL_HEIGHT = 680
const HALF_HEIGHT = 560
const HEIGHT_CHOICES = ['Auto', 'S', 'M', 'L']

/**
 * A custom section: the charts picked from any page with their + button (lib/sections), one per card, in the
 * reader's order, with a note on each. A section of this browser can be changed here; one that came from a link
 * is shown as it was sent until the reader saves a copy.
 */
export function SectionPage({ sec, setSec, isTab }: {
  /** the address's section: an id of this browser, a shared token, or nothing (the section being built) */
  sec: string | null
  setSec: (sec: string | null) => void
  /** whether this build has a tab: a link can hold a chart from a page a public build leaves out */
  isTab: (tab: string) => boolean
}) {
  const { sections, active } = useSections()
  const [shared, setShared] = useState<{ token: string; section: Section | null } | null>(null)
  const token = sec && !isSectionId(sec) ? sec : null

  useEffect(() => {
    if (!token) { setShared(null); return }
    let live = true
    decodeSection(token).then(
      (section) => {
        if (!live) return
        // the sender's own link, or one this reader already saved: open theirs, which they can change
        const own = sectionsNow().sections.find((s) => s.id === section.id && sameSection(s, section))
        if (own) { setActiveSection(own.id); setSec(own.id) } else setShared({ token, section })
      },
      () => { if (live) setShared({ token, section: null }) })
    return () => { live = false }
  }, [token, setSec])

  // the address follows the section being built, and a link's id picks it
  const seen = useRef<string | null>(null)
  useEffect(() => {
    if (token) return
    if (sec && sec !== seen.current) { seen.current = sec; if (sections.some((s) => s.id === sec)) setActiveSection(sec); return }
    if (active && sec !== active) { seen.current = active; setSec(active) }
  }, [sec, active, token, sections, setSec])

  if (token) {
    if (!shared || shared.token !== token) return <p role="status">Opening the shared section…</p>
    if (!shared.section) return <div className="sec-empty"><h2>This link could not be read</h2><p>It may have been cut off when it was copied. Ask for the link again.</p></div>
    return <SectionView section={shared.section} isTab={isTab} onSave={() => { const s = saveSection(shared.section!); setSec(s.id) }} />
  }
  const section = sec ? sections.find((s) => s.id === sec) : activeOf(sections, active)
  if (sec && !section) return <div className="sec-empty"><h2>This section is in another browser</h2><p>A section’s address only opens where it was made. Use its <strong>Copy link</strong> button to share it.</p></div>
  if (!section) return <div className="sec-empty">
    <h2>Build a section of your own</h2>
    <p>Press <strong>+</strong> beside any chart in the gallery, on any page. It joins a section with its sites, filters, dates and options as they are at that moment, so charts from different pages can sit side by side.</p>
    <button type="button" className="btn" onClick={() => createSection()}>Start an empty section</button>
  </div>
  return <SectionView key={section.id} section={section} isTab={isTab} onDeleted={() => setSec(null)} choose={sections.length > 1 ? { sections, onPick: setActiveSection } : undefined} />
}

function SectionView({ section, isTab, choose, onSave, onDeleted }: {
  section: Section
  isTab: (tab: string) => boolean
  choose?: { sections: Section[]; onPick: (id: string) => void }
  /** a shared section: read-only, with a way to keep a copy */
  onSave?: () => void
  onDeleted?: () => void
}) {
  const editable = !onSave
  const change = (fn: (s: Section) => Section) => updateSection(section.id, fn)
  // the link, made ahead of the click (a token takes a moment, and a copy has to start inside the click)
  const [link, setLink] = useState<{ url?: string; size?: number; error?: string }>({})
  useEffect(() => {
    let live = true
    Promise.all([sectionUrl(section), encodeSection(section)]).then(
      ([url, token]) => { if (live) setLink({ url, size: token.length }) },
      () => { if (live) setLink({ error: tooLongText }) })
    return () => { live = false }
  }, [section])
  const [note, setNote] = useState<{ text: string; undo?: () => void; link?: string } | null>(null)
  const [exporting, setExporting] = useState<ExportProgress | null>(null)
  const cancelled = useRef(false)
  const save = async (format: SectionFormat) => {
    cancelled.current = false
    setNote(null)
    setExporting({ done: 0, total: chartCount(section), title: '' })
    try {
      // loaded when asked for: the PowerPoint writer is large, and most readers never export
      const { exportSection } = await import('@/lib/sectionExport')
      const result = await exportSection(section, format, setExporting, () => cancelled.current)
      if (result) setNote({ text: result.missing.length ? `Saved, without ${result.missing.length} chart${result.missing.length === 1 ? '' : 's'} that could not be drawn: ${result.missing.join('; ')}` : 'Saved' })
    } catch (e) { setNote({ text: `Could not export: ${e instanceof Error ? e.message : e}` }) } finally { setExporting(null) }
  }
  const copy = () => { if (link.url) copyText(link.url).then(() => setNote({ text: `Link copied · ${link.size} characters. It holds the whole section, so later changes need a new link.` }), () => setNote({ text: 'Your browser refused the copy. Select this link and copy it:', link: link.url })) }
  // each change with a way back: a chart taken out loses its note and settings
  const undoable = (text: string, act: () => void) => { const before = sectionsNow(); act(); setNote({ text, undo: () => { restoreSections(before); setNote(null) } }) }

  return <>
    <header className="sec-head">
      <div className="sec-title">
        {editable
          ? <input className="sec-name" aria-label="Section name" value={section.name} maxLength={120} onChange={(e) => change((s) => ({ ...s, name: e.target.value }))} />
          : <h2 className="sec-name">{section.name}</h2>}
        {editable
          ? <textarea className="sec-about" aria-label="About this section" rows={1} maxLength={2000} placeholder="What this section shows (optional)" value={section.about ?? ''} onChange={(e) => change((s) => ({ ...s, about: e.target.value }))} />
          : section.about && <p className="sec-about">{section.about}</p>}
      </div>
      <div className="sec-tools">
        {choose && <Combobox label="Section" value={section.id} options={choose.sections.map((s) => ({ key: s.id, label: s.name, hint: `${s.items.length}` }))} onChange={choose.onPick} />}
        {onSave
          ? <button type="button" className="btn" onClick={onSave}>Save a copy to change it</button>
          : <button type="button" className="btn quiet" onClick={() => createSection()}>New section</button>}
        {editable && <button type="button" className="btn quiet" onClick={() => change((s) => addText(s))} title="A heading or a paragraph. Move it with ↑ ↓, or use ¶ on a chart to put one before it">Add text</button>}
        <button type="button" className="btn quiet" disabled={!section.items.length || !!exporting} onClick={() => save('pdf')} title="One PDF: a page for each chart with its title and note, still vector">PDF</button>
        <button type="button" className="btn quiet" disabled={!section.items.length || !!exporting} onClick={() => save('pptx')} title="One PowerPoint in the house style: a title and figure on each slide, your notes as speaker notes">Deck</button>
        <button type="button" className="btn" disabled={!link.url || section.items.length === 0} onClick={copy} title={link.error ?? 'One link holding this section, its notes and every chart’s settings'}>Copy link</button>
        {onDeleted && <button type="button" className="btn quiet" onClick={() => {
          if (section.items.length && !window.confirm(`Delete “${section.name}” and its ${section.items.length} chart${section.items.length === 1 ? '' : 's'}? Links already shared keep working.`)) return
          deleteSection(section.id); onDeleted()
        }}>Delete</button>}
      </div>
    </header>
    {!editable && <p className="sec-banner">A shared section: it shows what was sent, and nothing here changes it. Open a chart to explore it, or save a copy to change it or add to it.</p>}
    {link.error && <p className="sec-banner warn" role="alert">{link.error}</p>}
    {note && <p className="sec-banner" role="status">{note.text}{note.link && <input className="select sec-link" aria-label="Section link" readOnly value={note.link} onFocus={(e) => e.currentTarget.select()} />}{note.undo && <button type="button" className="btn quiet" onClick={note.undo}>Undo</button>}</p>}
    {exporting && <p className="sec-banner" role="status">{exporting.done < exporting.total ? `Drawing chart ${exporting.done + 1} of ${exporting.total}${exporting.title ? `: ${exporting.title}` : ''}` : 'Writing the file…'}
      <button type="button" className="btn quiet" onClick={() => { cancelled.current = true }}>Cancel</button></p>}
    {section.items.length === 0
      ? <div className="sec-empty"><h2>No charts yet</h2><p>Go to any page and press <strong>+</strong> beside a chart. It joins “{section.name}” as it is at that moment.</p></div>
      : <div className="sec-grid">{section.items.map((item, i) => (
        <SectionCard key={item.id} item={item} first={i === 0} last={i === section.items.length - 1} available={isTab(itemTab(item))}
          edit={editable ? {
            patch: (p) => change((s) => patchItem(s, item.id, p)),
            move: (by) => change((s) => moveItem(s, item.id, by)),
            remove: () => undoable(`Took out “${item.title || 'text'}”`, () => change((s) => removeItem(s, item.id))),
            textBefore: () => change((s) => addText(s, item.id)),
          } : undefined} />
      ))}</div>}
    {editable && chartCount(section) >= MAX_ITEMS && <p className="sec-banner">A section holds up to {MAX_ITEMS} charts. Start another for more.</p>}
  </>
}

interface CardEdit {
  patch: (p: Partial<Omit<SectionItem, 'id'>>) => void
  move: (by: -1 | 1) => void
  remove: () => void
  textBefore: () => void
}

function SectionCard({ item, first, last, available, edit }: { item: SectionItem; first: boolean; last: boolean; available: boolean; edit?: CardEdit }) {
  const order = edit && <>
    <button type="button" className="btn quiet icon-btn" disabled={first} aria-label={`Move ${item.title || 'text'} earlier`} title="Move earlier" onClick={() => edit.move(-1)}>↑</button>
    <button type="button" className="btn quiet icon-btn" disabled={last} aria-label={`Move ${item.title || 'text'} later`} title="Move later" onClick={() => edit.move(1)}>↓</button>
  </>
  const remove = edit && <button type="button" className="btn quiet x-btn" aria-label={`Take ${item.title || 'text'} out of the section`} title="Take out of this section" onClick={edit.remove}><XIcon /></button>
  if (isText(item)) return (
    <section className="sec-text">
      {edit
        ? <>
          <div className="sec-card-head">
            <input className="sec-text-head" aria-label="Heading" maxLength={120} placeholder="Heading" value={item.title} onChange={(e) => edit.patch({ title: e.target.value })} />
            <span className="sec-card-tools">{order}{remove}</span>
          </div>
          <textarea className="sec-note" aria-label={`Text under ${item.title || 'this heading'}`} rows={1} maxLength={2000} placeholder="Text (optional). **bold**, *italic* and - lists work." value={item.note ?? ''} onChange={(e) => edit.patch({ note: e.target.value })} />
        </>
        : <>{item.title && <h3 className="sec-text-head">{item.title}</h3>}{item.note && <div className="sec-note"><Markdown text={item.note} /></div>}</>}
    </section>
  )
  const height = item.height ? HEIGHTS[item.height] : item.wide ? FULL_HEIGHT : HALF_HEIGHT
  return (
    <article className={`panel sec-card${item.wide ? ' wide' : ''}`}>
      <header className="sec-card-head">
        {edit
          ? <input className="sec-card-title" aria-label="Chart title" value={item.title} maxLength={120} onChange={(e) => edit.patch({ title: e.target.value })} />
          : <h3 className="sec-card-title">{item.title}</h3>}
        <span className="sec-card-tools">
          {edit && <>
            <span className="sec-width"><Segmented label="Width" value={item.wide ? 'Full' : 'Half'} options={['Half', 'Full']} onChange={(v) => edit.patch({ wide: v === 'Full' })} /></span>
            <Segmented label="Height" value={item.height ?? 'Auto'} options={HEIGHT_CHOICES} title="Auto grows to fit the chart; S, M and L are set heights, and a taller chart scrolls" onChange={(v) => edit.patch({ height: v === 'Auto' ? undefined : v as ItemHeight })} />
            <button type="button" className="btn quiet icon-btn" aria-label={`Add text before ${item.title}`} title="Add a heading or text before this chart" onClick={edit.textBefore}>¶</button>
            {order}
          </>}
          <a className="btn quiet" href={itemHref(item, false)} target="_blank" rel="noreferrer" title="Open this chart on its page, with these settings, in a new tab">Open ↗</a>
          {remove}
        </span>
      </header>
      {available
        ? <SectionFrame item={item} height={height} fixed={!!item.height} onView={edit ? (view) => edit.patch({ view }) : undefined} onChart={edit ? (chart) => edit.patch({ chart }) : undefined} />
        : <p className="sec-frame-note">This chart comes from a page this copy of the gallery does not include ({itemTab(item) || 'unknown'}).</p>}
      {edit
        ? <textarea className="sec-note" aria-label={`Note on ${item.title}`} rows={1} maxLength={2000} placeholder="Add a note (optional)" value={item.note ?? ''} onChange={(e) => edit.patch({ note: e.target.value })} />
        : item.note && <div className="sec-note"><Markdown text={item.note} /></div>}
    </article>
  )
}
