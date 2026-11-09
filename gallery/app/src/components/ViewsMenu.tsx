import { useEffect, useRef, useState } from 'react'
import { ViewsIcon, XIcon } from '@/components/Icons'
import { PRESETS, applyView, copyText, currentView, sameView, savedViews, saveViews, viewUrl, type SavedView } from '@/lib/views'
import { shareableView } from '@/lib/sections'

/**
 * Header menu for views: built-in presets, the reader's saved views, and sharing — a link that
 * carries the whole view as one json-url token, and an iframe embed of the same view.
 */
export function ViewsMenu() {
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

  return (
    <div className="views-menu" ref={box}>
      <button type="button" className={`btn quiet icon-btn${open ? ' on' : ''}`} aria-haspopup="dialog" aria-expanded={open} aria-label="Views"
        title="Views: presets, saved views, copy link, embed" onClick={() => setOpen(!open)}><ViewsIcon /></button>
      {open && <div className="views-pop" role="dialog" aria-label="Views"><ViewsPanel onClose={() => setOpen(false)} /></div>}
    </div>
  )
}

/** the menu's contents, also shown from the phone's ⋯ menu (components/HeaderMore); `onClose` after a view is applied */
export function ViewsPanel({ onClose }: { onClose: () => void }) {
  const [saved, setSaved] = useState<SavedView[]>(savedViews)
  const [name, setName] = useState('')
  const [note, setNote] = useState('')

  // the token takes a moment, and a browser may only allow a copy begun inside the click: copyText starts it there
  const copy = (text: Promise<string>, done: (text: string) => string) => {
    let made = ''
    copyText(text.then((t) => (made = t))).then(() => setNote(done(made)), (e: unknown) => setNote(
      // the one way a token fails: json-url's length limit (a long list of hand-excluded filters)
      /maxLength/.test(String(e)) ? 'This view is too long for a link, usually a long list of hand-excluded filters. Clear some and try again.' : 'Could not copy; select the address bar instead'))
  }
  const view = () => shareableView(currentView())
  const shareLink = () => copy(view().then((v) => viewUrl(v)), (url) => `Link copied · ${url.length} characters`)
  const shareEmbed = () => copy(view().then((v) => viewUrl(v, true)).then((url) => `<iframe src="${url}" width="100%" height="760" style="border:0" loading="lazy" title="Aethmodular graph gallery"></iframe>`),
    () => 'Embed code copied (iframe, no header or filters)')
  const save = () => {
    const n = name.trim()
    if (!n) return
    const next = [...saved.filter((x) => x.name !== n), { name: n, view: currentView() }]
    setSaved(next); saveViews(next); setName(''); setNote(`Saved “${n}” in this browser`)
  }
  const remove = (n: string) => { const next = saved.filter((x) => x.name !== n); setSaved(next); saveViews(next) }
  const here = currentView()

  return (
        <>
          <div className="views-section">Presets</div>
          {PRESETS.map((p) => (
            <button key={p.key} type="button" className={`views-item ${sameView(p.view, here) ? 'on' : ''}`} title={p.note}
              onClick={() => { applyView(p.view); onClose() }}>
              <span>{p.label}</span><span className="views-note">{p.note}</span>
            </button>
          ))}
          <div className="views-section">Your views <span className="views-note">(this browser)</span></div>
          {saved.length === 0 && <div className="views-note views-pad">None yet: name the current view below.</div>}
          {saved.map((x) => (
            <div key={x.name} className="views-row">
              <button type="button" className={`views-item ${sameView(x.view, here) ? 'on' : ''}`} onClick={() => { applyView(x.view); onClose() }}><span>{x.name}</span></button>
              <button type="button" className="btn quiet x-btn" aria-label={`Delete ${x.name}`} onClick={() => remove(x.name)}><XIcon /></button>
            </div>
          ))}
          <form className="views-save" onSubmit={(e) => { e.preventDefault(); save() }}>
            <input className="select" placeholder="Name this view…" value={name} onChange={(e) => setName(e.target.value)} />
            <button type="submit" className="btn" disabled={!name.trim()}>Save</button>
          </form>
          <div className="views-section">Share this view</div>
          <div className="views-share">
            <button type="button" className="btn" onClick={shareLink} title="One link holding the tab, network, sites, fields, seasons, dates and hand-excluded filters">Copy link</button>
            <button type="button" className="btn" onClick={shareEmbed} title="An iframe snippet for a page or slide; opens without the header and the filter panel">Copy embed code</button>
          </div>
          {note && <div className="views-note views-pad" role="status">{note}</div>}
        </>
  )
}
