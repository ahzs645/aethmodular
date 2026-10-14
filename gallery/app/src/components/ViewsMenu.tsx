import { useEffect, useRef, useState } from 'react'
import { PRESETS, applyView, currentView, encodeView, sameView, savedViews, saveViews, viewUrl, type SavedView } from '@/lib/views'

/**
 * Header menu for views: built-in presets, the reader's saved views, and sharing — a link that
 * carries the whole view as one json-url token, and an iframe embed of the same view.
 */
export function ViewsMenu() {
  const [open, setOpen] = useState(false)
  const [saved, setSaved] = useState<SavedView[]>(savedViews)
  const [name, setName] = useState('')
  const [note, setNote] = useState('')
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [open])
  useEffect(() => { if (!open) setNote('') }, [open])

  const copy = async (text: string, done: string) => {
    try { await navigator.clipboard.writeText(text); setNote(done) } catch { setNote('Could not copy; select the address bar instead') }
  }
  const shareLink = async () => {
    const url = await viewUrl(currentView())
    await copy(url, `Link copied · ${(await encodeView(currentView())).length} characters`)
  }
  const shareEmbed = async () => {
    const url = await viewUrl(currentView(), true)
    await copy(`<iframe src="${url}" width="100%" height="760" style="border:0" loading="lazy" title="Aethmodular graph gallery"></iframe>`, 'Embed code copied (iframe, no header or filters)')
  }
  const save = () => {
    const n = name.trim()
    if (!n) return
    const next = [...saved.filter((x) => x.name !== n), { name: n, view: currentView() }]
    setSaved(next); saveViews(next); setName(''); setNote(`Saved “${n}” in this browser`)
  }
  const remove = (n: string) => { const next = saved.filter((x) => x.name !== n); setSaved(next); saveViews(next) }
  const here = currentView()

  return (
    <div className="views-menu" ref={box}>
      <button type="button" className="btn quiet" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(!open)}>Views ▾</button>
      {open && (
        <div className="views-pop" role="dialog" aria-label="Views">
          <div className="views-section">Presets</div>
          {PRESETS.map((p) => (
            <button key={p.key} type="button" className={`views-item ${sameView(p.view, here) ? 'on' : ''}`} title={p.note}
              onClick={() => { applyView(p.view); setOpen(false) }}>
              <span>{p.label}</span><span className="views-note">{p.note}</span>
            </button>
          ))}
          <div className="views-section">Your views <span className="views-note">(this browser)</span></div>
          {saved.length === 0 && <div className="views-note views-pad">None yet: name the current view below.</div>}
          {saved.map((x) => (
            <div key={x.name} className="views-row">
              <button type="button" className={`views-item ${sameView(x.view, here) ? 'on' : ''}`} onClick={() => { applyView(x.view); setOpen(false) }}><span>{x.name}</span></button>
              <button type="button" className="btn quiet" aria-label={`Delete ${x.name}`} onClick={() => remove(x.name)}>×</button>
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
        </div>
      )}
    </div>
  )
}
