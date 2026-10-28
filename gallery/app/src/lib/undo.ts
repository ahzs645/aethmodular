import { useSyncExternalStore } from 'react'
import { UrlHistory, describeChange } from './undoHistory.ts'

/**
 * ⌘Z / Ctrl+Z undoes the last change to the view and ⇧⌘Z / Ctrl+Y / Ctrl+⇧Z redoes it: sites, exclusions,
 * dates, seasons, fields, chart options, dashboard tiles, anything the gallery keeps in the URL (lib/url).
 * Every such write goes through history.replaceState, so wrapping it sees them all; App and the controls
 * already re-read the URL on hashchange / popstate / gallery:querychange, so restoring a URL restores the view.
 *
 * Writes within SETTLE_MS of each other are one step (a drag on a number, one click that rewrites several
 * keys). Only writes that follow the reader's own input (a click, key or edit in the last INPUT_MS) are
 * recorded, so the app filling in defaults or re-normalising a link is not something to undo. In a text
 * field the browser's own undo runs instead.
 */
const SETTLE_MS = 450
const INPUT_MS = 1500
const RESTORE_MS = 400

const steps = new UrlHistory()
const here = () => window.location.search + window.location.hash
let timer: ReturnType<typeof setTimeout> | undefined
let lastInput = 0
let quietUntil = 0
let installed = false
let nativeReplace: History['replaceState'] = () => {}

type Note = { id: number; text: string }
let note: Note | null = null
let version = 0
const listeners = new Set<() => void>()
const emit = () => { version++; listeners.forEach((l) => l()) }

function settleSoon() {
  clearTimeout(timer)
  timer = setTimeout(() => { if (steps.settle(here())) emit() }, SETTLE_MS)
}

function go(url: string) {
  clearTimeout(timer)
  quietUntil = Date.now() + RESTORE_MS
  nativeReplace.call(window.history, window.history.state, '', window.location.pathname + url)
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  window.dispatchEvent(new PopStateEvent('popstate'))
  window.dispatchEvent(new Event('gallery:querychange'))
}

export function undo() {
  const step = steps.undo(here())
  if (!step) { note = { id: Date.now(), text: 'Nothing to undo' }; emit(); return }
  go(step.from)
  note = { id: Date.now(), text: `Undid ${describeChange(step.from, step.to)}` }
  emit()
}

export function redo() {
  const step = steps.redo(here())
  if (!step) { note = { id: Date.now(), text: 'Nothing to redo' }; emit(); return }
  go(step.to)
  note = { id: Date.now(), text: `Redid ${describeChange(step.from, step.to)}` }
  emit()
}

/** text fields keep the browser's own undo */
const typing = (t: EventTarget | null) => {
  const el = t as HTMLElement | null
  if (!el) return false
  if (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true
  return el.tagName === 'INPUT' && !['checkbox', 'radio', 'range', 'button', 'submit', 'color'].includes((el as HTMLInputElement).type)
}

export function installUndo() {
  if (installed || typeof window === 'undefined') return
  installed = true
  // the first render writes the normalised link; that is where history starts
  quietUntil = Date.now() + 2000
  nativeReplace = window.history.replaceState
  const wrap = (native: History['replaceState']): History['replaceState'] => function (this: History, data, unused, url) {
    const before = here()
    native.call(this, data, unused, url)
    if (Date.now() < quietUntil || Date.now() - lastInput > INPUT_MS || here() === before) return
    steps.begin(before)
    settleSoon()
  }
  window.history.replaceState = wrap(window.history.replaceState)
  window.history.pushState = wrap(window.history.pushState)
  const mark = () => { lastInput = Date.now() }
  for (const type of ['pointerdown', 'keydown', 'input', 'change']) window.addEventListener(type, mark, true)
  window.addEventListener('keydown', (e) => {
    if (!(e.metaKey || e.ctrlKey) || e.altKey || typing(e.target)) return
    const key = e.key.toLowerCase()
    if (key === 'z' && !e.shiftKey) { e.preventDefault(); undo() }
    else if ((key === 'z' && e.shiftKey) || (key === 'y' && e.ctrlKey && !e.metaKey)) { e.preventDefault(); redo() }
  })
}

const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }
/** the note after the last undo or redo, and whether either is available (components/UndoToast) */
export function useUndo() {
  useSyncExternalStore(subscribe, () => version)
  return { note, canUndo: steps.canUndo || steps.pending, canRedo: steps.canRedo, undo, redo }
}
