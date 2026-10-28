import { useEffect, useState } from 'react'
import { useUndo } from '@/lib/undo'

const MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
export const UNDO_KEYS = MAC ? '⌘Z' : 'Ctrl+Z'
export const REDO_KEYS = MAC ? '⇧⌘Z' : 'Ctrl+Y'

/** a short note after ⌘Z / ⇧⌘Z (lib/undo): what came back, and the other way */
export function UndoToast() {
  const { note, canUndo, canRedo, undo, redo } = useUndo()
  const [shown, setShown] = useState<number | null>(null)
  useEffect(() => {
    if (!note) return
    setShown(note.id)
    const t = setTimeout(() => setShown(null), 3200)
    return () => clearTimeout(t)
  }, [note])
  if (!note || shown !== note.id) return null
  return (
    <div className="undo-toast" role="status" aria-live="polite">
      <span>{note.text}</span>
      <button type="button" className="btn quiet" disabled={!canUndo} onClick={undo} title={`Undo (${UNDO_KEYS})`}>Undo</button>
      <button type="button" className="btn quiet" disabled={!canRedo} onClick={redo} title={`Redo (${REDO_KEYS})`}>Redo</button>
    </div>
  )
}
