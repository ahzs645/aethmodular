import { useRef, type ReactNode } from 'react'
import { Select } from '@/components/ChartFrame'
import { XIcon } from '@/components/Icons'
import type { PanelPick } from '@/lib/panelPick'

const ADD = '\u0000add'

/**
 * The control for lib/panelPick usePanelPick, for a chart's options row.
 *
 * - `chips` (default): the shown panels in order, each with × to remove it; drag a chip onto another, or focus
 *   it and press ← / →, to reorder; an Add list offers the hidden ones (or pass `adder`, e.g. a FieldSelect,
 *   when the choice is wider than a list).
 * - `toggle`: every panel as a chip that is on or off, in the chart's own order (a small fixed set).
 *
 * `min` panels always stay shown.
 */
export function PanelPicker({ pick, labelOf = (id) => id, label = 'Panels', mode = 'chips', adder, min = 0, title }: {
  pick: PanelPick
  labelOf?: (id: string) => string
  label?: string
  mode?: 'chips' | 'toggle'
  /** a control of the chart's own for adding panels (chips mode), in place of the Add list */
  adder?: ReactNode
  min?: number
  title?: string
}) {
  const dragged = useRef<string | null>(null)
  if (mode === 'toggle') {
    return (
      <span className="control" title={title} style={{ flexWrap: 'wrap', gap: 4 }}>{label}
        {pick.all.map((id) => {
          const on = pick.has(id), locked = on && pick.ids.length <= min
          return <button key={id} type="button" className={`chip${on ? ' active' : ''}`} aria-pressed={on} disabled={locked}
            title={locked ? 'At least one panel stays shown' : on ? `Hide the ${labelOf(id)} panel` : `Show the ${labelOf(id)} panel`}
            onClick={() => pick.toggle(id)}>{labelOf(id)}</button>
        })}
      </span>
    )
  }
  const hidden = pick.all.filter((id) => !pick.has(id))
  return (
    <>
      <span className="control" title={title} style={{ flexWrap: 'wrap', gap: 4 }}>{label}
        {pick.ids.map((id) => (
          <span key={id} className="chip" draggable tabIndex={0} aria-label={`${labelOf(id)} panel: drag, or press left or right, to move it`}
            title="Drag onto another panel, or press ← / →, to reorder"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 2, cursor: 'grab', padding: '1px 3px 1px 9px' }}
            onDragStart={(e) => { dragged.current = id; e.dataTransfer.effectAllowed = 'move' }}
            onDragOver={(e) => { if (dragged.current && dragged.current !== id) e.preventDefault() }}
            onDrop={(e) => { e.preventDefault(); if (dragged.current) pick.moveTo(dragged.current, id); dragged.current = null }}
            onDragEnd={() => { dragged.current = null }}
            onKeyDown={(e) => {
              if (e.target !== e.currentTarget) return
              if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); pick.move(id, -1) }
              if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); pick.move(id, 1) }
            }}>
            {labelOf(id)}
            {pick.ids.length > min && <button type="button" className="btn quiet x-btn" onClick={() => pick.hide(id)}
              aria-label={`Remove the ${labelOf(id)} panel`} title="Remove this panel"><XIcon /></button>}
          </span>
        ))}
        {!pick.isDefault && <button type="button" className="btn quiet" onClick={pick.reset} title="Back to the chart's own panels">Reset</button>}
      </span>
      {adder ?? (hidden.length > 0 && <Select label="Add" value={ADD} options={[ADD, ...hidden]} title="Add a panel"
        optionLabel={(o) => (o === ADD ? 'Add a panel…' : labelOf(o))} onChange={(v) => { if (v !== ADD) pick.add(v) }} />)}
    </>
  )
}
