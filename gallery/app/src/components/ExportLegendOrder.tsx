import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ColorPopover } from '@/components/ExportColorOptions'
import { arrangeBlocks, flatKeys, mergeOrder, moved, type LegendBlock, type LegendRow } from '@/lib/legendOrder'
import { resetLegendOrder, setLegendOrder, useLegendOrder } from '@/lib/legendOrderPrefs'

/** the legend items in a panel, as drawn: one block per group and one for each legend's loose items, each label once */
function readBlocks(panel: HTMLElement): LegendBlock[] {
  const seen = new Set<string>()
  const rows = (els: Iterable<HTMLElement>) => [...els].flatMap((el): LegendRow[] => {
    const key = el.dataset.exportSeries!
    if (seen.has(key) || !el.getClientRects().length || el.closest('[data-export="skip"]')) return []
    seen.add(key)
    const shape = [...(el.querySelector('.swatch')?.classList ?? [])].find((c) => c !== 'swatch') ?? 'dot'
    return [{ key, label: el.dataset.legendText ?? key, color: el.dataset.exportColor ?? '', shape, off: el.classList.contains('off') }]
  })
  const blocks: LegendBlock[] = []
  for (const legend of panel.querySelectorAll<HTMLElement>('.legend')) {
    const own = (el: Element) => el.closest('.legend') === legend
    blocks.push({ name: null, rows: rows([...legend.querySelectorAll<HTMLElement>('.legend-item[data-export-series]')].filter((el) => own(el) && !el.closest('.legend-group'))) })
    for (const group of [...legend.querySelectorAll<HTMLElement>('.legend-group')].filter(own)) {
      blocks.push({ name: group.querySelector('.legend-group-name')?.textContent?.trim() ?? '', rows: rows(group.querySelectorAll<HTMLElement>(':scope > .legend-item[data-export-series]')) })
    }
  }
  return blocks.filter((b) => b.rows.length)
}

interface Drag { scope: string; index: number }

/**
 * One row of the list: drag it, or use the arrows (the keyboard's way). `scope` is what it can swap places with: the
 * items of its group, or the groups.
 */
function OrderRow({ id, scope, index, count, label, mark, off, head, drag, setDrag, onMove }: {
  id: string; scope: string; index: number; count: number; label: string; mark?: ReactNode; off?: boolean; head?: boolean
  drag: Drag | null; setDrag: (d: Drag | null) => void; onMove: (from: number, to: number, dir: 'up' | 'down') => void
}) {
  const [over, setOver] = useState(false)
  const same = drag?.scope === scope && drag.index !== index
  const cls = ['legend-order-row', head ? 'head' : '', off ? 'off' : '', drag?.scope === scope && drag.index === index ? 'dragging' : '',
    over && same ? (drag!.index < index ? 'over-after' : 'over-before') : ''].filter(Boolean).join(' ')
  return <div className={cls} data-row={id} draggable
    onDragStart={(e) => { e.stopPropagation(); e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', label); setDrag({ scope, index }) }}
    onDragOver={(e) => { if (!same) return; e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setOver(true) }}
    onDragLeave={() => setOver(false)}
    onDrop={(e) => { e.preventDefault(); setOver(false); if (same) onMove(drag!.index, index, drag!.index < index ? 'down' : 'up'); setDrag(null) }}
    onDragEnd={() => { setOver(false); setDrag(null) }}>
    <span className="legend-order-grip" aria-hidden="true">⠿</span>
    {mark}
    <span className="legend-order-label" title={label}>{label}</span>
    <button type="button" className="btn quiet" data-dir="up" disabled={index === 0} aria-label={`Move ${label} up`} onClick={() => onMove(index, index - 1, 'up')}>↑</button>
    <button type="button" className="btn quiet" data-dir="down" disabled={index === count - 1} aria-label={`Move ${label} down`} onClick={() => onMove(index, index + 1, 'down')}>↓</button>
  </div>
}

function OrderList({ blocks, onChange }: { blocks: LegendBlock[]; onChange: (next: LegendBlock[]) => void }) {
  const box = useRef<HTMLDivElement>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  // the list re-sorts under the arrow just pressed: keep the keyboard on that row
  const moving = useRef<{ id: string; dir: 'up' | 'down' } | null>(null)
  useLayoutEffect(() => {
    const m = moving.current
    if (!m) return
    moving.current = null
    const row = [...box.current!.querySelectorAll<HTMLElement>('[data-row]')].find((el) => el.dataset.row === m.id)
    const buttons = [...(row?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])]
    ;(buttons.find((b) => b.dataset.dir === m.dir) ?? buttons[0])?.focus({ preventScroll: true })
  }, [blocks])
  const groups = blocks.flatMap((b, i) => (b.name === null ? [] : [i]))
  const moveGroup = (from: number, to: number, dir: 'up' | 'down') => {
    moving.current = { id: `group:${blocks[groups[from]].name}`, dir }
    onChange(moved(blocks, groups[from], groups[to]))
  }
  const moveRow = (at: number) => (from: number, to: number, dir: 'up' | 'down') => {
    moving.current = { id: blocks[at].rows[from].key, dir }
    onChange(blocks.map((b, i) => (i === at ? { ...b, rows: moved(b.rows, from, to) } : b)))
  }
  return <div ref={box} className="legend-order-list">
    {blocks.map((b, at) => <div key={b.name === null ? `loose${at}` : `group:${b.name}`} className={b.name === null ? '' : 'legend-order-group'}>
      {b.name !== null && <OrderRow id={`group:${b.name}`} scope="groups" index={groups.indexOf(at)} count={groups.length} label={b.name} head drag={drag} setDrag={setDrag} onMove={moveGroup} />}
      {b.rows.map((r, i) => <OrderRow key={r.key} id={r.key} scope={`block${at}`} index={i} count={b.rows.length} label={r.label} off={r.off} drag={drag} setDrag={setDrag} onMove={moveRow(at)}
        mark={<span className={`swatch ${r.shape}`} style={{ background: r.color, color: r.color }} />} />)}
    </div>)}
  </div>
}

/**
 * The export dialog's Legend order: one list for the chart, however many legends it draws. Only the legend moves; while
 * the dialog is open the chart's `Legend` draws this order for the file, and the browser lays it out for every placement.
 * `drawn`: the composed figure, so the list reads the panel again after each drawing.
 */
export function LegendOrderPicker({ panel, chart, drawn }: { panel: HTMLElement; chart: string; drawn: unknown }) {
  const saved = useLegendOrder(chart)
  const [open, setOpen] = useState(false)
  const anchor = useRef<HTMLButtonElement>(null)
  const read = useMemo(() => readBlocks(panel), [panel, drawn]) // eslint-disable-line react-hooks/exhaustive-deps
  const blocks = useMemo(() => arrangeBlocks(read, saved?.order), [read, saved])
  const keys = flatKeys(blocks)
  if (keys.length < 2) return null
  const save = (next: string[], page = saved?.page ?? false) => setLegendOrder(chart, { order: mergeOrder(next, saved?.order), page })
  return <div className="export-row export-color-picker">
    <span className="combobox-label">Legend order</span>
    <button ref={anchor} type="button" className={`combobox-trigger${open ? ' open' : ''}`} aria-label="Legend order" aria-haspopup="dialog" aria-expanded={open}
      title="The order of the legend's items. The plot is not changed" onClick={() => setOpen(!open)}>
      {saved ? 'Custom' : 'Chart order'}
      <span aria-hidden="true">▾</span>
    </button>
    {open && anchor.current && <ColorPopover anchor={anchor.current} onClose={() => setOpen(false)} title="Legend order" label="Legend order">
      <OrderList blocks={blocks} onChange={(next) => save(flatKeys(next))} />
      <div className="export-color-actions">
        <button type="button" className="btn" onClick={() => save([...keys].reverse())}>Reverse</button>
        <button type="button" className="btn quiet" disabled={!saved} onClick={() => resetLegendOrder(chart)}>Reset</button>
      </div>
      <label className="export-same-color"><input type="checkbox" checked={!!saved?.page} onChange={(e) => save(keys, e.target.checked)} /> Use this order on the page too</label>
      <p className="export-color-note">Only the legend changes: the plot, its colors and what is drawn on top stay as they are. Remembered for this chart in this browser.</p>
    </ColorPopover>}
  </div>
}
