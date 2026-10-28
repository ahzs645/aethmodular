import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Combobox } from '@/components/Combobox'
import { prettyUnit } from '@/components/FieldSelect'
import { RATIO_GROUP, ratioKey, ratioUnit, type RatioDef } from '@/lib/viewFields'
import type { MetaFile } from '@/lib/types'
import { XIcon } from '@/components/Icons'

/**
 * "Define a ratio…" in any field picker (FieldSelect, the axis-title popover, the sidebar): opens
 * this dialog, and the ratio it makes joins every picker under "Ratios" and is chosen where the
 * dialog was opened. Presets (lib/viewFields RATIO_PRESETS) are always in the pickers already;
 * the dialog lists them too, for one click. App owns the list and the URL (`ratios`).
 */
export const DEFINE_RATIO = '__define_ratio__'

interface RatioCtx {
  /** open the dialog; `onPick` receives the ratio's field name */
  open: (onPick: (field: string) => void) => void
  /** the reader's own ratios (presets are not listed: they cannot be removed) */
  custom: RatioDef[]
  remove: (field: string) => void
}
const Ctx = createContext<RatioCtx | null>(null)
export const useRatios = () => useContext(Ctx)

export function RatiosProvider({ meta, presets, custom, onAdd, onRemove, count, children }: {
  /** the view's meta: fields to choose from (ratios excluded) and their units */
  meta: MetaFile
  /** presets whose two fields exist in this view */
  presets: RatioDef[]
  custom: RatioDef[]
  onAdd: (d: RatioDef) => void
  onRemove: (field: string) => void
  /** filters in view carrying both fields */
  count: (d: RatioDef) => number
  children: ReactNode
}) {
  const [pending, setPending] = useState<((field: string) => void) | null>(null)
  const open = useCallback((onPick: (field: string) => void) => setPending(() => onPick), [])
  const value = useMemo(() => ({ open, custom, remove: onRemove }), [open, custom, onRemove])
  return (
    <Ctx.Provider value={value}>
      {children}
      {pending && <RatioDialog meta={meta} presets={presets} count={count} onClose={() => setPending(null)}
        onPick={(d) => { onAdd(d); pending(ratioKey(d)); setPending(null) }} />}
    </Ctx.Provider>
  )
}

function RatioDialog({ meta, presets, count, onPick, onClose }: {
  meta: MetaFile; presets: RatioDef[]; count: (d: RatioDef) => number
  onPick: (d: RatioDef) => void; onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const options = useMemo(() => meta.field_groups.filter((g) => g.label !== RATIO_GROUP)
    .flatMap((g) => g.fields.filter((f) => meta.fields.includes(f)).map((f) => {
      const u = prettyUnit(meta.field_units?.[f])
      return { key: f, label: u ? `${f} (${u})` : f, group: g.label }
    })), [meta])
  const has = (f: string) => options.some((o) => o.key === f)
  const [num, setNum] = useState(() => (has('EC (FTIR)') ? 'EC (FTIR)' : options[0]?.key ?? ''))
  const [den, setDen] = useState(() => (has('PM2.5 mass') ? 'PM2.5 mass' : options[1]?.key ?? ''))
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    ref.current?.querySelector<HTMLButtonElement>('.ratio-preset, .btn.primary')?.focus()
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])
  const d = { num, den }
  const same = num === den
  const n = same ? 0 : count(d)
  const unit = ratioUnit(meta.field_units?.[num], meta.field_units?.[den])
  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div ref={ref} className="export-dialog ratio-dialog" role="dialog" aria-modal="true" aria-label="Define a ratio">
        <header>
          <div><strong>Define a ratio</strong><span className="export-title">one value per filter, usable in every chart</span></div>
          <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
        </header>
        {presets.length > 0 && (
          <section className="ratio-presets">
            <div className="row-label">Presets</div>
            <div className="ratio-preset-list">
              {presets.map((p) => (
                <button key={ratioKey(p)} type="button" className="chip ratio-preset" onClick={() => onPick(p)}
                  title={`${count(p).toLocaleString()} filters in view carry both`}>{ratioKey(p)}</button>
              ))}
            </div>
          </section>
        )}
        <section className="ratio-build">
          <div className="row-label">Your own</div>
          <div className="ratio-fraction">
            <Combobox label="Numerator" value={num} options={options} onChange={setNum} />
            <span className="ratio-over" aria-hidden>÷</span>
            <Combobox label="Denominator" value={den} options={options} onChange={setDen} />
            <button type="button" className="btn quiet" title="Swap numerator and denominator" onClick={() => { setNum(den); setDen(num) }}>⇄</button>
          </div>
          <p className="chart-note" style={{ margin: 0 }}>
            {same ? 'Choose two different measurements.'
              : `${ratioKey(d)} · ${unit === 'ratio' ? 'unitless' : prettyUnit(unit)} · ${n.toLocaleString()} filters in view carry both. A zero or missing denominator leaves the filter out.`}
          </p>
        </section>
        <div className="export-actions">
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="button" className="btn primary" disabled={same || n === 0} onClick={() => onPick(d)}>Add and use</button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
