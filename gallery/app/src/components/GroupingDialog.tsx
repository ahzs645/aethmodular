import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import * as d3 from 'd3'
import { Combobox } from '@/components/Combobox'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { applyGrouping, bandOf, BUILT_IN_GROUPINGS, colorsFor, describeGrouping, newGroupingId, rangeOptions, rangeText, SPLITS, type BuiltInGrouping, type GroupingDef, type SplitKind } from '@/lib/groupings'
import { Segmented } from '@/components/ChartFrame'
import { ColorChoice, ColorPopover } from '@/components/ExportColorOptions'
import { FONT, INK } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { XIcon } from '@/components/Icons'

/**
 * The reader's groupings (lib/groupings) for every chart: App owns the list and the URL (`grp`), this provider
 * opens the editor from any chart ("New grouping…", "Edit…") and hands the saved grouping's id back to it.
 */
interface GroupingsCtx {
  /** the reader's own groupings (URL and this browser) */
  list: GroupingDef[]
  /** groupings earlier analyses used whose measurement this view has (lib/groupings BUILT_IN_GROUPINGS) */
  builtIn: BuiltInGrouping[]
  /** a grouping by id, the reader's or built in */
  find: (id: string) => GroupingDef | null
  /** open the editor: `def` to edit one, otherwise a new grouping (on `field` if given); `onSaved` gets its id;
   *  `rows`: the filters the calling chart draws, so the preview and the quantiles match what it will show */
  edit: (opts?: EditOpts) => void
}
type EditOpts = { def?: GroupingDef; field?: string; onSaved?: (id: string) => void; rows?: FilterRow[]; copy?: boolean }
const Ctx = createContext<GroupingsCtx>({ list: [], builtIn: [], find: () => null, edit: () => {} })
export const useGroupings = () => useContext(Ctx)

export function GroupingsProvider({ meta, rows, list, onSave, onRemove, children }: {
  meta: MetaFile
  /** the filters in view: the editor's preview and its quantiles */
  rows: FilterRow[]
  list: GroupingDef[]
  onSave: (def: GroupingDef) => void
  onRemove: (id: string) => void
  children: ReactNode
}) {
  const [editing, setEditing] = useState<EditOpts | null>(null)
  const edit = useCallback((opts: EditOpts = {}) => setEditing(opts), [])
  const builtIn = useMemo(() => BUILT_IN_GROUPINGS.filter((d) => meta.fields.includes(d.field)), [meta.fields])
  const find = useCallback((id: string) => list.find((d) => d.id === id) ?? builtIn.find((d) => d.id === id) ?? null, [list, builtIn])
  const value = useMemo(() => ({ list, builtIn, find, edit }), [list, builtIn, find, edit])
  return (
    <Ctx.Provider value={value}>
      {children}
      {editing && <GroupingDialog meta={meta} rows={editing.rows ?? rows} start={editing.def} copy={editing.copy} field={editing.field} onClose={() => setEditing(null)}
        onSave={(d) => { onSave(d); editing.onSaved?.(d.id); setEditing(null) }}
        onRemove={editing.def && !editing.copy ? () => { onRemove(editing.def!.id); setEditing(null) } : undefined} />}
    </Ctx.Provider>
  )
}

/** one-click starting points, from splits the notebooks used; offered when the view has the field */
const PRESETS: { field: string; kind: SplitKind; cuts?: number[]; labels?: string[]; name: string }[] = [
  { field: 'Iron', kind: 'median', labels: ['Low iron', 'High iron'], name: 'Iron · high vs low' },
  { field: 'EC (FTIR)', kind: 'cutoffs', cuts: [2, 4, 8], name: 'EC (FTIR) · 2, 4, 8 µg/m³' },
  { field: 'HIPS BC', kind: 'tertiles', name: 'HIPS BC · tertiles' },
  { field: 'PM2.5 mass', kind: 'quartiles', name: 'PM2.5 · quartiles' },
]

function GroupingDialog({ meta, rows, start, copy = false, field: startField, onSave, onRemove, onClose }: {
  meta: MetaFile; rows: FilterRow[]; start?: GroupingDef; copy?: boolean; field?: string
  onSave: (d: GroupingDef) => void; onRemove?: () => void; onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const has = (f: string) => meta.fields.includes(f)
  const firstField = startField && has(startField) ? startField : has('Iron') ? 'Iron' : meta.fields[0] ?? ''
  const [draft, setDraft] = useState<GroupingDef>(() => start ?? { id: newGroupingId(), name: '', field: firstField, kind: 'median', cuts: [], labels: [] })
  // cutoffs as typed (a half-typed "1." must not be rewritten under the cursor)
  const [cutText, setCutText] = useState<string[]>(() => (start?.cuts ?? []).map(String))
  const set = (patch: Partial<GroupingDef>) => setDraft((d) => ({ ...d, ...patch }))
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !document.querySelector('.combobox-menu')) { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const def: GroupingDef = { ...draft, cuts: draft.kind === 'cutoffs' ? cutText.map(Number).filter((v, i) => cutText[i].trim() !== '' && Number.isFinite(v)) : [] }
  const values = useMemo(() => rows.map((r) => r[def.field]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)), [rows, def.field])
  const g = applyGrouping(def, rows)
  const counts = g.labels.map((l) => rows.filter((r) => g.labelOf(r) === l).length)
  // each band's colour from the ordered palette, what a band without its own colour shows
  const auto = colorsFor(g.labels.length)
  const unit = prettyUnit(meta.field_units?.[def.field])
  const median = values.length ? d3.median(values)! : NaN
  const splitName = SPLITS.find((s) => s.kind === def.kind)?.label ?? ''
  const name = draft.name.trim() || `${def.field} · ${def.kind === 'cutoffs' ? (def.cuts.length ? def.cuts.join(', ') : 'cutoffs') : splitName.toLowerCase()}`
  const ok = !!def.field && values.length > 0 && (def.kind !== 'cutoffs' || def.cuts.length > 0)

  const setKind = (kind: SplitKind) => {
    if (kind === 'cutoffs' && !cutText.length && Number.isFinite(median)) setCutText([String(+median.toPrecision(3))])
    set({ kind, labels: [] })
  }
  const usePreset = (p: (typeof PRESETS)[number]) => {
    setDraft((d) => ({ ...d, name: p.name, field: p.field, kind: p.kind, labels: p.labels ?? [] }))
    setCutText((p.cuts ?? []).map(String))
  }

  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div ref={ref} className="export-dialog ratio-dialog grouping-dialog" role="dialog" aria-modal="true" aria-label={start && !copy ? 'Edit a grouping' : 'New grouping'}>
        <header>
          <div><strong>{copy ? 'Your copy of a notebook grouping' : start ? 'Edit grouping' : 'New grouping'}</strong><span className="export-title">one measurement split into labelled bands, for panels or colour in every chart</span></div>
          <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
        </header>
        {!start && <section className="ratio-presets">
          <div className="row-label">Start from</div>
          <div className="ratio-preset-list">
            {PRESETS.filter((p) => has(p.field)).map((p) => (
              <button key={p.name} type="button" className="chip ratio-preset" onClick={() => usePreset(p)}>{p.name}</button>
            ))}
          </div>
        </section>}
        <section className="grouping-form">
          <FieldSelect label="Measurement" value={def.field} meta={meta} onChange={(f) => set({ field: f })} />
          <Combobox label="Split" value={def.kind} options={SPLITS.map((x) => ({ key: x.kind, label: x.label }))} onChange={setKind}
            title="Cutoffs: your own values · Median split to deciles: computed from the filters the chart draws, every site pooled. Which band a value equal to a cutoff joins is chosen below" />
          {def.kind === 'cutoffs' && <div className="grouping-cuts">
            <span className="combobox-label">Cutoffs{unit ? ` (${unit})` : ''}</span>
            {cutText.map((t, i) => (
              <span key={i} className="grouping-cut">
                <input className="select" type="number" step="any" value={t} aria-label={`Cutoff ${i + 1}`}
                  onChange={(e) => setCutText((c) => c.map((x, j) => (j === i ? e.target.value : x)))} />
                <button type="button" className="btn quiet x-btn" aria-label={`Remove cutoff ${i + 1}`} onClick={() => setCutText((c) => c.filter((_, j) => j !== i))}><XIcon /></button>
              </span>
            ))}
            <button type="button" className="btn quiet" onClick={() => setCutText((c) => [...c, Number.isFinite(median) ? String(+median.toPrecision(3)) : ''])}>+ cutoff</button>
          </div>}
          {g.cuts.length > 0 && <Segmented label="Equal to a cutoff" value={draft.equal ?? 'above'} options={['above', 'below'] as const}
            optionLabel={(o) => (o === 'above' ? 'Upper band (≥)' : 'Lower band (≤)')} onChange={(v) => set({ equal: v === 'below' ? 'below' : undefined })}
            title="Which band a value exactly on a cutoff joins: the upper one (bands are [lower, upper), as pandas cut with right=False) or the lower one ((lower, upper], right=True)" />}
          <label className="grouping-name"><span className="combobox-label">Name</span>
            <input className="select" value={draft.name} placeholder={name} onChange={(e) => set({ name: e.target.value })} />
          </label>
        </section>
        <Preview values={values} cuts={g.cuts} colors={g.colors} unit={unit} equal={def.equal} exact={def.kind === 'cutoffs'} />
        <section className="grouping-bands">
          {g.labels.map((l, i) => (
            <div key={i} className="grouping-band">
              <BandColor label={l} color={g.colors[i]} standard={auto[i]}
                onChange={(c) => set({ colors: Array.from({ length: g.labels.length }, (_, j) => (j === i ? c : draft.colors?.[j] ?? '')) })}
                onStandard={() => set({ colors: Array.from({ length: g.labels.length }, (_, j) => (j === i ? '' : draft.colors?.[j] ?? '')) })} />
              <input className="select" value={draft.labels[i] ?? ''} placeholder={l} aria-label={`Label of band ${i + 1}`}
                onChange={(e) => set({ labels: Array.from({ length: g.labels.length }, (_, j) => (j === i ? e.target.value : draft.labels[j] ?? '')) })} />
              <span className="grouping-range">{rangeText(g.cuts, i, rangeOptions(def))}{unit ? ` ${unit}` : ''}</span>
              <span className="grouping-count">{counts[i].toLocaleString()} filters</span>
            </div>
          ))}
          <p className="chart-note" style={{ margin: 0 }}>{(rows.length - values.length).toLocaleString()} of {rows.length.toLocaleString()} filters in view have no {def.field} and are left out of the groups.</p>
        </section>
        <div className="export-actions">
          {onRemove && <button type="button" className="btn" style={{ marginRight: 'auto' }} onClick={onRemove}>Delete grouping</button>}
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="button" className="btn primary" disabled={!ok} onClick={() => onSave({ ...def, name })}>{start && !copy ? 'Save' : 'Create and use'}</button>
        </div>
      </div>
    </div>,
    document.body,
  )
}

/**
 * A band's colour: its swatch is a button that opens the gallery's colour chooser (the export's, ColorChoice),
 * with the palette's colour for the band as its standard.
 */
function BandColor({ label, color, standard, onChange, onStandard }: { label: string; color: string; standard: string; onChange: (c: string) => void; onStandard: () => void }) {
  const [open, setOpen] = useState(false)
  const anchor = useRef<HTMLButtonElement>(null)
  return <>
    <button ref={anchor} type="button" className="grouping-swatch" style={{ background: color }} aria-haspopup="dialog" aria-expanded={open}
      aria-label={`Colour of ${label}: ${color}`} title={`${label}: ${color} · click to change`} onClick={() => setOpen(!open)} />
    {open && anchor.current && <ColorPopover anchor={anchor.current} onClose={() => setOpen(false)} title={`Colour of ${label}`} label={`Colour of ${label}`}>
      <ColorChoice label={label} value={color} standard={standard} onChange={onChange} onStandard={onStandard} />
    </ColorPopover>}
  </>
}

/** the distribution of the measurement in view, the cutoffs on it and each band's colour */
function Preview({ values, cuts, colors, unit, equal, exact }: { values: number[]; cuts: number[]; colors: string[]; unit: string; equal?: 'above' | 'below'; exact: boolean }) {
  const W = 560, H = 120, m = { l: 8, r: 8, t: 8, b: 22 }
  if (values.length < 2) return <p className="chart-note">No filters in view carry this measurement.</p>
  const sorted = [...values].sort((a, b) => a - b)
  const lo = d3.quantile(sorted, 0.005)!, hi = d3.quantile(sorted, 0.995)!
  const x = d3.scaleLinear().domain([Math.min(lo, ...cuts), Math.max(hi, ...cuts)]).nice().range([m.l, W - m.r])
  const bins = d3.bin().domain(x.domain() as [number, number]).thresholds(40)(sorted.filter((v) => v >= x.domain()[0] && v <= x.domain()[1]))
  const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length) ?? 1]).range([H - m.b, m.t])
  const band = (v: number) => bandOf(cuts, v, equal) ?? 0
  return (
    <svg className="grouping-preview" viewBox={`0 0 ${W} ${H}`} width="100%" fontFamily={FONT.family} role="img" aria-label="Distribution of the measurement with the cutoffs">
      {bins.map((b, i) => <rect key={i} x={x(b.x0!) + 0.5} width={Math.max(0, x(b.x1!) - x(b.x0!) - 1)} y={y(b.length)} height={H - m.b - y(b.length)}
        fill={colors[band(((b.x0 ?? 0) + (b.x1 ?? 0)) / 2)]} fillOpacity={0.75} />)}
      {cuts.map((c, i) => <g key={i}><line x1={x(c)} x2={x(c)} y1={m.t - 4} y2={H - m.b} stroke={INK.text} strokeWidth={1.5} strokeDasharray="4 3" />
        <text x={x(c) + 3} y={m.t + 6} fontSize={10} fill={INK.text}>{exact ? +c.toPrecision(6) : +c.toPrecision(3)}</text></g>)}
      <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke={INK.axis} />
      {x.ticks(6).map((t) => <text key={t} x={x(t)} y={H - 6} fontSize={10} textAnchor="middle" fill={INK.muted}>{t}</text>)}
      <text x={W - m.r} y={m.t + 6} fontSize={10} textAnchor="end" fill={INK.muted}>{unit}</text>
    </svg>
  )
}

/**
 * A chart's grouping picker: None, the reader's groupings and the notebooks', each with what it consists of on a
 * second line, then an add (+) button and Edit… (Build on this… for a notebook grouping, which saves a new one). `value` is the
 * id ('' for none). Charts show it always and their group choices only once a grouping is picked.
 */
export function GroupingPicker({ value, onChange, field, rows }: { value: string; onChange: (id: string) => void; field?: string; rows?: FilterRow[] }) {
  const { list, builtIn, find, edit } = useGroupings()
  const current = find(value)
  const fromNotebook = !!current && builtIn.some((d) => d.id === current.id)
  // None first: a chart offers its group choices (Per group, colour by Group …) only once a grouping is picked
  const available = (d: GroupingDef) => rows ? rows.filter(r => typeof r[d.field] === 'number' && Number.isFinite(r[d.field])).length : undefined
  const availability = (d: GroupingDef) => ({ hint: available(d) === undefined ? undefined : `${available(d)} filters`, disabled: available(d) === 0 && d.id !== value })
  const options = [
    { key: '', label: current ? 'None' : 'Select a grouping', detail: current ? 'clear the grouping' : 'pick one to split or colour by its bands' },
    ...list.map((d) => ({ key: d.id, label: d.name, detail: describeGrouping(d), ...availability(d), group: 'Your groupings' })),
    ...builtIn.map((d) => ({ key: d.id, label: d.name, detail: describeGrouping(d), ...availability(d), group: 'From the notebooks', title: `${d.source}${d.note ? ` · ${d.note}` : ''}` })),
  ]
  return (
    <span className="grouping-picker">
      <Combobox label="Grouping" value={current ? value : ''} options={options} minWidth={160} onChange={onChange}
        title="A measurement split into labelled bands (high vs low iron, rain classes, tertiles …), defined once for every chart" />
      <span className="grouping-actions">
        <button type="button" className="btn icon-btn" onClick={() => edit({ field, onSaved: onChange, rows })} title="Make a grouping from scratch" aria-label="New grouping">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true">
            <path d="M8 3v10M3 8h10" />
          </svg>
        </button>
        {current && (fromNotebook
          // a notebook grouping is not changed in place: building on it saves the reader's own copy
          ? <button type="button" className="btn" title="Open this notebook grouping in the editor; saving makes a new grouping of yours"
            onClick={() => edit({ def: { ...current, id: newGroupingId(), name: `${current.name} (my copy)` }, copy: true, onSaved: onChange, rows })}>Build on this…</button>
          : <button type="button" className="btn" title="Change this grouping (every chart using it follows)" onClick={() => edit({ def: current, onSaved: onChange, rows })}>Edit…</button>)}
      </span>
    </span>
  )
}

/**
 * A chart's grouping in one call: the chosen grouping (kept in the URL as `<chartId>.group`), applied to `rows`
 * (the filters the chart draws, so quantiles match it), and the picker for its options row.
 */
export function useChartGrouping(chartId: string, rows: FilterRow[], onFirstPick?: () => void) {
  const { find } = useGroupings()
  const [groupId, setGroupId] = useSearchParam<string>(`${chartId}.group`, '')
  const def = find(groupId)
  const grp = useMemo(() => (def ? applyGrouping(def, rows) : null), [def, rows])
  // filters without the grouping's measurement are in no band: say so wherever a chart leaves them out
  const missing = useMemo(() => (grp ? rows.filter((r) => grp.labelOf(r) === null).length : 0), [grp, rows])
  const missingNote = (what: string) => (def && missing > 0
    ? <p className="chart-note">No data: {missing.toLocaleString()} of {rows.length.toLocaleString()} filters have no {def.field}, {what}.</p> : null)
  // the first pick (from none) also switches the chart to the grouping, so it shows at once
  const picker = <GroupingPicker value={groupId} rows={rows} onChange={(id) => { if (!def && id) onFirstPick?.(); setGroupId(id) }} />
  return { grp, picker, missingNote }
}
