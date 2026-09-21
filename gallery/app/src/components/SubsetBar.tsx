import { useRef, useState } from 'react'
import { useDimensions } from '@/hooks/useGalleryData'
import { DateBrush, type DateRange } from '@/components/DateBrush'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Segmented, SwapButton, Toggle } from '@/components/ChartFrame'
import { AXES_MODES, type AxesOpts } from '@/lib/axes'
import { fmt } from '@/lib/stats'
import { usePrefs } from '@/lib/prefs'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'

export interface SubsetCaps {
  /** the tab's charts respond to the season filter */
  seasons: boolean
  /** the tab has more than one site to choose between */
  sites: boolean
  /** the tab's data carries exclusion flags */
  excluded: boolean
  /** the tab's data has dates the brush can cut */
  dates: boolean
  /** which axis fields the tab's charts share: an x/y pair, one measurement, or none */
  fields: 'pair' | 'single' | 'none'
  /** e.g. "filters" or "PMF filters" — what the count is counting */
  datasetLabel: string
  /** shown when a control is absent because the data can't support it */
  note?: string
}

export interface SubsetState {
  convention: string
  seasons: SeasonMeta[]
  activeSeasons: Set<string>
  seasonCounts: Record<string, number>
  sites: string[]
  activeSites: Set<string>
  siteCounts: Record<string, number>
  includeExcluded: boolean
  nExcluded: number
  dateRange: DateRange | null
  xField: string
  yField: string
  field: string
  axes: AxesOpts
  pinned: FilterRow | null
  pinnedInSubset: boolean
  shown: number
  total: number
}

export interface SubsetActions {
  onConvention: (v: string) => void
  onToggleSeason: (name: string) => void
  onAllSeasons: () => void
  onToggleSite: (name: string) => void
  onAllSites: () => void
  onIncludeExcluded: (v: boolean) => void
  onDateRange: (r: DateRange | null) => void
  onXField: (v: string) => void
  onYField: (v: string) => void
  onField: (v: string) => void
  onAxes: (v: AxesOpts) => void
  onUnpin: () => void
  onOpenPinned: () => void
  onReset: () => void
}

/**
 * Global subset controls, rendered to match what the active tab can actually
 * use. A control that does nothing is worse than no control, so on the PMF
 * tab the site chips, the excluded toggle and the axis pickers are absent and
 * a note says why.
 *
 * Everything here is mirrored into the URL hash (lib/url.ts), so the bar is
 * also the shareable state of the app.
 */
export function SubsetBar({
  meta,
  caps,
  brushRows,
  state: s,
  actions: a,
}: {
  meta: MetaFile
  caps: SubsetCaps
  /** rows the date brush draws — before the date filter, after the exclusion toggle */
  brushRows: FilterRow[]
  state: SubsetState
  actions: SubsetActions
}) {
  const [open, setOpen] = useState(true)
  const { explain } = usePrefs()
  const brushWrap = useRef<HTMLDivElement>(null)
  const { width: brushW } = useDimensions(brushWrap)

  const allSeasons = s.activeSeasons.size === s.seasons.length
  const allSites = s.activeSites.size === s.sites.length
  const isDefault =
    allSeasons && allSites && !s.includeExcluded && !s.dateRange && s.convention === meta.season_convention

  const summary = (
    <div className="subset-summary">
      <span className="pill">{s.shown} of {s.total} {caps.datasetLabel}</span>
      {caps.seasons && !allSeasons && <span>seasons: {[...s.activeSeasons].join(', ')}</span>}
      {caps.sites && !allSites && <span>sites: {[...s.activeSites].join(', ')}</span>}
      {s.dateRange && <span>{s.dateRange[0]} → {s.dateRange[1]}</span>}
      {s.includeExcluded && caps.excluded && <span>+{s.nExcluded} excluded</span>}
      {caps.fields === 'pair' && <span>{s.xField} → {s.yField}</span>}
      {caps.fields === 'single' && <span>{s.field}</span>}
      <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
        {!isDefault && (
          <button type="button" className="btn quiet" onClick={a.onReset} title="Clear every filter">
            reset
          </button>
        )}
        <button type="button" className="btn quiet" onClick={() => setOpen((o) => !o)}>
          {open ? 'collapse' : 'expand'}
        </button>
      </span>
    </div>
  )

  return (
    <div className="panel subset">
      {summary}
      {open && (
        <>
          {caps.fields === 'pair' && (
            <>
            <Row label="axes">
              <FieldSelect label="x" value={s.xField} meta={meta} onChange={a.onXField} />
              <SwapButton onClick={() => { a.onXField(s.yField); a.onYField(s.xField) }} />
              <FieldSelect label="y" value={s.yField} meta={meta} onChange={a.onYField} />
              {explain && <span className="row-note">shared by every chart on this tab</span>}
            </Row>
            <Row label="crossplot axes">
              <Toggle
                label="1:1 line"
                checked={s.axes.identity}
                onChange={(v) => a.onAxes({ ...s.axes, identity: v })}
                title="Squares the panels, shares the x and y domain, and turns on the Deming slope"
              />
              <Segmented
                value={s.axes.mode}
                options={AXES_MODES}
                onChange={(v) => a.onAxes({ ...s.axes, mode: v })}
                title="Data: 0 always visible, negatives kept · From 0,0: both axes start at zero · Show intercept: extend to the fitted lines' y-intercepts and mark them"
              />
              <Toggle label="log axes" checked={s.axes.log} onChange={(v) => a.onAxes({ ...s.axes, log: v })} />
              {explain && <span className="row-note">applies to the scatterplot, hexbin and connected scatter together</span>}
            </Row>
            </>
          )}
          {caps.fields === 'single' && (
            <Row label="measurement">
              <FieldSelect label="" value={s.field} meta={meta} onChange={a.onField} />
              {explain && (
                <span className="row-note">
                  {meta.coverage?.[s.field] ?? '—'} of {meta.sites.reduce((n, x) => n + x.n_filters, 0)} filters carry {s.field}
                  {prettyUnit(meta.field_units?.[s.field]) ? ` · ${prettyUnit(meta.field_units?.[s.field])}` : ''}
                </span>
              )}
            </Row>
          )}

          {caps.seasons && (
            <>
              <Row label="season calendar">
                <select className="select" value={s.convention} onChange={(e) => a.onConvention(e.target.value)}>
                  {Object.keys(meta.season_conventions).map((k) => (
                    <option key={k} value={k}>
                      {k}
                      {k === meta.season_convention ? ' (repo default)' : ''}
                    </option>
                  ))}
                </select>
                {explain && (
                  <span className="row-note">
                    February sits in <strong style={{ color: 'var(--ink-text)' }}>{s.seasons.find((x) => x.months.includes(2))?.name ?? '—'}</strong> under this calendar
                  </span>
                )}
              </Row>

              <Row label="seasons">
                <Chip active={allSeasons} color="var(--ink-muted)" onClick={a.onAllSeasons} label="all" />
                {s.seasons.map((se) => {
                  const n = s.seasonCounts[se.name] ?? 0
                  return (
                    <Chip
                      key={se.name}
                      active={s.activeSeasons.has(se.name)}
                      color={se.color}
                      disabled={n === 0}
                      onClick={() => n > 0 && a.onToggleSeason(se.name)}
                      label={`${se.name} · ${n}`}
                    />
                  )
                })}
              </Row>
            </>
          )}

          {caps.sites && (
            <Row label="sites">
              <Chip active={allSites} color="var(--ink-muted)" onClick={a.onAllSites} label="all" />
              {s.sites.map((name) => {
                const n = s.siteCounts[name] ?? 0
                return (
                  <Chip
                    key={name}
                    active={s.activeSites.has(name)}
                    color={meta.sites.find((x) => x.name === name)?.color ?? 'var(--ink-muted)'}
                    disabled={n === 0}
                    onClick={() => n > 0 && a.onToggleSite(name)}
                    label={`${name} · ${n}`}
                  />
                )
              })}
            </Row>
          )}

          {caps.dates && (
            <Row label="date range">
              <div ref={brushWrap} style={{ flex: '1 1 320px', minWidth: 240 }}>
                <DateBrush rows={brushRows} meta={meta} range={s.dateRange} onChange={a.onDateRange} width={brushW} />
              </div>
              <span className="brush-hint" style={{ width: 150 }}>
                {s.dateRange ? (
                  <>
                    {s.dateRange[0]} → {s.dateRange[1]}
                    <br />
                    <button type="button" className="btn quiet" onClick={() => a.onDateRange(null)}>clear</button>
                  </>
                ) : explain ? (
                  'drag across the timeline to keep only a date range; snaps to whole months'
                ) : (
                  'all dates'
                )}
              </span>
            </Row>
          )}

          <Row label="">
            {caps.excluded ? (
              <label className="toggle">
                <input type="checkbox" checked={s.includeExcluded} onChange={(e) => a.onIncludeExcluded(e.target.checked)} />
                include {s.nExcluded} excluded samples
              </label>
            ) : null}
            {caps.note && explain && <span className="row-note">{caps.note}</span>}
            <span className="row-tail">MAC = {meta.mac_value}</span>
          </Row>

          {s.pinned && (
            <PinnedCard row={s.pinned} meta={meta} fields={[s.xField, s.yField, s.field]} inSubset={s.pinnedInSubset} onUnpin={a.onUnpin} onOpen={a.onOpenPinned} />
          )}
        </>
      )}
      {!open && s.pinned && (
        <PinnedCard row={s.pinned} meta={meta} fields={[s.xField, s.yField, s.field]} inSubset={s.pinnedInSubset} onUnpin={a.onUnpin} onOpen={a.onOpenPinned} />
      )}
    </div>
  )
}

/**
 * The record behind a clicked point. Charts only ever show two or three
 * numbers per filter; this is where "what *is* that outlier?" gets answered
 * without opening the pickle.
 */
function PinnedCard({
  row,
  meta,
  fields,
  inSubset,
  onUnpin,
  onOpen,
}: {
  row: FilterRow
  meta: MetaFile
  fields: string[]
  inSubset: boolean
  onUnpin: () => void
  onOpen: () => void
}) {
  const shown = [...new Set([...fields, 'EC (FTIR)', 'EC (TOR)', 'HIPS BC', 'PM2.5 mass'])].filter((f) =>
    meta.fields.includes(f)
  )
  return (
    <div className="pinned">
      <span className="id">📌 {row.id}</span>
      <span>
        <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 5, background: row.color as string, marginRight: 5 }} />
        {row.site} · {row.date} · {row.season}
      </span>
      {shown.map((f) => (
        <span key={f} className="kv">
          <span>{f}</span> {fmt(row[f] as number | null, 3)}
        </span>
      ))}
      {row.excluded && <span className="reason">excluded: {row.exclusion_reason}</span>}
      {!inSubset && <span className="row-note">(outside the current subset)</span>}
      <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4 }}>
        <button type="button" className="btn" onClick={onOpen}>details</button>
        <button type="button" className="btn quiet" onClick={onUnpin}>unpin</button>
      </span>
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      {children}
    </div>
  )
}

function Chip({
  active,
  color,
  onClick,
  label,
  disabled = false,
}: {
  active: boolean
  color: string
  onClick: () => void
  label: string
  disabled?: boolean
}) {
  const cls = ['chip', active ? 'active' : '', disabled ? 'disabled' : ''].filter(Boolean).join(' ')
  return (
    <button
      type="button"
      className={cls}
      onClick={onClick}
      disabled={disabled}
      title={disabled ? 'no data under the current selection' : undefined}
      style={{ ['--chip' as string]: color }}
    >
      {label}
    </button>
  )
}
