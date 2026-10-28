import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { useFitHeight } from '@/lib/chartFit'
import { useTileSetting } from '@/lib/url'
import { chemText } from '@/lib/format'
import type { FilterRow, MetaFile } from '@/lib/types'
import { RATIO_GROUP } from '@/lib/viewFields'

const SKIP = new Set([RATIO_GROUP, 'Sampling'])
/** Full: every group with its filter count. Compact: the measured groups as a share, the rest on one line. Ticker: one scrolling line. */
const VIEWS = ['Full', 'Compact', 'Ticker'] as const
type View = (typeof VIEWS)[number]
const pctOf = (k: number, n: number) => `${Math.round((100 * k) / Math.max(1, n))} %`
/** a group's name without its qualifier ("Weather (sampling window)" → "Weather"), for the short views; the tooltip keeps it */
const shortLabel = (label: string) => chemText(label.replace(/\s*\([^)]*\)$/, ''))

/**
 * What the filters in view are, before any chart (OpenAQ's location page opens the same
 * way): the site and where it is, the dates and count, and for each group of measurements
 * how many filters have at least one measurement. The tooltip lists each field’s coverage. A dash on a site
 * card ("not measured") is explained here. With several sites in view, one line each.
 */
export function SiteSummary({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const fit = useFitHeight()
  const [view, setView] = useTileSetting<View>('sitesummary.view', 'Full', VIEWS)
  const sites = useMemo(() => [...d3.group(rows, (r) => r.site)].map(([site, rs]) => {
    const m = meta.sites.find((s) => s.name === site)
    const has = (f: string) => rs.filter((r) => typeof r[f] === 'number' && Number.isFinite(r[f])).length
    // the derived groups (ratios of other fields, the sampled volume) say nothing about what was measured
    const groups = meta.field_groups.filter((g) => !SKIP.has(g.label)).map((g) => {
      const counts = g.fields.map((f) => ({ f, n: has(f) }))
      const measured = counts.filter((c) => c.n > 0)
      return { label: g.label, of: g.fields.length, measured, filters: rs.filter(r => g.fields.some(f => typeof r[f] === 'number' && Number.isFinite(r[f]))).length }
    })
    return { site, code: rs[0].code, color: rs[0].color, location: m?.location, lat: m?.lat, lon: m?.lon, n: rs.length,
      from: d3.min(rs, (r) => r.date) ?? '', to: d3.max(rs, (r) => r.date) ?? '', months: new Set(rs.map((r) => r.date.slice(0, 7))).size, groups }
  }), [rows, meta])

  return (
    <ChartFrame id="sitesummary" title="Site — where, when, and what was measured" exportable={false}
      controls={<Segmented label="Show" value={view} options={VIEWS} onChange={setView}
        title="Full: every group with its filter count. Compact: measured groups as a share, the rest on one line. Ticker: one scrolling line (pauses on hover)" />}>
      {view === 'Ticker' ? <div className="site-summary"><SiteTicker sites={sites} /></div>
      : <div className="site-summary" style={fit === null ? undefined : { maxHeight: Math.max(60, fit), overflowY: 'auto' }}>
        {sites.length === 0 && <Empty>No filters in this subset.</Empty>}
        {view === 'Compact' ? sites.map((s) => {
          const measured = s.groups.filter((g) => g.of > 0 && g.measured.length), missing = s.groups.filter((g) => g.of > 0 && !g.measured.length)
          return (
            <div key={s.site} className="site-summary-row compact">
              <div className="site-summary-head">
                <span className="swatch" style={{ background: s.color }} />
                <strong>{s.site}</strong><span className="site-card-n">{s.code}</span>
                <span className="site-summary-meta">{s.n} filters · {s.from.slice(0, 7)} → {s.to.slice(0, 7)}</span>
              </div>
              <div className="site-summary-groups">
                {measured.map((g) => <span key={g.label} className="site-summary-group" tabIndex={0} title={groupTitle(g, s)}>{shortLabel(g.label)} <em>{pctOf(g.filters, s.n)}</em></span>)}
              </div>
              {missing.length > 0 && <div className="site-summary-meta">Not measured: {missing.map((g) => shortLabel(g.label)).join(', ')}</div>}
            </div>
          )
        }) : sites.map((s) => (
          <div key={s.site} className="site-summary-row">
            <div className="site-summary-head">
              <span className="swatch" style={{ background: s.color }} />
              <strong>{s.site}</strong><span className="site-card-n">{s.code}</span>
              <span className="site-summary-meta">{s.location !== s.site ? s.location : null}{s.lat != null && s.lon != null ? `${s.location && s.location !== s.site ? ' · ' : ''}${s.lat.toFixed(2)}°, ${s.lon.toFixed(2)}°` : ''}</span>
            </div>
            <div className="site-summary-meta">{s.n} filters · {s.from} → {s.to} · {s.months} months</div>
            <div className="site-summary-groups">
              {s.groups.filter((g) => g.of > 0).map((g) => (
                <span key={g.label} className={`site-summary-group${g.measured.length ? '' : ' none'}`}
                  tabIndex={0} title={groupTitle(g, s)}>
                  {chemText(g.label)} <em>{g.measured.length ? `${g.filters}/${s.n} filters` : 'not measured'}</em>
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>}
    </ChartFrame>
  )
}

type Site = { site: string; code: string; color: string; n: number; from: string; to: string; groups: { label: string; of: number; measured: { f: string; n: number }[]; filters: number }[] }

/** a group's coverage, field by field (every view's tooltip) */
const groupTitle = (g: Site['groups'][number], s: Site) => g.measured.length
  ? `${g.filters} of ${s.n} filters have at least one measurement in this group. ${g.measured.length} of ${g.of} measurement types are available.\n` + g.measured.map((c) => `${c.f}: ${c.n} of ${s.n} filters`).join('\n')
  : `${g.label}: not measured at ${s.site}`

/**
 * One line that scrolls: each site, its filters and dates, then its measured groups as a share of its filters.
 * Drawn twice so the loop is seamless; it pauses on hover and stands still for a reader who asks for less motion
 * (styles.css .site-ticker), where the line scrolls by hand instead.
 */
function SiteTicker({ sites }: { sites: Site[] }) {
  if (!sites.length) return <Empty>No filters in this subset.</Empty>
  // one site: its name and dates stay put above the line, which then carries only what was measured
  const one = sites.length === 1 ? sites[0] : null
  const head = (s: Site) => <span key={`${s.site}-h`} className="site-ticker-site"><span className="swatch" style={{ background: s.color }} /><strong>{s.site}</strong> {s.n} filters · {s.from.slice(0, 7)} → {s.to.slice(0, 7)}</span>
  const items = sites.flatMap((s) => [
    ...(one ? [] : [head(s)]),
    ...s.groups.filter((g) => g.of > 0 && g.measured.length).map((g) =>
      <span key={`${s.site}-${g.label}`} className="site-summary-group" title={groupTitle(g, s)}>{shortLabel(g.label)} <em>{pctOf(g.filters, s.n)}</em></span>),
  ])
  // about 0.18 s a character, so a long line scrolls at the same speed as a short one
  const chars = sites.reduce((k, s) => k + s.site.length + 40 + s.groups.reduce((a, g) => a + g.label.length + 6, 0), 0)
  const missing = one ? one.groups.filter((g) => g.of > 0 && !g.measured.length) : []
  return (<>
    {one && <div className="site-summary-head">{head(one)}</div>}
    <div className="site-ticker" aria-label="Sites and what was measured">
      <div className="site-ticker-track" style={{ animationDuration: `${Math.max(20, chars * 0.18)}s` }}>
        <span className="site-ticker-run">{items}</span>
        <span className="site-ticker-run" aria-hidden="true">{items}</span>
      </div>
    </div>
    {missing.length > 0 && <div className="site-summary-meta">Not measured: {missing.map((g) => shortLabel(g.label)).join(', ')}</div>}
  </>)
}
