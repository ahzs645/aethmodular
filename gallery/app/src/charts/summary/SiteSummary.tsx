import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useFitHeight } from '@/lib/chartFit'
import { chemText } from '@/lib/format'
import type { FilterRow, MetaFile } from '@/lib/types'
import { RATIO_GROUP } from '@/lib/viewFields'

const SKIP = new Set([RATIO_GROUP, 'Sampling'])

/**
 * What the filters in view are, before any chart (OpenAQ's location page opens the same
 * way): the site and where it is, the dates and count, and for each group of measurements
 * how many filters have at least one measurement. The tooltip lists each field’s coverage. A dash on a site
 * card ("not measured") is explained here. With several sites in view, one line each.
 */
export function SiteSummary({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const fit = useFitHeight()
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
    <ChartFrame id="sitesummary" title="Site — where, when, and what was measured" exportable={false}>
      <div className="site-summary" style={fit === null ? undefined : { maxHeight: Math.max(60, fit), overflowY: 'auto' }}>
        {sites.length === 0 && <Empty>No filters in this subset.</Empty>}
        {sites.map((s) => (
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
                  tabIndex={0} title={g.measured.length ? `${g.filters} of ${s.n} filters have at least one measurement in this group. ${g.measured.length} of ${g.of} measurement types are available.\n` + g.measured.map((c) => `${c.f}: ${c.n} of ${s.n} filters`).join('\n') : `${g.label}: not measured at ${s.site}`}>
                  {chemText(g.label)} <em>{g.measured.length ? `${g.filters}/${s.n} filters` : 'not measured'}</em>
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </ChartFrame>
  )
}
