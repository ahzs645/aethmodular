import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { AERONET_SUMMARY, hasAod, scaleHeight, type AeronetStations } from '@/lib/aeronet'
import { seasonOnly } from '@/lib/seasons'
import type { FilterRow, MetaFile } from '@/lib/types'

const siteColour = (meta: MetaFile, name: string) => meta.sites.find((s) => s.name === name)?.color ?? INK.muted

/**
 * One row per picked site: the AERONET station behind it, how many of its filters have a
 * retrieval in their window, and the median of each column quantity over those filters.
 * Clicking a site narrows the Filters panel's pick to it.
 */
export function AeronetSiteTable({ rows, meta, stations, current, onSite }: {
  rows: FilterRow[]; meta: MetaFile; stations: AeronetStations | null; current: string | null
  onSite?: (code: string) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const table = useMemo(() => d3.groups(rows, (r) => r.site).map(([site, rs]) => {
    const code = rs[0]?.code ?? ''
    const st = stations?.sites[code]
    const withAod = rs.filter(hasAod)
    return {
      site, code, st, n: rs.length, nAod: withAod.length,
      nInv: rs.filter((r) => typeof r[AERONET_SUMMARY[2].field] === 'number').length,
      cells: AERONET_SUMMARY.map((q) => d3.median(rs, (r) => (typeof r[q.field] === 'number' ? (r[q.field] as number) : undefined)) ?? null),
      h: d3.median(rs, (r) => scaleHeight(r) ?? undefined) ?? null,
    }
  }).filter((t) => t.nAod > 0).sort((a, b) => a.site.localeCompare(b.site)), [rows, stations])

  const cols = ['station', 'km', 'filters', 'AOD', 'inversion', ...AERONET_SUMMARY.map((q) => q.label), 'H (m)']
  const labelW = 190, stationW = 150, numW = Math.max(62, Math.min(90, (width - labelW - stationW) / (cols.length - 1)))
  const rowH = 24, head = 30
  const w = labelW + stationW + (cols.length - 1) * numW, h = head + table.length * rowH + 4
  const colX = (j: number) => (j === 0 ? labelW : labelW + stationW + (j - 1) * numW)

  return (
    <ChartFrame id="aeronet-sites" title="Sites compared — the column over each site's filters"
      subtitle="Per picked site: the AERONET station used (the one within 50 km whose daily AOD covers the most filter days), the filters in view, how many have an AOD and an inversion inside their sampling window, and the median over those filters. H is the absorption scale height, AAOD 675 / surface Fabs. Click a site to look at it alone."
      provenance="research/ftir_ec_phase3/AERONET_SPARTAN_2026-08-22.md (four sites), extended to every SPARTAN site with a station">
      <div ref={wrapRef} className="chart-wrap" style={{ overflowX: 'auto' }}>
        {table.length === 0 ? <Empty>No picked site has AERONET retrievals inside its filters' windows.</Empty> : (
          <svg width={w} height={h} fontFamily={FONT.family}>
            {cols.map((c, j) => <text key={c} x={colX(j) + 4} y={head - 10} fontSize={10.5} fill={INK.muted}>{c}</text>)}
            {table.map((t, i) => {
              const on = t.code === current
              const others = t.st?.candidates.filter((c) => c.station !== t.st?.station) ?? []
              return (
                <g key={t.site} transform={`translate(0,${head + i * rowH})`} style={{ cursor: onSite ? 'pointer' : undefined }} onClick={() => onSite?.(t.code)}
                  onMouseEnter={(e) => tip.show(e, [t.site, t.st ? `${t.st.station}, ${fmt(t.st.km, 1)} km` : 'station unknown',
                    ...others.map((c) => `also within 50 km: ${c.station} (${fmt(c.km, 1)} km, ${c.window_days_with_aod} window days with AOD)`),
                    onSite ? 'click to look at this site alone' : ''].filter(Boolean))} onMouseLeave={tip.hide}>
                  <rect width={w} height={rowH - 2} fill={on ? INK.tint : 'transparent'} />
                  <circle cx={10} cy={rowH / 2 - 1} r={5} fill={siteColour(meta, t.site)} />
                  <text x={22} y={rowH / 2} dy="0.1em" fontSize={12} fontWeight={on ? 650 : 400} fill={INK.text}>{t.site}</text>
                  <text x={colX(0) + 4} y={rowH / 2} dy="0.1em" fontSize={11} fill={INK.text}>{t.st?.station.replace(/_/g, ' ') ?? '—'}</text>
                  {[fmt(t.st?.km ?? null, 1), String(t.n), String(t.nAod), String(t.nInv), ...t.cells.map((v, k) => fmt(v, AERONET_SUMMARY[k].digits)), fmt(t.h, 0)].map((v, k) => (
                    <text key={k} x={colX(k + 1) + 4} y={rowH / 2} dy="0.1em" fontSize={11} fontFamily={FONT.mono} fill={INK.muted}>{v}</text>
                  ))}
                </g>
              )
            })}
          </svg>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/**
 * The effective absorption scale height H = AAOD 675 / (surface Fabs × 10⁻⁶): median and interquartile
 * range over the filters with both, one row per site (several sites) or per season (one site). A surface
 * Fabs too high for the column above it makes H small. Only comparable between sites measured the same
 * way; not a boundary-layer height.
 */
export function ScaleHeightStrip({ rows, meta, by = 'site', onSite }: {
  rows: FilterRow[]; meta: MetaFile; by?: 'site' | 'season'; onSite?: (code: string) => void
}) {
  const blocks = useMemo(() => {
    const key = (r: FilterRow) => (by === 'site' ? r.site : String(r.season ?? ''))
    const groups = d3.groups(rows, key).map(([name, rs]) => ({
      name, code: rs[0]?.code ?? '', hs: rs.map(scaleHeight).filter((v): v is number => v !== null).sort(d3.ascending),
    })).filter((g) => g.name && g.hs.length >= 5)
    // sites by their median; seasons in the calendar's order
    const order = meta.seasons.map((x) => x.name)
    groups.sort((a, b) => (by === 'site' ? (d3.median(a.hs) ?? 0) - (d3.median(b.hs) ?? 0) : order.indexOf(a.name) - order.indexOf(b.name)))
    const out: IntervalRow[] = groups.map((g) => {
      const mid = d3.quantileSorted(g.hs, 0.5), lo = d3.quantileSorted(g.hs, 0.25), hi = d3.quantileSorted(g.hs, 0.75)
      const color = by === 'site' ? siteColour(meta, g.name) : meta.seasons.find((x) => x.name === g.name)?.color ?? INK.muted
      return {
        key: g.name, label: by === 'site' ? g.name : seasonOnly(g.name), note: `n=${g.hs.length}`, onClick: by === 'site' && onSite ? () => onSite(g.code) : undefined,
        marks: [[{ mid, lo, hi, color }]],
        tip: [g.name, `median H ${fmt(mid ?? null, 0)} m (IQR ${fmt(lo ?? null, 0)}–${fmt(hi ?? null, 0)})`, `${g.hs.length} filters with AAOD 675 and Fabs`],
      }
    })
    return [{ key: by, rows: out }]
  }, [rows, meta, by, onSite])
  const any = blocks[0].rows.length > 0
  return (
    <ChartFrame id="aeronet-scale-height" title={`Column against surface — the absorption scale height by ${by}`}
      subtitle={`H = AAOD 675 nm (AERONET inversions in the filter's window) ÷ surface Fabs (HIPS, 633 nm; public BC × 10 where HIPS Fabs is absent). Median and interquartile range by ${by}, at least 5 filters each. Filters that read more absorption than the column above them supports sit low. H is not a boundary-layer height and only compares sites measured the same way.`}
      provenance="research/ftir_ec_phase3/AERONET_SPARTAN_2026-08-22.md: Addis H ≈ 230 m against 540–890 m at Delhi, Beijing and Pasadena">
      {any ? <IntervalStrip blocks={blocks} panels={[{ label: 'Scale height H (m)', format: (v) => fmt(v, 0) }]} noteWidth={50} />
        : <Empty>No {by} in view has at least 5 filters with both an AERONET inversion and a surface Fabs.</Empty>}
    </ChartFrame>
  )
}
