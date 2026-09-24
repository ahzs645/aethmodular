import { useId, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'
import { calendarLabel, mixedCalendars, seasonsForSite } from '@/siteSeasons'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const STATS = ['Median', 'Mean'] as const
const LAYOUTS = ['Grouped', 'Rings'] as const
/** A site-month with fewer filters than this is a stub, not a value: a median of two is a coin toss. */
const MIN_N = 3
const TAU = 2 * Math.PI

type Cell = { month: number; n: number; median: number | null; mean: number | null; q1: number | null; q3: number | null }
type Series = { name: string; color: string; cells: Cell[] }

/** Angle (radians, clockwise from 12 o'clock — d3.arc's convention) at which month m (1–12) starts. */
const monthStart = (m: number) => ((m - 1) * TAU) / 12

/** A point on the circle at radius r and clock angle a (radians from the top, clockwise). */
const polar = (r: number, a: number): [number, number] => [r * Math.sin(a), -r * Math.cos(a)]

/**
 * Splits a season's months into runs that are contiguous *around the circle*,
 * so Oct–Feb is one run crossing the Dec/Jan seam rather than two (Oct–Dec
 * and Jan–Feb). Returned as { start, length } with start in 1–12.
 */
function circularRuns(months: number[]): { start: number; length: number }[] {
  const set = new Set(months)
  if (set.size === 0) return []
  if (set.size === 12) return [{ start: 1, length: 12 }]
  const prev = (m: number) => (m === 1 ? 12 : m - 1)
  const next = (m: number) => (m === 12 ? 1 : m + 1)
  const runs: { start: number; length: number }[] = []
  // a run starts at any month whose predecessor is not in the season
  for (const m of [...set].sort(d3.ascending)) {
    if (set.has(prev(m))) continue
    let length = 1
    let cur = m
    while (set.has(next(cur))) {
      cur = next(cur)
      length++
    }
    runs.push({ start: m, length })
  }
  return runs
}

/**
 * Circular barplot as a seasonal clock — react-graph-gallery.com/circular-barplot.
 *
 * The monthly band chart lays the year on a straight axis, which puts
 * December and January at opposite ends although they are neighbours in the
 * same dry season. On a circle the dry season (Oct–Feb under `dry_feb`) is
 * one contiguous wedge and the eye reads it as such. The season sectors come
 * from meta.seasons, i.e. the calendar picked in the subset bar, so switching
 * to `belg_feb` visibly moves February out of the dry wedge.
 *
 * Two layouts: `Grouped` puts the sites side by side inside each month's
 * sector, sharing one radial scale; `Rings` gives each site its own ring with
 * its own baseline, still on a shared value scale so ring heights compare.
 */
export function SeasonalClock({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  // textPath needs a path id; useId keeps two clocks on one page from sharing one
  const uid = useId().replace(/:/g, '')

  const [stat, setStat] = useState<(typeof STATS)[number]>('Median')
  const [layout, setLayout] = useState<(typeof LAYOUTS)[number]>('Grouped')
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [hover, setHover] = useState<string | null>(null)

  const allSeries: Series[] = useMemo(
    () =>
      meta.sites
        .map((s) => {
          const cells: Cell[] = d3.range(1, 13).map((m) => {
            const vals = rows
              .filter((r) => r.site === s.name && r.month === m)
              .map((r) => r[field])
              .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
              .sort(d3.ascending)
            return {
              month: m,
              n: vals.length,
              median: vals.length ? (d3.quantile(vals, 0.5) ?? null) : null,
              mean: vals.length ? (d3.mean(vals) ?? null) : null,
              // an IQR of three points is still a stretch, but it is what the band chart quotes too
              q1: vals.length >= MIN_N ? (d3.quantile(vals, 0.25) ?? null) : null,
              q3: vals.length >= MIN_N ? (d3.quantile(vals, 0.75) ?? null) : null,
            }
          })
          return { name: s.name, color: s.color, cells }
        })
        .filter((s) => s.cells.some((c) => c.n > 0)),
    [rows, field, meta.sites]
  )
  const series = allSeries.filter((s) => !hidden.has(s.name))
  // a hovered site dims the other sites' bars; a hovered season (a background key, not a series) deepens its own sectors
  const siteHover = hover && series.some((s) => s.name === hover) ? hover : null
  const seasonHover = hover && meta.seasons.some((s) => s.name === hover) ? hover : null
  const dim = (name: string) => (siteHover === null || siteHover === name ? 1 : 0.15)
  const valueOf = (c: Cell) => (c.n >= MIN_N ? (stat === 'Median' ? c.median : c.mean) : null)

  // ---- geometry: a square, with room outside the ring for month and season labels
  const size = Math.max(280, Math.min(width, 520))
  const cx = size / 2
  const cy = size / 2
  const outerR = size / 2 - 64
  const hole = Math.round(outerR * 0.24)
  const monthLabelR = outerR + 14
  const seasonLabelR = outerR + 38

  const valued = series.flatMap((s) => s.cells.map(valueOf)).filter((v): v is number => v !== null)
  // never floor at zero: EC (FTIR) carries genuinely negative values near the detection limit
  const vMin = Math.min(0, d3.min(valued) ?? 0)
  const vMax = Math.max(vMin + 1e-9, (d3.max(valued) ?? 1) * 1.05)

  // Grouped: one radial scale from the hole to the rim. Rings: the same value
  // domain mapped into each ring's band, so a tall bar means the same thing
  // on every ring even though each ring has its own baseline.
  const ringGap = 5
  const band = layout === 'Rings' && series.length ? (outerR - hole) / series.length : outerR - hole
  const ringInner = (i: number) => (layout === 'Rings' ? hole + i * band : hole)
  const ringOuter = (i: number) => ringInner(i) + band - (layout === 'Rings' ? ringGap : 0)
  const rel = d3.scaleLinear().domain([vMin, vMax]).range([0, band - (layout === 'Rings' ? ringGap : 0)]).nice()
  const rOf = (i: number, v: number) => ringInner(i) + rel(v)
  const ticks = rel.ticks(3).filter((t) => t !== 0 && t >= rel.domain()[0] && t <= rel.domain()[1])

  // within a month's sector, the sub-sector for each site (Grouped only)
  const sectorPad = TAU / 12 / 14
  const subBand = d3.scaleBand<string>().domain(series.map((s) => s.name)).range([sectorPad, TAU / 12 - sectorPad]).paddingInner(0.14)

  const arc = d3.arc<{ r0: number; r1: number; a0: number; a1: number }>()
    .innerRadius((d) => d.r0)
    .outerRadius((d) => d.r1)
    .startAngle((d) => d.a0)
    .endAngle((d) => d.a1)

  // the shared wedges need one calendar; with several sites' calendars in force the clock is left unshaded
  const mixed = mixedCalendars(meta)
  const wedgeSeasons = mixed ? [] : meta.seasons
  const seasonOf = (m: number, site?: string): SeasonMeta | undefined =>
    (site ? seasonsForSite(meta, site) : wedgeSeasons).find((s) => s.months.includes(m))
  const emphasizeFebruary = meta.season_convention !== 'local'
  const unit = withUnit(field, meta.field_units)
  const stubCount = series.reduce((acc, s) => acc + s.cells.filter((c) => c.n > 0 && c.n < MIN_N).length, 0)

  /** The arc path a season label rides on. Top-half labels run clockwise and sit outside the path; bottom-half labels run the other way so they are not upside down, and hang inside it. */
  const labelPath = (start: number, length: number) => {
    const a0 = monthStart(start)
    const a1 = a0 + (length * TAU) / 12
    const midDeg = (((a0 + a1) / 2) * 180) / Math.PI % 360
    const bottom = midDeg > 90 && midDeg < 270
    const r = bottom ? seasonLabelR + 9 : seasonLabelR
    const [x0, y0] = polar(r, bottom ? a1 : a0)
    const [x1, y1] = polar(r, bottom ? a0 : a1)
    const large = a1 - a0 > Math.PI ? 1 : 0
    return `M${x0},${y0} A${r},${r} 0 ${large} ${bottom ? 0 : 1} ${x1},${y1}`
  }

  return (
    <ChartFrame
      id="seasonal-clock"
      title="Seasonal clock — the year as a circle"
      subtitle={`Monthly statistic of the measurement, laid around a circle so December and January sit next to each other. ${mixed ? 'Each site keeps its own calendar, so the sectors are left unshaded; hover a bar for its local season, or pick one site to see its seasons.' : `Shaded sectors use the ${calendarLabel(meta)}.`}`}
      provenance="react-graph-gallery.com/circular-barplot · not in the estate"
      controls={
        <>
          <Segmented label="bar" value={stat} options={STATS} onChange={setStat} title="Statistic of the measurement across the month's filters" />
          <Segmented label="layout" value={layout} options={LAYOUTS} onChange={setLayout} title="Grouped: sites side by side in each month. Rings: one ring per site, each with its own baseline, on a shared value scale" />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap centered">
        {allSeries.length === 0 ? (
          <Empty>No site has {field} in this subset.</Empty>
        ) : (
          <svg width={size} height={size} className="animated">
            <defs>
              {wedgeSeasons.flatMap((s) =>
                circularRuns(s.months).map((run, i) => (
                  <path key={`${s.name}-${i}`} id={`${uid}-season-${s.name.replace(/\W+/g, '')}-${i}`} d={labelPath(run.start, run.length)} fill="none" />
                ))
              )}
            </defs>
            <g transform={`translate(${cx},${cy})`} fontFamily={FONT.family}>
              {/* season sectors behind everything, from just inside the hole to just past the rim */}
              {d3.range(1, 13).map((m) => {
                const s = seasonOf(m)
                if (!s) return null
                return <path key={m} d={arc({ r0: Math.max(0, hole - 6), r1: outerR + 4, a0: monthStart(m), a1: monthStart(m + 1) }) ?? ''} fill={s.color} fillOpacity={seasonHover === null ? 0.12 : seasonHover === s.name ? 0.28 : 0.04} pointerEvents="none" />
              })}
              {/* month spokes */}
              {d3.range(1, 13).map((m) => {
                const [x0, y0] = polar(Math.max(0, hole - 6), monthStart(m))
                const [x1, y1] = polar(outerR + 4, monthStart(m))
                return <line key={m} x1={x0} y1={y0} x2={x1} y2={y1} stroke={INK.grid} strokeWidth={1} pointerEvents="none" />
              })}

              {/* radial axis: baseline per ring, faint tick rings, values printed once along the January spoke */}
              {(layout === 'Rings' ? series : series.slice(0, 1)).map((s, i) => (
                <g key={s.name} pointerEvents="none">
                  <circle r={ringInner(i)} fill="none" stroke={INK.axis} strokeOpacity={0.55} strokeWidth={1} />
                  {ticks.map((t) => (
                    <circle key={t} r={rOf(i, t)} fill="none" stroke={INK.axis} strokeOpacity={0.28} strokeWidth={0.8} strokeDasharray="2 3" />
                  ))}
                  {layout === 'Rings' && (
                    <text x={0} y={-(ringInner(i) + (ringOuter(i) - ringInner(i)) / 2)} dy="0.35em" textAnchor="middle" fontSize={10.5} fontWeight={600} fill={s.color} opacity={dim(s.name)} stroke="#fff" strokeWidth={3} paintOrder="stroke">
                      {s.name}
                    </text>
                  )}
                </g>
              ))}
              {series.length > 0 &&
                ticks.map((t) => {
                  // labels go on the outermost ring only; every ring shares the scale
                  const i = layout === 'Rings' ? series.length - 1 : 0
                  return (
                    <text key={t} x={4} y={-rOf(i, t)} dy="0.35em" textAnchor="start" fontSize={9.5} fontFamily={FONT.mono} fill={INK.muted} stroke="#fff" strokeWidth={3} paintOrder="stroke" pointerEvents="none">
                      {fmt(t, Math.abs(t) >= 10 ? 0 : 1)}
                    </text>
                  )
                })}

              {/* the bars */}
              {series.map((s, i) => (
                <g key={s.name} opacity={dim(s.name)}>
                  {s.cells.map((c) => {
                    const a0 = layout === 'Grouped' ? monthStart(c.month) + (subBand(s.name) ?? 0) : monthStart(c.month) + sectorPad
                    const a1 = layout === 'Grouped' ? a0 + subBand.bandwidth() : monthStart(c.month + 1) - sectorPad
                    const season = seasonOf(c.month, s.name)
                    const seasonLine = `season: ${season?.name ?? '—'} (${calendarLabel(meta)})`
                    if (c.n === 0) return null
                    const v = valueOf(c)
                    if (v === null) {
                      // fewer than MIN_N filters: a hollow stub says "there is something here" without asserting a value
                      const stub = Math.min(10, band * 0.12)
                      return (
                        <path
                          key={c.month}
                          d={arc({ r0: ringInner(i), r1: ringInner(i) + stub, a0, a1 }) ?? ''}
                          fill="none" stroke={s.color} strokeWidth={1.2} strokeDasharray="2 2"
                          onMouseEnter={(e) => tip.show(e, [`${s.name} · ${MONTHS[c.month - 1]}`, `only ${c.n} filter${c.n === 1 ? '' : 's'} — no ${stat.toLowerCase()} drawn (needs ${MIN_N})`, seasonLine])}
                          onMouseLeave={tip.hide}
                        />
                      )
                    }
                    // bars grow from the ring's zero; a negative statistic grows inward from it
                    const base = rOf(i, 0)
                    const tipR = rOf(i, v)
                    return (
                      <path
                        key={c.month}
                        d={arc({ r0: Math.min(base, tipR), r1: Math.max(base, tipR), a0, a1 }) ?? ''}
                        fill={s.color} fillOpacity={0.88} stroke="#fff" strokeWidth={0.6}
                        style={{ cursor: 'default' }}
                        onMouseEnter={(e) =>
                          tip.show(e, [
                            `${s.name} · ${MONTHS[c.month - 1]}`,
                            `${stat.toLowerCase()} ${unit} = ${fmt(v, 2)}`,
                            `n = ${c.n} filters`,
                            c.q1 !== null ? `IQR ${fmt(c.q1, 2)} – ${fmt(c.q3, 2)}` : 'IQR n/a',
                            seasonLine,
                          ])
                        }
                        onMouseLeave={tip.hide}
                      />
                    )
                  })}
                </g>
              ))}

              {/* month labels just outside the rim, January at the top */}
              {MONTHS.map((name, idx) => {
                const [x, y] = polar(monthLabelR, monthStart(idx + 1) + TAU / 24)
                return (
                  <text key={name} x={x} y={y} dy="0.35em" textAnchor="middle" fontSize={11} fill={emphasizeFebruary && idx === 1 ? INK.text : INK.muted} fontWeight={emphasizeFebruary && idx === 1 ? 600 : 400} pointerEvents="none">
                    {name}
                  </text>
                )
              })}

              {/* season names curved along the outside of their wedge */}
              {wedgeSeasons.flatMap((s) =>
                circularRuns(s.months).map((_, i) => (
                  <text key={`${s.name}-${i}`} fontSize={10.5} fontWeight={600} fill={s.color} opacity={seasonHover === null || seasonHover === s.name ? 1 : 0.35} pointerEvents="none">
                    <textPath href={`#${uid}-season-${s.name.replace(/\W+/g, '')}-${i}`} startOffset="50%" textAnchor="middle">
                      {s.name}
                    </textPath>
                  </text>
                ))
              )}
            </g>
          </svg>
        )}
        {stubCount > 0 && (
          <Note>
            {stubCount} site-month{stubCount === 1 ? '' : 's'} with fewer than {MIN_N} filters {stubCount === 1 ? 'is' : 'are'} drawn as a hollow stub rather than a {stat.toLowerCase()}.
          </Note>
        )}
        <Legend
          items={[
            ...allSeries.map((s) => ({ label: s.name, color: s.color })),
            ...wedgeSeasons.map((s) => ({ label: s.name, color: s.color, shape: 'band' as const })),
          ]}
          hidden={hidden}
          onToggle={(l) => { if (allSeries.some((s) => s.name === l)) setHidden((h) => toggleIn(h, l)) }}
          onHover={setHover}
          highlighted={siteHover ?? seasonHover}
          note={emphasizeFebruary ? `bars share one value scale · February changes bins across Ethiopian conventions` : `bars share one value scale · ${calendarLabel(meta)}`}
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
