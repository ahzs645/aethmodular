import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { calendarLabel, seasonsForSite } from '@/siteSeasons'

// Stacked bars are the recommended view (PI decision, 23 Sep 2026); the
// hierarchy encodings stay selectable for the within-site reading.
const MODES = ['Stacked bars', 'Treemap', 'Circle pack', 'Donut'] as const
const STATS = ['Mean', 'Median'] as const
const GROUPINGS = ['per site', 'by season'] as const
const UNIT_MODES = ['concentration (µg/m³)', 'composition (%)'] as const

interface Kid { name: string; family: string; value: number; median: number; n: number }
/** one stacked bar: a site, or a site × season */
interface Bar { key: string; label: string; site: string; siteColor: string; barColor: string; pm25_mean: number | null; nRows: number; children: Kid[] }
interface Node { name: string; family?: string; children?: Node[]; value?: number; median?: number; n?: number }

/**
 * Treemap / circle pack / donut / stacked bars — PM2.5 composition per site.
 *
 * The estate has almost nothing in this category (1 stackplot, 2 pie calls),
 * because matplotlib makes it awkward. It is the natural reading of the
 * ChemSpec columns, so it goes in, with react-graph-gallery's two-level
 * hierarchy: measurement family (carbon / ions / metals & crustal) → species,
 * so "how much of Addis is crustal" is a border, not a sum in your head.
 * Stacked bars are the one encoding that compares *across* sites at a
 * glance; the others compare within one.
 */
export function CompositionTreemap({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [mode, setMode] = useState<(typeof MODES)[number]>('Stacked bars')
  const [stat, setStat] = useState<(typeof STATS)[number]>('Mean')
  const [grouping, setGrouping] = useState<(typeof GROUPINGS)[number]>('per site')
  const [unitMode, setUnitMode] = useState<(typeof UNIT_MODES)[number]>('concentration (µg/m³)')
  const [twoLevel, setTwoLevel] = useState(true)
  const lg = useLegend()
  const fam = useLegend()

  const panelW = Math.max(240, Math.floor((width - 26) / 2))
  const panelH = 260
  const UNITS = 'µg/m³'

  // Species that make up mass, in the order the exporter groups them. PM2.5
  // mass is the denominator, not a slice, so it is excluded from the parts.
  const speciesFields = useMemo(() => {
    const wanted = new Set(['Ions', 'Metals & crustal'])
    const fromGroups = meta.field_groups.filter((g) => wanted.has(g.label)).flatMap((g) => g.fields)
    return ['OC (ChemSpec FTIR)', 'EC (ChemSpec FTIR)', ...fromGroups].filter((f) => meta.fields.includes(f))
  }, [meta])
  const familyOf = (f: string) => (f.startsWith('OC') || f.startsWith('EC') ? 'Carbon' : meta.field_groups.find((g) => g.fields.includes(f))?.label ?? 'Other')
  const families = useMemo(() => [...new Set(speciesFields.map(familyOf))], [speciesFields]) // eslint-disable-line react-hooks/exhaustive-deps

  // Built from the live subset so the season/site filters above apply here too.
  const composition = useMemo(() => {
    const kidsOf = (sub: FilterRow[]) =>
      speciesFields
        .map((f): Kid | null => {
          const vals = sub.map((r) => r[f]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)).sort(d3.ascending)
          if (!vals.length) return null
          return { name: f, family: familyOf(f), value: d3.mean(vals) ?? 0, median: d3.quantile(vals, 0.5) ?? 0, n: vals.length }
        })
        .filter((c): c is Kid => !!c)
    const pmOf = (sub: FilterRow[]) => {
      const pm = sub.map((r) => r['PM2.5 mass']).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      return pm.length ? (d3.mean(pm) ?? null) : null
    }
    const sites = meta.sites
      .filter((s) => rows.some((r) => r.site === s.name))
      .map((s) => {
        const sub = rows.filter((r) => r.site === s.name)
        return { code: s.code, name: s.name, color: s.color, pm25_mean: pmOf(sub), children: kidsOf(sub) }
      })
    // One bar per season per site, from the season each row already carries
    // (App relabels it under the calendar picked in the subset bar), ordered as
    // that site's calendar lists them; under the per-site calendar the names are
    // site-qualified ("Addis Ababa · Dry …"), so the prefix is dropped for the label.
    const seasonBars: Bar[] = meta.sites
      .filter((s) => rows.some((r) => r.site === s.name))
      .flatMap((s) => {
        const sub = rows.filter((r) => r.site === s.name)
        const cal = seasonsForSite(meta, s.name)
        const order = [...cal.map((c) => c.name), ...new Set(sub.map((r) => r.season).filter((n) => !cal.some((c) => c.name === n)))]
        return order
          .map((name): Bar | null => {
            const ss = sub.filter((r) => r.season === name)
            const kids = kidsOf(ss)
            // a season with filters but no speciated species (e.g. Delhi winter) has no bar to stack
            if (!ss.length || !kids.length) return null
            const label = name.startsWith(`${s.name} · `) ? name.slice(s.name.length + 3) : name
            return { key: `${s.code}|${name}`, label, site: s.name, siteColor: s.color, barColor: cal.find((c) => c.name === name)?.color ?? INK.muted, pm25_mean: pmOf(ss), nRows: ss.length, children: kids }
          })
          .filter((b): b is Bar => !!b)
      })
    return { units: UNITS, sites, seasonBars }
  }, [rows, meta, speciesFields]) // eslint-disable-line react-hooks/exhaustive-deps

  const color = useMemo(() => d3.scaleOrdinal<string>().domain(speciesFields).range(d3.schemeTableau10.concat(d3.schemeSet3 as any)), [speciesFields])
  const familyColor = useMemo(() => d3.scaleOrdinal<string>().domain(families).range(['#5b6470', '#2171b5', '#b2182b', '#0f766e']), [families])
  const val = (c: { value: number; median: number }) => (stat === 'Mean' ? c.value : c.median)
  // the family legend is live wherever families are drawn; hiding a family or a species drops it from every total
  const famOn = twoLevel || mode === 'Stacked bars'
  const vis = (k: Kid) => lg.show(k.name) && (!famOn || fam.show(k.family))
  const dimK = (k: { name: string; family?: string }) => lg.dim(k.name) * (famOn ? fam.dim(k.family ?? '') : 1)
  const famDimOf = (f: string) => (famOn ? fam.dim(f, 0.3) : 1)
  const shownSites = composition.sites.map((s) => ({ ...s, children: s.children.filter(vis) }))
  const bySeason = grouping === 'by season'
  const pct = unitMode === 'composition (%)'
  const bars: Bar[] = bySeason
    ? composition.seasonBars.map((b) => ({ ...b, children: b.children.filter(vis) }))
    : shownSites.map((s) => ({ key: s.code, label: s.name, site: s.name, siteColor: s.color, barColor: s.color, pm25_mean: s.pm25_mean, nRows: rows.filter((r) => r.site === s.name).length, children: s.children }))

  /** family → species hierarchy (or flat), summed on the chosen statistic */
  const rootFor = (siteName: string, kids: Kid[]) => {
    const data: Node = twoLevel
      ? { name: siteName, children: families.map((fam) => ({ name: fam, children: kids.filter((k) => k.family === fam) })).filter((f) => f.children!.length) }
      : { name: siteName, children: kids }
    return d3.hierarchy<Node>(data).sum((d) => (d.children ? 0 : val(d as Kid))).sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  }
  const leafTip = (leaf: d3.HierarchyNode<Node>, total: number) => [
    leaf.data.name,
    `${leaf.data.family ?? ''}`,
    `${fmt(leaf.value, 3)} ${UNITS}`,
    `${(((leaf.value ?? 0) / total) * 100).toFixed(1)} % of speciated mass`,
    `n = ${leaf.data.n}`,
  ]

  return (
    <ChartFrame
      id="composition"
      title="Stacked bars — what the PM2.5 is made of, per site and season"
      subtitle="Speciated mass from the ChemSpec columns, grouped by measurement family (carbon, ions, metals & crustal). Concentration stacks the mean of each species in µg/m³, so absolute mass differences between sites or seasons show; composition (%) stacks each species' share of the speciated total, so a shift such as less carbon in one season shows. PM2.5 mass is not a slice; the gap between speciated and measured PM2.5 is unmeasured organic matter and water. Treemap, circle pack and donut remain for the within-site reading."
      provenance="1 stackplot + 2 pie calls in the estate · react-graph-gallery.com/barplot (stacked) · /treemap · /circular-packing · /donut"
      controls={
        <>
          <Segmented label="encoding" value={mode} options={MODES} onChange={setMode} />
          {mode === 'Stacked bars' && <Segmented label="bars" value={grouping} options={GROUPINGS} onChange={setGrouping} title="One bar per site, or one per season within each site" />}
          {mode === 'Stacked bars' && <Segmented label="units" value={unitMode} options={UNIT_MODES} onChange={setUnitMode} />}
          <Segmented label="statistic" value={stat} options={STATS} onChange={setStat} />
          {mode !== 'Stacked bars' && <Toggle label="group by family" checked={twoLevel} onChange={setTwoLevel} title="Two levels of hierarchy: family → species" />}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        {composition.sites.length === 0 && <Empty>No sites in this subset.</Empty>}
        {mode === 'Stacked bars' ? (
          <div style={{ width: '100%' }}>
            {bySeason && bars.length > 0 && (
              <p style={{ margin: '0 0 6px', fontSize: 11.5, color: INK.muted }}>
                One bar per season within each site, seasons from {calendarLabel(meta)} (pick the calendar in the subset bar). Each bar is the {stat.toLowerCase()} over that season's filters, so n differs by bar; hover a bar for it.
              </p>
            )}
            <StackedBars bars={bars} grouped={bySeason} pct={pct} families={families} val={val} dim={dimK} famDim={fam.dim} color={color} familyColor={familyColor} width={width} tip={tip} units={UNITS} />
          </div>
        ) : (
          shownSites.map((site) => {
            const kids = site.children.filter((c) => val(c) > 0)
            const total = d3.sum(kids, val)
            const root = rootFor(site.name, kids)
            return (
              <div key={site.code} style={{ width: panelW }}>
                <p className="facet-title">
                  <span className="swatch" style={{ background: site.color }} />
                  {site.name}
                  <span style={{ fontWeight: 400, color: 'var(--ink-muted)', fontSize: 11.5 }}>
                    PM2.5 {fmt(site.pm25_mean, 1)} µg/m³ · speciated {fmt(total, 1)}
                  </span>
                </p>
                <svg width={panelW} height={panelH} className="animated">
                  {kids.length > 0 && mode === 'Treemap' && (() => {
                    d3.treemap<Node>().size([panelW - 1, panelH - 1]).paddingInner(2).paddingOuter(twoLevel ? 3 : 0).paddingTop(twoLevel ? 16 : 0).round(true)(root)
                    return (
                      <>
                        {twoLevel && root.children?.map((fam: any) => (
                          <g key={fam.data.name} pointerEvents="none">
                            <rect x={fam.x0} y={fam.y0} width={Math.max(0, fam.x1 - fam.x0)} height={Math.max(0, fam.y1 - fam.y0)} fill={familyColor(fam.data.name)} fillOpacity={0.1 * famDimOf(fam.data.name)} stroke={familyColor(fam.data.name)} strokeOpacity={0.5 * famDimOf(fam.data.name)} rx={3} />
                            {fam.x1 - fam.x0 > 60 && <text x={fam.x0 + 5} y={fam.y0 + 12} fontSize={10} fontWeight={600} fill={familyColor(fam.data.name)} fontFamily={FONT.family}>{fam.data.name} · {(((fam.value ?? 0) / total) * 100).toFixed(0)}%</text>}
                          </g>
                        ))}
                        {root.leaves().map((leaf: any, i) => {
                          const w = leaf.x1 - leaf.x0
                          const h = leaf.y1 - leaf.y0
                          if (w <= 0 || h <= 0) return null
                          const pct = ((leaf.value ?? 0) / total) * 100
                          return (
                            <g key={i} opacity={dimK(leaf.data)} onMouseEnter={(e) => tip.show(e, leafTip(leaf, total))} onMouseLeave={tip.hide}>
                              <rect x={leaf.x0} y={leaf.y0} width={w} height={h} fill={color(leaf.data.name)} fillOpacity={0.85} stroke="#fff" rx={2} />
                              {w > 54 && h > 20 && (
                                <text x={leaf.x0 + 5} y={leaf.y0 + 14} fontSize={10.5} fill="#fff" fontFamily={FONT.family} pointerEvents="none">{leaf.data.name}</text>
                              )}
                              {w > 54 && h > 33 && (
                                <text x={leaf.x0 + 5} y={leaf.y0 + 27} fontSize={9.5} fill="#fff" fillOpacity={0.85} fontFamily={FONT.mono} pointerEvents="none">{pct.toFixed(1)}%</text>
                              )}
                            </g>
                          )
                        })}
                      </>
                    )
                  })()}
                  {kids.length > 0 && mode === 'Circle pack' && (() => {
                    d3.pack<Node>().size([panelW - 2, panelH - 2]).padding(twoLevel ? 4 : 2)(root)
                    return (
                      <g transform="translate(1,1)">
                        {twoLevel && root.children?.map((fam: any) => (
                          <g key={fam.data.name} pointerEvents="none">
                            <circle cx={fam.x} cy={fam.y} r={fam.r} fill={familyColor(fam.data.name)} fillOpacity={0.08 * famDimOf(fam.data.name)} stroke={familyColor(fam.data.name)} strokeOpacity={0.5 * famDimOf(fam.data.name)} />
                            {fam.r > 28 && <text x={fam.x} y={fam.y - fam.r + 11} textAnchor="middle" fontSize={9.5} fontWeight={600} fill={familyColor(fam.data.name)} fontFamily={FONT.family}>{fam.data.name}</text>}
                          </g>
                        ))}
                        {root.leaves().map((leaf: any, i) => (
                          <g key={i} opacity={dimK(leaf.data)} onMouseEnter={(e) => tip.show(e, leafTip(leaf, total))} onMouseLeave={tip.hide}>
                            <circle cx={leaf.x} cy={leaf.y} r={leaf.r} fill={color(leaf.data.name)} fillOpacity={0.85} stroke="#fff" />
                            {leaf.r > 18 && <text x={leaf.x} y={leaf.y} dy="0.35em" textAnchor="middle" fontSize={Math.min(10.5, leaf.r / 2.4)} fill="#fff" fontFamily={FONT.family} pointerEvents="none">{leaf.data.name}</text>}
                          </g>
                        ))}
                      </g>
                    )
                  })()}
                  {mode === 'Donut' && (() => {
                    const r = Math.min(panelW, panelH) / 2 - 8
                    const arcs = d3.pie<Kid>().value(val).sort(null)(kids)
                    const arc = d3.arc<d3.PieArcDatum<Kid>>().innerRadius(r * 0.55).outerRadius(r)
                    // family ring: a thin outer band per family, the two-level reading of a donut
                    const famArcs = d3.pie<{ name: string; v: number }>().value((d) => d.v).sort(null)(families.map((f) => ({ name: f, v: d3.sum(kids.filter((k) => k.family === f), val) })))
                    const famArc = d3.arc<d3.PieArcDatum<{ name: string; v: number }>>().innerRadius(r + 2).outerRadius(r + 7)
                    return (
                      <g transform={`translate(${panelW / 2},${panelH / 2})`}>
                        {arcs.map((a, i) => (
                          <path
                            key={i} d={arc(a) ?? ''} fill={color(a.data.name)} fillOpacity={0.88 * dimK(a.data)} stroke="#fff" strokeWidth={1}
                            onMouseEnter={(e) => tip.show(e, [a.data.name, a.data.family, `${fmt(val(a.data), 3)} ${UNITS}`, `${((val(a.data) / total) * 100).toFixed(1)} %`, `n = ${a.data.n}`])}
                            onMouseLeave={tip.hide}
                          />
                        ))}
                        {twoLevel && famArcs.map((a, i) => (
                          <path key={`f${i}`} d={famArc(a) ?? ''} fill={familyColor(a.data.name)} fillOpacity={0.7 * famDimOf(a.data.name)} onMouseEnter={(e) => tip.show(e, [a.data.name, `${fmt(a.data.v, 2)} ${UNITS}`, `${((a.data.v / total) * 100).toFixed(1)} % of speciated mass`])} onMouseLeave={tip.hide} />
                        ))}
                        <text textAnchor="middle" dy="0.35em" fontSize={12} fill={INK.muted} fontFamily={FONT.mono}>{fmt(total, 1)}</text>
                      </g>
                    )
                  })()}
                </svg>
              </div>
            )
          })
        )}
        {tip.node}
      </div>
      {famOn && <Legend items={families.map((f) => ({ label: f, color: familyColor(f), shape: 'band' as const }))} {...fam.props} />}
      <Legend items={speciesFields.map((f) => ({ label: f, color: color(f), shape: 'square' as const }))} {...lg.props} />
    </ChartFrame>
  )
}

/**
 * Stacked bars, species in family order with family boundaries marked. `pct`
 * normalises each bar to its speciated total (composition %); otherwise the
 * stack is the mean species concentration in µg/m³ on one shared axis.
 * `grouped` lays season bars out in per-site clusters with the site underneath.
 */
function StackedBars({ bars, grouped, pct, families, val, dim, famDim, color, familyColor, width, tip, units }: {
  bars: Bar[]
  grouped: boolean
  pct: boolean
  families: string[]
  val: (c: Kid) => number
  dim: (k: Kid) => number
  famDim: (family: string) => number
  color: d3.ScaleOrdinal<string, string>
  familyColor: d3.ScaleOrdinal<string, string>
  width: number
  tip: ReturnType<typeof useTooltip>
  units: string
}) {
  const margin = { top: 16, right: 20, bottom: grouped ? 72 : 40, left: 60 }
  const innerW = Math.max(240, width - margin.left - margin.right)
  const innerH = 320
  // a spacer slot between site clusters so the seasons of one site read as a group
  const slots: string[] = []
  const siteOrder = [...new Set(bars.map((b) => b.site))]
  siteOrder.forEach((site, i) => {
    if (grouped && i > 0) slots.push(`__gap${i}`)
    for (const b of bars) if (b.site === site) slots.push(b.key)
  })
  const x = d3.scaleBand<string>().domain(slots).range([0, innerW]).padding(grouped ? 0.22 : 0.3)
  const bw = x.bandwidth()
  const gapW = x.step() - bw
  const kidsOf = (b: Bar) => families.flatMap((f) => b.children.filter((k) => k.family === f && val(k) > 0))
  const maxTotal = d3.max(bars, (b) => d3.sum(kidsOf(b), val)) ?? 1
  const y = d3.scaleLinear().domain([0, pct ? 1 : maxTotal || 1]).range([innerH, 0]).nice()
  const yLabel = pct ? 'share of speciated mass (%)' : `mean concentration (${units})`
  const famText = (v: number, total: number) => (pct ? `${Math.round((v / total) * 100)}%` : fmt(v, 1))
  return (
    <svg width={width} height={innerH + margin.top + margin.bottom} className="animated">
      <g transform={`translate(${margin.left},${margin.top})`}>
        {y.ticks(5).map((t) => (
          <g key={t} transform={`translate(0,${y(t)})`}>
            <line x2={innerW} stroke={INK.grid} />
            <text x={-8} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.muted} fontFamily={FONT.family}>{pct ? `${Math.round(t * 100)}%` : fmt(t, 1)}</text>
          </g>
        ))}
        <text transform={`translate(${-margin.left + 13},${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill={INK.axis} fontFamily={FONT.family}>{yLabel}</text>
        {bars.map((bar) => {
          const kids = kidsOf(bar)
          const total = d3.sum(kids, val)
          const norm = pct ? total || 1 : 1
          const where = grouped ? `${bar.site} · ${bar.label}` : bar.site
          let acc = 0
          const bx = x(bar.key) ?? 0
          const famBounds: { name: string; v: number; y0: number; y1: number }[] = []
          let famAcc = 0
          for (const f of families) {
            const v = d3.sum(kids.filter((k) => k.family === f), val)
            if (v > 0) famBounds.push({ name: f, v, y0: famAcc / norm, y1: (famAcc + v) / norm })
            famAcc += v
          }
          // "Belg (Feb-May, short rains)" -> "Belg" over "Feb-May"; the full name is in the <title>
          const paren = bar.label.indexOf(' (')
          const seasonName = paren > 0 ? bar.label.slice(0, paren) : bar.label
          const seasonMonths = paren > 0 ? bar.label.slice(paren + 2).replace(/\)$/, '').split(',')[0] : ''
          return (
            <g key={bar.key}>
              {kids.map((k) => {
                const y0 = acc / norm
                acc += val(k)
                const y1 = acc / norm
                const h = y(y0) - y(y1)
                const share = total ? (val(k) / total) * 100 : 0
                return (
                  <g key={k.name} opacity={dim(k)} onMouseEnter={(e) => tip.show(e, [k.name, `${where} · ${k.family}`, `${fmt(val(k), 3)} ${units}`, `${share.toFixed(1)} % of the speciated ${fmt(total, 1)} ${units}`, `PM2.5 ${fmt(bar.pm25_mean, 1)} ${units} · n = ${k.n} of ${bar.nRows} filters`])} onMouseLeave={tip.hide}>
                    <rect x={bx} y={y(y1)} width={bw} height={Math.max(0, h)} fill={color(k.name)} fillOpacity={0.88} stroke="#fff" strokeWidth={0.6} />
                    {h > 13 && bw > 70 && <text x={bx + bw / 2} y={y(y1) + h / 2} dy="0.35em" textAnchor="middle" fontSize={9.5} fill="#fff" fontFamily={FONT.family} pointerEvents="none">{k.name} {pct ? `${share.toFixed(0)}%` : fmt(val(k), 1)}</text>}
                  </g>
                )
              })}
              {famBounds.map((f) => (
                <g key={f.name} pointerEvents="none">
                  <rect x={bx + bw + 3} y={y(f.y1)} width={Math.min(5, gapW / 3)} height={Math.max(0, y(f.y0) - y(f.y1))} fill={familyColor(f.name)} fillOpacity={0.8 * famDim(f.name)} rx={1} />
                  {y(f.y0) - y(f.y1) > 12 && gapW > 26 && <text x={bx + bw + 10} y={(y(f.y0) + y(f.y1)) / 2} dy="0.35em" fontSize={9} fill={familyColor(f.name)} fontFamily={FONT.family}>{famText(f.v, total)}</text>}
                </g>
              ))}
              {grouped ? (
                <>
                  <rect x={bx} y={innerH + 3} width={bw} height={3} fill={bar.barColor} rx={1}><title>{bar.label}</title></rect>
                  <text x={bx + bw / 2} y={innerH + 17} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>{seasonName}</text>
                  {seasonMonths && <text x={bx + bw / 2} y={innerH + 29} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.family}>{seasonMonths}</text>}
                  <text x={bx + bw / 2} y={innerH + 41} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>{pct ? `n=${bar.nRows}` : fmt(total, 1)}</text>
                </>
              ) : (
                <>
                  <text x={bx + bw / 2} y={innerH + 16} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={bar.siteColor} fontFamily={FONT.family}>{bar.label}</text>
                  <text x={bx + bw / 2} y={innerH + 30} textAnchor="middle" fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>{fmt(total, 1)} of PM2.5 {fmt(bar.pm25_mean, 1)} {units}</text>
                </>
              )}
            </g>
          )
        })}
        {grouped && siteOrder.map((site) => {
          const mine = bars.filter((b) => b.site === site)
          const x0 = x(mine[0].key) ?? 0
          const x1 = (x(mine[mine.length - 1].key) ?? 0) + bw
          return (
            <g key={site} pointerEvents="none">
              <line x1={x0} x2={x1} y1={innerH + 50} y2={innerH + 50} stroke={mine[0].siteColor} strokeWidth={1.5} />
              <text x={(x0 + x1) / 2} y={innerH + 64} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={mine[0].siteColor} fontFamily={FONT.family}>{site}</text>
            </g>
          )
        })}
      </g>
    </svg>
  )
}
