import { memo, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useChartOptions } from '@/lib/chartOptions'
import { averageLinkage, corrDistance, leafOrder, type Clade } from '@/lib/cluster'
import { useColourScheme } from '@/lib/colourSchemes'
import { f2, fmt } from '@/lib/format'
import { num, type Table } from '@/lib/table'
import { FONT, INK } from '@/lib/theme'

/** the five SPARTAN targets carry a leading '*' in ftir_52's matrix */
const isTarget = (s: string) => s.startsWith('*')
const label = (s: string) => (isTarget(s) ? s.slice(1) : s)
const TREE_H = 72
const NEAREST = ['20', '40', '80'] as const

/**
 * ftir_52's site × site correlation of median spectra (spline baseline), reordered by average-linkage
 * clustering on 1 − r with the dendrogram above: the tree the page used to say it lacked. All sites,
 * or one SPARTAN target and its nearest IMPROVE sites, reclustered on their own.
 */
export function SiteSimilarityMatrix({ table, provenance }: { table: Table | undefined; provenance: React.ReactNode }) {
  const full = useMemo(() => {
    const cols = table && !Array.isArray(table) ? table as Record<string, unknown[]> : null
    const names = (cols?.['Unnamed: 0'] ?? []).map(String)
    if (!cols || names.length < 3) return null
    // columns by position: the matrix is square in one order, and its rows already say JPL where a column may still say Pasadena
    const keys = Object.keys(cols).filter((k) => k !== 'Unnamed: 0')
    if (keys.length !== names.length) return null
    return { names, r: names.map((_, i) => keys.map((k) => num((cols[k] as unknown[])[i] as number))) }
  }, [table])
  const targets = useMemo(() => (full?.names ?? []).filter(isTarget), [full])
  const o = useChartOptions('network-site-corr', [
    { key: 'sites', label: 'Sites', default: 'all', choices: [{ value: 'all', label: 'All' }, { value: 'near', label: 'Nearest to a target' }] },
    { key: 'target', label: 'Target', style: 'select', default: targets[0] ?? '*Addis', when: (c) => c.sites === 'near', choices: targets.map((t) => ({ value: t, label: label(t) })) },
    { key: 'n', label: 'Nearest', default: '40', when: (c) => c.sites === 'near', choices: [...NEAREST] },
  ])

  // the sites in view, clustered and put in leaf order
  const view = useMemo(() => {
    if (!full) return null
    let keep = full.names.map((_, i) => i)
    if (o.values.sites === 'near') {
      const t = full.names.indexOf(String(o.values.target))
      if (t >= 0) keep = [...keep].sort((a, b) => (full.r[t][b] ?? -1) - (full.r[t][a] ?? -1)).slice(0, Number(o.values.n) + 1)
    }
    const r = keep.map((i) => keep.map((j) => full.r[i][j]))
    const tree = averageLinkage(corrDistance(r))
    const order = leafOrder(tree)
    const off = r.flatMap((row, i) => row.filter((v, j): v is number => i !== j && v != null))
    // the low end at the 2nd percentile, so one Arctic outlier does not wash out the rest
    const lo = off.length ? d3.quantile(off.sort(d3.ascending), 0.02)! : 0
    return { names: keep.map((i) => full.names[i]), r, tree, order, domain: [Math.min(lo, 0.99), 1] as [number, number] }
  }, [full, o.values.sites, o.values.target, o.values.n])

  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const colours = useColourScheme('network-site-corr', 'sequential')
  const [hover, setHover] = useState<[number, number] | null>(null)

  const layout = useMemo(() => {
    if (!view) return null
    const n = view.order.length
    const left0 = 64
    const cell = Math.max(2.5, Math.min(18, (width - left0 - 12) / n))
    const font = Math.min(11, cell * 0.85)
    // every name once the rows are tall enough to read; otherwise the SPARTAN targets only
    const allLabels = font >= 6.5
    const left = allLabels ? Math.max(left0, 8 + font * 0.62 * Math.max(...view.names.map((s) => label(s).length))) : left0
    const side = cell * n
    const pos = new Map(view.order.map((idx, k) => [idx, k]))
    // dendrogram: leaves at their cell centre, internal nodes at their children's mid and merge height
    const maxH = view.tree.height || 1
    // square-root heights: the Arctic outliers join last, far above the rest, and would flatten the tree
    const y = d3.scaleSqrt().domain([0, maxH]).range([TREE_H, 4])
    const links: string[] = []
    const place = (c: Clade): number => {
      if (!c.children) return left + (pos.get(c.idx!)! + 0.5) * cell
      const xs = c.children.map(place)
      const cy = y(c.height)
      c.children.forEach((ch, i) => links.push(`M${xs[i]},${ch.children ? y(ch.height) : TREE_H}V${cy}`))
      links.push(`M${Math.min(...xs)},${cy}H${Math.max(...xs)}`)
      return (Math.min(...xs) + Math.max(...xs)) / 2
    }
    place(view.tree)
    return { n, cell, font, allLabels, left, top: TREE_H + 6, side, links, y, maxH }
  }, [view, width])

  const scale = useMemo(() => {
    const ramp = colours.stops, dom = view?.domain ?? [0.9, 1]
    return d3.scaleLinear<string>().domain(ramp.map((_, i) => dom[0] + (i / (ramp.length - 1)) * (dom[1] - dom[0]))).range(ramp).clamp(true)
  }, [colours.stops, view])

  const title = 'Site × site similarity, clustered — which IMPROVE sites share a spectrum, and where the SPARTAN sites join the tree'
  if (!view || !layout) return <ChartFrame id="network-site-corr" title={title} provenance={provenance}><Empty>The export has no site × site matrix; re-run ftir_52 and the Research spectral export.</Empty></ChartFrame>
  const { n, cell, font, allLabels, left, top, side } = layout
  const nm = (k: number) => label(view.names[view.order[k]])
  const rAt = (a: number, b: number) => view.r[view.order[a]][view.order[b]]
  const move = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const j = Math.floor((e.clientX - box.left) / cell), i = Math.floor((e.clientY - box.top) / cell)
    if (i < 0 || j < 0 || i >= n || j >= n) return
    if (!hover || hover[0] !== i || hover[1] !== j) setHover([i, j])
    const v = rAt(i, j)
    tip.show(e, [`${nm(i)} × ${nm(j)}`, `r = ${v == null ? '—' : fmt(v, 4)}`, `1 − r = ${v == null ? '—' : fmt(1 - v, 4)}`])
  }
  return (
    <ChartFrame id="network-site-corr" title={title}
      subtitle={`Pearson r between every pair of sites' median spline-baseline spectra (IMPROVE sites with ≥ 20 spectra, and the five SPARTAN targets in bold), ordered by average-linkage clustering on 1 − r; the tree above joins sites at the mean distance between their groups (square-root scale). Most pairs sit above 0.9, so the colour runs from the 2nd percentile of the off-diagonal r (${f2(view.domain[0])}) to 1. Sites: Nearest to a target keeps that target and its closest IMPROVE sites and clusters them again on their own. Point at a cell for the pair.`}
      provenance={provenance} controls={o.controls()}>
      <div ref={wrapRef} className="chart-wrap" onMouseLeave={() => { setHover(null); tip.hide() }}>
        <svg width={left + side + 12} height={top + side + 8} fontFamily={FONT.family} style={{ display: 'block' }}>
          <g fill="none" stroke={INK.axis} strokeWidth={1}>{layout.links.map((d, i) => <path key={i} d={d} />)}</g>
          <text x={left - 6} y={layout.y(layout.maxH)} dy="0.35em" textAnchor="end" fontSize={10} fill={INK.muted}>1 − r {fmt(layout.maxH, 3)}</text>
          <text x={left - 6} y={TREE_H} dy="0.35em" textAnchor="end" fontSize={10} fill={INK.muted}>0</text>
          <Cells view={view} left={left} top={top} cell={cell} scale={scale} />
          {view.order.map((idx, k) => {
            const s = view.names[idx], t = isTarget(s)
            if (!allLabels && !t) return null
            return <text key={s} x={left - 4} y={top + (k + 0.5) * cell} dy="0.35em" textAnchor="end" fontSize={t ? Math.max(font, 10) : font}
              fontWeight={t ? 700 : 400} fill={INK.text}>{label(s)}{!allLabels && ' ▸'}</text>
          })}
          {hover && <g fill="none" stroke={INK.text} strokeWidth={1} pointerEvents="none">
            <rect x={left} y={top + hover[0] * cell} width={side} height={cell} />
            <rect x={left + hover[1] * cell} y={top} width={cell} height={side} />
          </g>}
          <rect x={left} y={top} width={side} height={side} fill="transparent" onMouseMove={move} />
        </svg>
        {tip.node}
      </div>
      <div className="legend" style={{ justifyContent: 'center' }}>
        <ColorLegend scale={scale as unknown as { domain(): number[]; (v: number): string }} scheme={colours.scheme} label="r of median spectra" format={(v) => fmt(v, 3)} />
        <span className="legend-note">{n} sites</span>
      </div>
    </ChartFrame>
  )
}

/** the cells, drawn once per view and colour ramp (hover only moves the outline) */
const Cells = memo(function Cells({ view, left, top, cell, scale }: {
  view: { r: (number | null)[][]; order: number[] }; left: number; top: number; cell: number; scale: (v: number) => string
}) {
  return <g shapeRendering="crispEdges">
    {view.order.flatMap((a, i) => view.order.map((b, j) => {
      const v = view.r[a][b]
      return v == null ? null : <rect key={`${i}-${j}`} x={left + j * cell} y={top + i * cell} width={cell + 0.3} height={cell + 0.3} fill={scale(v)} />
    }))}
  </g>
})
