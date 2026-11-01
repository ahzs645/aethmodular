import { useCallback, useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { HeatMatrix, type MatrixItem } from '@/components/HeatMatrix'
import { useDimensions } from '@/hooks/useGalleryData'
import { useChartOptions } from '@/lib/chartOptions'
import { averageLinkage, corrDistance, leafOrder, type Clade } from '@/lib/cluster'
import { useColourScheme } from '@/lib/colourSchemes'
import { f2, fmt } from '@/lib/format'
import { FixedSetNote } from '@/lib/markedRows'
import { num, type Table } from '@/lib/table'
import { INK } from '@/lib/theme'

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

  const colours = useColourScheme('network-site-corr', 'sequential')
  const measureRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(measureRef)

  const layout = useMemo(() => {
    if (!view) return null
    const n = view.order.length
    const cell = Math.max(2.5, Math.min(18, (width - 64 - 12) / n))
    const font = Math.min(11, cell * 0.85)
    // every name once the rows are tall enough to read; otherwise the SPARTAN targets only
    const allLabels = font >= 6.5
    const pos = new Map(view.order.map((idx, k) => [idx, k]))
    // dendrogram: leaves at their cell centre, internal nodes at their children's mid and merge height
    const maxH = view.tree.height || 1
    // square-root heights: the Arctic outliers join last, far above the rest, and would flatten the tree
    const y = d3.scaleSqrt().domain([0, maxH]).range([TREE_H, 4])
    const links: string[] = []
    const place = (c: Clade): number => {
      if (!c.children) return (pos.get(c.idx!)! + 0.5) * cell
      const xs = c.children.map(place)
      const cy = y(c.height)
      c.children.forEach((ch, i) => links.push(`M${xs[i]},${ch.children ? y(ch.height) : TREE_H}V${cy}`))
      links.push(`M${Math.min(...xs)},${cy}H${Math.max(...xs)}`)
      return (Math.min(...xs) + Math.max(...xs)) / 2
    }
    place(view.tree)
    return { n, cell, font, allLabels, links, y, maxH }
  }, [view, width])

  const scale = useMemo(() => {
    const ramp = colours.stops, dom = view?.domain ?? [0.9, 1]
    return d3.scaleLinear<string>().domain(ramp.map((_, i) => dom[0] + (i / (ramp.length - 1)) * (dom[1] - dom[0]))).range(ramp).clamp(true)
  }, [colours.stops, view])

  // rows and columns in leaf order; memoised with the cell lookup so a hover never rebuilds the cells
  const axes = useMemo(() => {
    if (!view || !layout) return null
    const rows: MatrixItem[] = view.order.map((idx) => {
      const s = view.names[idx], t = isTarget(s)
      return { key: s, title: label(s), label: layout.allLabels || t ? `${label(s)}${layout.allLabels ? '' : ' ▸'}` : '',
        weight: t ? 700 : 400, size: t ? Math.max(layout.font, 10) : layout.font }
    })
    const cols: MatrixItem[] = view.order.map((idx) => ({ key: view.names[idx], title: label(view.names[idx]), label: '' }))
    return { rows, cols }
  }, [view, layout])
  const cellAt = useCallback((i: number, j: number) => (view ? view.r[view.order[i]][view.order[j]] ?? undefined : undefined), [view])
  const tipAt = useCallback((i: number, j: number) => {
    if (!view) return null
    const v = view.r[view.order[i]][view.order[j]]
    const nm = (k: number) => label(view.names[view.order[k]])
    return [`${nm(i)} × ${nm(j)}`, `r = ${v == null ? '—' : fmt(v, 4)}`, `1 − r = ${v == null ? '—' : fmt(1 - v, 4)}`]
  }, [view])
  const legend = useMemo(() => ({ scale: scale as unknown as { domain(): number[]; (v: number): string }, scheme: colours.scheme, label: 'r of median spectra', format: (v: number) => fmt(v, 3) }), [scale, colours.scheme])

  const title = 'Site × site similarity, clustered — which IMPROVE sites share a spectrum, and where the SPARTAN sites join the tree'
  if (!view || !layout || !axes) return <ChartFrame id="network-site-corr" title={title} provenance={provenance}><Empty>The export has no site × site matrix; re-run ftir_52 and the Research spectral export.</Empty></ChartFrame>
  const { n, cell } = layout
  return (
    <ChartFrame id="network-site-corr" title={title}
      subtitle={`Pearson r between every pair of sites' median spline-baseline spectra (IMPROVE sites with ≥ 20 spectra, and the five SPARTAN targets in bold), ordered by average-linkage clustering on 1 − r; the tree above joins sites at the mean distance between their groups (square-root scale). Most pairs sit above 0.9, so the colour runs from the 2nd percentile of the off-diagonal r (${f2(view.domain[0])}) to 1. Sites: Nearest to a target keeps that target and its closest IMPROVE sites and clusters them again on their own. Point at a cell for the pair.`}
      provenance={provenance} controls={o.controls()}>
      <div ref={measureRef}>
        <HeatMatrix rows={axes.rows} columns={axes.cols} cell={cellAt} tip={tipAt} color={scale} cellLabels="never"
          cellWidth={cell} cellHeight="square" gap={0} crisp hover="cross" colLabels="none" rowLabelSize={layout.font} maxLabelWidth={400}
          fitRows={false} before={TREE_H + 6} center legend={legend} legendExtra={<span className="legend-note">{n} sites</span>}
          footer={<FixedSetNote what="ftir_52's saved site × site matrix of median spectra" />}>
          {(l) => <g pointerEvents="none">
            <g fill="none" stroke={INK.axis} strokeWidth={1} transform={`translate(${l.left},0)`}>{layout.links.map((d, i) => <path key={i} d={d} />)}</g>
            {/* the tree's scale in the label gutter; flush left when the gutter is too narrow for it */}
            <text x={l.left < 72 ? 0 : l.left - 6} y={layout.y(layout.maxH)} dy="0.35em" textAnchor={l.left < 72 ? 'start' : 'end'} fontSize={10} fill={INK.muted}>1 − r {fmt(layout.maxH, 3)}</text>
            <text x={l.left - 6} y={TREE_H} dy="0.35em" textAnchor="end" fontSize={10} fill={INK.muted}>0</text>
          </g>}
        </HeatMatrix>
      </div>
    </ChartFrame>
  )
}
