import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { XAxis } from '@/components/Axes'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend, type LegendItem } from '@/components/Legend'
import { useChartOptions } from '@/lib/chartOptions'
import { useColourScheme } from '@/lib/colourSchemes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { RESTATES, isBookkeeping, sameProduct } from '@/fieldLineage'
import { averageLinkage, type Clade } from '@/lib/cluster'
import { corrMatrix, fmt } from '@/lib/stats'
import { useStableRows } from '@/lib/stable'
import { INK, FONT } from '@/lib/theme'
import { WEATHER_GROUP } from '@/lib/weather'
import { PMF_GROUP } from '@/lib/pmfGroups'
import { RATIO_GROUP } from '@/lib/viewFields'
import type { FilterRow, MetaFile } from '@/lib/types'
import { ExcludedNote } from '@/lib/markedRows'

const VIEWS = ['Network', 'Tree'] as const
const THRESHOLDS = ['0.5', '0.6', '0.7', '0.8', '0.9']
/**
 * Which fields take part. Chemistry is the measured species; weather, the PMF factor shares
 * (model outputs), ratios (restate their parts) and the XRF elements that repeat an ion
 * (S beside sulfate) would otherwise make most of the lines and say little.
 */
const INCLUDE = ['Chemistry', 'With weather', 'Everything'] as const
const NOT_CHEMISTRY = [WEATHER_GROUP, PMF_GROUP, RATIO_GROUP, 'Elements (XRF)', 'Sampling']
const isWeather = (f: string) => /\((ERA5|station)\)$/.test(f)
const NETWORK_HEIGHT = 460
const ALONE = '#9aa5b1'

/** One edge: two field indices and the signed r between them. */
interface Edge {
  a: number
  b: number
  r: number
}

interface SimNode extends d3.SimulationNodeDatum {
  i: number
}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  r: number
}

/** Rough pixel width of a label at 10.5 px sans: sizes gutters and keeps labels off each other. */
const labelW = (s: string) => s.length * 6.2

/**
 * Which measurements move together, as groups. The correlogram shows every pair as a cell;
 * this one answers the question it raises: which species rise and fall together.
 *
 * - Groups: average-linkage clustering on 1 − |r|, cut where the members' average |r| with
 *   each other reaches the threshold. A group is two or more species; the rest have no
 *   strong partner. Both views colour by these groups and list them under the plot.
 * - Network: a line joins two species with |r| ≥ the threshold (thicker = stronger, dashed =
 *   one rises as the other falls). Members of a group are pulled together.
 * - Tree: the clustering itself (a dendrogram), cut at the same threshold.
 *
 * The matrix is computed exactly as the correlogram computes it — same RESTATES / sameProduct
 * filtering, same ≥ 20 numeric values rule, same corrMatrix() (lib/stats) — so the two charts
 * can never disagree on an r.
 */
export function SpeciesGraph({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [hover, setHover] = useState<number | null>(null)
  const [hoverGroup, setHoverGroup] = useState<number | null>(null)
  const colours = useColourScheme('species-graph', 'categorical', 'Tableau 10')
  // a phone gets the chart at its own width: shorter forces, names cut short (full on hover),
  // and the network keeps a readable width, scrolling sideways inside the card (.chart-wrap)
  const w = Math.max(280, width)
  const narrow = w < 560
  const short = (f: string) => (narrow && f.length > 18 ? `${f.slice(0, 17)}…` : f)
  const netH = narrow ? 520 : NETWORK_HEIGHT
  const nw = narrow ? Math.max(w, 600) : w
  const k = Math.min(1, Math.max(0.45, nw / 700))

  const siteCodes = useMemo(
    () => ['All', ...meta.sites.filter((s) => rows.some((r) => r.code === s.code)).map((s) => s.code)],
    [meta.sites, rows]
  )
  const o = useChartOptions('species-graph', [
    { key: 'site', label: 'Site', style: 'select', default: 'ETAD', choices: siteCodes.map((c) => ({ value: c, label: c === 'All' ? 'All sites' : meta.sites.find((s) => s.code === c)?.name ?? c })) },
    { key: 'include', label: 'Fields', choices: INCLUDE, default: 'Chemistry',
      title: 'Chemistry: the measured species. With weather: and the weather over each filter. Everything: also the PMF factor shares, ratios and XRF elements that repeat an ion' },
    { key: 'view', label: 'View', choices: VIEWS, default: 'Network', title: 'Network: lines between strongly correlated species. Tree: the grouping itself (a dendrogram)' },
    { key: 'threshold', label: 'Strong at |r| ≥', style: 'select', choices: THRESHOLDS, default: '0.7',
      title: 'Lines join pairs at least this correlated; a group\'s members average at least this with each other' },
  ])
  const effectiveSite = o.values.site as string
  const include = o.values.include as (typeof INCLUDE)[number]
  const view = o.values.view as (typeof VIEWS)[number]
  const thresholdS = o.values.threshold as string
  const threshold = Number(thresholdS)

  // ---- families: field → measurement group label (shown in the tooltip)
  const familyOf = useMemo(() => {
    const label = new Map<string, string>()
    for (const g of meta.field_groups) for (const f of g.fields) label.set(f, g.label)
    return (f: string) => label.get(f) ?? 'Other'
  }, [meta.field_groups])

  // ---- the correlation matrix, identical to Correlogram.tsx
  // the site's rows, kept as the same array while their values are unchanged: a pick of other
  // sites in the sidebar does not recompute this site's matrix, layout and tree (lib/stable)
  const siteRows = useMemo(() => (effectiveSite === 'All' ? rows : rows.filter((r) => r.code === effectiveSite)), [rows, effectiveSite])
  const sub = useStableRows(siteRows, meta.fields)
  const block = useMemo(() => {
    const wanted = (f: string) => {
      if (include === 'Everything') return true
      const fam = familyOf(f)
      if (include === 'With weather' && (fam === WEATHER_GROUP || isWeather(f))) return true
      return !NOT_CHEMISTRY.includes(fam) && !isWeather(f)
    }
    const fields = meta.fields
      .filter((f) => !(f in RESTATES) && !isBookkeeping(f) && wanted(f))
      .filter((f) => sub.filter((r) => typeof r[f] === 'number').length >= 20)
    return { fields, matrix: corrMatrix(sub, fields, sameProduct) }
  }, [sub, meta.fields, include, familyOf])

  // ---- the groups: average linkage on d = 1 − |r| (a missing r counts as unrelated, d = 1),
  // cut at 1 − threshold. Groups (≥ 2 species) are numbered top to bottom in the tree's leaf order
  const tree = useMemo(() => {
    const { fields, matrix } = block
    const n = fields.length
    if (n < 2) return null
    const dist = matrix.map((row) => row.map((r) => (r === null ? 1 : 1 - Math.abs(r))))
    const root = d3.hierarchy<Clade>(averageLinkage(dist), (d) => d.children)
    const cut = 1 - threshold
    const groupOf = new Array<number>(n).fill(-1)
    const members: number[][] = []
    const walk = (node: d3.HierarchyNode<Clade>) => {
      if (!node.children || node.data.height <= cut + 1e-9) {
        const leaves = node.leaves().map((l) => l.data.idx ?? -1)
        if (leaves.length >= 2) {
          for (const i of leaves) groupOf[i] = members.length
          members.push(leaves)
        }
      } else node.children.forEach(walk)
    }
    walk(root)
    const groups = members.map((m) => {
      const rs: number[] = []
      let best: { a: number; b: number; r: number } | null = null
      for (let x = 0; x < m.length; x++) for (let y = x + 1; y < m.length; y++) {
        const r = matrix[m[x]][m[y]]
        if (r === null) continue
        rs.push(Math.abs(r))
        if (!best || Math.abs(r) > Math.abs(best.r)) best = { a: m[x], b: m[y], r }
      }
      return { members: m, meanR: d3.mean(rs) ?? NaN, best }
    })
    return { root, groupOf, groups }
  }, [block, threshold])

  // indexed directly rather than via scaleOrdinal, so group 1 is always the set's first colour; wraps past its last
  const groupColor = (g: number) => (g < 0 ? ALONE : colours.stops[g % colours.stops.length])
  const groupOf = (i: number) => tree?.groupOf[i] ?? -1

  // ---- edges above the threshold, degrees, and each species' strongest partner
  const graph = useMemo(() => {
    const { fields, matrix } = block
    const n = fields.length
    const edges: Edge[] = []
    const degree = new Array<number>(n).fill(0)
    const strongest = new Array<{ j: number; r: number } | null>(n).fill(null)
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const r = matrix[i][j]
        if (r === null) continue // by-construction pairs are null in the matrix
        if (Math.abs(r) >= threshold) {
          edges.push({ a: i, b: j, r })
          degree[i]++
          degree[j]++
        }
        for (const [p, q] of [[i, j], [j, i]]) {
          const cur = strongest[p]
          if (!cur || Math.abs(r) > Math.abs(cur.r)) strongest[p] = { j: q, r }
        }
      }
    }
    return { edges, degree, strongest }
  }, [block, threshold])

  // every edge is already ≥ threshold, so widths are scaled from the threshold up
  const widthOf = useMemo(() => d3.scaleLinear().domain([threshold, 1]).range([1, 5]).clamp(true), [threshold])

  // what is lit: a hovered species and its partners, or a hovered group's members
  const isPartner = (i: number) => graph.edges.some((e) => (e.a === hover && e.b === i) || (e.b === hover && e.a === i))
  const lit = (i: number) =>
    hoverGroup !== null ? groupOf(i) === hoverGroup : hover === null || i === hover || isPartner(i)
  const edgeLit = (e: Edge) =>
    hoverGroup !== null ? groupOf(e.a) === hoverGroup && groupOf(e.b) === hoverGroup : hover === null || e.a === hover || e.b === hover
  const groupName = (g: number) => (g < 0 ? 'no group' : `group ${g + 1}`)

  const nodeTip = (i: number): string[] => {
    const s = graph.strongest[i]
    const g = groupOf(i)
    return [
      block.fields[i],
      `${familyOf(block.fields[i])} · ${g < 0 ? `in no group at |r| ≥ ${thresholdS}` : groupName(g)}`,
      `${graph.degree[i]} partner${graph.degree[i] === 1 ? '' : 's'} at |r| ≥ ${thresholdS}`,
      s ? `strongest: ${block.fields[s.j]} · r = ${fmt(s.r, 3)}` : 'no correlated partner',
    ]
  }
  const edgeTip = (e: Edge) => [`${block.fields[e.a]} — ${block.fields[e.b]}`,
    `r = ${fmt(e.r, 3)} · R² = ${fmt(e.r * e.r, 3)}`, e.r > 0 ? 'rise and fall together' : 'one rises as the other falls']

  // ---- Network: run the force layout to completion synchronously so the positions are
  // deterministic and there is no animation loop. Species with no partner are listed under the
  // plot instead of floating as loose dots.
  const network = useMemo(() => {
    const { fields } = block
    const connected = fields.map((_, i) => i).filter((i) => graph.degree[i] > 0)
    const nodes: SimNode[] = connected.map((i) => ({ i }))
    const index = new Map(connected.map((i, k) => [i, k]))
    const links: SimLink[] = graph.edges.map((e) => ({ source: index.get(e.a)!, target: index.get(e.b)!, r: e.r }))
    const gOf = tree?.groupOf ?? []
    // a group's members drift toward the group's middle, so each group reads as one patch of colour
    const pullTogether = (alpha: number) => {
      const mid = new Map<number, { x: number; y: number; n: number }>()
      for (const nd of nodes) {
        const g = gOf[nd.i] ?? -1
        if (g < 0) continue
        const m = mid.get(g) ?? { x: 0, y: 0, n: 0 }
        m.x += nd.x ?? 0; m.y += nd.y ?? 0; m.n++
        mid.set(g, m)
      }
      for (const nd of nodes) {
        const m = mid.get(gOf[nd.i] ?? -1)
        if (!m) continue
        nd.vx = (nd.vx ?? 0) + (m.x / m.n - (nd.x ?? 0)) * 0.12 * alpha
        nd.vy = (nd.vy ?? 0) + (m.y / m.n - (nd.y ?? 0)) * 0.12 * alpha
      }
    }
    const sim = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        'link',
        d3
          .forceLink<SimNode, SimLink>(links)
          // strongest pairs sit closest: (1 − |r|) rescaled over the visible range
          .distance((l) => k * (60 + 100 * ((1 - Math.abs(l.r)) / Math.max(0.05, 1 - threshold))))
          // a line between two groups pulls less than one inside a group
          .strength((l) => ((gOf[(l.source as SimNode).i] ?? -1) >= 0 && gOf[(l.source as SimNode).i] === gOf[(l.target as SimNode).i] ? 0.7 : 0.15))
      )
      .force('charge', d3.forceManyBody().strength(-240 * k))
      .force('center', d3.forceCenter(nw / 2, netH / 2))
      // a weak pull to the middle keeps small separate groups on the canvas instead of flung to an edge
      .force('x', d3.forceX<SimNode>(nw / 2).strength(0.04))
      .force('y', d3.forceY<SimNode>(netH / 2).strength(0.07))
      .force('group', pullTogether)
      .force('collide', d3.forceCollide<SimNode>((n) => 14 + labelW(fields[n.i]) * 0.3 * k).iterations(2))
      .stop()
    for (let t = 0; t < 300; t++) sim.tick()

    // clamp into the box: the invariant check flags any circle past the edge
    const pad = 18
    const pos = new Map<number, { x: number; y: number }>()
    for (const n of nodes) pos.set(n.i, { x: Math.max(pad, Math.min(nw - pad, n.x ?? 0)), y: Math.max(pad, Math.min(netH - pad, n.y ?? 0)) })

    // labels: the best-connected species first, each on the side away from the middle when that
    // fits, and none drawn over an earlier one (the rest show when their species is hovered)
    const placed: { x0: number; x1: number; y0: number; y1: number }[] = []
    const label = new Map<number, { x: number; anchor: 'start' | 'end' }>()
    const order = [...pos.keys()].sort((a, b) => graph.degree[b] - graph.degree[a])
    for (const i of order) {
      const p = pos.get(i)!
      const r = nodeR(graph.degree[i])
      const lw = labelW(short(fields[i]))
      const sides: ('start' | 'end')[] = p.x >= nw / 2 ? ['start', 'end'] : ['end', 'start']
      for (const side of sides) {
        const x0 = side === 'start' ? p.x + r + 4 : p.x - r - 4 - lw
        const box = { x0, x1: x0 + lw, y0: p.y - 7, y1: p.y + 7 }
        if (box.x0 < 2 || box.x1 > nw - 2) continue
        if (placed.some((q) => box.x0 < q.x1 && box.x1 > q.x0 && box.y0 < q.y1 && box.y1 > q.y0)) continue
        placed.push(box)
        label.set(i, { x: side === 'start' ? p.x + r + 4 : p.x - r - 4, anchor: side })
        break
      }
    }
    const alone = fields.map((_, i) => i).filter((i) => graph.degree[i] === 0)
    return { pos, label, alone }
  }, [block, graph, tree, nw, threshold, netH, k, narrow]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Tree: d3.cluster gives the vertical spread; the horizontal position is the merge
  // height, so the axis reads 1 − |r|
  const dendro = useMemo(() => {
    if (!tree) return null
    const n = block.fields.length
    const leafH = 18
    const mt = 22
    const ml = 16
    const mr = Math.ceil(Math.max(0, ...block.fields.map((f) => labelW(short(f))))) + 18
    const mb = 58
    const plotH = n * leafH
    const plotW = Math.max(200, w - ml - mr)
    d3.cluster<Clade>().size([plotH, plotW]).separation(() => 1)(tree.root)
    const xs = d3.scaleLinear().domain([1, 0]).range([0, plotW])
    const px = (d: d3.HierarchyNode<Clade>) => xs(d.data.height)
    return { xs, px, ml, mt, plotW, plotH, height: mt + plotH + mb }
  }, [tree, block, w, narrow]) // eslint-disable-line react-hooks/exhaustive-deps

  const nSpecies = block.fields.length
  const groups = tree?.groups ?? []
  const nGrouped = groups.reduce((s, g) => s + g.members.length, 0)
  const anyNegative = graph.edges.some((e) => e.r < 0)
  const siteName = effectiveSite === 'All' ? 'all sites' : meta.sites.find((s) => s.code === effectiveSite)?.name ?? effectiveSite
  const edgeKey: LegendItem[] = [
    { label: 'rise and fall together', color: INK.axis, shape: 'line' },
    ...(anyNegative ? [{ label: 'one rises as the other falls', color: INK.negative, shape: 'dashed' as const }] : []),
  ]

  return (
    <ChartFrame
      id="species-graph"
      title="Species network — which measurements move together"
      subtitle="Each dot is a measurement. A line joins two that rise and fall together from filter to filter (|r| at or above the threshold; thicker is stronger). Dots of one colour form a group: species whose correlations with each other average at least the threshold, listed under the plot. Grey dots have a partner or two but belong to no group. 'Tree' shows the same grouping as a dendrogram. Hover a dot or a group to pick it out."
      provenance="react-graph-gallery.com/network-chart · /dendrogram · groups: average-linkage clustering on 1 − |r|"
      controls={<>{o.controls()}{colours.picker}<span className="control hint">{nSpecies} species · {groups.length} group{groups.length === 1 ? '' : 's'}</span></>}
    >
      <div ref={wrapRef} className="chart-wrap">
        {nSpecies < 3 || !tree ? (
          <Empty>Fewer than three species with ≥20 filters in this subset.</Empty>
        ) : view === 'Network' ? (
          network.pos.size === 0 ? <Empty>No pair reaches |r| ≥ {thresholdS} here; try a lower threshold.</Empty> : (
          <svg width={nw} height={netH} fontFamily={FONT.family}>
            {graph.edges.map((e, n) => {
              const p = network.pos.get(e.a)
              const q = network.pos.get(e.b)
              if (!p || !q) return null
              const inside = groupOf(e.a) >= 0 && groupOf(e.a) === groupOf(e.b)
              return (
                <line
                  key={n}
                  x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                  stroke={e.r < 0 ? INK.negative : inside ? groupColor(groupOf(e.a)) : INK.axis}
                  strokeDasharray={e.r < 0 ? '5 3' : undefined}
                  strokeWidth={widthOf(Math.abs(e.r))}
                  strokeOpacity={edgeLit(e) ? (inside ? 0.6 : 0.35) : 0.06}
                  style={{ transition: 'stroke-opacity 0.2s' }}
                  onMouseEnter={(ev) => tip.show(ev, edgeTip(e))}
                  onMouseLeave={tip.hide}
                />
              )
            })}
            {[...network.pos.entries()].map(([i, p]) => (
              <circle
                key={i}
                cx={p.x} cy={p.y} r={nodeR(graph.degree[i])}
                fill={groupColor(groupOf(i))} stroke={INK.panel} strokeWidth={1.5}
                opacity={lit(i) ? 1 : 0.15}
                style={{ transition: 'opacity 0.2s', cursor: 'default' }}
                onMouseEnter={(ev) => { setHover(i); tip.show(ev, nodeTip(i)) }}
                onMouseLeave={() => { setHover(null); tip.hide() }}
              />
            ))}
            {/* labels in their own pass, above every node; a hovered species and its partners always show theirs */}
            {[...network.pos.entries()].map(([i, p]) => {
              const f = block.fields[i]
              const r = nodeR(graph.degree[i])
              const forced = (hover !== null && (i === hover || isPartner(i))) || (hoverGroup !== null && groupOf(i) === hoverGroup)
              const at = network.label.get(i) ?? (forced ? (p.x + r + 4 + labelW(f) <= nw - 4 ? { x: p.x + r + 4, anchor: 'start' as const } : { x: p.x - r - 4, anchor: 'end' as const }) : null)
              if (!at) return null
              return (
                <text
                  key={`label-${i}`}
                  x={at.x} y={p.y} dy="0.32em"
                  textAnchor={at.anchor}
                  fontSize={10.5} fill={INK.text}
                  fontWeight={hover === i ? 600 : 400}
                  stroke={INK.panel} strokeWidth={3} paintOrder="stroke"
                  opacity={lit(i) ? 1 : 0.15}
                  style={{ transition: 'opacity 0.2s' }}
                  pointerEvents="none"
                >
                  {forced ? f : short(f)}
                </text>
              )
            })}
          </svg>)
        ) : dendro ? (
          <svg width={w} height={dendro.height} fontFamily={FONT.family}>
            <g transform={`translate(${dendro.ml},${dendro.mt})`}>
              {/* right-angled links: out from the parent's height, then across to the child's;
                  a link inside a group takes the group's colour */}
              {tree.root.links().map((l, n) => {
                const g = l.target.leaves().map((x) => groupOf(x.data.idx ?? -1))
                const one = g.every((x) => x >= 0 && x === g[0]) && l.source.data.height <= 1 - threshold + 1e-9 ? g[0] : -1
                return (
                  <path
                    key={n}
                    d={`M${dendro.px(l.source)},${l.source.x} V${l.target.x} H${dendro.px(l.target)}`}
                    fill="none"
                    stroke={one >= 0 ? groupColor(one) : INK.axis}
                    strokeWidth={one >= 0 ? 1.6 : 1}
                    opacity={hoverGroup === null || one === hoverGroup ? 1 : 0.25}
                  />
                )
              })}
              <line x1={dendro.xs(1 - threshold)} x2={dendro.xs(1 - threshold)} y1={-6} y2={dendro.plotH} stroke={INK.muted} strokeDasharray="4 3" />
              <text x={dendro.xs(1 - threshold)} y={-10} textAnchor="middle" fontSize={10} fill={INK.muted}>groups cut at |r| ≥ {thresholdS}</text>
              {tree.root.leaves().map((l) => {
                const i = l.data.idx ?? -1
                const f = block.fields[i]
                return (
                  <g
                    key={i}
                    opacity={lit(i) ? 1 : 0.25}
                    style={{ cursor: 'default' }}
                    onMouseEnter={(ev) => { setHover(i); tip.show(ev, nodeTip(i)) }}
                    onMouseLeave={() => { setHover(null); tip.hide() }}
                  >
                    <circle cx={dendro.px(l)} cy={l.x} r={4.5} fill={groupColor(groupOf(i))} stroke={INK.panel} strokeWidth={1} />
                    <text x={dendro.px(l) + 8} y={l.x} dy="0.32em" fontSize={10.5} fill={groupOf(i) < 0 ? INK.muted : INK.text} fontWeight={hover === i ? 600 : 400}>
                      {short(f)}
                    </text>
                  </g>
                )
              })}
              <XAxis scale={dendro.xs} y={dendro.plotH + 12} label="1 − |r|: joined further right = more alike" tickCount={5} format={(v) => fmt(v, 1)} />
            </g>
          </svg>
        ) : null}
        {tree && nSpecies >= 3 && <>
          {/* the groups, in words: what the colours mean, and the answer the chart is for */}
          <div className="species-groups" data-export="text">
            <div className="species-groups-head">
              {groups.length === 0
                ? `No group at |r| ≥ ${thresholdS} at ${siteName}: no species move together this strongly.`
                : `At ${siteName}, ${nGrouped} of ${nSpecies} species fall into ${groups.length} group${groups.length === 1 ? '' : 's'} that move together:`}
            </div>
            {groups.map((g, n) => (
              <div key={n} className={`species-group${hoverGroup === n ? ' on' : ''}`}
                onMouseEnter={() => setHoverGroup(n)} onMouseLeave={() => setHoverGroup(null)}>
                <span className="swatch" style={{ background: groupColor(n) }} />
                <strong>{n + 1}</strong>
                <span className="species-group-names">{g.members.map((i) => block.fields[i]).join(', ')}</span>
                <span className="species-group-r" title={g.best ? `strongest pair: ${block.fields[g.best.a]} and ${block.fields[g.best.b]}, r = ${fmt(g.best.r, 3)}` : undefined}>
                  average |r| {fmt(g.meanR, 2)}
                </span>
              </div>
            ))}
            {view === 'Network' && network.alone.length > 0 && (
              <div className="species-alone">
                No partner at |r| ≥ {thresholdS}: {network.alone.map((i) => block.fields[i]).join(', ')}
              </div>
            )}
          </div>
          {view === 'Network' && <Legend items={edgeKey} note="bigger dot = more partners" />}
        </>}
        <ExcludedNote />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** a node's radius: bigger with more partners, capped so a hub does not cover its neighbours */
const nodeR = (degree: number) => Math.min(13, 5 + 1.5 * degree)
