import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { XAxis } from '@/components/Axes'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { Legend, useLegend, type LegendItem } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { RESTATES, isBookkeeping, sameProduct } from '@/fieldLineage'
import { fmt, regression } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const VIEWS = ['Network', 'Arc', 'Dendrogram'] as const
const THRESHOLDS = ['0.5', '0.6', '0.7', '0.8', '0.9']
const CUTS = ['0.2', '0.3', '0.4', '0.5']
const NETWORK_HEIGHT = 480
const SIGNS = ['positive r', 'negative r']
const signLabel = (r: number) => (r > 0 ? SIGNS[0] : SIGNS[1])

/** One edge of the species graph: two field indices and the signed r between them. */
interface Edge {
  a: number
  b: number
  r: number
}

/** A node of the merge tree; leaves carry the species index, internal nodes the merge height. */
interface Clade {
  idx?: number
  height: number
  children?: Clade[]
}

interface SimNode extends d3.SimulationNodeDatum {
  i: number
}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  r: number
}

/**
 * Rough pixel width of a label at 10.5 px sans. Used to size gutters and to
 * choose which side of a node a label goes, so nothing draws outside the
 * <svg> (the README's rendering invariant is checked in the browser).
 */
const labelW = (s: string) => s.length * 6.2

/**
 * Agglomerative clustering with average linkage on a distance matrix.
 * Start with singletons; repeatedly merge the two clusters whose mean
 * pairwise distance is smallest; record that mean as the merge height.
 * n ≤ ~30 species, so the O(n³) scan is nothing.
 */
function averageLinkage(dist: number[][]): Clade {
  let clusters: { members: number[]; node: Clade }[] = dist.map((_, i) => ({ members: [i], node: { idx: i, height: 0 } }))
  while (clusters.length > 1) {
    let best = Infinity
    let bi = 0
    let bj = 1
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        let sum = 0
        for (const a of clusters[i].members) for (const b of clusters[j].members) sum += dist[a][b]
        const avg = sum / (clusters[i].members.length * clusters[j].members.length)
        if (avg < best) {
          best = avg
          bi = i
          bj = j
        }
      }
    }
    const merged = {
      members: [...clusters[bi].members, ...clusters[bj].members],
      node: { height: best, children: [clusters[bi].node, clusters[bj].node] },
    }
    // keep the merged cluster where the first partner sat, so leaf order stays stable
    clusters = clusters.flatMap((c, k) => (k === bi ? [merged] : k === bj ? [] : [c]))
  }
  return clusters[0].node
}

/**
 * Three react-graph-gallery views over the correlogram's data. The
 * correlogram shows every pair as a cell; here only the pairs above a
 * threshold survive, as a graph, so "which species move together" is a
 * shape rather than a scan. The dendrogram uses every pair (average linkage
 * on 1 − |r|) and needs no threshold.
 *
 * The matrix is computed exactly as the correlogram computes it — same
 * RESTATES / sameProduct filtering, same ≥ 20 numeric values rule,
 * same regression() — so the two charts can never disagree on an r.
 */
export function SpeciesGraph({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [siteCode, setSiteCode] = useState('ETAD')
  const [view, setView] = useState<(typeof VIEWS)[number]>('Network')
  const [thresholdS, setThresholdS] = useState('0.7')
  const [cutS, setCutS] = useState('0.3')
  const [hover, setHover] = useState<number | null>(null)
  // families and edge signs toggle in Network/Arc; dendrogram clusters only highlight, since
  // dropping species would re-cut the tree and renumber every cluster
  const lg = useLegend()
  const cl = useLegend()
  const off = view === 'Dendrogram' ? null : lg.hidden
  const threshold = Number(thresholdS)
  const cut = Number(cutS)
  const w = Math.max(400, width)

  const siteCodes = useMemo(
    () => ['All', ...meta.sites.filter((s) => rows.some((r) => r.code === s.code)).map((s) => s.code)],
    [meta.sites, rows]
  )
  const effectiveSite = siteCodes.includes(siteCode) ? siteCode : siteCodes[0] ?? 'All'

  // ---- the correlation matrix, identical to Correlogram.tsx
  const block = useMemo(() => {
    const sub = effectiveSite === 'All' ? rows : rows.filter((r) => r.code === effectiveSite)
    const fields = meta.fields
      .filter((f) => !(f in RESTATES) && !isBookkeeping(f))
      .filter((f) => sub.filter((r) => typeof r[f] === 'number').length >= 20)
    const matrix = fields.map((fa) =>
      fields.map((fb) => {
        if (sameProduct(fa, fb)) return null
        const xs: number[] = []
        const ys: number[] = []
        for (const r of sub) {
          const a = r[fa]
          const b = r[fb]
          if (typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b)) {
            xs.push(a)
            ys.push(b)
          }
        }
        const st = regression(xs, ys)
        return st ? st.r : null
      })
    )
    return { code: effectiveSite, fields, matrix }
  }, [rows, effectiveSite, meta.fields])

  // ---- families: field → measurement group label, coloured by Tableau10
  const { familyOf, familyRank } = useMemo(() => {
    const label = new Map<string, string>()
    const rank = new Map<string, number>()
    meta.field_groups.forEach((g, k) => {
      rank.set(g.label, k)
      for (const f of g.fields) label.set(f, g.label)
    })
    const familyOf = (f: string) => label.get(f) ?? 'Other'
    // 'Other' (a field in no group) sorts after every declared family
    const familyRank = (f: string) => rank.get(familyOf(f)) ?? meta.field_groups.length
    return { familyOf, familyRank }
  }, [meta.field_groups])
  const familyColor = useMemo(
    () => d3.scaleOrdinal<string, string>(d3.schemeTableau10).domain([...meta.field_groups.map((g) => g.label), 'Other']),
    [meta.field_groups]
  )

  // ---- edges above the threshold, degrees, and each species' strongest partner
  const graph = useMemo(() => {
    const { fields, matrix } = block
    const n = fields.length
    const vis = fields.map((f) => !off?.has(familyOf(f)))
    const edges: Edge[] = []
    const degree = new Array<number>(n).fill(0)
    const strongest = new Array<{ j: number; r: number } | null>(n).fill(null)
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const r = matrix[i][j]
        if (r === null || !vis[i] || !vis[j]) continue // by-construction pairs are null in the matrix
        if (Math.abs(r) >= threshold && !off?.has(signLabel(r))) {
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
    return { edges, degree, strongest, vis }
  }, [block, threshold, off, familyOf])

  const widthOf = useMemo(
    // every edge is already ≥ threshold, so widths are scaled from the
    // threshold up rather than from 0 — otherwise all edges look the same
    () => d3.scaleLinear().domain([threshold, 1]).range([1, 5]).clamp(true),
    [threshold]
  )
  const signColor = (r: number) => (r > 0 ? INK.positive : INK.negative)
  const touches = (e: Edge) => hover === null || e.a === hover || e.b === hover
  const neighbour = (i: number) => hover === null || i === hover || graph.edges.some((e) => (e.a === hover && e.b === i) || (e.b === hover && e.a === i))
  const famDim = (i: number) => (!lg.hover || SIGNS.includes(lg.hover) || lg.hover === familyOf(block.fields[i]) ? 1 : 0.15)
  const edgeDim = (e: Edge) => (!lg.hover ? 1 : SIGNS.includes(lg.hover) ? lg.dim(signLabel(e.r)) : famDim(e.a) === 1 || famDim(e.b) === 1 ? 1 : 0.15)

  const nodeTip = (i: number): string[] => {
    const s = graph.strongest[i]
    return [
      block.fields[i],
      familyOf(block.fields[i]),
      `${graph.degree[i]} partner${graph.degree[i] === 1 ? '' : 's'} at |r| ≥ ${thresholdS}`,
      s ? `strongest: ${block.fields[s.j]} · r = ${fmt(s.r, 3)}` : 'no correlated partner',
    ]
  }

  // ---- Network: run the force layout to completion synchronously so the
  // positions are deterministic and there is no animation loop to manage.
  const network = useMemo(() => {
    const { fields } = block
    const isolated = fields.map((_, i) => i).filter((i) => graph.vis[i] && graph.degree[i] === 0)
    const connected = fields.map((_, i) => i).filter((i) => graph.degree[i] > 0)

    // Species with no edges sit in a strip at the bottom, wrapped into rows
    // as needed, so the reader sees what is uncorrelated at this threshold
    // without them drifting off-canvas under forceManyBody.
    const strip: { i: number; x: number; row: number }[] = []
    let sx = 14
    let row = 0
    for (const i of isolated) {
      const need = 12 + labelW(fields[i]) + 18
      if (sx + need > w - 12 && sx > 14) {
        sx = 14
        row++
      }
      strip.push({ i, x: sx + 6, row })
      sx += need
    }
    const stripH = isolated.length ? 30 + (row + 1) * 20 : 0
    const simH = NETWORK_HEIGHT - stripH

    const nodes: SimNode[] = connected.map((i) => ({ i }))
    const index = new Map(connected.map((i, k) => [i, k]))
    const links: SimLink[] = graph.edges.map((e) => ({ source: index.get(e.a)!, target: index.get(e.b)!, r: e.r }))
    const sim = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        'link',
        d3
          .forceLink<SimNode, SimLink>(links)
          // strongest pairs sit closest: (1 − |r|) rescaled over the visible range
          .distance((l) => 70 + 110 * ((1 - Math.abs(l.r)) / Math.max(0.05, 1 - threshold)))
      )
      .force('charge', d3.forceManyBody().strength(-260))
      .force('center', d3.forceCenter(w / 2, simH / 2))
      // a weak pull to the middle keeps small disconnected clusters (sulfate–ammonium,
      // lead–zinc) on the canvas instead of flung to an edge and clamped there
      .force('x', d3.forceX<SimNode>(w / 2).strength(0.03))
      .force('y', d3.forceY<SimNode>(simH / 2).strength(0.06))
      // collide on the node plus part of its label, so names do not stack on each other
      .force('collide', d3.forceCollide<SimNode>((n) => 14 + labelW(fields[n.i]) * 0.35).iterations(2))
      .stop()
    for (let k = 0; k < 300; k++) sim.tick()

    // clamp into the box: the invariant check flags any circle past the edge
    const pad = 18
    const pos = new Map<number, { x: number; y: number }>()
    for (const n of nodes) pos.set(n.i, { x: Math.max(pad, Math.min(w - pad, n.x ?? 0)), y: Math.max(pad, Math.min(simH - pad, n.y ?? 0)) })
    return { pos, strip, stripH, simH }
  }, [block, graph, w, threshold])

  // ---- Arc: nodes on a line ordered by family then name; arc height ∝ span
  const arc = useMemo(() => {
    const order = block.fields
      .map((_, i) => i)
      .filter((i) => graph.vis[i])
      .sort((i, j) => familyRank(block.fields[i]) - familyRank(block.fields[j]) || block.fields[i].localeCompare(block.fields[j]))
    const maxLabel = Math.max(0, ...block.fields.map(labelW))
    // labels hang down-left at 45°, so they need ~0.71 × their length both ways
    const ml = Math.ceil(0.71 * maxLabel) + 14
    const mr = 16
    const x = d3.scalePoint<number>().domain(order).range([ml, w - mr])
    const spans = graph.edges.map((e) => Math.abs((x(e.a) ?? 0) - (x(e.b) ?? 0)))
    const maxArc = spans.length ? 0.35 * Math.max(...spans) : 30
    const top = 14
    const y0 = top + maxArc
    const height = y0 + 12 + Math.ceil(0.71 * maxLabel) + 22
    const path = (e: Edge) => {
      const x1 = x(e.a) ?? 0
      const x2 = x(e.b) ?? 0
      const span = Math.abs(x2 - x1)
      // a quadratic whose peak is 0.35 × span above the line (peak = half the control offset)
      return `M${x1},${y0} Q${(x1 + x2) / 2},${y0 - 0.7 * span} ${x2},${y0}`
    }
    return { order, x, y0, height, path }
  }, [block, graph, w, familyRank])

  // ---- Dendrogram: average linkage on d = 1 − |r|; a missing r counts as
  // unrelated (d = 1) rather than being dropped from the tree.
  const dendro = useMemo(() => {
    const { fields, matrix } = block
    const n = fields.length
    if (n < 2) return null
    const dist = matrix.map((row) => row.map((r) => (r === null ? 1 : 1 - Math.abs(r))))
    const root = d3.hierarchy<Clade>(averageLinkage(dist), (d) => d.children)
    const leafH = 18
    const mt = 12
    const ml = 16
    const mr = Math.ceil(Math.max(0, ...fields.map(labelW))) + 18
    const mb = 58
    const plotH = n * leafH
    const plotW = Math.max(200, w - ml - mr)
    // d3.cluster gives the vertical spread; the horizontal position is
    // overridden with the merge height so the x axis reads 1 − |r|.
    d3.cluster<Clade>().size([plotH, plotW]).separation(() => 1)(root)
    const xs = d3.scaleLinear().domain([1, 0]).range([0, plotW])
    const px = (d: d3.HierarchyNode<Clade>) => xs(d.data.height)

    // clusters at the cut: every subtree whose merge height ≤ cut is one
    // cluster, numbered top to bottom so colours read in leaf order
    const clusterOf = new Map<number, number>()
    let k = 0
    const walk = (node: d3.HierarchyNode<Clade>) => {
      if (!node.children || node.data.height <= cut) {
        for (const l of node.leaves()) clusterOf.set(l.data.idx ?? -1, k)
        k++
      } else node.children.forEach(walk)
    }
    walk(root)

    return { root, xs, px, ml, mt, mr, mb, plotW, plotH, height: mt + plotH + mb, clusterOf, nClusters: k }
  }, [block, w, cut])

  // indexed directly rather than via scaleOrdinal, whose colour order would
  // depend on which cluster happened to be painted first; wraps past 10
  const clusterColor = (c: number) => d3.schemeTableau10[c % d3.schemeTableau10.length]

  const familiesPresent = useMemo(() => {
    const seen = new Set(block.fields.map(familyOf))
    return [...meta.field_groups.map((g) => g.label), 'Other'].filter((l) => seen.has(l))
  }, [block.fields, familyOf, meta.field_groups])

  const legendItems: LegendItem[] =
    view === 'Dendrogram' && dendro
      ? d3.range(dendro.nClusters).map((c) => ({ label: `cluster ${c + 1}`, color: clusterColor(c), shape: 'dot' as const }))
      : [
          ...familiesPresent.map((l) => ({ label: l, color: familyColor(l), shape: 'dot' as const })),
          { label: 'positive r', color: INK.positive, shape: 'line' as const },
          { label: 'negative r', color: INK.negative, shape: 'line' as const },
        ]

  const nSpecies = block.fields.length
  const nEdges = graph.edges.length

  return (
    <ChartFrame
      id="species-graph"
      title="Species network · arc · dendrogram — which measurements move together"
      subtitle="The correlogram above shows every retained pair as a cell; this shows pairs above a threshold as a graph — line colour is the sign of r, line width its strength, node colour the measurement family. The dendrogram groups species by average-linkage clustering on 1 − |r|. Duplicate ChemSpec FTIR carbon fields are omitted from both views. Hover a species to isolate its partners."
      provenance="react-graph-gallery.com/network-chart · /arc-diagram · /dendrogram · not in the estate (0 flow figures)"
      controls={
        <>
          <Select label="site" value={block.code} options={siteCodes} onChange={setSiteCode} optionLabel={(c) => (c === 'All' ? 'All sites' : meta.sites.find((s) => s.code === c)?.name ?? c)} />
          <Segmented label="view" value={view} options={VIEWS} onChange={setView} />
          {view !== 'Dendrogram' && <Select label="|r| ≥" value={thresholdS} options={THRESHOLDS} onChange={setThresholdS} />}
          {view === 'Dendrogram' && <Select label="cut at" value={cutS} options={CUTS} onChange={setCutS} title="distance 1 − |r| at which the tree is cut into clusters" />}
          <span className="control">
            {nSpecies} species · {nEdges} edges
            {view === 'Dendrogram' && dendro ? ` · ${dendro.nClusters} clusters at ${cutS}` : ''}
          </span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {nSpecies < 3 ? (
          <Empty>Fewer than three species with ≥20 filters in this subset.</Empty>
        ) : view === 'Network' ? (
          <svg width={w} height={NETWORK_HEIGHT} fontFamily={FONT.family}>
            {graph.edges.map((e, k) => {
              const p = network.pos.get(e.a)
              const q = network.pos.get(e.b)
              if (!p || !q) return null
              return (
                <line
                  key={k}
                  x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                  stroke={signColor(e.r)}
                  strokeWidth={widthOf(Math.abs(e.r))}
                  strokeOpacity={(touches(e) ? 0.7 : 0.15) * edgeDim(e)}
                  style={{ transition: 'stroke-opacity 0.2s' }}
                  onMouseEnter={(ev) => tip.show(ev, [`${block.fields[e.a]} — ${block.fields[e.b]}`, `r = ${fmt(e.r, 3)} · R² = ${fmt(e.r * e.r, 3)}`])}
                  onMouseLeave={tip.hide}
                />
              )
            })}
            {[...network.pos.entries()].map(([i, p]) => {
              const f = block.fields[i]
              const r = Math.min(16, 6 + 2 * graph.degree[i])
              const on = neighbour(i)
              return (
                <g
                  key={i}
                  opacity={(on ? 1 : 0.15) * famDim(i)}
                  style={{ transition: 'opacity 0.2s', cursor: 'default' }}
                  onMouseEnter={(ev) => { setHover(i); tip.show(ev, nodeTip(i)) }}
                  onMouseLeave={() => { setHover(null); tip.hide() }}
                >
                  <circle cx={p.x} cy={p.y} r={r} fill={familyColor(familyOf(f))} stroke="#fff" strokeWidth={1.5} />
                </g>
              )
            })}
            {/* labels in their own pass, above every node, so no circle paints over a name */}
            {[...network.pos.entries()].map(([i, p]) => {
              const f = block.fields[i]
              const r = Math.min(16, 6 + 2 * graph.degree[i])
              // label points away from the middle (out of the cluster), unless that side would leave the canvas
              const fitsRight = p.x + r + 4 + labelW(f) <= w - 4
              const fitsLeft = p.x - r - 4 - labelW(f) >= 4
              const right = fitsRight && (p.x >= w / 2 || !fitsLeft)
              return (
                <g key={`label-${i}`} opacity={(neighbour(i) ? 1 : 0.15) * famDim(i)} style={{ transition: 'opacity 0.2s' }}>
                  <text
                    x={right ? p.x + r + 4 : p.x - r - 4}
                    y={p.y} dy="0.32em"
                    textAnchor={right ? 'start' : 'end'}
                    fontSize={10.5} fill={INK.text}
                    fontWeight={hover === i ? 600 : 400}
                    stroke="#fff" strokeWidth={3} paintOrder="stroke"
                    pointerEvents="none"
                  >
                    {f}
                  </text>
                </g>
              )
            })}
            {network.strip.length > 0 && (
              <g transform={`translate(0,${network.simH})`}>
                <line x1={12} x2={w - 12} y1={0} y2={0} stroke={INK.grid} />
                <text x={14} y={14} fontSize={10} fill={INK.muted}>
                  uncorrelated with everything at |r| ≥ {thresholdS}:
                </text>
                {network.strip.map(({ i, x, row }) => {
                  const f = block.fields[i]
                  const y = 30 + row * 20
                  return (
                    <g
                      key={i}
                      opacity={(hover === null || hover === i ? 1 : 0.15) * famDim(i)}
                      style={{ cursor: 'default' }}
                      onMouseEnter={(ev) => { setHover(i); tip.show(ev, nodeTip(i)) }}
                      onMouseLeave={() => { setHover(null); tip.hide() }}
                    >
                      <circle cx={x} cy={y} r={6} fill="#fff" stroke={INK.neutral} strokeWidth={2} />
                      <text x={x + 10} y={y} dy="0.32em" fontSize={10.5} fill={INK.muted}>{f}</text>
                    </g>
                  )
                })}
              </g>
            )}
          </svg>
        ) : view === 'Arc' ? (
          <svg width={w} height={arc.height} fontFamily={FONT.family}>
            <line x1={arc.x.range()[0]} x2={arc.x.range()[1]} y1={arc.y0} y2={arc.y0} stroke={INK.grid} />
            {graph.edges.map((e, k) => (
              <path
                key={k}
                d={arc.path(e)}
                fill="none"
                stroke={signColor(e.r)}
                strokeWidth={widthOf(Math.abs(e.r))}
                strokeOpacity={(touches(e) ? 0.65 : 0.12) * edgeDim(e)}
                style={{ transition: 'stroke-opacity 0.2s' }}
                onMouseEnter={(ev) => tip.show(ev, [`${block.fields[e.a]} — ${block.fields[e.b]}`, `r = ${fmt(e.r, 3)} · R² = ${fmt(e.r * e.r, 3)}`])}
                onMouseLeave={tip.hide}
              />
            ))}
            {arc.order.map((i) => {
              const f = block.fields[i]
              const x = arc.x(i) ?? 0
              const deg = graph.degree[i]
              const r = Math.min(12, 5 + deg)
              return (
                <g
                  key={i}
                  opacity={(neighbour(i) ? 1 : 0.15) * famDim(i)}
                  style={{ transition: 'opacity 0.2s', cursor: 'default' }}
                  onMouseEnter={(ev) => { setHover(i); tip.show(ev, nodeTip(i)) }}
                  onMouseLeave={() => { setHover(null); tip.hide() }}
                >
                  {deg === 0 ? (
                    <circle cx={x} cy={arc.y0} r={5} fill="#fff" stroke={INK.neutral} strokeWidth={2} />
                  ) : (
                    <circle cx={x} cy={arc.y0} r={r} fill={familyColor(familyOf(f))} stroke="#fff" strokeWidth={1.5} />
                  )}
                  <text
                    transform={`translate(${x},${arc.y0 + 12}) rotate(-45)`}
                    textAnchor="end" dy="0.32em"
                    fontSize={10.5} fill={deg === 0 ? INK.muted : INK.text}
                    fontWeight={hover === i ? 600 : 400}
                  >
                    {f}
                  </text>
                </g>
              )
            })}
          </svg>
        ) : dendro ? (
          <svg width={w} height={dendro.height} fontFamily={FONT.family}>
            <g transform={`translate(${dendro.ml},${dendro.mt})`}>
              {/* right-angled links: out from the parent's height, then across to the child's */}
              {dendro.root.links().map((l, k) => (
                <path
                  key={k}
                  d={`M${dendro.px(l.source)},${l.source.x} V${l.target.x} H${dendro.px(l.target)}`}
                  fill="none"
                  stroke={INK.axis}
                  strokeWidth={1}
                />
              ))}
              <line
                x1={dendro.xs(cut)} x2={dendro.xs(cut)} y1={0} y2={dendro.plotH}
                stroke={INK.muted} strokeDasharray="4 3"
              />
              <text x={dendro.xs(cut) + 4} y={10} fontSize={10} fill={INK.muted}>cut {cutS}</text>
              {dendro.root.leaves().map((l) => {
                const i = l.data.idx ?? -1
                const f = block.fields[i]
                const c = dendro.clusterOf.get(i) ?? 0
                const s = graph.strongest[i]
                return (
                  <g
                    key={i}
                    opacity={cl.dim(`cluster ${c + 1}`)}
                    style={{ cursor: 'default' }}
                    onMouseEnter={(ev) => {
                      setHover(i)
                      tip.show(ev, [f, familyOf(f), `cluster ${c + 1} at cut ${cutS}`, s ? `nearest: ${block.fields[s.j]} · r = ${fmt(s.r, 3)}` : 'no correlated partner'])
                    }}
                    onMouseLeave={() => { setHover(null); tip.hide() }}
                  >
                    <circle cx={dendro.px(l)} cy={l.x} r={4.5} fill={clusterColor(c)} stroke="#fff" strokeWidth={1} />
                    <text x={dendro.px(l) + 8} y={l.x} dy="0.32em" fontSize={10.5} fill={INK.text} fontWeight={hover === i ? 600 : 400}>
                      {f}
                    </text>
                  </g>
                )
              })}
              <XAxis scale={dendro.xs} y={dendro.plotH + 12} label="1 − |r|" tickCount={5} format={(v) => fmt(v, 1)} />
            </g>
          </svg>
        ) : null}
        <Legend
          items={legendItems}
          {...(view === 'Dendrogram' ? { highlighted: cl.hover, onHover: cl.setHover } : lg.props)}
          note={
            view === 'Dendrogram'
              ? 'leaf colour = cluster at the cut · lower on the axis = more correlated'
              : 'node size = number of partners · hover a node to isolate its partners'
          }
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
