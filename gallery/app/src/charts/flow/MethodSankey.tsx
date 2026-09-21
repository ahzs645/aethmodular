import { useMemo, useRef, useState } from 'react'
import { sankey, sankeyLinkHorizontal, sankeyLeft } from 'd3-sankey'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const STAGE3 = ['HIPS coverage', 'Season'] as const

type N = { id: string; label: string; color: string; stage: number }
type L = { source: string; target: string; value: number }

/**
 * Sankey — how filters reach a reported EC.
 *
 * The estate has zero flow figures in 827, and the census's flow/README
 * describes exactly this chart: filters → analytical method → …, making
 * visible how many filters carry both FTIR and TOR EC (the calibration pool)
 * versus FTIR only. The third stage is a choice: HIPS coverage completes the
 * measurement chain; season shows whether a method gap is seasonal.
 */
export function MethodSankey({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [stage3, setStage3] = useState<(typeof STAGE3)[number]>('HIPS coverage')
  const [hoverNode, setHoverNode] = useState<string | null>(null)
  const height = 460

  const has = (r: FilterRow, f: string) => typeof r[f] === 'number' && Number.isFinite(r[f] as number)

  const graph = useMemo(() => {
    const carbon = (r: FilterRow) => {
      const f = has(r, 'EC (FTIR)')
      const t = has(r, 'EC (TOR)')
      return f && t ? 'FTIR + TOR' : f ? 'FTIR only' : t ? 'TOR only' : 'no carbon EC'
    }
    const third = (r: FilterRow) => (stage3 === 'HIPS coverage' ? (has(r, 'HIPS Fabs') ? 'HIPS Fabs' : 'no HIPS') : r.season)

    const carbonOrder = ['FTIR + TOR', 'FTIR only', 'TOR only', 'no carbon EC']
    const carbonColor: Record<string, string> = { 'FTIR + TOR': INK.accent, 'FTIR only': '#6baed6', 'TOR only': '#9e9ac8', 'no carbon EC': INK.neutral }
    const thirdOrder = stage3 === 'HIPS coverage' ? ['HIPS Fabs', 'no HIPS'] : meta.seasons.map((s) => s.name)
    const thirdColor = (k: string) =>
      stage3 === 'HIPS coverage' ? (k === 'HIPS Fabs' ? INK.deming : INK.neutral) : meta.seasons.find((s) => s.name === k)?.color ?? INK.muted

    const nodes: N[] = [
      ...meta.sites.filter((s) => rows.some((r) => r.site === s.name)).map((s) => ({ id: `s:${s.name}`, label: s.name, color: s.color, stage: 0 })),
      ...carbonOrder.map((c) => ({ id: `c:${c}`, label: c, color: carbonColor[c], stage: 1 })),
      ...thirdOrder.map((t) => ({ id: `t:${t}`, label: t, color: thirdColor(t), stage: 2 })),
    ]
    const count = new Map<string, number>()
    for (const r of rows) {
      const c = carbon(r)
      const t = third(r)
      const k1 = `s:${r.site}|c:${c}`
      const k2 = `c:${c}|t:${t}`
      count.set(k1, (count.get(k1) ?? 0) + 1)
      count.set(k2, (count.get(k2) ?? 0) + 1)
    }
    const links: L[] = [...count.entries()].map(([k, value]) => {
      const [source, target] = k.split('|')
      return { source, target, value }
    })
    const used = new Set(links.flatMap((l) => [l.source, l.target]))
    return { nodes: nodes.filter((n) => used.has(n.id)), links }
  }, [rows, meta, stage3])

  const layout = useMemo(() => {
    if (!graph.links.length) return null
    const margin = { top: 10, right: 130, bottom: 10, left: 110 }
    const gen = sankey<N, L>()
      .nodeId((d) => d.id)
      .nodeWidth(14)
      .nodePadding(14)
      .nodeAlign(sankeyLeft)
      .nodeSort((a, b) => a.stage - b.stage) // keep stage order; within a stage, d3 sorts by size
      .extent([[margin.left, margin.top], [Math.max(400, width) - margin.right, height - margin.bottom]])
    return gen({ nodes: graph.nodes.map((n) => ({ ...n })), links: graph.links.map((l) => ({ ...l })) })
  }, [graph, width])

  const total = rows.length
  const touches = (l: any) => hoverNode === null || l.source.id === hoverNode || l.target.id === hoverNode

  return (
    <ChartFrame
      id="sankey"
      title="Sankey — which filters reach which measurement"
      subtitle="Every filter in the subset flows from its site through its carbon-EC coverage (both FTIR and TOR, one, or neither) into either its HIPS coverage or its season. The 'FTIR + TOR' band is the calibration pool; 'FTIR only' is where a calibrated FTIR EC is the only EC there will ever be. Hover a node to isolate its flows."
      provenance="zero flow figures in the estate; described in gallery/app/src/charts/flow/README.md · react-graph-gallery.com/sankey-diagram"
      controls={
        <>
          <Segmented label="third stage" value={stage3} options={STAGE3} onChange={setStage3} />
          <span className="control">{total} filters</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {!layout ? (
          <Empty>No filters in this subset.</Empty>
        ) : (
          <svg width={Math.max(400, width)} height={height}>
            <g fill="none">
              {layout.links.map((l: any, i) => (
                <path
                  key={i}
                  d={sankeyLinkHorizontal()(l) ?? ''}
                  stroke={(l.source as N).color}
                  strokeOpacity={touches(l) ? 0.42 : 0.06}
                  strokeWidth={Math.max(1, l.width)}
                  style={{ transition: 'stroke-opacity 0.2s' }}
                  onMouseEnter={(e) =>
                    tip.show(e, [
                      `${(l.source as N).label} → ${(l.target as N).label}`,
                      `${l.value} filters`,
                      `${((l.value / (l.source as any).value) * 100).toFixed(0)} % of ${(l.source as N).label}`,
                      `${((l.value / total) * 100).toFixed(1)} % of the subset`,
                    ])
                  }
                  onMouseLeave={tip.hide}
                />
              ))}
            </g>
            {layout.nodes.map((n: any) => (
              <g
                key={n.id}
                onMouseEnter={(e) => { setHoverNode(n.id); tip.show(e, [n.label, `${n.value} filters`, `${((n.value / total) * 100).toFixed(1)} % of the subset`]) }}
                onMouseLeave={() => { setHoverNode(null); tip.hide() }}
                style={{ cursor: 'default' }}
              >
                <rect x={n.x0} y={n.y0} width={n.x1 - n.x0} height={Math.max(1, n.y1 - n.y0)} fill={n.color} rx={2} />
                <text
                  x={n.stage === 2 ? n.x1 + 6 : n.x0 - 6}
                  y={(n.y0 + n.y1) / 2}
                  dy="0.35em"
                  textAnchor={n.stage === 2 ? 'start' : 'end'}
                  fontSize={11.5}
                  fill={INK.text}
                  fontFamily={FONT.family}
                >
                  {n.label}
                  <tspan fill={INK.muted} fontFamily={FONT.mono} fontSize={10}> {n.value}</tspan>
                </text>
              </g>
            ))}
          </svg>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
