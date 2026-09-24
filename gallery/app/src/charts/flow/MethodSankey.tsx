import { useMemo, useRef, useState } from 'react'
import { sankey, sankeyLinkHorizontal, sankeyLeft } from 'd3-sankey'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const STAGE3 = ['HIPS coverage', 'Season'] as const
const FEED_BOTH = 'FTIR EC: both exports'
const FEED_INHOUSE = 'FTIR EC: in-house only'
const FEED_CHEMSPEC = 'FTIR EC: ChemSpec only'
const FEED_NONE = 'no FTIR EC'
const HIPS_YES = 'HIPS Fabs'
const HIPS_NO = 'no HIPS Fabs'

type N = { id: string; label: string; color: string; stage: number }
type L = { source: string; target: string; value: number }

/**
 * Sankey — how filters reach a reported EC.
 *
 * The estate has zero flow figures in 827, and the census's flow/README
 * describes exactly this chart: filters → analytical method → …, making
 * visible which FTIR EC export contains a filter. The in-house and public
 * ChemSpec EC columns are the same analytical product, not two methods.
 * The third stage is a choice: HIPS coverage or season.
 *
 * "Only filters with both FTIR EC and HIPS" (default on, Ann 23 Sep 2026)
 * restricts the flow to the filters that can enter an FTIR-vs-HIPS
 * comparison. The count line under the controls states every total the
 * reader might add up (subset, with FTIR EC, with HIPS, with both, shown),
 * each defined, so the node numbers can be reconciled.
 */
export function MethodSankey({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [stage3, setStage3] = useState<(typeof STAGE3)[number]>('HIPS coverage')
  const [hoverNode, setHoverNode] = useState<string | null>(null)
  const [bothOnly, setBothOnly] = useState(true)
  const height = 460

  const has = (r: FilterRow, f: string) => typeof r[f] === 'number' && Number.isFinite(r[f] as number)
  const hasFtir = (r: FilterRow) => has(r, 'EC (FTIR)') || has(r, 'EC (ChemSpec FTIR)')
  const hasHips = (r: FilterRow) => has(r, 'HIPS Fabs')

  // every count the chart shows, each with one definition, so the totals reconcile
  const counts = useMemo(() => {
    let ftir = 0, hips = 0, both = 0
    for (const r of rows) {
      const f = hasFtir(r), h = hasHips(r)
      if (f) ftir++
      if (h) hips++
      if (f && h) both++
    }
    return { subset: rows.length, ftir, hips, both, noFtir: rows.length - ftir }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows])
  const shownRows = useMemo(
    () => (bothOnly ? rows.filter((r) => hasFtir(r) && hasHips(r)) : rows),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, bothOnly]
  )

  const graph = useMemo(() => {
    const carbon = (r: FilterRow) => {
      const f = has(r, 'EC (FTIR)')
      const t = has(r, 'EC (ChemSpec FTIR)')
      return f && t ? FEED_BOTH : f ? FEED_INHOUSE : t ? FEED_CHEMSPEC : FEED_NONE
    }
    const third = (r: FilterRow) => (stage3 === 'HIPS coverage' ? (hasHips(r) ? HIPS_YES : HIPS_NO) : r.season)

    const carbonOrder = [FEED_BOTH, FEED_INHOUSE, FEED_CHEMSPEC, FEED_NONE]
    const carbonColor: Record<string, string> = { [FEED_BOTH]: INK.accent, [FEED_INHOUSE]: '#6baed6', [FEED_CHEMSPEC]: '#9e9ac8', [FEED_NONE]: INK.neutral }
    const thirdOrder = stage3 === 'HIPS coverage' ? [HIPS_YES, HIPS_NO] : meta.seasons.map((s) => s.name)
    const thirdColor = (k: string) =>
      stage3 === 'HIPS coverage' ? (k === HIPS_YES ? INK.deming : INK.neutral) : meta.seasons.find((s) => s.name === k)?.color ?? INK.muted

    const nodes: N[] = [
      ...meta.sites.filter((s) => shownRows.some((r) => r.site === s.name)).map((s) => ({ id: `s:${s.name}`, label: s.name, color: s.color, stage: 0 })),
      ...carbonOrder.map((c) => ({ id: `c:${c}`, label: c, color: carbonColor[c], stage: 1 })),
      ...thirdOrder.map((t) => ({ id: `t:${t}`, label: t, color: thirdColor(t), stage: 2 })),
    ]
    const count = new Map<string, number>()
    for (const r of shownRows) {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shownRows, meta, stage3])

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

  const total = shownRows.length
  const touches = (l: any) => hoverNode === null || l.source.id === hoverNode || l.target.id === hoverNode

  return (
    <ChartFrame
      id="sankey"
      title="Sankey — which filters reach which measurement"
      subtitle="Each filter flows from its site through availability in the in-house FTIR EC export, the public ChemSpec FTIR EC export, both, or neither, then to HIPS coverage or season. Every number is a count of filters. By default only filters with both an FTIR EC value (either export) and HIPS Fabs are shown, the set that can enter an FTIR-vs-HIPS comparison; untick the box to see the whole subset. The two EC feeds report the same FTIR product; none of these SPARTAN filters has an independent TOR EC result. Hover a node to isolate its flows."
      provenance="zero flow figures in the estate; described in gallery/app/src/charts/flow/README.md · react-graph-gallery.com/sankey-diagram"
      controls={
        <>
          <Segmented label="third stage" value={stage3} options={STAGE3} onChange={setStage3} />
          <Toggle label="only filters with both FTIR EC and HIPS" checked={bothOnly} onChange={setBothOnly} title="Drop filters missing an FTIR EC value (in both exports) or missing HIPS Fabs" />
          <span className="control">{total} filters shown</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        <p className="chart-note" style={{ marginTop: 0 }}>
          Filters in the current subset: <strong>{counts.subset}</strong> · with an FTIR EC value (in-house or ChemSpec export): <strong>{counts.ftir}</strong>
          {' '}(without: {counts.noFtir}) · with HIPS Fabs: <strong>{counts.hips}</strong> · with both FTIR EC and HIPS Fabs: <strong>{counts.both}</strong>
          {' '}· drawn below: <strong>{total}</strong>{bothOnly ? ' (the both set)' : ' (the whole subset)'}. Node and link numbers are filter counts within what is drawn.
        </p>
        {!layout ? (
          <Empty>{bothOnly ? 'No filter in this subset has both FTIR EC and HIPS Fabs.' : 'No filters in this subset.'}</Empty>
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
                      `${((l.value / total) * 100).toFixed(1)} % of the ${total} filters drawn`,
                    ])
                  }
                  onMouseLeave={tip.hide}
                />
              ))}
            </g>
            {layout.nodes.map((n: any) => (
              <g
                key={n.id}
                onMouseEnter={(e) => { setHoverNode(n.id); tip.show(e, [n.label, `${n.value} filters`, `${((n.value / total) * 100).toFixed(1)} % of the ${total} filters drawn`]) }}
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
