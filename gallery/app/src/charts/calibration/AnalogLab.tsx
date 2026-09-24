import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import { PREPROCESSING, label } from '@/lib/labels'
import type { AnalogLab as AnalogLabData } from '@/lib/types'
import { cohortColor } from './common'

/**
 * The analog lab: does the committed spectral-analog ranking agree with
 * simpler similarity metrics? Left, a 2-D PCA of normalised spectra with the
 * pool subsample, the committed top-N highlighted and every Addis filter
 * starred — the picture of "is Addis inside the pool at all". Right, the
 * committed rank against an alternative metric's rank, with Spearman ρ; a
 * tight diagonal means the metrics would pick the same cohort.
 */
export function AnalogLab({ bySpace, cutoff }: { bySpace: Record<string, AnalogLabData>; cutoff: number }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const lg = useLegend()
  const TOP = `committed top ${cutoff}`, REST = 'rest of pool', ADDIS = 'Addis filters (★)'
  const cls = (rank: number) => (rank < cutoff ? TOP : REST)

  const spaces = Object.keys(bySpace)
  const [space, setSpace] = useState(spaces[0] ?? 'raw')
  const lab = bySpace[space]
  const metrics = lab ? lab.metrics.filter((m) => m !== 'committed' && lab.ranks[m]) : []
  const [metric, setMetric] = useState(metrics.includes('mahalanobis_pca') ? 'mahalanobis_pca' : metrics[0] ?? '')
  const m = metrics.includes(metric) ? metric : metrics[0] ?? ''

  const cols = width > 760 ? 2 : 1
  const panelW = Math.floor((width - (cols - 1) * 12) / cols)
  const S = Math.max(220, Math.min(440, panelW - MARGIN.left - MARGIN.right))

  const pca = useMemo(() => {
    if (!lab) return null
    const all = [...lab.pool_xy, ...lab.addis_xy]
    // a handful of extreme spectra (PC2 ≈ −2 against a bulk at ±0.1) used to
    // squash the whole pool into one pixel row; the axes follow the 0.5–99.5 %
    // bulk and the extremes are pinned hollow at the edge, counted below
    const q = (i: 0 | 1, p: number) => d3.quantile(all.map((d) => d[i]).sort(d3.ascending), p) ?? 0
    const x = d3.scaleLinear().domain([q(0, 0.005), q(0, 0.995)]).range([0, S]).nice()
    const y = d3.scaleLinear().domain([q(1, 0.005), q(1, 0.995)]).range([S, 0]).nice()
    // closest-point detection over the 2,500-point subsample: one Delaunay
    // and one hit rect instead of 2,500 mouse handlers (react-graph-gallery's
    // scatterplot-with-voronoi pattern)
    const cx = (p: [number, number]) => Math.max(0, Math.min(S, x(p[0])))
    const cy = (p: [number, number]) => Math.max(0, Math.min(S, y(p[1])))
    const outside = (p: [number, number]) => x(p[0]) < 0 || x(p[0]) > S || y(p[1]) < 0 || y(p[1]) > S
    const nOutside = all.filter(outside).length
    const delaunay = d3.Delaunay.from(lab.pool_xy, cx, cy)
    return { x, y, cx, cy, outside, nOutside, delaunay }
  }, [lab, S])
  const [hoverI, setHoverI] = useState<number | null>(null)
  const onPcaMove = (e: React.MouseEvent<SVGRectElement>) => {
    if (!lab || !pca) return
    const [mx, my] = d3.pointer(e)
    const i = pca.delaunay.find(mx, my)
    const p = lab.pool_xy[i]
    if (!p || !lg.show(cls(lab.sample_idx[i])) || Math.hypot(pca.cx(p) - mx, pca.cy(p) - my) > 14) { setHoverI(null); tip.hide(); return }
    setHoverI(i)
    const rank = lab.sample_idx[i]
    tip.show(e, [`pool filter, committed rank ${rank + 1}`, rank < cutoff ? `inside the top ${cutoff}` : 'rest of pool', `PC1 ${fmt(p[0], 2)} · PC2 ${fmt(p[1], 2)}`])
  }

  const rankPts = useMemo(() => {
    if (!lab || !m) return []
    const c = lab.ranks.committed
    const a = lab.ranks[m]
    return c.map((cr, i) => ({ c: cr, a: a[i] }))
  }, [lab, m])
  const rx = d3.scaleLinear().domain([0, lab?.n ?? 1]).range([0, S]).nice()
  const ry = d3.scaleLinear().domain([0, lab?.n ?? 1]).range([S, 0]).nice()

  return (
    <ChartFrame
      id="analog-lab"
      title="Analog lab — does the committed spectral-analog ranking survive other metrics?"
      subtitle={`Left: PCA of normalised spectra, pool subsample with the committed top-${cutoff} analogs in colour and every Addis filter as a star. Right: committed rank against the chosen metric's rank; Spearman ρ says how far the two selections would agree.`}
      provenance="calibration_explorer /api/analog_lab · Analogs tab"
      controls={
        <>
          {spaces.length > 1 && <Segmented label="preprocessing" value={label(PREPROCESSING, space)} options={spaces.map((s) => label(PREPROCESSING, s))} onChange={(v) => setSpace(spaces.find((s) => label(PREPROCESSING, s) === v) ?? space)} />}
          {metrics.length > 0 && <Select label="alternative metric" value={m} options={metrics} onChange={setMetric} />}
          {lab && m && <span className="control">ρ = {fmt(lab.agreement[m] ?? null, 3)} · {lab.labels[m] ?? m}</span>}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {!lab || !pca ? (
          <Empty>No analog-lab export for this preprocessing.</Empty>
        ) : (
          <div className="facet-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            <div>
              <p className="facet-title">PCA of normalised spectra (PC1 {Math.round(lab.explained[0] * 100)} %, PC2 {Math.round(lab.explained[1] * 100)} %)</p>
              <svg width={S + MARGIN.left + MARGIN.right} height={S + MARGIN.top + MARGIN.bottom}>
                <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                  <YAxis scale={pca.y} x={0} label="PC2" gridWidth={S} tickCount={5} />
                  <XAxis scale={pca.x} y={S} label="PC1" tickCount={5} />
                  {lab.pool_xy.map((p, i) => {
                    const rank = lab.sample_idx[i]
                    const inCohort = rank < cutoff
                    if (!lg.show(cls(rank))) return null
                    const hot = hoverI === i
                    const pinned = pca.outside(p)
                    return <circle key={i} cx={pca.cx(p)} cy={pca.cy(p)} r={hot ? 5 : inCohort ? 3 : 2} fill={pinned ? '#fff' : inCohort ? cohortColor('analogs') : INK.muted} fillOpacity={hot ? 1 : (inCohort ? 0.8 : 0.3) * lg.dim(cls(rank))} strokeOpacity={hot ? 1 : lg.dim(cls(rank))} stroke={hot ? INK.text : pinned ? INK.muted : 'none'} strokeWidth={hot ? 1.5 : pinned ? 1 : 0} pointerEvents="none" />
                  })}
                  {lg.show(ADDIS) && lab.addis_xy.map((p, i) => (
                    <text key={`a${i}`} x={pca.cx(p)} y={pca.cy(p)} dy="0.35em" textAnchor="middle" fontSize={11} fill={INK.text} fillOpacity={lg.dim(ADDIS)} fontFamily={FONT.family} pointerEvents="none">★</text>
                  ))}
                  <rect x={0} y={0} width={S} height={S} fill="transparent" onMouseMove={onPcaMove} onMouseLeave={() => { setHoverI(null); tip.hide() }} />
                </g>
              </svg>
            </div>
            <div>
              <p className="facet-title">committed rank vs {lab.labels[m] ?? m}</p>
              <svg width={S + MARGIN.left + MARGIN.right} height={S + MARGIN.top + MARGIN.bottom}>
                <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                  <YAxis scale={ry} x={0} label="alternative rank" gridWidth={S} tickCount={5} />
                  <XAxis scale={rx} y={S} label="committed rank" tickCount={5} />
                  <line x1={rx(0)} y1={ry(0)} x2={rx(lab.n)} y2={ry(lab.n)} stroke={INK.identity} strokeDasharray="5 4" />
                  <rect x={0} y={ry(cutoff)} width={rx(cutoff)} height={S - ry(cutoff)} fill={cohortColor('analogs')} fillOpacity={0.08} />
                  {rankPts.map((p, i) => lg.show(cls(p.c)) && (
                    <circle key={i} cx={rx(p.c)} cy={ry(p.a)} r={1.6} fill={p.c < cutoff ? cohortColor('analogs') : INK.muted} fillOpacity={(p.c < cutoff ? 0.8 : 0.3) * lg.dim(cls(p.c))} pointerEvents="none" />
                  ))}
                </g>
              </svg>
            </div>
          </div>
        )}
        <Legend items={[{ label: TOP, color: cohortColor('analogs') }, { label: REST, color: INK.muted }, { label: ADDIS, color: INK.text }]} {...lg.props} note={lab ? `pool subsample ${lab.pool_xy.length} of ${lab.n.toLocaleString()} · ranks every ${lab.rank_stride}th` : undefined} />
        {pca && pca.nOutside > 0 && <p className="chart-note">{pca.nOutside} extreme spectra fall outside the 0.5–99.5 % PCA range and are pinned hollow at the edge of the axes.</p>}
        {lab && (
          <table className="placement" style={{ maxWidth: 560 }}>
            <thead><tr><th>metric</th><th>Spearman ρ with committed</th></tr></thead>
            <tbody>
              {Object.entries(lab.agreement).map(([k, v]) => (
                <tr key={k} className="link" onClick={() => setMetric(k)}><td className="f">{lab.labels[k] ?? k}</td><td className="v">{fmt(v, 3)}</td></tr>
              ))}
            </tbody>
          </table>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
