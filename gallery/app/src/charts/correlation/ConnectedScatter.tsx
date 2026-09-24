import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { seasonsForSite } from '@/siteSeasons'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { pairDomain, type AxesOpts } from '@/lib/axes'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const STEPS = ['Month of year', 'Year'] as const

/**
 * Connected scatterplot — the census classified 31 figures as this and none
 * were built. The pair from the bar is aggregated to a median per step
 * (calendar month or year) and the steps are joined in order, so the path
 * traces how the *relationship* moves through the seasons rather than how
 * each variable moves on its own. A loop means hysteresis: the same x in
 * Belg and Kiremt sits at different y.
 *
 * Follows the bar's axes policy: with the 1:1 line on, the panel is square
 * and both axes share a domain, so a path that runs along the diagonal is
 * a site where the two methods agree all year.
 */
export function ConnectedScatter({ rows, meta, xField, yField, axes }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; axes: AxesOpts }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const [step, setStep] = useState<(typeof STEPS)[number]>('Month of year')
  const [labels, setLabels] = useState(true)
  const { hidden, setHidden, dim, hover, setHover } = useLegend()
  const minN = 4
  const opts: AxesOpts = { ...axes, log: false, mode: axes.mode === 'Show intercept' ? 'Data' : axes.mode }

  const series = useMemo(() => {
    const keyOf = (r: FilterRow) => (step === 'Month of year' ? r.month ?? 0 : r.year ?? 0)
    return meta.sites
      .filter((s) => !hidden.has(s.name))
      .map((s) => {
        const byKey = new Map<number, { xs: number[]; ys: number[] }>()
        for (const r of rows) {
          if (r.site !== s.name) continue
          const x = r[xField]
          const y = r[yField]
          if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) continue
          const k = keyOf(r)
          if (!k) continue
          const g = byKey.get(k) ?? { xs: [], ys: [] }
          g.xs.push(x)
          g.ys.push(y)
          byKey.set(k, g)
        }
        const pts = [...byKey.entries()]
          .filter(([, g]) => g.xs.length >= minN)
          .sort((a, b) => a[0] - b[0])
          .map(([k, g]) => ({ key: k, x: d3.median(g.xs) ?? 0, y: d3.median(g.ys) ?? 0, n: g.xs.length }))
        return { name: s.name, color: s.color, pts }
      })
      .filter((s) => s.pts.length >= 2)
  }, [rows, meta.sites, xField, yField, step, hidden])

  const all = series.flatMap((s) => s.pts)
  const MAX_SQUARE = 560
  const availW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerW = Math.min(availW, MAX_SQUARE)
  const innerH = innerW
  const height = innerH + MARGIN.top + MARGIN.bottom

  const domain = pairDomain(all.map((p) => p.x), all.map((p) => p.y), opts)
  const x = d3.scaleLinear().domain(domain.x).range([0, innerW]).nice()
  const y = d3.scaleLinear().domain(domain.y).range([innerH, 0]).nice()
  const [d0, d1] = x.domain() as [number, number]
  const line = d3.line<{ x: number; y: number }>().x((p) => x(p.x)).y((p) => y(p.y)).curve(d3.curveCatmullRom.alpha(0.6))
  const label = (k: number) => (step === 'Month of year' ? MONTHS[k - 1] : String(k))

  // Greedy label thinning: a label is drawn only if it is at least 14 px from
  // every label already placed, across all series. First and last steps of a
  // series always get theirs, so the direction of travel stays readable.
  const placed: [number, number][] = []
  const labelOk = (px: number, py: number, force: boolean) => {
    if (!force && placed.some(([qx, qy]) => Math.abs(qx - px) < 34 && Math.abs(qy - py) < 14)) return false
    placed.push([px, py])
    return true
  }
  const seasonOf = (m: number, site: string) => seasonsForSite(meta, site).find((s) => s.months.includes(m))?.name

  return (
    <ChartFrame
      id="connected"
      title="Connected scatter — the pair's path through the year"
      subtitle="Median x and median y per step, joined in time order, one path per site. A closed loop is hysteresis: the same loading in the dry season sits at a different y than in the rains. With the 1:1 line on, a path hugging the diagonal is a site where the two methods agree all year. Steps with fewer than 4 filters are skipped."
      provenance="31 figures classified as connected scatter, none built until now · react-graph-gallery.com/connected-scatter-plot"
      controls={
        <>
          <Segmented label="step" value={step} options={STEPS} onChange={setStep} />
          <Toggle label="labels" checked={labels} onChange={setLabels} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap centered">
        {all.length < 2 ? (
          <Empty>Not enough steps with ≥{minN} filters carrying both fields.</Empty>
        ) : (
          <svg width={innerW + MARGIN.left + MARGIN.right} height={height}>
            <defs>
              {series.map((s) => (
                <marker key={s.name} id={`arrow-${s.name.replace(/\s/g, '')}`} viewBox="0 0 10 10" refX={8} refY={5} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={s.color} />
                </marker>
              ))}
            </defs>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={withUnit(yField, meta.field_units)} gridWidth={innerW} />
              <XAxis scale={x} y={innerH} label={withUnit(xField, meta.field_units)} />
              {opts.identity && (
                <line x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} stroke={INK.identity} strokeWidth={1.5} strokeDasharray="5 4" />
              )}
              {series.map((s) => (
                <g key={s.name} opacity={dim(s.name)}>
                  <path
                    d={line(s.pts) ?? ''}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={2}
                    strokeOpacity={0.75}
                    markerEnd={`url(#arrow-${s.name.replace(/\s/g, '')})`}
                  />
                  {s.pts.map((p, i) => (
                    <g key={p.key}>
                      <circle
                        cx={x(p.x)}
                        cy={y(p.y)}
                        r={i === 0 ? 5.5 : 4}
                        fill={i === 0 ? '#fff' : s.color}
                        stroke={s.color}
                        strokeWidth={i === 0 ? 2.2 : 1}
                        onMouseEnter={(e) =>
                          tip.show(e, [
                            `${s.name} · ${label(p.key)}`,
                            `${xField} median = ${fmt(p.x)}`,
                            `${yField} median = ${fmt(p.y)}`,
                            `ratio y/x = ${fmt(p.x !== 0 ? p.y / p.x : null, 3)}`,
                            `n = ${p.n}`,
                            ...(step === 'Month of year' ? [seasonOf(p.key, s.name) ?? ''] : []),
                          ])
                        }
                        onMouseLeave={tip.hide}
                      />
                      {labels && labelOk(x(p.x) + 7, y(p.y) - 6, i === 0 || i === s.pts.length - 1) && (
                        <text x={x(p.x) + 7} y={y(p.y) - 6} fontSize={10} fill={s.color} fontFamily={FONT.mono} pointerEvents="none" stroke="#fff" strokeWidth={3} paintOrder="stroke">
                          {label(p.key)}
                        </text>
                      )}
                    </g>
                  ))}
                </g>
              ))}
            </g>
          </svg>
        )}
        <Legend
          items={meta.sites.map((s) => ({ label: s.name, color: s.color, shape: 'line' as const }))}
          hidden={hidden}
          onToggle={(l) => setHidden((h) => toggleIn(h, l))}
          onHover={setHover}
          highlighted={hover}
          note={`hollow marker = first step · arrow = last · crowded labels thinned, hover for the rest${opts.identity ? ' · grey dashed = 1:1' : ''}`}
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
