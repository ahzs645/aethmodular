import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { MetaFile, PmfFile, PmfRow } from '@/lib/types'

const MODES = ['Relative (%)', 'Absolute (µg/m³)'] as const

/**
 * Source apportionment by season, plus which source wins each filter.
 *
 * `dominant_source` is only meaningful on normalised fractions — on the raw GF
 * columns the dominant fraction tops out near 0.24 and no filter ever crosses
 * a 30 % threshold. After normalisation the mean is ~46 %, which is what the
 * threshold line here is drawn against.
 */
export function SourceSeasonality({ pmf, rows: subsetRows, meta }: { pmf: PmfFile; rows: PmfRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 400

  const [mode, setMode] = useState<(typeof MODES)[number]>('Relative (%)')
  const [threshold, setThreshold] = useState('30')

  const labels = pmf.sources.map((s) => s.label)
  const keyOfLabel = useMemo(() => new Map(pmf.sources.map((s) => [s.label, s.key])), [pmf.sources])
  const colorOf = useMemo(() => new Map(pmf.sources.map((s) => [s.label, s.color])), [pmf.sources])
  const relative = mode === 'Relative (%)'

  const groups = useMemo(() => {
    const names = meta.seasons.map((s) => s.name)
    return names
      .map((name) => {
        const sub = subsetRows.filter((r) => r.season === name)
        if (!sub.length) return null
        const means: Record<string, number> = {}
        for (const l of labels) {
          const vals = sub.map((r) => (relative ? r.fraction[l] : r.ugm3[l])).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
          means[l] = vals.length ? (d3.mean(vals) ?? 0) : 0
        }
        return { name, n: sub.length, means, rows: sub }
      })
      .filter((g): g is NonNullable<typeof g> => !!g)
  }, [subsetRows, meta.seasons, labels, relative])

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom

  const x = d3.scaleBand<string>().domain(groups.map((g) => g.name)).range([0, innerW]).padding(0.34)
  const yMax = d3.max(groups, (g) => d3.sum(labels, (l) => g.means[l])) ?? 1
  const y = d3.scaleLinear().domain([0, yMax * 1.04]).range([innerH, 0]).nice()

  const thr = Number(threshold) / 100
  const crossing = subsetRows.filter((r) => (r.dominant_fraction ?? 0) >= thr).length

  return (
    <ChartFrame
      id="sourceseason"
      title="Source apportionment by season"
      subtitle={`Mean source mix per Ethiopian season, stacked. ${crossing} of ${subsetRows.length} filters have a single source above ${threshold} % — on the un-normalised GF columns that count would be zero, which is the trap AGENTS.md warns about.`}
      provenance={`normalised via normalize_gf_fractions · calendar: ${meta.season_convention}`}
      controls={
        <>
          <Segmented label="scale" value={mode} options={MODES} onChange={setMode} />
          <Select label="dominance threshold" value={threshold} options={['20', '30', '40', '50', '60']} onChange={setThreshold} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {!groups.length ? (
          <Empty>No PMF filters in this subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={relative ? 'mean share' : 'mean µg/m³'} gridWidth={innerW} format={(v: number) => (relative ? `${Math.round(v * 100)}%` : String(v))} />
              <XAxis scale={x} y={innerH} />
              {groups.map((g) => {
                let acc = 0
                return (
                  <g key={g.name}>
                    {labels.map((l) => {
                      const v = g.means[l]
                      const y0 = y(acc)
                      acc += v
                      const y1 = y(acc)
                      const dominant = g.rows.filter((r) => r.dominant_source === keyOfLabel.get(l)).length
                      return (
                        <rect
                          key={l}
                          x={x(g.name) ?? 0} y={y1} width={x.bandwidth()} height={Math.max(0, y0 - y1)}
                          fill={colorOf.get(l) ?? INK.muted} fillOpacity={0.88} stroke="#fff" strokeWidth={0.8}
                          onMouseEnter={(e) => tip.show(e, [`${l} · ${g.name}`, relative ? `mean share ${(v * 100).toFixed(1)} %` : `mean ${fmt(v, 2)} µg/m³`, `dominant on ${dominant} of ${g.n} filters`])}
                          onMouseLeave={tip.hide}
                        />
                      )
                    })}
                    <text x={(x(g.name) ?? 0) + x.bandwidth() / 2} y={innerH + 30} textAnchor="middle" fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>
                      n={g.n}
                    </text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <Legend items={pmf.sources.map((s) => ({ label: s.label, color: s.color, shape: 'square' as const }))} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
