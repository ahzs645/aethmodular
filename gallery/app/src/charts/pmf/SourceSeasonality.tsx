import { useMemo, useRef, useState } from 'react'
import { seasonOnly } from '@/lib/seasons'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { MetaFile, PmfFile, PmfRow } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { PAPER_COLORS, PAPER_ORDER, PAPER_SEASONS, paperSeasonOf } from '@/lib/pmfPaper'

const MODES = ['Relative (%)', 'Absolute (µg/m³)'] as const

/**
 * Source apportionment by season, plus which source wins each filter.
 *
 * `dominant_source` is only meaningful on normalised fractions — on the raw GF
 * columns the dominant fraction tops out near 0.24 and no filter ever crosses
 * a 30 % threshold. After normalisation the mean is ~46 %; the threshold control sets
 * the share used for the dominated-filter count in the subtitle.
 */
export function SourceSeasonality({ pmf, rows: subsetRows, meta }: { pmf: PmfFile; rows: PmfRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = useFitHeight() ?? 400

  const [mode, setMode] = useState<(typeof MODES)[number]>('Relative (%)')
  const [threshold, setThreshold] = useState('30')
  // as on the stacked chart: the sidebar's calendar or the PMF paper's four seasons; gallery or paper colours
  const [calendar, setCalendar] = useState<'Season calendar' | 'Paper (four seasons)'>('Season calendar')
  const [palette, setPalette] = useState<'Gallery' | 'Paper'>('Gallery')
  const paperSeasons = calendar === 'Paper (four seasons)', paper = palette === 'Paper'
  const lg = useLegend()

  const ordered = paper ? [...pmf.sources].sort((a, b) => PAPER_ORDER.indexOf(a.label) - PAPER_ORDER.indexOf(b.label)) : pmf.sources
  const labels = ordered.map((s) => s.label)
  const keyOfLabel = useMemo(() => new Map(pmf.sources.map((s) => [s.label, s.key])), [pmf.sources])
  const colorOf = useMemo(() => new Map(pmf.sources.map((s) => [s.label, paper ? PAPER_COLORS[s.label] ?? s.color : s.color])), [pmf.sources, paper])
  const relative = mode === 'Relative (%)'

  const groups = useMemo(() => {
    const names = paperSeasons ? PAPER_SEASONS.map((s) => s.name) : meta.seasons.map((s) => s.name)
    return names
      .map((name) => {
        const sub = subsetRows.filter((r) => (paperSeasons ? paperSeasonOf(r.date)?.name : r.season) === name)
        if (!sub.length) return null
        const means: Record<string, number> = {}
        for (const l of labels) {
          const vals = sub.map((r) => (relative ? r.fraction[l] : r.ugm3[l])).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
          means[l] = vals.length ? (d3.mean(vals) ?? 0) : 0
        }
        return { name, n: sub.length, means, rows: sub }
      })
      .filter((g): g is NonNullable<typeof g> => !!g)
  }, [subsetRows, meta.seasons, labels, relative, paperSeasons])

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom

  const x = d3.scaleBand<string>().domain(groups.map((g) => g.name)).range([0, innerW]).padding(0.34)
  const shown = labels.filter(lg.show)
  const yMax = d3.max(groups, (g) => d3.sum(shown, (l) => g.means[l])) || 1
  const y = d3.scaleLinear().domain([0, yMax * 1.04]).range([innerH, 0]).nice()

  const thr = Number(threshold) / 100
  const crossing = subsetRows.filter((r) => (r.dominant_fraction ?? 0) >= thr).length

  return (
    <ChartFrame
      id="sourceseason"
      title="Source apportionment by season"
      subtitle={`Mean source mix per selected season, stacked. ${crossing} of ${subsetRows.length} filters have a single source above ${threshold} % — on the un-normalised GF columns that count would be zero, which the analysis notes explain.`}
      provenance={`normalised via normalize_gf_fractions · calendar: ${meta.season_convention}`}
      controls={
        <>
          <Segmented label="Scale" value={mode} options={MODES} onChange={setMode} />
          {/* the share one source needs before a filter counts as dominated: counted under each bar and in the note below */}
          <Select label="Count filters dominated above" value={threshold} options={['20', '30', '40', '50', '60']} optionLabel={(o) => `${o} %`} onChange={setThreshold} />
          <Select label="Seasons" value={calendar} options={['Season calendar', 'Paper (four seasons)']} onChange={(v) => setCalendar(v as typeof calendar)}
            title="The sidebar's season calendar, or the PMF paper's four seasons (First Dry, Less Rainy, More Rainy, Second Dry)" />
          <Segmented label="Colours" value={palette} options={['Gallery', 'Paper'] as const} onChange={setPalette} title="Paper: the PMF paper's source colours and stacking order" />
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
                    {shown.map((l) => {
                      const v = g.means[l]
                      const y0 = y(acc)
                      acc += v
                      const y1 = y(acc)
                      const dominant = g.rows.filter((r) => r.dominant_source === keyOfLabel.get(l)).length
                      return (
                        <rect
                          key={l}
                          x={x(g.name) ?? 0} y={y1} width={x.bandwidth()} height={Math.max(0, y0 - y1)}
                          fill={colorOf.get(l) ?? INK.muted} fillOpacity={0.88 * lg.dim(l)} stroke="#fff" strokeWidth={0.8}
                          onMouseEnter={(e) => tip.show(e, [`${l} · ${seasonOnly(g.name)}`, relative ? `mean share ${(v * 100).toFixed(1)} %` : `mean ${fmt(v, 2)} µg/m³`, `dominant on ${dominant} of ${g.n} filters`, `above ${threshold} % on ${g.rows.filter((r) => (r.fraction[l] ?? 0) >= thr).length} of ${g.n}`])}
                          onMouseLeave={tip.hide}
                        />
                      )
                    })}
                    <text x={(x(g.name) ?? 0) + x.bandwidth() / 2} y={innerH + 30} textAnchor="middle" fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>
                      n={g.n} · {g.rows.filter((r) => (r.dominant_fraction ?? 0) >= thr).length} above {threshold} %
                    </text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <Legend items={ordered.map((s) => ({ label: s.label, color: colorOf.get(s.label) ?? s.color, shape: 'square' as const }))} {...lg.props} />
        <p className="chart-note" style={{ display: 'block' }}>{crossing} of {subsetRows.length} filters have one source above {threshold} % of the day's PMF total (the count under each bar is that season's).</p>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
