import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const NORMS = ['÷ max site median', '÷ pooled median'] as const

/**
 * Radar — a chemical fingerprint per site, one polygon each.
 *
 * react-graph-gallery files this under *ranking*; here it answers the
 * question the composition treemap can only half-answer: not "what is Addis
 * made of" but "on which species does Addis differ from Delhi". Each axis is
 * a species, each polygon a site's median on that species, normalised so a
 * trace metal and PM2.5 mass share a scale. Not in the estate at all —
 * matplotlib radar is painful enough that nobody drew one.
 */
export function SiteRadar({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const groupNames = meta.field_groups.map((g) => g.label)
  const [group, setGroup] = useState(groupNames.includes('Ions') ? 'Ions' : groupNames[0] ?? '')
  const [norm, setNorm] = useState<(typeof NORMS)[number]>('÷ max site median')
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const species = useMemo(
    () => (meta.field_groups.find((g) => g.label === group)?.fields ?? []).filter((f) => !f.includes('uncertainty') && !f.includes('MDL')),
    [meta.field_groups, group]
  )

  const { sites, axisMax } = useMemo(() => {
    const medians = new Map<string, Map<string, { m: number; n: number }>>()
    const pooled = new Map<string, number>()
    for (const f of species) {
      const all = rows.map((r) => r[f]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      pooled.set(f, d3.median(all) ?? 0)
    }
    const sites = meta.sites
      .map((s) => {
        const sub = rows.filter((r) => r.site === s.name)
        const m = new Map<string, { m: number; n: number }>()
        for (const f of species) {
          const vals = sub.map((r) => r[f]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
          if (vals.length >= 5) m.set(f, { m: d3.median(vals) ?? 0, n: vals.length })
        }
        medians.set(s.name, m)
        return { name: s.name, color: s.color, m }
      })
      .filter((s) => s.m.size >= 3)
    const axisMax = new Map<string, number>()
    for (const f of species) {
      const denom = norm === '÷ pooled median' ? pooled.get(f) ?? 0 : d3.max(sites, (s) => s.m.get(f)?.m ?? 0) ?? 0
      axisMax.set(f, denom > 0 ? denom : 1)
    }
    return { sites, axisMax, pooled }
  }, [rows, meta.sites, species, norm])

  const shown = sites.filter((s) => !hidden.has(s.name))
  const size = Math.min(520, Math.max(320, width - 40))
  // gutter sized from the longest species label so nothing runs off the canvas:
  // labels sit 12 px outside the rim and need ~6.6 px per character beyond that
  const labelPad = Math.min(160, 26 + 8 * Math.max(6, ...species.map((f) => f.length)))
  const R = size / 2 - labelPad
  const cx = size / 2
  const cy = size / 2
  const angle = d3.scaleBand<string>().domain(species).range([0, 2 * Math.PI])
  // with "÷ pooled median" a site can exceed 1 (it is above the pooled median);
  // let the ring scale grow to the largest ratio so nothing leaves the chart
  const rMax = norm === '÷ pooled median' ? Math.max(1.5, d3.max(shown, (s) => d3.max(species, (f) => (s.m.get(f)?.m ?? 0) / (axisMax.get(f) ?? 1)) ?? 0) ?? 1) : 1
  const radial = d3.scaleLinear().domain([0, rMax]).range([0, R])
  const rings = norm === '÷ pooled median' ? [0.5, 1, 1.5, 2].filter((v) => v <= rMax) : [0.25, 0.5, 0.75, 1]

  const polar = (f: string, v: number): [number, number] => {
    const a = (angle(f) ?? 0) - Math.PI / 2
    return [cx + radial(v) * Math.cos(a), cy + radial(v) * Math.sin(a)]
  }
  const pathFor = (s: (typeof sites)[number]) => {
    const pts = species.map((f) => polar(f, Math.min(rMax, (s.m.get(f)?.m ?? 0) / (axisMax.get(f) ?? 1))))
    return d3.line<[number, number]>().curve(d3.curveLinearClosed)(pts) ?? ''
  }

  return (
    <ChartFrame
      id="radar"
      title="Radar — each site's chemical fingerprint"
      subtitle="One polygon per site: the site median of every species in the chosen family, normalised per axis. '÷ max site median' puts the highest site at the rim on every axis; '÷ pooled median' draws the ring at 1 where a site matches the estate as a whole, so anything outside the ring is enriched."
      provenance="not in the estate · react-graph-gallery.com/radar-chart"
      controls={
        <>
          <Select label="species family" value={group} options={groupNames} onChange={setGroup} />
          <Segmented label="normalise" value={norm} options={NORMS} onChange={setNorm} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', justifyContent: 'center' }}>
        {species.length < 3 || sites.length === 0 ? (
          <Empty>Need at least 3 species with ≥5 filters per site.</Empty>
        ) : (
          <svg width={size} height={size} className="animated">
            {rings.map((v) => (
              <circle key={v} cx={cx} cy={cy} r={radial(v)} fill="none" stroke={v === 1 && norm === '÷ pooled median' ? INK.axis : INK.grid} strokeWidth={v === 1 && norm === '÷ pooled median' ? 1.4 : 1} strokeDasharray={v === 1 && norm === '÷ pooled median' ? '4 3' : undefined} />
            ))}
            {/* ring labels sit halfway between the first two spokes, so they never overprint the top axis title */}
            {rings.map((v) => {
              const half = Math.PI / Math.max(3, species.length) - Math.PI / 2
              return (
                <text key={`t${v}`} x={cx + radial(v) * Math.cos(half) + 3} y={cy + radial(v) * Math.sin(half)} dy="0.35em" fontSize={9.5} fill={INK.muted} fontFamily={FONT.mono} stroke="#fff" strokeWidth={3} paintOrder="stroke">
                  {norm === '÷ pooled median' ? `${v}×` : `${Math.round(v * 100)}%`}
                </text>
              )
            })}
            {species.map((f) => {
              const [x1, y1] = polar(f, rMax)
              const a0 = (angle(f) ?? 0) - Math.PI / 2
              const [lx, ly] = [cx + (R + 12) * Math.cos(a0), cy + (R + 12) * Math.sin(a0)]
              const a = (angle(f) ?? 0) - Math.PI / 2
              const anchor = Math.abs(Math.cos(a)) < 0.2 ? 'middle' : Math.cos(a) > 0 ? 'start' : 'end'
              return (
                <g key={f}>
                  <line x1={cx} y1={cy} x2={x1} y2={y1} stroke={INK.grid} />
                  <text x={lx} y={ly} dy="0.35em" textAnchor={anchor} fontSize={11} fill={INK.text} fontFamily={FONT.family}>
                    {f}
                  </text>
                </g>
              )
            })}
            {shown.map((s) => (
              <path key={s.name} d={pathFor(s)} fill={s.color} fillOpacity={0.14} stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
            ))}
            {shown.map((s) =>
              species.map((f) => {
                const rec = s.m.get(f)
                if (!rec) return null
                const ratio = rec.m / (axisMax.get(f) ?? 1)
                const [px, py] = polar(f, Math.min(rMax, ratio))
                return (
                  <circle
                    key={`${s.name}-${f}`}
                    cx={px} cy={py} r={3.5} fill={s.color} stroke="#fff" strokeWidth={1}
                    onMouseEnter={(e) =>
                      tip.show(e, [
                        `${s.name} · ${f}`,
                        `median = ${fmt(rec.m, 3)} µg/m³`,
                        norm === '÷ pooled median' ? `${fmt(ratio, 2)}× the pooled median` : `${Math.round(ratio * 100)} % of the highest site`,
                        `n = ${rec.n}`,
                      ])
                    }
                    onMouseLeave={tip.hide}
                  />
                )
              })
            )}
          </svg>
        )}
        {tip.node}
      </div>
      <Legend items={sites.map((s) => ({ label: s.name, color: s.color }))} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} />
    </ChartFrame>
  )
}
