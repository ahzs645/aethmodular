import { useId, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { useAddisStatusBySample } from '@/lib/filterSpectrum'
import { IMPROVE_GREY, INK, MARGIN, TRACE, themed } from '@/lib/theme'
import { calendarOf, groupColor, groupLabel, members, siteName, siteWhere, type SimilarityMeta, type Traces } from '@/lib/similarity'

const TOP_GREY = IMPROVE_GREY
const groupGrey = () => themed('#39404d', '#b4bcc6')

export function SpectraTraces({ meta, traces, target, topFilters, compare, mask, hoverFilter, pinned, onPin, method, controls,
  targetFilters, targetLabel, regionParam = 'sregion', scoreLabel = 'mean r', provenance, id, source,
  pinInstructions = 'Hover a trace to identify it; click a trace or filter-table row to pin it and open its details in the sidebar.' }: {
  meta: SimilarityMeta; traces: Traces | null; target: number; topFilters: { filter: number; r: number; rank?: number }[]; compare: number | null
  mask: string; hoverFilter: number | null; pinned: number | null; onPin: (f: number | null) => void; method: string; controls: React.ReactNode
  targetFilters?: number[]; targetLabel?: string; regionParam?: string; scoreLabel?: string; provenance?: string; id?: string; source?: string
  pinInstructions?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [hoverTrace, setHoverTrace] = useState<number | null>(null)
  const addisStatus = useAddisStatusBySample()
  const [regions, setRegions] = useSearchParam<'gaps' | 'shaded'>(regionParam, 'gaps', ['gaps', 'shaded'])
  const [bg, setBg] = useSearchParam<'most similar' | 'selected group'>('sbg', 'most similar', ['most similar', 'selected group'])
  const height = 420
  const innerW = Math.max(260, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  // a mask this export lacks keeps every channel rather than failing
  const keep = meta.masks.find((m) => m.key === mask)?.bins ?? meta.bin_wn.map(() => true)
  const ignoresAny = keep.some((k) => !k)
  const wn = meta.bin_wn
  const clip = useId()
  const targetMembers = useMemo(() => targetFilters ?? members(meta, target), [targetFilters, meta, target])
  const targetName = targetLabel ?? groupLabel(meta, target)
  const useGroup = compare != null && bg === 'selected group'
  // stable legend labels (counts go in `detail`) so a hidden series stays hidden as N changes
  const lg = useLegend()
  const BACK = useGroup ? groupLabel(meta, compare!) : 'most similar filters, other sites'
  const wholeSite = meta.groups[target].season == null
  const cal = calendarOf(meta, meta.groups[target].site)
  // calendar order, "no date" last
  const seasonOrder = (label: string) => { const i = cal.findIndex((c) => label.endsWith(c.name)); return i < 0 ? cal.length : i }
  const tKey = (f: number) => !wholeSite ? groupLabel(meta, target)
    // grouped under the site's name, as the gallery's other legends are; filters the frozen ledger has no date for
    // fall in no season and are listed as "no date"
    : `${siteName(meta.sites[meta.groups[target].site])} · ${meta.filters.season[f] >= 0 ? cal[meta.filters.season[f]].name : 'no date'}`

  const drawn = useMemo(() => {
    if (!traces) return null
    const tgt = targetMembers.map((f) => ({ f, v: traces.of(f) }))
    const back = (useGroup ? members(meta, compare!) : topFilters.map((t) => t.filter)).map((f) => ({ f, v: traces.of(f) }))
    const lit = hoverTrace ?? hoverFilter
    const hover = lit != null && lit !== pinned ? { f: lit, v: traces.of(lit) } : null
    const pin = pinned != null ? { f: pinned, v: traces.of(pinned) } : null
    return { tgt, back, hover, pin }
  }, [traces, meta, targetMembers, topFilters, useGroup, compare, hoverFilter, hoverTrace, pinned])
  const rankOf = useMemo(() => new Map(topFilters.map((t, i) => [t.filter, { rank: t.rank ?? i + 1, r: t.r }])), [topFilters])
  const targetSet = useMemo(() => new Set(targetMembers), [targetMembers])
  const vis = useMemo(() => drawn && {
    tgt: drawn.tgt.filter((t) => lg.show(tKey(t.f))),
    back: lg.show(BACK) ? drawn.back : [],
    // tKey and lg.show derive from target and lg.hidden
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawn, lg.hidden, BACK, target])

  const x = d3.scaleLinear().domain([wn[0], wn[wn.length - 1]]).range([0, innerW])
  const y = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (const t of [...(vis?.tgt ?? []), ...(vis?.back ?? [])]) {
      for (let i = 0; i < t.v.length; i++) {
        // scale to the scored channels in both modes; shaded curves are clipped
        if (!keep[i]) continue
        if (t.v[i] < lo) lo = t.v[i]
        if (t.v[i] > hi) hi = t.v[i]
      }
    }
    return d3.scaleLinear().domain(Number.isFinite(lo) ? [lo, hi] : [0, 1]).range([innerH, 0]).nice()
  }, [vis, keep, innerH])
  const line = d3.line<number>().defined((_, i) => regions === 'shaded' || keep[i]).x((_, i) => x(wn[i])).y((v) => y(v))
  const ignored = useMemo(() => {
    const runs: [number, number][] = []
    let start = -1
    keep.forEach((k, i) => {
      if (!k && start < 0) start = i
      if ((k || i === keep.length - 1) && start >= 0) { runs.push([start, k ? i - 1 : i]); start = -1 }
    })
    return runs
  }, [keep])

  /** The trace under the pointer: nearest in y at the pointer's wavenumber, within 6 px. */
  const nearest = (mx: number, my: number): number | null => {
    if (!vis) return null
    const step = wn[1] - wn[0]
    const i = Math.max(0, Math.min(wn.length - 1, Math.round((x.invert(mx) - wn[0]) / step)))
    if (regions === 'gaps' && !keep[i]) return null
    let best: number | null = null
    let dist = 6
    for (const t of [...vis.back, ...vis.tgt]) {
      const d = Math.abs(y(t.v[i]) - my)
      if (d <= dist) { dist = d; best = t.f }
    }
    return best
  }
  const describe = (f: number): string[] => {
    const s = meta.sites[meta.filters.site[f]]
    const k = meta.filters.season[f]
    const hit = rankOf.get(f)
    return [
      meta.filters.sample_id[f],
      `${s.site} · ${siteWhere(s)}`,
      `${meta.filters.date[f] ?? 'no date'} · ${k < 0 ? 'no season' : calendarOf(meta, meta.filters.site[f])[k].name}`,
      targetSet.has(f) ? 'target filter' : hit ? `#${hit.rank} most similar · ${scoreLabel} ${hit.r.toFixed(4)}` : `member of ${groupLabel(meta, compare ?? target)}`,
      ...(addisStatus.get(meta.filters.sample_id[f]) && !addisStatus.get(meta.filters.sample_id[f])!.in_test_set
        ? [`not in the Addis test set: ${addisStatus.get(meta.filters.sample_id[f])!.note.replace(/^in the spectra run but not in the Addis test set: /, '')}`] : []),
    ]
  }
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const [mx, my] = d3.pointer(e)
    const f = nearest(mx, my)
    setHoverTrace(f)
    if (f == null) tip.hide()
    else tip.show(e, [...describe(f), pinned === f ? 'click to unpin' : 'click to pin'])
  }

  const tColor = (f: number) => wholeSite ? (meta.filters.season[f] >= 0 ? cal[meta.filters.season[f]].color : INK.muted) : groupColor(meta, target, INK.accent)
  const backLabel = useGroup ? `${groupLabel(meta, compare!)} (${drawn?.back.length ?? 0})` : `${topFilters.length} most similar filters from other sites`
  const tLabel = `${targetName} (${drawn?.tgt.length ?? 0})`
  return (
    <ChartFrame
      id={id} source={source}
      title="Spectra — target filters against their closest matches"
      subtitle={`Every ${targetName} filter in colour, drawn over ${useGroup ? `every ${groupLabel(meta, compare!)} filter in dark grey` : `its ${topFilters.length} most similar filters from other sites in grey`}. ${method} baseline-corrected absorbance. Gaps or shading mark the channels the selection mask leaves out of the score. The y-axis spans the scored channels only.`}
      provenance={provenance ?? 'export_similarity.py · traces_{method}.bin (8-channel means) and top_{method}_{mask}.bin'}
      controls={<>
        {/* the page's controls shape the most-similar set; a grey selected group does not use them */}
        {!useGroup && controls}
        {compare != null && <Segmented label="Grey" value={bg} options={['most similar', 'selected group'] as const} onChange={setBg} />}
        {ignoresAny && <Segmented label="Ignored channels" value={regions} options={['gaps', 'shaded'] as const} onChange={setRegions} />}
      </>}
    >
      <div ref={ref} className="chart-wrap">
        {!drawn ? <Empty>Loading spectra…</Empty> : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {regions === 'shaded' && ignored.map(([a, b]) => (
                <rect key={a} x={x(wn[a])} width={Math.max(0, x(wn[b]) - x(wn[a]))} y={0} height={innerH} fill={themed('#efeeea', '#2a2a27')} />
              ))}
              <YAxis scale={y} x={0} label="Baseline-corrected absorbance" gridWidth={innerW} tickCount={6} />
              <XAxis scale={x} y={innerH} label="Wavenumber (cm⁻¹)" tickCount={8} />
              <clipPath id={clip}><rect width={innerW} height={innerH} /></clipPath>
              <g clipPath={`url(#${clip})`}>
              {vis!.back.map((t) => <path key={`b${t.f}`} d={line(Array.from(t.v)) ?? ''} fill="none" stroke={useGroup ? groupGrey() : TOP_GREY} strokeOpacity={(useGroup ? 0.3 : 0.22) * lg.dim(BACK, 0.2)} strokeWidth={0.7} />)}
              {vis!.tgt.map((t) => <path key={`t${t.f}`} d={line(Array.from(t.v)) ?? ''} fill="none" stroke={tColor(t.f)} strokeOpacity={0.55 * lg.dim(tKey(t.f), 0.12)} strokeWidth={0.8} />)}
              {drawn.pin && <path d={line(Array.from(drawn.pin.v)) ?? ''} fill="none" {...TRACE.pinned} />}
              {drawn.hover && <path d={line(Array.from(drawn.hover.v)) ?? ''} fill="none" {...TRACE.hovered} />}
              </g>
              <rect x={0} y={0} width={innerW} height={innerH} fill="transparent" style={{ cursor: hoverTrace != null ? 'pointer' : 'crosshair' }}
                onMouseMove={onMove} onMouseLeave={() => { setHoverTrace(null); tip.hide() }}
                onClick={() => { if (hoverTrace != null) onPin(pinned === hoverTrace ? null : hoverTrace) }} />
            </g>
          </svg>
        )}
        <Legend {...lg.props} items={[
          { label: BACK, color: useGroup ? groupGrey() : TOP_GREY, shape: 'line', detail: `n=${drawn?.back.length ?? 0}` },
          ...[...d3.rollup(drawn?.tgt ?? [], (v) => v.length, (t) => tKey(t.f))]
            .sort((p, q) => seasonOrder(p[0]) - seasonOrder(q[0]))
            .map(([label, n]) => ({
            label, color: tColor((drawn!.tgt.find((t) => tKey(t.f) === label))!.f), shape: 'line' as const, detail: `n=${n}`,
          })),
        ]} />
        {pinned != null && (
          <p style={{ fontSize: 13, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <span><strong>Pinned</strong> {describe(pinned).join(' · ')}</span>
            <button type="button" className="btn quiet" onClick={() => onPin(null)}>Unpin</button>
          </p>
        )}
        <Note>{tLabel}. {backLabel}. {pinInstructions}</Note>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
