import { useEffect, useMemo, useRef, useState } from 'react'
import { CountPicker, clampCount } from '@/components/Pickers'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { SpectraTraces } from '@/components/SimilaritySpectra'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { downloadCsv } from '@/lib/export'
import { useSearchParam } from '@/lib/url'
import { baseFilterId, useHighlight, type FilterRecord } from '@/lib/highlight'
import { FONT, INK } from '@/lib/theme'
import { f3 } from '@/lib/format'
import { IntervalStrip } from '@/components/IntervalStrip'
import { HeatCell } from '@/components/HeatCell'
import { ColorLegend } from '@/components/ColorLegend'
import { ANALYSIS_CALENDAR_NOTE } from '@/lib/seasons'
import { PREPROCESSING } from '@/lib/labels'
import {
  PairMatrix, loadSimilarityMeta, TopLists, Traces, calendarOf, fetchBinary, groupColor, groupIndex, groupKey, groupLabel, siteName, siteWhere,
  type SimilarityMeta,
} from '@/lib/similarity'

const TOP_N = ['50', '100', '250', '500'] as const
/** the stored list holds each target's 500 most similar filters, so any count up to that can be drawn */
const TOP_MAX = 500
const MIN_N = ['1', '5', '10', '20'] as const
const SHOW = ['15', '30', '60'] as const
const TEST_COLOR = '#9a6b2f'
const splitName = (s: string) => (s === 'test' ? 'outer test site' : s === 'external' ? 'Addis target' : 'training site')

/** Fetch and decode one exported binary; `null` until the named file has arrived. */
function useBinary<T>(name: string | null, decode: (b: ArrayBuffer) => T): { value: T | null; error: string } {
  const [state, setState] = useState<{ name: string; value: T } | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    if (!name) return
    let live = true
    setError('')
    fetchBinary(name)
      .then((b) => { if (live) setState({ name, value: decode(b) }) })
      .catch((e) => { if (live) setError(String(e)) })
    return () => { live = false }
    // decode is derived from the same inputs as name
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name])
  return { value: state && state.name === name ? state.value : null, error }
}

interface Ranked { g: number; r: number }

/**
 * Spectral similarity finder. Pick any site, or one season at a site, and see
 * which other sites, site-seasons and individual filters look most like it
 * under the same signed-Pearson metric the analog selections use.
 */
export function SimilarityPage() {
  const [meta, setMeta] = useState<SimilarityMeta | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let live = true
    loadSimilarityMeta().then((d) => { if (live) setMeta(d) }).catch((e) => { if (live) setError(String(e)) })
    return () => { live = false }
  }, [])
  if (error) return <ChartFrame title="Spectral similarity" exportable={false}><Empty>Could not load the similarity export: {error}. Run <code>uv run --locked --no-sync python gallery/data/export_similarity.py</code> and reload.</Empty></ChartFrame>
  if (!meta) return <p>Loading the spectral similarity export…</p>
  return <SimilarityView meta={meta} />
}

function SimilarityView({ meta }: { meta: SimilarityMeta }) {
  const index = useMemo(() => groupIndex(meta), [meta])
  const siteCodes = useMemo(() => meta.sites.map((s) => s.site), [meta])
  const maskKeys = useMemo(() => meta.masks.map((m) => m.key), [meta])

  const [site, setSite] = useSearchParam<string>('ssite', 'ETAD', siteCodes)
  const [season, setSeason] = useSearchParam<string>('sseason', 'all')
  const [archiveMode, setArchiveMode] = useSearchParam<string>('sarchive', 'hide', ['hide', 'show'])
  const showArchived = archiveMode === 'show'
  const methods = meta.methods.filter(m => showArchived || m !== 'VIBES')
  const [methodValue, setMethod] = useSearchParam<string>('smethod', methods[0], methods)
  const method = methods.includes(methodValue) ? methodValue : methods[0]
  const grid = meta.method_grids?.[method]
  const activeMeta = grid ? { ...meta, bin_wn: grid.bin_wn, masks: grid.masks } : meta
  const [mask, setMask] = useSearchParam<string>('smask', meta.default_mask, maskKeys)
  const [by, setBy] = useSearchParam<'site-seasons' | 'sites'>('sby', 'site-seasons', ['site-seasons', 'sites'])
  const [minN, setMinN] = useSearchParam<string>('smin', '10', MIN_N)
  const [show, setShow] = useSearchParam<string>('sshow', '30', SHOW)
  const [testParam, setTestParam] = useSearchParam<string>('stest', 'on', ['on', 'off'])
  const [sameParam, setSameParam] = useSearchParam<string>('ssame', 'on', ['on', 'off'])
  const [topRaw, setTopN] = useSearchParam<string>('stop', '500')
  const topN = clampCount(topRaw, TOP_MAX, 500)
  const [compareKey, setCompareKey] = useSearchParam<string>('scmp', '')
  const includeTest = testParam === 'on'
  const includeSame = sameParam === 'on'

  const siteIdx = siteCodes.indexOf(site)
  const calendar = calendarOf(meta, siteIdx)
  const target = index.get(season === 'all' ? site : `${site}:${season}`) ?? index.get(site)!
  const tg = meta.groups[target]
  const compare = compareKey ? index.get(compareKey) ?? null : null

  const G = meta.groups.length
  const F = meta.filters.site.length
  const pairs = useBinary(`pairs_${method}_${mask}.bin`, (b) => new PairMatrix(new Int16Array(b), G, meta.scale))
  const tops = useBinary(`top_${method}_${mask}.bin`, (b) => new TopLists(b, G, meta.top_k, meta.scale))
  const traces = useBinary(`traces_${method}.bin`, (b) => new Traces(b, F, activeMeta.bin_wn.length))
  const loadError = pairs.error || tops.error || traces.error

  const selfR = pairs.value?.get(target, target) ?? null
  const ranked = useMemo<Ranked[]>(() => {
    const P = pairs.value
    if (!P) return []
    const out: Ranked[] = []
    for (let g = 0; g < G; g++) {
      const grp = meta.groups[g]
      if (g === target || (by === 'sites') !== (grp.season == null)) continue
      if (grp.n < Number(minN)) continue
      if (!includeTest && meta.sites[grp.site].split === 'test') continue
      if (!includeSame && grp.site === tg.site) continue
      const r = P.get(target, g)
      if (r != null) out.push({ g, r })
    }
    return out.sort((a, b) => b.r - a.r)
  }, [pairs.value, meta, G, target, by, minN, includeTest, includeSame, tg.site])

  // the stored list is the 500 best from other sites; the test-site switch filters it
  const topFilters = useMemo(() => {
    const all = tops.value?.of(target) ?? []
    const kept = includeTest ? all : all.filter((t) => meta.sites[meta.filters.site[t.filter]].split !== 'test')
    return { rows: kept.slice(0, Number(topN)), available: kept.length }
  }, [tops.value, target, includeTest, topN, meta])

  const [hoverFilter, setHoverFilter] = useState<number | null>(null)
  const { pinnedId, setPinned, openRecord, dismissSample } = useHighlight()
  const filterIndex = useMemo(() => new Map(meta.filters.sample_id.map((id, f) => [baseFilterId(id), f])), [meta])
  const pinned = pinnedId == null ? null : filterIndex.get(baseFilterId(pinnedId)) ?? null
  const targetLabel = groupLabel(meta, target)
  // a mask this method's grid lacks falls back to its first mask rather than failing the page
  const maskMeta = activeMeta.masks.find((m) => m.key === mask) ?? activeMeta.masks[0]
  const pinnedRecord = useMemo<FilterRecord | null>(() => {
    if (pinned == null) return null
    const site = meta.sites[meta.filters.site[pinned]]
    const season = meta.filters.season[pinned]
    const rank = topFilters.rows.findIndex((row) => row.filter === pinned)
    const isTarget = meta.filters.site[pinned] === tg.site && (tg.season == null || season === tg.season)
    return {
      id: meta.filters.sample_id[pinned],
      source: 'Spectral similarity',
      site: `${site.site} · ${siteName(site)}`,
      date: meta.filters.date[pinned] ?? undefined,
      fields: [
        ['Site', site.site],
        ['Site name', siteName(site)],
        ['Latitude (°)', site.lat],
        ['Longitude (°)', site.lon],
        ['Elevation (m)', site.elevation],
        ['Land use', site.land_use],
        ['Setting', site.setting],
        ['Sampling date', meta.filters.date[pinned]],
        ['Season', season < 0 ? 'Undated' : calendarOf(meta, meta.filters.site[pinned])[season].name],
        ['Site role', site.split === 'train' ? 'Calibration set site' : splitName(site.split)],
        ['Compared with', targetLabel],
        ['Filter role', isTarget ? 'Target filter' : rank >= 0 ? 'Most similar filter' : 'Comparison filter'],
        ['Similarity rank (displayed list)', rank >= 0 ? rank + 1 : null],
        ['Mean signed Pearson r', rank >= 0 ? topFilters.rows[rank].r.toFixed(4) : null],
        ['Spectra', method === 'AIRSpec' ? 'Spline baseline (AIRSpec)' : method],
        ['Selection mask', maskMeta.label],
      ],
      note: 'Similarity scores compare this filter with every target filter over the selected channels. Seasons are fixed calendar month bins. Spectra in the main chart are 8-channel means for display.',
    }
  }, [pinned, meta, tg.site, tg.season, topFilters.rows, targetLabel, method, maskMeta.label])
  const previousTarget = useRef(target)
  useEffect(() => {
    const changedTarget = previousTarget.current !== target
    previousTarget.current = target
    if (changedTarget) {
      setHoverFilter(null)
      if (pinnedRecord) dismissSample()
    } else if (pinnedRecord) {
      openRecord(pinnedRecord)
    }
  }, [target, pinnedRecord, openRecord, dismissSample])
  const onPin = (f: number | null) => setPinned(f == null ? null : meta.filters.sample_id[f])
  const seasonOptions = ['all', ...calendar.map((_, i) => String(i)).filter((i) => index.has(`${site}:${i}`))]
  const setTarget = (g: number) => {
    const grp = meta.groups[g]
    setSite(meta.sites[grp.site].site)
    setSeason(grp.season == null ? 'all' : String(grp.season))
    setCompareKey('')
  }

  return (
    <PageToc>
      <ExplainOnly>
        <ChartFrame title="Spectral similarity finder" exportable={false}>
          <p style={{ fontSize: 14, lineHeight: 1.6 }}>
            Pick a <strong>target</strong>: any of the {meta.sites.length} sites, or one season at a site. Every other site and site-season is scored by the
            <strong> mean signed Pearson r</strong> between each of its filters and each target filter, over the channels the selection mask keeps. That is the
            metric the 500-filter analog selections rank by. Individual filters are scored the same way, against every target filter, and ranked across the other sites.
          </p>
          <Note>Filters from the frozen full-profile AIRSpec / VIBES run: {F.toLocaleString()} filters ({(F - meta.sites[0].n).toLocaleString()} IMPROVE calibration and {meta.sites[0].n} Addis). Blanks and injections are excluded. The masks change only which channels are compared. Nothing is refit and no filter is excluded.</Note>
        </ChartFrame>
      </ExplainOnly>

      <ChartFrame
        title="Target — the site or season everything else is compared with"
        exportable={false}
        tip={<>
          <p><strong>Within-group r</strong> is the mean r between the target's own filters. It is the dashed line in the ranking below: a group scoring above it is as close to the target as the target's own filters are to each other.</p>
          {site === 'ETAD' && <p>{ANALYSIS_CALENDAR_NOTE}</p>}
          {site === 'ETAD' && meta.undated.length > 0 && <p>{meta.undated.length} Addis filters have no date in the frozen ledger ({meta.undated.join(', ')}). They count toward all-year Addis but toward no season.</p>}
        </>}
        controls={<>
          <Select label="Site" value={site} options={siteCodes} onChange={(v) => { setSite(v); setSeason('all'); setCompareKey('') }}
            optionLabel={(o) => { const s = meta.sites[siteCodes.indexOf(o)]; return `${s.site} · ${siteName(s)} (${s.n}${s.split === 'test' ? ', outer test' : ''})` }} />
          <Select label="Season" value={seasonOptions.includes(season) ? season : 'all'} options={seasonOptions}
            onChange={(v) => { setSeason(v); setCompareKey('') }}
            optionLabel={(o) => o === 'all' ? `All year (${meta.sites[siteIdx].n})` : `${calendar[Number(o)].name} (${meta.groups[index.get(`${site}:${o}`)!].n})`} />
          <Toggle label="Show archived VIBES fit (4000–1425)" checked={showArchived} onChange={v => setArchiveMode(v ? 'show' : 'hide')} />
<span className="control">{method.endsWith('-r70') ? `Completed rank 70 · ${grid?.bin_wn.length ? (method.includes('cut') ? 'shared channels' : 'full available range') : ''}` : method === 'VIBES' ? 'Archived truncated correction' : 'Spline baseline'}</span>
          <Select label="Spectra" value={method} options={methods} onChange={setMethod} optionLabel={v => PREPROCESSING[v] ?? v} />
          <Select label="Selection mask" value={mask} options={maskKeys} onChange={setMask} optionLabel={(k) => `${activeMeta.masks.find((m) => m.key === k)?.label ?? k} (${activeMeta.masks.find((m) => m.key === k)?.n_channels ?? '?'} ch)`} />
        </>}
      >
        <p style={{ fontSize: 13, margin: '2px 0', color: INK.muted }}>
          <strong style={{ color: groupColor(meta, target, INK.text) }}>{targetLabel}</strong> · <span title={siteWhere(meta.sites[tg.site])}>{siteName(meta.sites[tg.site])}</span> · {tg.n} filters · {splitName(meta.sites[tg.site].split)} · within-group r <strong style={{ color: INK.text }}>{f3(selfR)}</strong>
        </p>
        {loadError && <Note>Could not load part of the export: {loadError}</Note>}
        {method === 'VIBES' && <Note>Archived VIBES fit (4000–1425 cm⁻¹): these similarity scores and traces use the earlier truncated fit. Current full-range VIBES comparisons are on <a href="#tab=meeting">Follow up meeting</a>.</Note>}
      </ChartFrame>

      <GroupRanking meta={meta} ranked={ranked} target={target} selfR={selfR} compare={compare} onCompare={(g) => setCompareKey(g == null ? '' : groupKey(meta, g))}
        onTarget={setTarget} loading={!pairs.value}
        controls={<>
          <Segmented label="Compare" value={by} options={['site-seasons', 'sites'] as const} onChange={setBy} />
          <Segmented label="Min filters" value={minN} options={MIN_N} onChange={setMinN} />
          <Segmented label="Rows" value={show} options={SHOW} onChange={setShow} />
          <Toggle label="Outer test sites" checked={includeTest} onChange={(v) => setTestParam(v ? 'on' : 'off')} />
          <Toggle label="Target's own site" checked={includeSame} onChange={(v) => setSameParam(v ? 'on' : 'off')} />
        </>}
        show={Number(show)} maskLabel={maskMeta.label} method={method} />

      <SeasonMatrix meta={meta} pairs={pairs.value} target={target} columns={ranked.slice(0, 16).map((r) => r.g)} />

      <SpectraTraces meta={activeMeta} traces={traces.value} target={target} topFilters={topFilters.rows} compare={compare} mask={mask}
        hoverFilter={hoverFilter} pinned={pinned} onPin={onPin} method={method}
        controls={<>
          <CountPicker label="Most similar" value={topN} presets={TOP_N} max={TOP_MAX} onChange={setTopN} noun="most similar filters" />
          <Toggle label="Outer test sites" checked={includeTest} onChange={(v) => setTestParam(v ? 'on' : 'off')} />
        </>} />

      <TopComposition meta={meta} rows={topFilters.rows} available={topFilters.available} targetLabel={targetLabel} />

      <FilterTable meta={meta} rows={topFilters.rows} targetLabel={targetLabel} maskLabel={maskMeta.label} method={method} onHover={setHoverFilter} pinned={pinned} onPin={onPin} />

      <ExplainOnly>
        <ChartFrame title="Provenance" exportable={false}>
          <p style={{ fontSize: 13 }}>Exported by <code>gallery/data/export_similarity.py</code> from run signature <code>{meta.source_signature}</code>. Group scores remove self-pairs, where a filter is on both sides, so a site is not credited for matching its own filters. Filter scores are checked against <code>seasonal_analogs.mean_correlation_scores</code> to within 1e-9 when exported. Scores are stored to four decimal places. Traces are 8-channel means (~10 cm⁻¹), for display only.</p>
          <p style={{ fontSize: 13 }}>Seasons: Addis uses the project's canonical calendar: Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep. Every IMPROVE site uses meteorological seasons. These are fixed month bins, not observed weather (see <code>docs/site-seasonality.md</code>).</p>
          <details><summary style={{ cursor: 'pointer', fontSize: 12 }}>Source files and SHA-256</summary>{Object.entries(meta.source_sha256).map(([name, hash]) => <p key={name} style={{ fontSize: 11, overflowWrap: 'anywhere' }}><code>{name}</code><br />{hash}</p>)}<p style={{ fontSize: 11 }}><code>export_similarity.py</code><br />{meta.exporter_sha256}</p></details>
        </ChartFrame>
      </ExplainOnly>
    </PageToc>
  )
}

function GroupRanking({ meta, ranked, target, selfR, compare, onCompare, onTarget, loading, controls, show, maskLabel, method }: {
  meta: SimilarityMeta; ranked: Ranked[]; target: number; selfR: number | null; compare: number | null
  onCompare: (g: number | null) => void; onTarget: (g: number) => void; loading: boolean; controls: React.ReactNode
  show: number; maskLabel: string; method: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const rows = ranked.slice(0, show)
  const labelW = Math.min(230, Math.max(150, width * 0.3))
  const nW = 64
  const rowH = 20
  const innerW = Math.max(160, width - labelW - nW - 16)
  const vals = rows.map((r) => r.r).concat(selfR != null ? [selfR] : [])
  const lo = d3.min(vals) ?? 0
  const hi = d3.max(vals) ?? 1
  const pad = Math.max(0.002, (hi - lo) * 0.08)
  const x = d3.scaleLinear().domain([lo - pad, Math.min(1, hi + pad)]).range([0, innerW]).nice()
  const title = `Most similar groups to ${groupLabel(meta, target)}`
  return (
    <ChartFrame
      title="Most similar groups — mean Pearson r between every filter pair"
      subtitle={`Ranked by mean signed r against ${groupLabel(meta, target)}, ${method}, ${maskLabel}. Click a row to overlay that group's spectra below. The dashed line is the target's own within-group r.`}
      provenance="export_similarity.py · pairs_{method}_{mask}.bin"
      controls={<>{controls}<button type="button" className="btn quiet" onClick={() => downloadCsv(
        ['rank', 'group', 'site', 'site_name', 'season', 'split', 'n_filters', 'mean_r', 'target', 'method', 'mask'],
        ranked.map((r, i) => { const g = meta.groups[r.g]; return [i + 1, groupLabel(meta, r.g), meta.sites[g.site].site, siteName(meta.sites[g.site]), g.season == null ? 'all year' : calendarOf(meta, g.site)[g.season].name, meta.sites[g.site].split, g.n, r.r.toFixed(4), groupLabel(meta, target), method, maskLabel] }),
        title)}>CSV</button></>}
    >
      <div ref={ref} className="chart-wrap">
        {loading ? <Empty>Loading scores…</Empty> : rows.length === 0 ? <Empty>No group passes these filters.</Empty> : (
          <IntervalStrip labelWidth={labelW} rowHeight={rowH} noteWidth={nW} minPanelWidth={160}
            panels={[{ label: 'Mean Pearson r', domain: x.domain() as [number, number], stems: true, format: (v) => v.toFixed(3),
              marker: selfR != null && x(selfR) >= 0 && x(selfR) <= innerW ? { value: selfR, label: `target within-group r ${selfR.toFixed(3)}` } : undefined }]}
            blocks={[{ key: 'ranked', rows: rows.map((r) => {
              const g = meta.groups[r.g], s = meta.sites[g.site], sel = compare === r.g
              return { key: String(r.g), label: `${groupLabel(meta, r.g)}${s.split === 'test' ? ' †' : ''}`, labelColor: s.split === 'test' ? TEST_COLOR : undefined,
                note: `n=${g.n}`, selected: sel, onClick: () => onCompare(sel ? null : r.g),
                tip: [groupLabel(meta, r.g), siteWhere(s), `mean r ${r.r.toFixed(4)}`, `${g.n} filters · ${splitName(s.split)}`, selfR != null ? `${r.r >= selfR ? '+' : ''}${(r.r - selfR).toFixed(4)} vs target's own r` : '', sel ? 'click to clear the overlay' : 'click to overlay its spectra'].filter(Boolean),
                marks: [[{ mid: r.r, color: groupColor(meta, r.g, INK.accent), r: 4.5 }]] }
            }) }]} />
        )}
        <Note>{ranked.length} groups pass these filters, of which the top {rows.length} are shown. † marks an outer test site. Small groups score noisily, so raise "Min filters" before reading much into the top rows.</Note>
        {compare != null && (
          <p style={{ fontSize: 13, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            Overlaying <strong style={{ color: groupColor(meta, compare, INK.accent) }}>{groupLabel(meta, compare)}</strong> below.
            <button type="button" className="btn" onClick={() => onTarget(compare)}>Make it the target</button>
            <button type="button" className="btn quiet" onClick={() => onCompare(null)}>Clear</button>
          </p>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** Rows: the target site as a whole and each of its seasons. Columns: the top-ranked groups. */
function SeasonMatrix({ meta, pairs, target, columns }: { meta: SimilarityMeta; pairs: PairMatrix | null; target: number; columns: number[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const site = meta.groups[target].site
  const rowGroups = useMemo(() => meta.groups.map((g, i) => ({ g, i })).filter(({ g }) => g.site === site).map(({ i }) => i), [meta, site])
  const cells = useMemo(() => {
    if (!pairs) return []
    return rowGroups.flatMap((r) => columns.map((c) => ({ r, c, v: pairs.get(r, c) })))
  }, [pairs, rowGroups, columns])
  const labelW = 190
  const top = 120
  const cw = columns.length ? Math.max(24, Math.min(64, (width - labelW - 10) / columns.length)) : 0
  const ch = 28
  const vals = cells.map((c) => c.v).filter((v): v is number => v != null)
  const color = d3.scaleSequential(d3.interpolateBlues).domain([d3.min(vals) ?? 0, d3.max(vals) ?? 1])
  return (
    <ChartFrame
      title="Season by season — is the match year-round or one season only?"
      subtitle={`Rows are ${meta.sites[site].label} as a whole and each of its seasons. Columns are the top-ranked groups from the chart above. Each cell is the mean r between the two groups.`}
      provenance="export_similarity.py · pairs_{method}_{mask}.bin"
    >
      <div ref={ref} className="chart-wrap" style={{ overflowX: 'auto' }}>
        {!pairs || columns.length === 0 ? <Empty>{pairs ? 'No columns to show.' : 'Loading scores…'}</Empty> : (
          <svg width={Math.max(width, labelW + cw * columns.length + 10)} height={top + rowGroups.length * ch + 8}>
            <g fontFamily={FONT.family}>
              {columns.map((c, j) => (
                <text key={c} transform={`translate(${labelW + j * cw + cw / 2},${top - 6}) rotate(-45)`} fontSize={10.5} onMouseMove={(e) => tip.show(e, [groupLabel(meta, c), siteWhere(meta.sites[meta.groups[c].site]), `${meta.groups[c].n} filters`])} onMouseLeave={tip.hide} fill={meta.sites[meta.groups[c].site].split === 'test' ? TEST_COLOR : INK.text}>{groupLabel(meta, c)}</text>
              ))}
              {rowGroups.map((r, i) => (
                <text key={r} x={labelW - 8} y={top + i * ch + ch / 2} dy="0.35em" textAnchor="end" fontSize={11.5} fontWeight={r === target ? 700 : 400} fill={groupColor(meta, r, INK.text)}>
                  {groupLabel(meta, r)} <tspan fill={INK.muted} fontFamily={FONT.mono} fontSize={10}>n={meta.groups[r].n}</tspan>
                </text>
              ))}
              {cells.map(({ r, c, v }) => {
                const i = rowGroups.indexOf(r)
                const j = columns.indexOf(c)
                return (
                  <HeatCell key={`${r}-${c}`} x={labelW + j * cw} y={top + i * ch} width={cw - 1} height={ch - 1} fill={v == null ? null : color(v)} outline={r === target}
                    label={cw >= 40 && v != null ? v.toFixed(3) : null} onEnter={(e) => tip.show(e, [`${groupLabel(meta, r)} vs ${groupLabel(meta, c)}`, `mean r ${v == null ? '—' : v.toFixed(4)}`, siteWhere(meta.sites[meta.groups[c].site])])} onLeave={tip.hide} />
                )
              })}
            </g>
          </svg>
        )}
        {vals.length > 0 && <div className="legend"><ColorLegend scale={color} label="mean Pearson r between the two groups" width={200} format={(v) => v.toFixed(3)} /></div>}
        {vals.length > 0 && <Note>The outlined row is the current target.</Note>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}


function TopComposition({ meta, rows, available, targetLabel }: { meta: SimilarityMeta; rows: { filter: number; r: number }[]; available: number; targetLabel: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const bySite = useMemo(() => {
    const m = new Map<number, number[]>()
    for (const { filter } of rows) {
      const s = meta.filters.site[filter]
      if (!m.has(s)) m.set(s, [])
      m.get(s)!.push(filter)
    }
    return [...m.entries()].map(([site, fs]) => ({ site, fs })).sort((a, b) => b.fs.length - a.fs.length)
  }, [rows, meta])
  const shown = bySite.slice(0, 20)
  const rest = bySite.slice(20).reduce((n, s) => n + s.fs.length, 0)
  const total = rows.length
  const effective = total ? 1 / d3.sum(bySite, (s) => (s.fs.length / total) ** 2) : 0
  const top5 = total ? d3.sum(bySite.slice(0, 5), (s) => s.fs.length) / total : 0
  const labelW = 150
  const rowH = 18
  const innerW = Math.max(160, width - labelW - 70)
  const x = d3.scaleLinear().domain([0, d3.max(shown, (s) => s.fs.length) ?? 1]).range([0, innerW]).nice()
  const seasonsSeen = new Map<string, string>()
  const lg = useLegend()
  return (
    <ChartFrame
      title="Where the most similar filters come from — by site, stacked by season"
      subtitle={`The ${total} filters from other sites most similar to ${targetLabel}, counted by source site and split by each filter's local season.`}
      provenance="export_similarity.py · top_{method}_{mask}.bin"
    >
      <div ref={ref} className="chart-wrap">
        {total === 0 ? <Empty>Loading…</Empty> : (
          <svg width={width} height={shown.length * rowH + 30}>
            <g transform={`translate(${labelW},6)`} fontFamily={FONT.family}>
              {shown.map((s, i) => {
                const cal = calendarOf(meta, s.site)
                const counts = cal.map((c, k) => ({ c, n: s.fs.filter((f) => meta.filters.season[f] === k).length }))
                const undated = s.fs.filter((f) => meta.filters.season[f] < 0).length
                let acc = 0
                return (
                  <g key={s.site} transform={`translate(0,${i * rowH})`}
                    onMouseMove={(e) => tip.show(e, [`${meta.sites[s.site].site} · ${splitName(meta.sites[s.site].split)}`, siteWhere(meta.sites[s.site]), `${s.fs.length} of ${total} (${Math.round((100 * s.fs.length) / total)}%)`, ...counts.filter((c) => c.n).map((c) => `${c.c.name}: ${c.n}`), ...(undated ? [`undated: ${undated}`] : [])])}
                    onMouseLeave={tip.hide}>
                    <text x={-8} y={rowH / 2} dy="0.35em" textAnchor="end" fontSize={11.5} fill={meta.sites[s.site].split === 'test' ? TEST_COLOR : INK.text}>{meta.sites[s.site].label}{meta.sites[s.site].split === 'test' ? ' †' : ''}</text>
                    {counts.map(({ c, n }) => {
                      seasonsSeen.set(c.name, c.color)
                      const x0 = acc
                      acc += n
                      return n && lg.show(c.name) ? <rect key={c.name} x={x(x0)} y={2} width={x(acc) - x(x0)} height={rowH - 4} fill={c.color} opacity={lg.dim(c.name, 0.2)} /> : null
                    })}
                    {undated > 0 && <rect x={x(acc)} y={2} width={x(acc + undated) - x(acc)} height={rowH - 4} fill={INK.neutral} />}
                    <text x={x(s.fs.length) + 5} y={rowH / 2} dy="0.35em" fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>{s.fs.length}</text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <Legend {...lg.props} items={[...seasonsSeen.entries()].map(([label, color]) => ({ label, color, shape: 'square' as const }))} />
        {total > 0 && <Note>{total} filters from {bySite.length} sites{rest ? `, of which ${bySite.length - 20} smaller contributors (${rest} filters) are not drawn` : ''}. The top 5 sites supply {Math.round(top5 * 100)}%, and the effective number of sites (1/Σp²) is {effective.toFixed(1)}: a lower number means the list is more concentrated. † marks an outer test site.</Note>}
        {available < meta.top_k && <Note>Outer test sites are hidden, so {available} of the {meta.top_k} stored most-similar filters remain.</Note>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

function FilterTable({ meta, rows, targetLabel, maskLabel, method, onHover, pinned, onPin }: {
  meta: SimilarityMeta; rows: { filter: number; r: number }[]; targetLabel: string; maskLabel: string; method: string
  onHover: (f: number | null) => void; pinned: number | null; onPin: (f: number | null) => void
}) {
  const [all, setAll] = useState(false)
  const shown = all ? rows : rows.slice(0, 25)
  const seasonName = (f: number) => { const k = meta.filters.season[f]; return k < 0 ? 'undated' : calendarOf(meta, meta.filters.site[f])[k].name }
  const title = `Most similar filters to ${targetLabel}`
  return (
    <ChartFrame title="Most similar filters — ranked individual filters from other sites" exportable={false}
      subtitle={`Each filter's mean signed r against every ${targetLabel} filter (${method}, ${maskLabel}). This is the score the 500-filter analog selections rank by. Selected here from all sites, training and outer test alike, unless the test-site switch is off.`}
      controls={<button type="button" className="btn quiet" onClick={() => downloadCsv(
        ['rank', 'sample_id', 'site', 'site_name', 'latitude', 'longitude', 'split', 'date', 'season', 'mean_r', 'target', 'method', 'mask'],
        rows.map((t, i) => { const s = meta.sites[meta.filters.site[t.filter]]; return [i + 1, meta.filters.sample_id[t.filter], s.site, siteName(s), s.lat, s.lon, s.split, meta.filters.date[t.filter], seasonName(t.filter), t.r.toFixed(4), targetLabel, method, maskLabel] }),
        title)}>CSV ({rows.length})</button>}>
      <div style={{ overflowX: 'auto' }} onMouseLeave={() => onHover(null)}>
        <table className="census-table">
          <thead><tr><th>#</th><th>Filter</th><th>Site</th><th>Date</th><th>Season</th><th>Mean r</th></tr></thead>
          <tbody>{shown.map((t, i) => {
            const s = meta.sites[meta.filters.site[t.filter]]
            return (
              <tr key={t.filter} onMouseEnter={() => onHover(t.filter)} onClick={() => onPin(pinned === t.filter ? null : t.filter)} style={{ cursor: 'pointer', background: pinned === t.filter ? '#eef4fb' : undefined }}>
                <td className="mono">{i + 1}</td>
                <td className="mono">{meta.filters.sample_id[t.filter]}</td>
                <td title={siteWhere(s)} style={{ color: s.split === 'test' ? TEST_COLOR : undefined }}>{s.site}{s.split === 'test' ? ' †' : ''} <span style={{ color: INK.muted, fontSize: 12 }}>{siteName(s)}</span></td>
                <td className="mono">{meta.filters.date[t.filter] ?? '—'}</td>
                <td>{seasonName(t.filter)}</td>
                <td className="mono">{t.r.toFixed(4)}</td>
              </tr>
            )
          })}</tbody>
        </table>
      </div>
      {rows.length > 25 && <button type="button" className="btn quiet" onClick={() => setAll((v) => !v)}>{all ? 'Show the first 25' : `Show all ${rows.length}`}</button>}
    </ChartFrame>
  )
}
