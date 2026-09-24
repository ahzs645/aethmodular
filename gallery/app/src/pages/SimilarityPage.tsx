import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Segmented, Select, Toggle } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { downloadCsv } from '@/lib/export'
import { useSearchParam } from '@/lib/url'
import { FONT, INK, MARGIN } from '@/lib/theme'
import {
  BASE, PairMatrix, TopLists, Traces, calendarOf, fetchBinary, groupColor, groupIndex, groupKey, groupLabel, members, siteName, siteWhere,
  type SimilarityMeta,
} from '@/lib/similarity'

const TOP_N = ['50', '100', '250', '500'] as const
const MIN_N = ['1', '5', '10', '20'] as const
const SHOW = ['15', '30', '60'] as const
const TEST_COLOR = '#9a6b2f'
// grey for the top list; a darker grey for an overlaid group, so neither can be
// mistaken for a season colour on the target
const TOP_GREY = '#8a8f98'
const GROUP_GREY = '#39404d'
const f3 = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(3))
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
    const controller = new AbortController()
    fetch(BASE + 'similarity.json', { signal: controller.signal })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: SimilarityMeta) => { if (d.schema_version !== 1) throw new Error('Unsupported data version'); setMeta(d) })
      .catch((e) => { if (e.name !== 'AbortError') setError(String(e)) })
    return () => controller.abort()
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
  const [method, setMethod] = useSearchParam<string>('smethod', meta.methods[0], meta.methods)
  const [mask, setMask] = useSearchParam<string>('smask', meta.default_mask, maskKeys)
  const [by, setBy] = useSearchParam<'site-seasons' | 'sites'>('sby', 'site-seasons', ['site-seasons', 'sites'])
  const [minN, setMinN] = useSearchParam<string>('smin', '10', MIN_N)
  const [show, setShow] = useSearchParam<string>('sshow', '30', SHOW)
  const [testParam, setTestParam] = useSearchParam<string>('stest', 'on', ['on', 'off'])
  const [sameParam, setSameParam] = useSearchParam<string>('ssame', 'on', ['on', 'off'])
  const [topN, setTopN] = useSearchParam<string>('stop', '500', TOP_N)
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
  const traces = useBinary(`traces_${method}.bin`, (b) => new Traces(b, F, meta.bin_wn.length))
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
  const [pinned, setPinned] = useState<number | null>(null)
  useEffect(() => setPinned(null), [target])
  const targetLabel = groupLabel(meta, target)
  const maskMeta = meta.masks.find((m) => m.key === mask)!
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
          {site === 'ETAD' && meta.undated.length > 0 && <p>{meta.undated.length} Addis filters have no date in the frozen ledger ({meta.undated.join(', ')}). They count toward all-year Addis but toward no season.</p>}
        </>}
        controls={<>
          <Select label="Site" value={site} options={siteCodes} onChange={(v) => { setSite(v); setSeason('all'); setCompareKey('') }}
            optionLabel={(o) => { const s = meta.sites[siteCodes.indexOf(o)]; return `${s.site} · ${siteName(s)} (${s.n}${s.split === 'test' ? ', outer test' : ''})` }} />
          <Select label="Season" value={seasonOptions.includes(season) ? season : 'all'} options={seasonOptions}
            onChange={(v) => { setSeason(v); setCompareKey('') }}
            optionLabel={(o) => o === 'all' ? `All year (${meta.sites[siteIdx].n})` : `${calendar[Number(o)].name} (${meta.groups[index.get(`${site}:${o}`)!].n})`} />
          <Segmented label="Spectra" value={method} options={meta.methods} onChange={setMethod} />
          <Select label="Selection mask" value={mask} options={maskKeys} onChange={setMask} optionLabel={(k) => `${meta.masks.find((m) => m.key === k)!.label} (${meta.masks.find((m) => m.key === k)!.n_channels} ch)`} />
        </>}
      >
        <p style={{ fontSize: 13, margin: '2px 0', color: INK.muted }}>
          <strong style={{ color: groupColor(meta, target, INK.text) }}>{targetLabel}</strong> · <span title={siteWhere(meta.sites[tg.site])}>{siteName(meta.sites[tg.site])}</span> · {tg.n} filters · {splitName(meta.sites[tg.site].split)} · within-group r <strong style={{ color: INK.text }}>{f3(selfR)}</strong>
        </p>
        {loadError && <Note>Could not load part of the export: {loadError}</Note>}
      </ChartFrame>

      <GroupRanking meta={meta} ranked={ranked} target={target} selfR={selfR} compare={compare} onCompare={(g) => setCompareKey(g == null ? '' : groupKey(meta, g))}
        onTarget={setTarget} loading={!pairs.value}
        controls={<>
          <Segmented label="Compare" value={by} options={['site-seasons', 'sites'] as const} onChange={setBy} />
          <Segmented label="Min filters" value={minN} options={MIN_N} onChange={setMinN} />
          <Segmented label="Rows" value={show} options={SHOW} onChange={setShow} />
          <Toggle label="outer test sites" checked={includeTest} onChange={(v) => setTestParam(v ? 'on' : 'off')} />
          <Toggle label="target's own site" checked={includeSame} onChange={(v) => setSameParam(v ? 'on' : 'off')} />
        </>}
        show={Number(show)} maskLabel={maskMeta.label} method={method} />

      <SeasonMatrix meta={meta} pairs={pairs.value} target={target} columns={ranked.slice(0, 16).map((r) => r.g)} />

      <SpectraTraces meta={meta} traces={traces.value} target={target} topFilters={topFilters.rows} compare={compare} mask={mask}
        hoverFilter={hoverFilter} pinned={pinned} onPin={setPinned} method={method}
        controls={<>
          <Segmented label="Most similar" value={topN} options={TOP_N} onChange={setTopN} />
          <Toggle label="outer test sites" checked={includeTest} onChange={(v) => setTestParam(v ? 'on' : 'off')} />
        </>} />

      <TopComposition meta={meta} rows={topFilters.rows} available={topFilters.available} targetLabel={targetLabel} />

      <FilterTable meta={meta} rows={topFilters.rows} targetLabel={targetLabel} maskLabel={maskMeta.label} method={method} onHover={setHoverFilter} pinned={pinned} onPin={setPinned} />

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
  const h = rows.length * rowH + 40
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
          <svg width={width} height={h}>
            <g transform={`translate(${labelW},24)`} fontFamily={FONT.family}>
              <XAxis scale={x} y={rows.length * rowH + 4} tickCount={6} format={(v) => (+v).toFixed(3)} />
              {x.ticks(6).map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={-4} y2={rows.length * rowH} stroke={INK.grid} />)}
              {selfR != null && x(selfR) >= 0 && x(selfR) <= innerW && (
                <g>
                  <line x1={x(selfR)} x2={x(selfR)} y1={-8} y2={rows.length * rowH} stroke={INK.fit} strokeDasharray="4 3" />
                  <text x={x(selfR)} y={-12} textAnchor="middle" fontSize={10} fill={INK.muted}>target within-group r {selfR.toFixed(3)}</text>
                </g>
              )}
              {rows.map((r, i) => {
                const g = meta.groups[r.g]
                const s = meta.sites[g.site]
                const color = groupColor(meta, r.g, INK.accent)
                const sel = compare === r.g
                const y = i * rowH + rowH / 2
                return (
                  <g key={r.g} style={{ cursor: 'pointer' }} onClick={() => onCompare(sel ? null : r.g)}
                    onMouseMove={(e) => tip.show(e, [groupLabel(meta, r.g), siteWhere(s), `mean r ${r.r.toFixed(4)}`, `${g.n} filters · ${splitName(s.split)}`, selfR != null ? `${r.r >= selfR ? '+' : ''}${(r.r - selfR).toFixed(4)} vs target's own r` : '', sel ? 'click to clear the overlay' : 'click to overlay its spectra'].filter(Boolean))}
                    onMouseLeave={tip.hide}>
                    <rect x={-labelW} y={i * rowH} width={labelW + innerW + nW} height={rowH} fill={sel ? '#eef4fb' : i % 2 ? '#fafbfc' : 'transparent'} />
                    <text x={-8} y={y} dy="0.35em" textAnchor="end" fontSize={11.5} fill={s.split === 'test' ? TEST_COLOR : INK.text} fontWeight={sel ? 600 : 400}>
                      {groupLabel(meta, r.g)}{s.split === 'test' ? ' †' : ''}
                    </text>
                    <line x1={0} x2={x(r.r)} y1={y} y2={y} stroke={color} strokeOpacity={0.45} strokeWidth={1.5} />
                    <circle cx={x(r.r)} cy={y} r={sel ? 5.5 : 4.5} fill={color} stroke={sel ? INK.text : '#fff'} strokeWidth={1} />
                    <text x={innerW + 8} y={y} dy="0.35em" fontSize={10.5} fill={INK.muted} fontFamily={FONT.mono}>n={g.n}</text>
                  </g>
                )
              })}
            </g>
          </svg>
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
  const mid = ((d3.min(vals) ?? 0) + (d3.max(vals) ?? 1)) / 2
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
                  <g key={`${r}-${c}`} onMouseMove={(e) => tip.show(e, [`${groupLabel(meta, r)} vs ${groupLabel(meta, c)}`, `mean r ${v == null ? '—' : v.toFixed(4)}`, siteWhere(meta.sites[meta.groups[c].site])])} onMouseLeave={tip.hide}>
                    <rect x={labelW + j * cw} y={top + i * ch} width={cw - 1} height={ch - 1} fill={v == null ? INK.empty : color(v)} stroke={r === target ? INK.text : 'none'} strokeWidth={r === target ? 1 : 0} />
                    {cw >= 40 && v != null && <text x={labelW + j * cw + cw / 2} y={top + i * ch + ch / 2} dy="0.35em" textAnchor="middle" fontSize={10} fontFamily={FONT.mono} fill={v > mid ? '#fff' : INK.text} pointerEvents="none">{v.toFixed(3)}</text>}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        {vals.length > 0 && <Note>Colour runs from {f3(d3.min(vals))} (light) to {f3(d3.max(vals))} (dark). The outlined row is the current target.</Note>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

function SpectraTraces({ meta, traces, target, topFilters, compare, mask, hoverFilter, pinned, onPin, method, controls }: {
  meta: SimilarityMeta; traces: Traces | null; target: number; topFilters: { filter: number; r: number }[]; compare: number | null
  mask: string; hoverFilter: number | null; pinned: number | null; onPin: (f: number | null) => void; method: string; controls: React.ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [hoverTrace, setHoverTrace] = useState<number | null>(null)
  const [regions, setRegions] = useSearchParam<'gaps' | 'shaded'>('sregion', 'gaps', ['gaps', 'shaded'])
  const [bg, setBg] = useSearchParam<'most similar' | 'selected group'>('sbg', 'most similar', ['most similar', 'selected group'])
  const height = 420
  const innerW = Math.max(260, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const keep = meta.masks.find((m) => m.key === mask)!.bins
  const wn = meta.bin_wn
  const useGroup = compare != null && bg === 'selected group'
  // stable legend labels (counts go in `detail`) so a hidden series stays hidden as N changes
  const lg = useLegend()
  const BACK = useGroup ? groupLabel(meta, compare!) : 'most similar filters, other sites'
  const wholeSite = meta.groups[target].season == null
  const cal = calendarOf(meta, meta.groups[target].site)
  // calendar order, "undated" last
  const seasonOrder = (label: string) => { const i = cal.findIndex((c) => label.endsWith(c.name)); return i < 0 ? cal.length : i }
  const tKey = (f: number) => !wholeSite ? groupLabel(meta, target)
    : `${meta.sites[meta.groups[target].site].site} · ${meta.filters.season[f] >= 0 ? cal[meta.filters.season[f]].name : 'undated'}`

  const drawn = useMemo(() => {
    if (!traces) return null
    const tgt = members(meta, target).map((f) => ({ f, v: traces.of(f) }))
    const back = (useGroup ? members(meta, compare!) : topFilters.map((t) => t.filter)).map((f) => ({ f, v: traces.of(f) }))
    const lit = hoverTrace ?? hoverFilter
    const hover = lit != null && lit !== pinned ? { f: lit, v: traces.of(lit) } : null
    const pin = pinned != null ? { f: pinned, v: traces.of(pinned) } : null
    return { tgt, back, hover, pin }
  }, [traces, meta, target, topFilters, useGroup, compare, hoverFilter, hoverTrace, pinned])
  const rankOf = useMemo(() => new Map(topFilters.map((t, i) => [t.filter, { rank: i + 1, r: t.r }])), [topFilters])
  const targetSet = useMemo(() => new Set(members(meta, target)), [meta, target])
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
      `${meta.filters.date[f] ?? 'undated'} · ${k < 0 ? 'no season' : calendarOf(meta, meta.filters.site[f])[k].name}`,
      targetSet.has(f) ? 'target filter' : hit ? `#${hit.rank} most similar · mean r ${hit.r.toFixed(4)}` : `member of ${groupLabel(meta, compare ?? target)}`,
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
  const tLabel = `${groupLabel(meta, target)} (${drawn?.tgt.length ?? 0})`
  return (
    <ChartFrame
      title="Spectra — target filters against their closest matches"
      subtitle={`Every ${groupLabel(meta, target)} filter in colour, drawn over ${useGroup ? `every ${groupLabel(meta, compare!)} filter in dark grey` : `its ${topFilters.length} most similar filters from other sites in grey`}. ${method} baseline-corrected absorbance. Gaps or shading mark the channels the selection mask leaves out of the score. The y-axis spans the scored channels only.`}
      provenance="export_similarity.py · traces_{method}.bin (8-channel means) and top_{method}_{mask}.bin"
      controls={<>
        {controls}
        {compare != null && <Segmented label="Grey" value={bg} options={['most similar', 'selected group'] as const} onChange={setBg} />}
        <Segmented label="Ignored channels" value={regions} options={['gaps', 'shaded'] as const} onChange={setRegions} />
      </>}
    >
      <div ref={ref} className="chart-wrap">
        {!drawn ? <Empty>Loading spectra…</Empty> : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {regions === 'shaded' && ignored.map(([a, b]) => (
                <rect key={a} x={x(wn[a])} width={Math.max(0, x(wn[b]) - x(wn[a]))} y={0} height={innerH} fill="#efeeea" />
              ))}
              <YAxis scale={y} x={0} label="Baseline-corrected absorbance" gridWidth={innerW} tickCount={6} />
              <XAxis scale={x} y={innerH} label="Wavenumber (cm⁻¹)" tickCount={8} />
              <clipPath id="similarity-trace-clip"><rect width={innerW} height={innerH} /></clipPath>
              <g clipPath="url(#similarity-trace-clip)">
              {vis!.back.map((t) => <path key={`b${t.f}`} d={line(Array.from(t.v)) ?? ''} fill="none" stroke={useGroup ? GROUP_GREY : TOP_GREY} strokeOpacity={(useGroup ? 0.3 : 0.22) * lg.dim(BACK, 0.2)} strokeWidth={0.7} />)}
              {vis!.tgt.map((t) => <path key={`t${t.f}`} d={line(Array.from(t.v)) ?? ''} fill="none" stroke={tColor(t.f)} strokeOpacity={0.55 * lg.dim(tKey(t.f), 0.12)} strokeWidth={0.8} />)}
              {drawn.pin && <path d={line(Array.from(drawn.pin.v)) ?? ''} fill="none" stroke={INK.fit} strokeWidth={2.2} />}
              {drawn.hover && <path d={line(Array.from(drawn.hover.v)) ?? ''} fill="none" stroke={INK.negative} strokeWidth={1.8} />}
              </g>
              <rect x={0} y={0} width={innerW} height={innerH} fill="transparent" style={{ cursor: hoverTrace != null ? 'pointer' : 'crosshair' }}
                onMouseMove={onMove} onMouseLeave={() => { setHoverTrace(null); tip.hide() }}
                onClick={() => { if (hoverTrace != null) onPin(pinned === hoverTrace ? null : hoverTrace) }} />
            </g>
          </svg>
        )}
        <Legend {...lg.props} items={[
          { label: BACK, color: useGroup ? GROUP_GREY : TOP_GREY, shape: 'line', detail: `n=${drawn?.back.length ?? 0}` },
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
        <Note>{tLabel}. {backLabel}. Hover a trace to identify it and click to pin it; hovering a row in the filter table traces that filter too.</Note>
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
