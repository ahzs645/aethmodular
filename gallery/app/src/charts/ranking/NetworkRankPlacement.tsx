import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { Combobox } from '@/components/Combobox'
import { DrawerShell } from '@/components/DrawerShell'
import { XIcon } from '@/components/Icons'
import { XAxis, YAxis } from '@/components/Axes'
import { AxisFieldsProvider, MeasurementControl } from '@/components/AxisFieldPopover'
import { HitLayer } from '@/components/HitLayer'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useChartOptions } from '@/lib/chartOptions'
import { useFitHeight } from '@/lib/chartFit'
import { useExport, useLoaded } from '@/lib/data'
import { fmt } from '@/lib/format'
import { textWidth } from '@/lib/fitLabel'
import { siteCodeColor } from '@/lib/siteColours'
import { useMac } from '@/lib/mac'
import { ExcludedNote, useCountedFilter, useExclMode, useKnownFilter } from '@/lib/markedRows'
import { networkColor } from '@/lib/networks'
import { loadSpartanReference, type SpartanIndex } from '@/lib/spartan'
import type { SpartanReference } from '@/lib/referencePopulation'
import { groupQuantiles, improveQuantity, NETWORK_RANK_PRESETS, partsOf, rankWithActiveSites, rankedReference, referenceQuantity, spartanQuantity,
  type ActiveRankSite, type NetworkRankSite, type Population, type RankNetworks, type ReferenceFile } from '@/lib/referenceRank'
import { ratioUnit, viewRows } from '@/lib/viewFields'
import { useSearchParam } from '@/lib/url'
import { FONT, IMPROVE_GREY, INK, MARGIN, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const FILE = 'improve_site_quantiles.json'
const NO_IMPROVE: ReadonlySet<string> = new Set()
const POPULATIONS: { value: Population; label: string }[] = [
  { value: 'paired', label: 'OC, EC and Fabs > 0' },
  { value: 'own', label: 'Every filter with it' },
]
const val = (v: number) => fmt(v, Math.abs(v) >= 10 ? 1 : Math.abs(v) >= 1 ? 2 : 3)
const keyOf = (s: Pick<NetworkRankSite, 'network' | 'code'>) => `${s.network}:${s.code}`

/** All eligible sites, ranked once per network + code. The notebook placement is also available. */
export function NetworkRankPlacement({ rows, meta, field: selectedField, improveCodes = NO_IMPROVE }: {
  rows: FilterRow[]; meta: MetaFile; field: string; improveCodes?: ReadonlySet<string>
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const fit = useFitHeight()
  const m = useMac()
  const lg = useLegend()
  const [open, setOpen] = useState<string | null>(null)
  // Keep old notebook-view links, while new selections use the shared axis picker and store the field itself.
  const [quantityPick, setQuantityPick] = useSearchParam<string>('netrank.quantity', 'field')
  const o = useChartOptions('netrank', [
    { key: 'networks', label: 'Comparison networks', default: 'both', choices: [
      { value: 'both', label: 'IMPROVE + SPARTAN' }, { value: 'spartan', label: 'SPARTAN only' }, { value: 'improve', label: 'IMPROVE only' },
    ], title: 'Choose the ranking background. Active sites from Samples stay visible in their own colours in every mode, replacing matching background dots' },
    { key: 'order', label: 'Ranking', default: 'together', when: c => c.networks === 'both', hidden: 'together', choices: [
      { value: 'together', label: 'Rank together' }, { value: 'reference', label: 'Place in IMPROVE ranking' },
    ], title: 'Rank together orders both networks by median. Place in IMPROVE ranking reproduces the notebook: each SPARTAN site is placed where its median would enter the IMPROVE ordering' },
    { key: 'scope', label: 'SPARTAN sites', default: 'all', when: c => c.networks !== 'improve', choices: [
      { value: 'all', label: 'All sites' }, { value: 'view', label: 'Sites / filters in view' },
    ], title: 'All sites uses the full bundled network across all seasons and dates. Sites / filters in view follows the gallery selection, source, dates and seasons' },
    { key: 'colours', label: 'SPARTAN background colours', default: 'network', when: c => c.networks !== 'improve', choices: [
      { value: 'network', label: 'One network colour' }, { value: 'site', label: 'Site colours' },
    ], title: 'Colours for other SPARTAN sites. Active sites always keep their Samples colours' },
    { key: 'pop', label: 'Filters', default: 'paired', choices: POPULATIONS,
      title: 'The same population rule on both sides. OC, EC and Fabs > 0 reproduces ftir_51 §11; Every filter with it uses each quantity wherever reported (both ratio parts must be positive)' },
    { key: 'minN', label: 'IMPROVE sites with ≥', default: '50', when: c => c.networks !== 'spartan',
      choices: ['10', '30', '50', '100'].map(v => ({ value: v, label: `${v} filters` })) },
    { key: 'spMinN', label: 'SPARTAN sites with ≥', default: '10', when: c => c.networks !== 'improve',
      choices: ['1', '10', '30', '50', '100'].map(v => ({ value: v, label: `${v} ${v === '1' ? 'filter' : 'filters'}` })) },
    { key: 'scale', label: 'Scale', default: 'Linear', choices: ['Linear', 'Log'] },
  ])
  const networks = o.values.networks as RankNetworks
  const needsImprove = networks !== 'spartan', needsSpartan = networks !== 'improve'
  const allSpartan = needsSpartan && o.values.scope === 'all'
  const chosenField = NETWORK_RANK_PRESETS.find(p => p.value === quantityPick)?.field ?? (quantityPick === 'field' ? selectedField : quantityPick)
  const field = networks === 'spartan' ? spartanQuantity({ parts: partsOf(chosenField) }).parts.join(' / ') : chosenField
  const pickerMeta = useMemo<MetaFile>(() => {
    const presets = NETWORK_RANK_PRESETS.map(p => p.field)
    const presetSet = new Set<string>(presets)
    const fields = [...new Set([...presets, ...meta.fields, chosenField])]
    const extra = fields.filter(f => !presetSet.has(f) && !meta.field_groups.some(g => g.fields.includes(f)))
    const field_units = { ...meta.field_units }
    const coverage = { ...meta.coverage }
    for (const f of presets) {
      const [a, b] = partsOf(f)
      field_units[f] = ratioUnit(meta.field_units[a], meta.field_units[b])
      // The network background can have a ratio even when the current Samples selection has no pairs.
      delete coverage[f]
    }
    return { ...meta, fields, field_units, coverage, field_groups: [
      { label: 'Site ratios', fields: presets },
      ...meta.field_groups.map(g => ({ ...g, fields: g.fields.filter(f => !presetSet.has(f)) })).filter(g => g.fields.length),
      ...(extra.length ? [{ label: 'Current measurement', fields: extra }] : []),
    ] }
  }, [meta, chosenField])
  const axisValue = useMemo(() => ({ x: '', y: '', field: chosenField }), [chosenField])
  const pop = o.values.pop as Population
  const minN = Number(o.values.minN), spMinN = Number(o.values.spMinN)
  const referenceOrder = networks === 'both' && o.values.order === 'reference'
  const networkLabel = networks === 'both' ? 'IMPROVE + SPARTAN' : networks === 'spartan' ? 'SPARTAN' : 'IMPROVE'

  const { data: ref, error } = useExport<ReferenceFile>(needsImprove ? FILE : null, d => d.schema_version === 1 && d.quantities ? null : 'unexpected schema')
  const { data: spIndex, error: indexError } = useExport<SpartanIndex>(needsSpartan ? 'spartan_network/index.json' : null)
  const spCodes = useMemo(() => new Set(spIndex?.sites.map(s => s.code)), [spIndex])
  const sp: { data: SpartanReference | null; error: string } = useLoaded(allSpartan ? 'spartan-reference' : null, loadSpartanReference)
  const q = useMemo(() => referenceQuantity(field, m.mac), [field, m.mac])
  const quantity = 'reason' in q ? null : q
  const spQuantity = useMemo(() => quantity ? spartanQuantity(quantity) : null, [quantity])
  // Concentration powers must cancel to compare a mass view with IMPROVE's concentration quantiles.
  const massClash = needsImprove && !!quantity && meta.basis === 'mass' && quantity.concPower !== 0
  const improve = useMemo(() => ref && quantity ? rankedReference(ref, pop, quantity, minN) : [], [ref, quantity, pop, minN])
  const isCounted = useCountedFilter(), exclMode = useExclMode(), isKnown = useKnownFilter()
  const spRows = useMemo(() => {
    if (!allSpartan) return rows.filter(r => spCodes.has(r.code))
    const clean = (sp.data?.rows ?? []).filter(r => isCounted(r.id) && (isKnown(r.id) || !r.excluded || exclMode === 'include'))
    return viewRows(clean, { mac: m.mac, exportMac: m.exportMac, basis: meta.basis ?? 'conc', units: sp.data?.field_units ?? {}, ratios: [] })
  }, [allSpartan, rows, spCodes, sp.data, isCounted, isKnown, exclMode, m.mac, m.exportMac, meta.basis])
  const siteInfo = useMemo(() => {
    const out = new Map<string, { name: string; color: string }>()
    const colors = new Map(spIndex?.sites.map((s, i) => [s.code, networkColor(i)]))
    for (const r of spRows) if (!out.has(r.code)) {
      const own = meta.sites.find(s => s.code === r.code)
      out.set(r.code, { name: own?.name ?? r.site, color: siteCodeColor(meta, r.code, colors.get(r.code) ?? INK.accent) })
    }
    return out
  }, [spRows, spIndex, meta])
  const spartan = useMemo(() => !needsSpartan || !spQuantity ? []
    : groupQuantiles(spRows, r => r.code, spQuantity, pop, spMinN)
      .map(g => ({ ...g, code: g.key, name: siteInfo.get(g.key)!.name })), [needsSpartan, spQuantity, spRows, pop, spMinN, siteInfo])
  const activeInfo = useMemo(() => new Map(rows.map(r => [r.code, { name: r.site, color: r.color }])), [rows])
  const active = useMemo<ActiveRankSite[]>(() => {
    if (!quantity) return []
    const out: ActiveRankSite[] = []
    for (const network of ['SPARTAN', 'IMPROVE'] as const) {
      const own = rows.filter(r => improveCodes.has(r.code) === (network === 'IMPROVE'))
      const q = network === 'SPARTAN' ? spartanQuantity(quantity) : improveQuantity(quantity)
      for (const g of groupQuantiles(own, r => r.code, q, pop, 1)) {
        const info = activeInfo.get(g.key)!
        out.push({ code: g.key, name: info.name, color: info.color, network, n: g.n, p25: g.p25, p50: g.p50, p75: g.p75 })
      }
    }
    return out
  }, [rows, improveCodes, quantity, pop, activeInfo])
  const { sites: ranked, nReference: N } = useMemo(() => rankWithActiveSites(improve, spartan, networks, referenceOrder ? 'reference' : 'together', active),
    [improve, spartan, networks, referenceOrder, active])
  const colorOf = (s: NetworkRankSite) => s.active ? s.color! : s.network === 'IMPROVE' ? IMPROVE_GREY
    : o.values.colours === 'site' ? siteInfo.get(s.code)?.color ?? INK.accent : INK.accent
  const shown = useMemo(() => ranked.filter(s => s.active || lg.show(s.network)), [ranked, lg.hidden]) // eslint-disable-line react-hooks/exhaustive-deps
  const height = fit ?? 440
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = Math.max(100, height - MARGIN.top - MARGIN.bottom)
  const lo = d3.min(shown, s => s.p25) ?? 0, hi = d3.max(shown, s => s.p75) ?? 1
  const log = o.values.scale === 'Log' && lo > 0
  const x = useMemo(() => d3.scaleLinear().domain([0, N + 2]).range([0, innerW]), [N, innerW])
  const y = useMemo(() => (log ? d3.scaleLog().domain([lo / 1.15, hi * 1.15]).range([innerH, 0]).nice()
    : d3.scaleLinear().domain([Math.min(0, lo), hi > 0 ? hi * 1.05 : 1]).range([innerH, 0]).nice()), [log, lo, hi, innerH])
  const xy = useMemo(() => shown.map(s => [x(s.rank), y(s.p50)] as const), [shown, x, y])
  const rankText = (s: NetworkRankSite) => !N ? 'no eligible comparison sites'
    : s.rank > N ? `above all ${N} ${referenceOrder ? 'IMPROVE' : networkLabel} sites`
      : `rank ${s.rank} of ${N}${referenceOrder ? ' in IMPROVE' : ''}`
  const openSite = ranked.find(s => keyOf(s) === open)
  const scopeText = allSpartan ? 'SPARTAN uses all bundled filters, across all dates and seasons; exclusions follow Filters.' : 'SPARTAN follows the sites, dates, seasons, source and exclusions in Filters.'
  const parts = quantity?.parts ?? [field]
  const carbon = parts.some(p => /^(EC|OC) \(/.test(p))
  const units = parts.length === 2 ? { ...meta.field_units, [field]: ratioUnit(meta.field_units[parts[0]], meta.field_units[parts[1]]) } : meta.field_units
  const loading = (needsImprove && !ref && !error) || (needsSpartan && !spIndex && !indexError) || (allSpartan && !sp.data && !sp.error)
  const loadError = (needsImprove && error) || (needsSpartan && indexError) || (allSpartan && sp.error)
  const blocked = !quantity || massClash || loading || !!loadError
  const selected = shown.filter(s => s.active)
  const missingActive = [...activeInfo].filter(([code]) => !active.some(s => s.code === code)).map(([, s]) => s.name)
  const star = d3.symbol(d3.symbolStar, 130)() ?? ''

  return <AxisFieldsProvider meta={pickerMeta} value={axisValue} set={(_, value) => setQuantityPick(value)}>
    <ChartFrame id="netrank" title={`Network rank — site medians of ${field}`}
    subtitle={`${networkLabel} comparison background: IMPROVE is grey; other SPARTAN sites are coloured. Active Samples sites always appear as stars in their Samples colours, replacing their background dots. ${referenceOrder ? 'Sites are placed in the IMPROVE ordering.' : 'The comparison networks are ranked together, lowest median first.'} Click a site for its right sidebar.${carbon ? ' Carbon: SPARTAN FTIR; IMPROVE thermal-optical (TOR).' : ''}`}
    provenance={<>Site-rank notebooks: follow_up_mass_and_ratio_plots, site_ratios_mass_space, ftir_51 §11 · IMPROVE: {FILE} · SPARTAN: bundled network filter records, or rows in view</>}
    tip={<>
      {carbon ? <p>Carbon methods: SPARTAN EC and OC are FTIR-derived; IMPROVE EC and OC are thermal-optical (TOR). This comparison does not establish SPARTAN EC accuracy. ChemSpec carbon republishes FTIR and is counted once.</p> : null}
      <p>Each ratio is computed on the same filter before taking site quantiles. {scopeText} IMPROVE uses fixed exported quantiles and does not follow the gallery's dates, seasons or exclusions.</p>
      <p>Active stars always use the current Samples filters and colours, including in the other network's comparison mode. Background minimum counts do not hide an active site with a valid value. Its own background dot is replaced; a site's filtered median is placed among the comparison sites other than itself.</p>
      {needsImprove && ref && <p>{ref.populations[pop].rule}. {ref.note} {ref.data_use}</p>}
      <p>Notebook choices: fAbs/EC, OC/EC, OC/fAbs, iron/fAbs, iron/EC and iron/PM2.5. The older ChemSpec “thermal” duplication is superseded. Iron/PM2.5 is a fraction here (multiply by 100 for percent). HIPS BC uses the selected MAC, {m.mac} m²/g.</p>
    </>}
    controls={<>{o.controls()}{!blocked && ranked.length > 0 && <Combobox label="Site details" value="" search
      options={[{ key: '', label: 'Choose a site…' }, ...ranked.map(s => ({ key: keyOf(s), label: `${s.name} (${s.code})`, group: s.network, hint: rankText(s) }))]}
      onChange={key => { if (key) { tip.hide(); setOpen(key) } }} />}</>}>
    <div ref={wrapRef} className="chart-wrap">
      {!quantity ? <Empty>{field} cannot be ranked: {(q as { reason: string }).reason}. Choose another measurement or ratio.</Empty>
        : massClash ? <Empty>IMPROVE quantiles use concentration. Switch Units to concentration, choose a ratio whose concentration units cancel, or choose SPARTAN only.</Empty>
        : loadError ? <LoadState what="network site data" error={loadError} />
        : loading ? <LoadState what="network site data" />
        : !ranked.length ? <Empty>No site meets the filter population and minimum count for {field}. Try “Every filter with it” or a lower minimum.</Empty>
        : <>
          <svg width={width} height={height} aria-label={`${networkLabel} sites ranked by median ${field}`}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`} fontFamily={FONT.family}>
              <YAxis scale={y} x={0} label={withUnit(field, units)} gridWidth={innerW} pick="field" format={log ? (v: number) => d3.format('~g')(v) : undefined} />
              <XAxis scale={x} y={innerH} label={`${referenceOrder ? 'IMPROVE' : networkLabel} sites, ranked by median`} format={(v: number) => Number.isInteger(v) && v >= 1 && v <= N ? String(v) : ''} />
              {shown.map(s => {
                const color = colorOf(s), picked = !!s.active
                return <g key={keyOf(s)} data-network-site={keyOf(s)} data-active-site={picked ? 'true' : undefined} data-export-site={s.name} data-export-color={color} opacity={picked ? 1 : lg.dim(s.network)}>
                  <line x1={x(s.rank)} x2={x(s.rank)} y1={y(Math.max(y.domain()[0], s.p25))} y2={y(s.p75)} stroke={color} strokeOpacity={s.network === 'IMPROVE' ? 0.35 : 0.65} strokeWidth={s.network === 'IMPROVE' ? 1 : 1.6} />
                  {picked ? <path d={star} transform={`translate(${x(s.rank)},${y(s.p50)})`} fill={color} stroke={INK.panel} strokeWidth={0.8} />
                    : <circle cx={x(s.rank)} cy={y(s.p50)} r={s.network === 'IMPROVE' ? 2.5 : 4.5} fill={color} />}
                </g>
              })}
              <HitLayer items={shown} xy={xy} width={innerW} height={innerH} hit={12}
                onHover={(e, s) => tip.show(e, [`${s.name} (${s.code}) · ${s.network}${s.active ? ' · active Samples site' : ''}`, `median ${val(s.p50)} · IQR ${val(s.p25)}–${val(s.p75)}`, `n = ${s.n} · ${rankText(s)}`, 'click for site details in the sidebar'])}
                onLeave={tip.hide} onPick={s => { tip.hide(); setOpen(keyOf(s)) }} />
              <g data-active-site-labels="" pointerEvents="none">
                {selected.length <= 8 && selected.map(s => {
                  // the name turns to the star's left when it would run past the plot (measured at 10.5 px, lib/fitLabel)
                  const end = x(s.rank) + 9 + textWidth(s.name, 10.5, 600) > innerW
                  return <text key={keyOf(s)} x={x(s.rank) + (end ? -9 : 9)} y={y(s.p50) - 9}
                    textAnchor={end ? 'end' : 'start'} fontSize={10.5} fontWeight={600}
                    fill={INK.text} paintOrder="stroke" stroke={INK.panel} strokeWidth={3} strokeLinejoin="round">{s.name}</text>
                })}
              </g>
            </g>
          </svg>
          <Legend items={[
            ...(needsImprove ? [{ label: 'IMPROVE', color: IMPROVE_GREY, shape: 'dot' as const, detail: `${improve.length} sites` }] : []),
            ...(needsSpartan ? [{ label: 'SPARTAN', color: INK.accent, shape: 'dot' as const, detail: `${spartan.length} of ${spIndex?.sites.length ?? 0} sites${o.values.colours === 'site' ? ' · site colours' : ''}` }] : []),
          ]} {...lg.props} note={needsSpartan ? scopeText : 'IMPROVE uses fixed exported site quantiles.'} />
          <Legend items={active.map(s => ({ label: s.name, color: s.color, shape: 'diamond' as const, detail: `active · n=${s.n}` }))}
            note="Active Samples sites stay visible in every comparison mode and keep their Samples colours." />
          {missingActive.length > 0 && <p className="chart-note">Active sites without a valid {field} value under this filter population: {missingActive.join(', ')}.</p>}
          {o.values.scale === 'Log' && !log && <p className="chart-note">Using a linear scale because a displayed lower quartile is zero or negative.</p>}
        </>}
      {(!quantity || massClash || (!loading && !loadError && !ranked.length)) && <MeasurementControl />}
      {tip.node}
    </div>
    <ExcludedNote fields={parts} />
    {!blocked && openSite && <DrawerShell portal label={`${openSite.name} (${openSite.code}) · ${openSite.network}`} onClose={() => setOpen(null)}>
      <header className="drawer-head"><div>
        <div className="drawer-id"><span className="swatch" style={{ background: colorOf(openSite) }} />{openSite.name}</div>
        <div className="drawer-sub">{openSite.code} · {openSite.network}{openSite.active ? ' · active Samples site' : ''}</div>
      </div><button type="button" className="btn quiet x-btn" aria-label="Close site details" onClick={() => setOpen(null)}><XIcon /></button></header>
      <section className="drawer-section"><h3>{withUnit(field, units)}</h3><p>{rankText(openSite)}</p>
        <table className="placement"><tbody>
          <tr><th>Filters</th><td>{openSite.n.toLocaleString()}</td></tr>
          <tr><th>25th percentile</th><td>{val(openSite.p25)}</td></tr>
          <tr><th>Median</th><td>{val(openSite.p50)}</td></tr>
          <tr><th>75th percentile</th><td>{val(openSite.p75)}</td></tr>
        </tbody></table>
      </section>
      <section className="drawer-section"><h3>Population</h3>
        <p>{POPULATIONS.find(p => p.value === pop)?.label}. {openSite.active ? 'Current Samples filters, dates, seasons, source and exclusions. The star keeps its Samples colour in every comparison mode.'
          : openSite.network === 'IMPROVE' ? 'All dates and seasons in the IMPROVE quantile export; the gallery selection does not change these statistics.' : scopeText}</p>
        <p>{openSite.network === 'IMPROVE' ? `Reference quantity: ${quantity?.label}. Carbon is measured by thermal-optical reflectance (TOR).` : 'Carbon is FTIR-derived. ChemSpec carbon is the same published product and is counted once.'}</p>
        {openSite.network === 'IMPROVE' && ref && <p>{ref.data_use}</p>}
      </section>
    </DrawerShell>}
    </ChartFrame>
  </AxisFieldsProvider>
}
