import { useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import { SidebarControls } from '@/lib/pageSidebar'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { Segmented, Toggle } from '@/components/ChartFrame'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { useSearchParam } from '@/lib/url'
import { useSiteMode } from '@/lib/siteFocus'
import { SiteModeHead } from '@/components/SiteModeHead'
import type { SectionPageProps } from '../../types'
import { useAethFields } from '../lib'
import { CHANNELS, NEAREST_TO_HIPS, bccField, fallbackSites, pairSites, useAethExtra, usePairs, type Channel } from '../filters/data'
import { ChannelAgreement } from '../filters/ChannelAgreement'
import { ConstraintCheck } from '../filters/ConstraintCheck'
import { ImpliedMac } from '../filters/ImpliedMac'
import { IronTest } from '../filters/IronTest'
import { MacTimeline } from '../filters/MacTimeline'
import { MethodAgreement } from '../filters/MethodAgreement'
import { useMarkedRows } from '@/lib/markedRows'

/**
 * Sub-page 'filters' of the Aethalometer tab: the MA350 against the filters it ran beside, HIPS Fabs
 * (633 nm) and FTIR EC, the MAC each pairing implies, and whether iron explains the absorption EC
 * does not. Driven by the Filters panel's site pick like the AERONET tab: one site picked gives its
 * detail, several the comparison (click a site to open it alone).
 *
 * Every filter carries the aethalometer averaged over its own sampling window (../lib.ts,
 * gallery/data/export_aethalometer.py); smoothed channels and spot attenuation come from
 * gallery/data/export_aeth_filters.py. The channel chosen here (IR by default, as the notebooks;
 * Red is the nearest to HIPS) drives every chart on the page.
 */
export function FilterComparison({ rows: allRows, meta, field, axes, setField }: SectionPageProps) {
  const [ch, setCh] = useSearchParam<Channel>('af.ch', 'IR', CHANNELS)
  const [sm, setSm] = useSearchParam<'0' | '1'>('af.sm', '0', ['0', '1'])
  const smoothed = sm === '1'
  const codes = useMemo(() => [...new Set(allRows.map((r) => r.code))], [allRows])
  const extra = useAethExtra(codes)
  // the aethalometer's own counts per filter (overlap hours), for the header
  const fields = useAethFields(codes)
  const allPairs = usePairs(allRows, meta, extra, ch, smoothed)
  // Filters → excluded: Mark (lib/markedRows): these sites' excluded filters, paired the same way: × on the point
  // charts, a count under the summaries, never in a fit, median or count
  const markedAll = useMarkedRows()
  const markedRows = useMemo(() => markedAll.filter((r) => codes.includes(r.code)), [markedAll, codes])
  const allMarkedPairs = usePairs(markedRows, meta, extra, ch, smoothed)
  const allSites = useMemo(() => pairSites(allPairs, meta), [allPairs, meta])
  // several sites: the comparison, or one of them on its own (Page › analysis, or a click in the MAC strip),
  // drawn from that site's filters only, as a pick of it alone would
  const mode = useSiteMode(allSites)
  const one = mode.one
  const focusKey = mode.focused ? one?.code : ''
  const sites = useMemo(() => (one ? [one] : allSites), [one, allSites])
  const rows = useMemo(() => mode.rowsOf(allRows), [allRows, focusKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const pairs = useMemo(() => mode.rowsOf(allPairs), [allPairs, focusKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const markedPairs = useMemo(() => mode.rowsOf(allMarkedPairs), [allMarkedPairs, focusKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const sigma = extra.index?.sigma[ch] ?? null
  const wavelengths = extra.index?.wavelengths_nm ?? meta.wavelengths_nm

  // the free crossplot: x an aethalometer field (the page's channel until another is picked), y the tab's measurement
  // (in the URL, so a link reopens it; empty: follow the channel)
  const [axX, setAxX] = useSearchParam<string>('af.x', '')
  const xField = axX || bccField(ch, smoothed && ch === 'IR')
  const axisValue = useMemo(() => ({ x: xField, y: field, field }), [xField, field])
  const setAxis = useCallback((role: AxisRole, f: string) => (role === 'x' ? setAxX(f) : setField(f)), [setField]) // eslint-disable-line react-hooks/exhaustive-deps
  const scatterRows = useMemo(() => rows.filter((r) => typeof r[xField] === 'number'), [rows, xField])

  if (allSites.length === 0) {
    return <div className="panel"><p className="frame-empty">No picked site has aethalometer values over its filters. The MA350 ran beside the filters at the four focus sites (Beijing, Delhi, JPL, Addis Ababa): pick one or more of them in the Filters panel.</p></div>
  }
  const span = `${d3.min(pairs, (p) => p.date)} to ${d3.max(pairs, (p) => p.date)}`
  const overlap = fields ? d3.median(pairs, (p) => fields.counts[p.id]?.[0]) : undefined
  const nSmoothed = pairs.filter((p) => p.bcKind === 'smoothed').length
  const rawSites = fallbackSites(pairs)
  return (
    <>
      <SiteModeHead mode={mode} panel="Filters" title={one ? `${one.name} · the aethalometer against its filters` : `${sites.length} sites compared`} controls={<SidebarControls title="Aethalometer channel">
          <Segmented label="Channel" value={ch} options={CHANNELS} onChange={setCh}
            optionLabel={(c) => `${c}${c === NEAREST_TO_HIPS ? ' ≈ HIPS' : ''}`} title={`MA350 channels: ${CHANNELS.map((c) => `${c} ${wavelengths[c]} nm`).join(', ')}. HIPS reads at 633 nm.`} />
          <Toggle label="Smoothed BCc" checked={smoothed} onChange={(v) => setSm(v ? '1' : '0')}
            title="The instrument's smoothed BCc (ETAD_Aug13 used Red and IR smoothed). Beijing, Delhi and JPL carry only some smoothed channels: where a filter has none, the raw BCc is used and the axis labels say so" />
        </SidebarControls>}>
        {pairs.length} filters, {span}, each with the MA350 averaged over its own sampling window
        {overlap ? ` (median ${Math.round(overlap)} h of daily bins inside it; at least 12 h kept)` : ''}.
        The pickles are legacy daily aggregates whose collection windows were never verified (processed_sites/README.md).
        {!one && ' Click a site in the MAC strip (or choose One site under Page › analysis) for its detail.'}
        {smoothed && rawSites.length ? ` Smoothed ${ch} BCc for ${nSmoothed} of them; no smoothed ${ch} series at ${rawSites.join(', ')}${nSmoothed ? ' (for some or all filters)' : ''}, where the raw BCc is used and the axis labels say so.` : ''}
      </SiteModeHead>
      <ImpliedMac pairs={pairs} marked={markedPairs} meta={meta} sites={sites} ch={ch} sigma={sigma} onSite={!one ? mode.open : undefined} />
      <MethodAgreement pairs={pairs} marked={markedPairs} meta={meta} sites={sites} ch={ch} />
      <ChannelAgreement rows={rows} sites={sites} extra={extra.values} wavelengths={wavelengths} ch={ch} onChannel={setCh} />
      <MacTimeline pairs={pairs} marked={markedPairs} meta={meta} sites={sites} ch={ch} sigma={sigma} />
      <ConstraintCheck pairs={pairs} marked={markedPairs} sites={sites} ch={ch} sigma={sigma} />
      <IronTest pairs={pairs} marked={markedPairs} meta={meta} sites={sites} ch={ch} sigma={sigma} />
      <AxisFieldsProvider meta={meta} value={axisValue} set={setAxis}>
        <Scatterplot rows={scatterRows} meta={meta} xField={xField} yField={field} axes={{ ...axes, identity: false, mode: 'Data' }}
          id="aeth-filters-scatter" title={`${field} against ${xField} — any pair, per site (click an axis title to change it)`} />
      </AxisFieldsProvider>
    </>
  )
}
