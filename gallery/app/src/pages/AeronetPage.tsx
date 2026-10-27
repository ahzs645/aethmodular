import { useCallback, useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { AeronetTimeline } from '@/charts/aeronet/AeronetTimeline'
import { AeronetSiteTable, ScaleHeightStrip } from '@/charts/aeronet/AeronetSites'
import { EtaBySite } from '@/charts/aeronet/EtaBySite' // aeronet-sites
import { useExport } from '@/lib/data'
import { AAOD675, hasAod, type AeronetStations } from '@/lib/aeronet'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile } from '@/lib/types'

/**
 * The sun photometer over the filters, driven by the Filters panel's site pick (no picker of its
 * own), laid out like the Weather tab.
 *
 * One site picked: the daily column (AOD, fine fraction, absorption AOD, SSA) above the chosen
 * measurement, then the measurement against any AERONET quantity over the filters' windows. The
 * views the Addis notebooks built (addis_04_aeronet, aeronet_addis_diagnostics,
 * aeronet_method_agreement) for any SPARTAN site with a station.
 *
 * Several sites picked: the comparison: which station stands for each site and how much of its
 * filter record it covers, the absorption scale height by site, and the same scatter.
 *
 * Every filter carries the mean of the retrievals inside its own sampling window
 * (gallery/data/export_aeronet.py); dates, seasons and hand edits apply as on every tab.
 */
export function AeronetPage({ rows, picked, onPick, meta, field, axes, setField }: {
  rows: FilterRow[]
  /** the Filters panel's SPARTAN pick, and a way to set it (absent outside a SPARTAN network) */
  picked: Set<string>
  onPick?: (codes: string[]) => void
  meta: MetaFile; field: string; axes: AxesOpts
  setField: (f: string) => void
}) {
  const stations = useExport<AeronetStations>('aeronet/stations.json').data

  // the picked sites' filters that have a retrieval in their window
  const axRows = useMemo(() => rows.filter((r) => hasAod(r) || typeof r[AAOD675] === 'number'), [rows])
  const sites = useMemo(() => {
    const n = d3.rollup(axRows, (v) => v.length, (r) => r.site)
    return meta.sites.filter((s) => n.has(s.name)).map((s) => ({ name: s.name, code: s.code, n: n.get(s.name)! }))
  }, [axRows, meta.sites])
  const one = sites.length === 1 ? sites[0] : null
  const station = one ? stations?.sites[one.code] : undefined

  const [previous, setPrevious] = useState<string[] | null>(null)
  useEffect(() => { if (sites.length > 1) setPrevious(null) }, [sites.length])
  const openSite = useCallback((code: string) => {
    if (!onPick) return
    setPrevious([...picked])
    onPick([code])
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [onPick, picked])

  // the scatter's x is an AERONET quantity of its own; its y is the tab's measurement
  const [axX, setAxX] = useState(AAOD675)
  const axisValue = useMemo(() => ({ x: axX, y: field, field }), [axX, field])
  const setAxis = useCallback((role: AxisRole, f: string) => (role === 'x' ? setAxX(f) : setField(f)), [setField])
  const scatter = (title: string, id: string) => (
    <AxisFieldsProvider meta={meta} value={axisValue} set={setAxis}>
      <Scatterplot rows={axRows.filter((r) => typeof r[axX] === 'number')} meta={meta} xField={axX} yField={field}
        axes={{ ...axes, identity: false, mode: 'Data' }} id={id} title={title} />
    </AxisFieldsProvider>
  )

  if (sites.length === 0) {
    return <div className="panel"><p className="frame-empty">No picked site has AERONET retrievals inside its filters' sampling windows. AERONET is pulled for SPARTAN sites with a station within 50 km: pick one or more SPARTAN sites in the Filters panel.</p></div>
  }

  const level = stations ? `Level ${stations.level}` : 'AERONET'
  if (one) {
    const span = `${d3.min(axRows, (r) => r.date)} to ${d3.max(axRows, (r) => r.date)}`
    return (
      <>
        <div className="panel weather-head">
          <div>
            <h2 style={{ margin: 0 }}>{one.name} · the column over the filters</h2>
            <p className="subtitle" style={{ margin: '4px 0 0' }}>
              {one.n} filters, {span}, each with the mean of the AERONET retrievals inside its own sampling window
              {station ? ` at ${station.station.replace(/_/g, ' ')}, ${station.km.toFixed(1)} km away (of all ${station.filters} filters from this site, ${station.with_aod} have AOD and ${station.with_inversion} an inversion)` : ''}.
              {' '}{level}: daylight, cloud-free retrievals only. Pick more sites in the Filters panel to compare them.
            </p>
          </div>
          <div className="weather-head-controls">
            {previous && previous.length > 1 && onPick && <button type="button" className="btn" onClick={() => { onPick(previous); setPrevious(null) }}
              title="Put the Filters panel's pick back to the sites you were comparing">← Back to comparing {previous.length} sites</button>}
          </div>
        </div>
        <AeronetTimeline rows={axRows} meta={meta} field={field} site={one} station={station?.station.replace(/_/g, ' ')} />
        {scatter(`${one.name} · ${field} against the column`, 'aeronet-site-scatter')}
        <ScaleHeightStrip rows={axRows} meta={meta} by="season" />
        {/* aeronet-sites: PM2.5 / AOD (η) by season (charts/aeronet/EtaBySite.tsx) */}
        <EtaBySite rows={axRows} meta={meta} />
      </>
    )
  }

  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>AERONET and the filters · {sites.length} sites compared</h2>
          <p className="subtitle" style={{ margin: '4px 0 0' }}>
            The sites picked in the Filters panel, each filter with the mean of the {level} retrievals inside its sampling window at the
            site's AERONET station. Click a site in the table{onPick ? ' (or pick just one in the Filters panel)' : ''} for its daily timeline.
          </p>
        </div>
      </div>
      <AeronetSiteTable rows={axRows} meta={meta} stations={stations} current={null} onSite={onPick ? openSite : undefined} />
      <ScaleHeightStrip rows={axRows} meta={meta} onSite={onPick ? openSite : undefined} />
      {scatter(`${field} against the column — the picked sites`, 'aeronet-compare-scatter')}
      {/* aeronet-sites: PM2.5 / AOD (η) by site and season (charts/aeronet/EtaBySite.tsx) */}
      <EtaBySite rows={axRows} meta={meta} />
    </>
  )
}
