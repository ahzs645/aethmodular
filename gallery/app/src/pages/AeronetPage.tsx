import { useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import { SidebarControls } from '@/lib/pageSidebar'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { Segmented } from '@/components/ChartFrame'
import { Scatterplot } from '@/charts/correlation/Scatterplot'
import { AeronetTimeline } from '@/charts/aeronet/AeronetTimeline'
import { AeronetSiteTable, ScaleHeightStrip } from '@/charts/aeronet/AeronetSites'
import { AeronetScaleHeightTime } from '@/charts/aeronet/AeronetScaleHeightTime'
import { AeronetSsaSpectrum } from '@/charts/aeronet/AeronetSsaSpectrum'
import { AeronetTyping } from '@/charts/aeronet/AeronetTyping'
import { AeronetNonAbsorbing } from '@/charts/aeronet/AeronetNonAbsorbing'
import { AeronetEom } from '@/charts/aeronet/AeronetEom'
import { AeronetClimatology } from '@/charts/aeronet/AeronetClimatology'
import { EtaBySite } from '@/charts/aeronet/EtaBySite'
import { useExport } from '@/lib/data'
import { usePrefs } from '@/lib/prefs'
import { useSearchParam } from '@/lib/url'
import { useSiteMode } from '@/lib/siteFocus'
import { SiteModeHead } from '@/components/SiteModeHead'
import { RATIO_GROUP } from '@/lib/viewFields'
import { AAOD675, hasAod, LEVEL_PARAM, LEVELS, levelLabel, stationsPath, useAeronet, withLevel, type AeronetLevel, type AeronetStations } from '@/lib/aeronet'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile } from '@/lib/types'

/**
 * The sun photometer over the filters, driven by the Samples panel's site pick (no picker of its
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
 * Then, either way, the surface–column notebooks' figures (surface_columnar_linkage, _figures): the
 * scale height through time, the SSA spectrum, aerosol typing on the Ångström planes, the
 * non-absorbing fraction against SSA, the Eom et al. tests and the monthly climatology.
 *
 * Every filter carries the mean of the retrievals inside its own sampling window
 * (gallery/data/export_aeronet.py); dates, seasons and hand edits apply as on every tab. The Level
 * toggle (`aerlev`) swaps this tab's AERONET values for Level 2.0; the other tabs keep Level 1.5.
 */
export function AeronetPage({ rows: rows15, meta, field, axes, setField }: {
  rows: FilterRow[]
  /** unused: one site opens through the Page › analysis choice (lib/siteFocus), not by narrowing the pick */
  picked?: Set<string>
  onPick?: (codes: string[]) => void
  meta: MetaFile; field: string; axes: AxesOpts
  setField: (f: string) => void
}) {
  const [level, setLevel] = useSearchParam<AeronetLevel>(LEVEL_PARAM, '15', LEVELS)
  const { explain } = usePrefs()
  const stations = useExport<AeronetStations>(stationsPath(level)).data
  // Level 2.0: the same rows with this tab's AERONET fields (and the reader's ratios on them) from aeronet/lev20/
  const codes = useMemo(() => [...new Set(rows15.map((r) => r.code))], [rows15])
  const lev20 = useAeronet(level === '20' ? codes : [], '20')
  const ratioFields = useMemo(() => meta.field_groups?.find((g) => g.label === RATIO_GROUP)?.fields ?? [], [meta.field_groups])
  const rows = useMemo(() => (level === '20' ? withLevel(rows15, lev20, ratioFields) : rows15), [level, rows15, lev20, ratioFields])

  // the picked sites' filters that have a retrieval in their window
  const allAxRows = useMemo(() => rows.filter((r) => hasAod(r) || typeof r[AAOD675] === 'number'), [rows])
  const sites = useMemo(() => {
    const n = d3.rollup(allAxRows, (v) => v.length, (r) => r.site)
    return meta.sites.filter((s) => n.has(s.name)).map((s) => ({ name: s.name, code: s.code, n: n.get(s.name)! }))
  }, [allAxRows, meta.sites])
  // several sites picked: the comparison, or one of them on its own (Page › analysis, or a click on a site);
  // the one-site view draws that site's filters only, as a pick of it alone would
  const mode = useSiteMode(sites)
  const one = mode.one
  const focusKey = mode.focused ? one?.code : ''
  const axRows = useMemo(() => mode.rowsOf(allAxRows), [allAxRows, focusKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const inView = useMemo(() => mode.rowsOf(rows), [rows, focusKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const inView15 = useMemo(() => mode.rowsOf(rows15), [rows15, focusKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const station = one ? stations?.sites[one.code] : undefined

  // how much absorption each level leaves in view: Level 2.0 keeps an inversion's AAOD only at AOD 440 ≥ 0.4
  const absorption = useMemo(() => ({
    l15: inView15.filter((r) => typeof r[AAOD675] === 'number').length,
    here: inView.filter((r) => typeof r[AAOD675] === 'number').length,
  }), [inView15, inView])

  // the scatter's x is an AERONET quantity of its own (in the URL); its y is the tab's measurement
  const [axX, setAxX] = useSearchParam<string>('aeronet.x', AAOD675)
  const axisValue = useMemo(() => ({ x: axX, y: field, field }), [axX, field])
  const setAxis = useCallback((role: AxisRole, f: string) => (role === 'x' ? setAxX(f) : setField(f)), [setField]) // eslint-disable-line react-hooks/exhaustive-deps
  const scatter = (title: string, id: string) => (
    <AxisFieldsProvider meta={meta} value={axisValue} set={setAxis}>
      <Scatterplot rows={axRows.filter((r) => typeof r[axX] === 'number')} meta={meta} xField={axX} yField={field}
        axes={{ ...axes, identity: false, mode: 'Data' }} id={id} title={title} />
    </AxisFieldsProvider>
  )

  const levelControl = <Segmented label="AERONET" value={level} options={LEVELS} onChange={setLevel} optionLabel={levelLabel}
    title="Level 1.5: cloud-screened, not final-calibrated, every inversion. Level 2.0: final, months behind, absorption only at AOD 440 ≥ 0.4. The other tabs keep Level 1.5." />
  const levelNote = level === '20' && (
    <p className="subtitle" style={{ margin: '6px 0 0' }}>
      <strong>Level 2.0</strong>: final-calibrated, but it lags by months and keeps an inversion's absorption (AAOD, SSA, AAE) only at AOD 440 ≥ 0.4.
      {' '}{absorption.here} of the filters in view have an absorption AOD at Level 2.0, against {absorption.l15} at Level 1.5
      {absorption.here < Math.max(20, absorption.l15 * 0.3) ? ' — too few for the absorption charts below (scale height, SSA, typing, w) to say much here; read those at Level 1.5, and Level 2.0 as the check on the AOD side.' : '.'}
    </p>
  )

  if (level === '20' && !lev20) {
    return <div className="panel"><div className="weather-head"><p className="frame-empty">Loading the Level 2.0 AERONET…</p><SidebarControls title="AERONET">{levelControl}</SidebarControls></div></div>
  }
  if (sites.length === 0) {
    return <div className="panel">
      <div className="weather-head"><p className="frame-empty">No picked site has {levelLabel(level)} AERONET retrievals inside its filters' sampling windows. AERONET is pulled for SPARTAN sites with a station within 50 km: pick one or more SPARTAN sites in the Samples panel.</p>
        <SidebarControls title="AERONET">{levelControl}</SidebarControls></div>
    </div>
  }

  const span = one ? `${d3.min(axRows, (r) => r.date)} to ${d3.max(axRows, (r) => r.date)}` : ''
  return (
    <>
      <SidebarControls title="AERONET">{levelControl}</SidebarControls>
      {/* the comparison's heading is reading guidance (only with Explain on); one site's heading and the Level 2.0 caveat always show */}
      {one ? <SiteModeHead mode={mode} title={`${one.name} · the column over the filters`} extra={levelNote}>
        {one.n} filters, {span}, each with the mean of the AERONET retrievals inside its own sampling window
        {station ? ` at ${station.station.replace(/_/g, ' ')}, ${station.km.toFixed(1)} km away (of all ${station.filters} filters from this site, ${station.with_aod} have AOD and ${station.with_inversion} an inversion)` : ''}.
        {' '}{levelLabel(level)}: daylight, cloud-free retrievals only.
      </SiteModeHead> : (explain || levelNote) && <SiteModeHead mode={mode} title={explain ? `AERONET and the filters · ${sites.length} sites compared` : undefined} extra={levelNote}>
        {explain && <>The sites picked in the Samples panel, each filter with the mean of the {levelLabel(level)} retrievals inside its sampling window at the
          site's AERONET station. Click a site in the table (or choose One site under Page › analysis) for its daily timeline.</>}
      </SiteModeHead>}
      {one ? <>
        <AeronetTimeline rows={axRows} meta={meta} field={field} site={one} station={station?.station.replace(/_/g, ' ')} level={level} />
        {scatter(`${one.name} · ${field} against the column`, 'aeronet-site-scatter')}
        <ScaleHeightStrip rows={axRows} meta={meta} by="season" />
      </> : <>
        <AeronetSiteTable rows={axRows} meta={meta} stations={stations} current={null} onSite={mode.open} />
        <ScaleHeightStrip rows={axRows} meta={meta} onSite={mode.open} />
        {scatter(`${field} against the column — the picked sites`, 'aeronet-compare-scatter')}
      </>}
      {/* PM2.5 / AOD (η) by site and season, or by season at one site (charts/aeronet/EtaBySite.tsx) */}
      <EtaBySite rows={axRows} meta={meta} />
      <AeronetScaleHeightTime rows={axRows} meta={meta} />
      <AeronetClimatology sites={one ? [one] : sites} level={level} />
      <AeronetSsaSpectrum rows={axRows} meta={meta} />
      <AeronetTyping rows={axRows} meta={meta} />
      <AeronetNonAbsorbing rows={axRows} meta={meta} />
      <AeronetEom rows={axRows} meta={meta} />
    </>
  )
}
