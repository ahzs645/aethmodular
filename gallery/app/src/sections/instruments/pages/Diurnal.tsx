import { useMemo } from 'react'
import { ChartFrame, Segmented, Select } from '@/components/ChartFrame'
import { ExportState } from '@/components/ExportState'
import { useSearchParam } from '@/lib/url'
import { seasonsForSite } from '@/siteSeasons'
import type { SectionPageProps } from '../../types'
import { BrcTest, ChannelHealth, EcAgainstBc880 } from '../diurnal/BrcTest'
import { ColumnCycle, DaylightWindow } from '../diurnal/ColumnCharts'
import { DIURNAL_DIR, EXPORT_COMMAND, hourRuns, useBrc, useColumnIndex, useSiteFile, useSurfaceIndex, utcLabel, type ColumnFile, type SurfaceFile } from '../diurnal/data'
import { AaeCycle, ChannelCycle, ColumnCoupling, CoverageCheck, IntradayCv, MorningShare, SurfaceVsColumnAae } from '../diurnal/SurfaceCharts'

type Split = 'seasons' | 'together'

/**
 * Sub-page 'diurnal' of the Aethalometer tab: the day inside a 24 h filter.
 *
 * The column side (AERONET retrievals by local hour, every SPARTAN site with a station) and the ftir_28
 * brown-carbon test (the committed daily MA350 record) are always here. The surface side, hour-of-day
 * cycles from the MA350 minute files, appears once gallery/data/export_aeth_diurnal.py has run where
 * the Drive is mounted; until then the page says so and shows what was found earlier.
 */
export function Diurnal({ rows, meta }: SectionPageProps) {
  const column = useColumnIndex()
  const surface = useSurfaceIndex()
  const brc = useBrc()

  // the sites the Filters panel has in view, and those this page has anything for
  const inView = useMemo(() => {
    const codes = new Set(rows.map((r) => r.code))
    return meta.sites.filter((s) => codes.has(s.code))
  }, [rows, meta.sites])
  const withData = inView.filter((s) => column.data?.sites[s.code] || surface.data?.sites[s.code])
  const codes = withData.map((s) => s.code)
  const fallback = codes.includes('ETAD') ? 'ETAD' : codes[0] ?? ''
  // the options arrive with the index, so the key takes any value and an unknown one falls back here
  const [focus, setFocus] = useSearchParam<string>('dn.site', '')
  const code = codes.includes(focus) ? focus : fallback
  const [split, setSplit] = useSearchParam<Split>('dn.split', 'seasons', ['seasons', 'together'])
  const site = meta.sites.find((s) => s.code === code) ?? null
  const seasons = useMemo(() => (site ? seasonsForSite(meta, site.name) : []), [meta, site])

  const colFile = useSiteFile<ColumnFile>('aeronet', column.data?.sites[code] ? code : null)
  const surfFile = useSiteFile<SurfaceFile>('aeth', surface.data?.sites[code] ? code : null)
  const shade = colFile?.inv_hours ?? column.data?.sites[code]?.inv_hours ?? surfFile?.coverage?.hours ?? []
  const addisRows = useMemo(() => rows.filter((r) => r.code === 'ETAD'), [rows])
  const showBrc = inView.some((s) => s.code === 'ETAD')

  if (column.error && surface.error && brc.error) {
    return <ExportState title="Diurnal" what="the diurnal export" error={column.error} command={EXPORT_COMMAND} />
  }
  const splitOn = split === 'seasons' && seasons.length > 1
  const offset = colFile?.offset ?? surfFile?.offset ?? column.data?.sites[code]?.offset
  const surfaceSites = Object.keys(surface.data?.sites ?? {})

  return (
    <>
      <div className="panel weather-head">
        <div>
          <p className="subtitle" style={{ margin: 0 }}>
            A filter integrates 24 hours; the sun photometer sees only clear daylight, and the surface can swing several-fold between night and day.
            {site ? <> Looking at <b>{site.name}</b>{offset != null ? ` (local standard time, ${utcLabel(offset)})` : ''}{shade.length ? `, whose AERONET inversions fall at ${hourRuns(shade)}` : ''}.</> : ' No site in view has hourly data: pick SPARTAN sites in the Filters panel.'}
            {' '}Seasons follow {site ? `${site.name}'s` : "each site's"} calendar as set in the Filters panel.
          </p>
        </div>
        <div className="weather-head-controls">
          {codes.length > 1 && <Select label="Site" value={code} options={codes} onChange={setFocus} optionLabel={(c) => meta.sites.find((s) => s.code === c)?.name ?? c} />}
          {seasons.length > 1 && <Segmented label="Seasons" value={split} options={['seasons', 'together'] as Split[]} onChange={setSplit}
            optionLabel={(k) => (k === 'seasons' ? 'Split' : 'Together')} />}
        </div>
      </div>

      {column.data && <DaylightWindow index={column.data} sites={inView} focus={code} onFocus={setFocus} />}
      {colFile && site && <ColumnCycle file={colFile} site={site} seasons={seasons} split={splitOn} />}
      {column.error && <ExportState title="Diurnal · the column" what="the AERONET hour-of-day export" error={column.error} command={`${EXPORT_COMMAND} --skip-aeth`} />}

      {site && (surfFile
        ? <>
            <ChannelCycle file={surfFile} site={site} seasons={seasons} split={splitOn} shade={shade} />
            <AaeCycle file={surfFile} site={site} seasons={seasons} split={splitOn} shade={shade} />
            <CoverageCheck file={surfFile} site={site} seasons={seasons} split={splitOn} />
            <MorningShare file={surfFile} site={site} seasons={seasons} split={splitOn} />
            <ColumnCoupling file={surfFile} site={site} seasons={seasons} split={splitOn} shade={shade} />
            <SurfaceVsColumnAae file={surfFile} site={site} seasons={seasons} shade={shade} />
            <IntradayCv file={surfFile} site={site} seasons={seasons} />
          </>
        : !surface.loading && <SurfaceMissing site={site.name} code={code} exported={surfaceSites} searched={surface.data?.searched ?? {}} />)}

      {showBrc && brc.data && <>
        <BrcTest brc={brc.data} />
        <ChannelHealth brc={brc.data} />
        <EcAgainstBc880 rows={addisRows} />
      </>}
    </>
  )
}

/** what the surface side will show, why it is not here yet, and what was found when it was last run by hand */
function SurfaceMissing({ site, code, exported, searched }: { site: string; code: string; exported: string[]; searched: Record<string, string[]> }) {
  const files = searched[code]
  return (
    <ChartFrame id="diurnal-surface-missing" title={`${site} · the aethalometer through the day — not exported yet`} exportable={false}
      subtitle="Hour-of-day cycles need the MA350 minute records, which live on Google Drive; the committed processed_sites pickles are daily."
      provenance="research/ftir_hips_chem/addis_05_diurnal_wavelength_analysis.ipynb, multisite_diurnal_wavelength_analysis.ipynb, notebooks/analysis/diurnal_pub_figures.ipynb, research/ftir_ec_phase3/scripts/ma350_diurnal_coverage.py, notebooks/analysis/meteorology/aeronet_addis_{diagnostics,deep_dive}.ipynb, pw_mac_calibration.ipynb">
      <div className="subtitle" style={{ lineHeight: 1.55 }}>
        <p style={{ marginTop: 0 }}>
          <b>Not exported yet:</b> run <code>gallery/data/export_aeth_diurnal.py</code> where the Drive is mounted
          (<code>{EXPORT_COMMAND}</code>), then reload. It writes <code>public/data/{DIURNAL_DIR}/aeth/</code>.
          {exported.length > 0
            ? <> Exported so far for {exported.join(', ')}; {code} has no minute file.</>
            : files?.length ? <> For {code} it looks under <code>aethalometry_dir()</code> for {files.map((f, i) => <span key={f}>{i ? ' or ' : ''}<code>{f}</code></span>)}.</>
              : <> No minute file is registered for {code}; pass one with <code>--minute {code}=PATH</code>.</>}
        </p>
        <p>Once it has run, this page adds, for the site: BCc at the five wavelengths by local hour and season, and each over IR; AAE(UV, IR) and AAE(Blue, IR) by hour from absorption;
          Red BCc's 24 h mean against its mean over AERONET's hours, day by day; the share of a 24 h filter's black carbon from 06–10 and the other parts of the day;
          surface BC, AOD and BC/AOD at the photometer's moments by hour; surface AAE against the column's Ångström exponent; and each day's within-day variability of BC against AOD.</p>
        {(code === 'ETAD' || code === 'USPA') && <p style={{ marginBottom: 0 }}>
          Recorded when the coverage script was last run by hand (research/ftir_ec_phase3/AERONET_SPARTAN_2026-08-22.md, not recomputed here):
          the median per-day ratio of 24 h to AERONET-hours Red BCc was <b>1.05</b> from daily means and 0.92 from daily medians at Addis (hours 07–10 and 14–17, 751 days),
          and 1.09 and 1.07 at Pasadena (06–18, 327 days); hourly mean Red BCc at Addis ran from 3,987 ng/m³ at 00:00 to 19,096 at 04:00.
        </p>}
      </div>
    </ChartFrame>
  )
}
