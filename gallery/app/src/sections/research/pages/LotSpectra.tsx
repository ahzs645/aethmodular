import type { SectionPageProps } from '../../types'
import { SampleControls } from '@/lib/pageSidebar'
import { patchHash } from '@/lib/url'
import { BandMeans } from './lots/BandMeans'
import { EXPORTER, NOTEBOOK_CALENDAR, useLotExport, useLots, type Lots } from './lots/data'
import { EscapeByLot } from './lots/EscapeByLot'
import { LotPanels } from './lots/LotPanels'
import { Waiting } from './spectralCommon'

/**
 * The Samples settings that show char_12's own 239 Addis filters: Addis alone, its calendar (February in Dry), its
 * sampling dates (which leaves out the spectra scanned after it ran), every season, the export's exclusions included
 * and no PMF pick. Hand exclusions stay as they are. One hash patch, so ⌘Z puts the reader's back.
 */
function applyCohortSamples(cohort: Lots['cohort']) {
  patchHash({
    net: null, ssites: 'ETAD', sites: null, cal: NOTEBOOK_CALENDAR, cs: null, rr: null, seas: null, seasons: null,
    from: cohort.from, to: cohort.to, excl: '1', incl: null, pmfon: null, pmfonly: null,
  })
}

/** the preset at the top of the Samples tab, and whether the filters shown are char_12's 239 now */
function CohortPreset({ lots }: { lots: Lots }) {
  const { cohort, notebook } = lots
  if (!cohort.n) return null
  return <SampleControls>
    <div className="row">
      <span className="row-label">preset</span>
      <button type="button" className={`btn${notebook ? ' active' : ''}`} aria-pressed={notebook} onClick={() => applyCohortSamples(cohort)}
        title={`char_12's own ${cohort.n} Addis filters: Addis alone, its calendar (Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep), its dates (${cohort.from} to ${cohort.to}), every season, the export's exclusions included. Hand exclusions are kept. The charts' numbers then compare like for like with char_12's printed ones.`}>
        char_12's {cohort.n}
      </button>
      <span className="method-note" style={{ display: 'block' }}>{notebook ? `showing char_12's ${cohort.n}: its printed numbers compare like for like` : `${lots.members.length} filters with a spectrum shown`}</span>
    </div>
  </SampleControls>
}

/**
 * PTFE lot comparison for any site or sites (char_12 §4: 04_lot_effect, 04b_lot_airspec_view, lot_effect.csv,
 * lot_bands_within_dry.csv). Per-filter spectra in three views (raw, char_12's 2nd derivative, AIRSpec
 * area-normalised) and char_12's envelope-escape score, for Addis and the SPARTAN pulls (Beijing, Delhi, JPL,
 * Bahir Dar), exported by gallery/data/export_research_lot_spectra.py; every centre, band, box and test is
 * recomputed in the browser over the filters in view (common/lotSpectra.ts), split by lot and held to a season.
 */
export function LotSpectra({ rows, meta }: SectionPageProps) {
  const { data: exp, error } = useLotExport()
  const lots = useLots(exp, rows, meta)
  if (!exp || !lots) return <Waiting error={error} what="the lot spectra" exporter={EXPORTER} />
  return <>
    <CohortPreset lots={lots} />
    <LotPanels exp={exp} lots={lots} id="ls-deriv" prefix="ls.d" figure="figure 04 (left)" defaults={{ view: 'deriv', region: 'ch', band: '5-95' }} />
    <EscapeByLot exp={exp} lots={lots} />
    <LotPanels exp={exp} lots={lots} id="ls-airspec" prefix="ls.a" figure="figure 04b" defaults={{ view: 'airspec', region: 'full', band: '10-90' }} />
    <BandMeans exp={exp} lots={lots} />
  </>
}
