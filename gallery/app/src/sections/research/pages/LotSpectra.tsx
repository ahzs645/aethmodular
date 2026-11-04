import type { SectionPageProps } from '../../types'
import { SampleControls } from '@/lib/pageSidebar'
import { patchHash } from '@/lib/url'
import { BandMeans } from './lots/BandMeans'
import { EXPORTER, NOTEBOOK_CALENDAR, useLotExport, useLots, type Lots } from './lots/data'
import type { LotExport } from '../common/lotSpectra'
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

/**
 * Bishoftu beside Addis: both sites, the shared Ethiopian calendar (so the season each spectra panel holds fixed has
 * one name at both), every date and season, and the spectra charts split per site. Bishoftu's filters are all lot
 * 251 (Oct 2025 – Feb 2026); the Addis spectra of those months have no lot, so its lot 251 is the earlier years'.
 * Exclusions and the PMF pick stay as they are.
 */
const BISHOFTU = ['ETAD', 'ETBI']
function applyBishoftuSamples() {
  patchHash({ net: null, ssites: BISHOFTU.join(','), sites: null, cal: NOTEBOOK_CALENDAR, cs: null, rr: null, seas: null, seasons: null, from: null, to: null },
    { 'ls.d.layout': 'site', 'ls.a.layout': 'site', 'ls.e.layout': null })
}

/** the presets at the top of the Samples tab, and which one the filters shown match now */
function LotPresets({ lots, exp }: { lots: Lots; exp: LotExport }) {
  const { cohort, notebook } = lots
  const codes = new Set(lots.members.map((m) => m.code))
  const bishoftu = codes.size === BISHOFTU.length && BISHOFTU.every((c) => codes.has(c))
  const nBishoftu = exp.filters.code.filter((c) => c === 'ETBI').length
  return <SampleControls>
    <div className="row">
      <span className="row-label">presets</span>
      <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 6 }}>
        {cohort.n > 0 && <button type="button" className={`btn${notebook ? ' active' : ''}`} aria-pressed={notebook} onClick={() => applyCohortSamples(cohort)}
          title={`char_12's own ${cohort.n} Addis filters: Addis alone, its calendar (Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep), its dates (${cohort.from} to ${cohort.to}), every season, the export's exclusions included. Hand exclusions are kept. The charts' numbers then compare like for like with char_12's printed ones.`}>
          char_12's {cohort.n}
        </button>}
        {nBishoftu > 0 && <button type="button" className={`btn${bishoftu ? ' active' : ''}`} aria-pressed={bishoftu} onClick={applyBishoftuSamples}
          title={`Bishoftu's ${nBishoftu} spectra beside Addis: both sites, the Ethiopian calendar with February in Dry at both, every date and season, and the spectra charts per site. Bishoftu is lot 251 only (Oct 2025 – Feb 2026), so it has no lot contrast of its own: compare its lot 251 with Addis's, season held fixed. The Addis spectra of Bishoftu's months have no lot, so Addis's lot 251 is from earlier years. Exclusions stay as they are.`}>
          Bishoftu vs Addis
        </button>}
      </span>
      <span className="method-note" style={{ display: 'block' }}>
        {notebook ? `showing char_12's ${cohort.n}: its printed numbers compare like for like`
          : bishoftu ? `Bishoftu and Addis: Bishoftu is lot 251 only, so compare its panel with Addis's lot 251`
          : `${lots.members.length} filters with a spectrum shown`}
      </span>
    </div>
  </SampleControls>
}

/**
 * PTFE lot comparison for any site or sites (char_12 §4: 04_lot_effect, 04b_lot_airspec_view, lot_effect.csv,
 * lot_bands_within_dry.csv). Per-filter spectra in three views (raw, char_12's 2nd derivative, AIRSpec
 * area-normalised) and char_12's envelope-escape score, for Addis and the SPARTAN pulls (Beijing, Delhi, JPL,
 * Bishoftu), exported by gallery/data/export_research_lot_spectra.py; every centre, band, box and test is
 * recomputed in the browser over the filters in view (common/lotSpectra.ts), split by lot and held to a season.
 */
export function LotSpectra({ rows, meta }: SectionPageProps) {
  const { data: exp, error } = useLotExport()
  const lots = useLots(exp, rows, meta)
  if (!exp || !lots) return <Waiting error={error} what="the lot spectra" exporter={EXPORTER} />
  return <>
    <LotPresets lots={lots} exp={exp} />
    <LotPanels exp={exp} lots={lots} id="ls-deriv" prefix="ls.d" figure="figure 04 (left)" defaults={{ view: 'deriv', region: 'ch', band: '5-95' }} />
    <EscapeByLot exp={exp} lots={lots} />
    <LotPanels exp={exp} lots={lots} id="ls-airspec" prefix="ls.a" figure="figure 04b" defaults={{ view: 'airspec', region: 'full', band: '10-90' }} />
    <BandMeans exp={exp} lots={lots} />
  </>
}
