import { Segmented } from '@/components/ChartFrame'
import type { SectionPageProps } from '../../types'
import { MEMBER_SETS } from '../common/lotSpectra'
import { BandMeans } from './lots/BandMeans'
import { EXPORTER, SET_LABEL, SET_TITLE, useLotExport, useLots, useMemberSet } from './lots/data'
import { EscapeByLot } from './lots/EscapeByLot'
import { LotPanels } from './lots/LotPanels'
import { Waiting } from './spectralCommon'

/**
 * PTFE lot comparison for any site or sites (char_12 §4: 04_lot_effect, 04b_lot_airspec_view, lot_effect.csv,
 * lot_bands_within_dry.csv). Per-filter spectra in three views (raw, char_12's 2nd derivative, AIRSpec
 * area-normalised) and char_12's envelope-escape score, for Addis and the SPARTAN pulls (Beijing, Delhi, JPL,
 * Bahir Dar), exported by gallery/data/export_research_lot_spectra.py; every centre, band, box and test is
 * recomputed in the browser over the filters in view (common/lotSpectra.ts), split by lot and held to a season.
 */
export function LotSpectra({ rows, meta }: SectionPageProps) {
  const { data: exp, error } = useLotExport()
  const [set, setSet] = useMemberSet()
  const lots = useLots(exp, rows, meta, set)
  if (!exp || !lots) return <Waiting error={error} what="the lot spectra" exporter={EXPORTER} />
  const pick = <Segmented label="Filters" value={set} options={MEMBER_SETS} onChange={setSet} optionLabel={(o) => SET_LABEL[o]} title={SET_TITLE} />
  return <>
    <LotPanels exp={exp} lots={lots} id="ls-deriv" prefix="ls.d" figure="figure 04 (left)" setPick={pick} defaults={{ view: 'deriv', region: 'ch', band: '5-95' }} />
    <EscapeByLot exp={exp} lots={lots} setPick={pick} />
    <LotPanels exp={exp} lots={lots} id="ls-airspec" prefix="ls.a" figure="figure 04b" setPick={pick} defaults={{ view: 'airspec', region: 'full', band: '10-90' }} />
    <BandMeans exp={exp} lots={lots} setPick={pick} />
  </>
}
