import { ChartFrame, Segmented } from '@/components/ChartFrame'
import { fmt } from '@/lib/format'
import type { SectionPageProps } from '../../types'
import { MEMBER_SETS } from '../common/lotSpectra'
import { BandMeans } from './lots/BandMeans'
import { EXPORTER, FOLDER, NB, SET_LABEL, SET_TITLE, useLotExport, useLots, useMemberSet } from './lots/data'
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
  const f = exp.filters
  const perSite = [...new Set(f.code)].map((c) => `${lots.siteName(c)} ${f.code.filter((x) => x === c).length}`).join(', ')
  const check = exp.check, nb = check.saved.lot_effect ?? []
  const agree = nb.length === check.lot_effect.length && nb.every((s, i) => {
    const r = check.lot_effect[i]
    return s.n248 === r.n248 && s.n251 === r.n251 && Math.abs(s.median248 - r.median248) < 1e-6 && Math.abs(s.median251 - r.median251) < 1e-6 && Math.abs((s.p ?? 0) - (r.p ?? 0)) < 1e-5
  })
  return <>
    <ChartFrame id="ls-intro" title="PTFE lot comparison — char_12 §4 for any site" exportable={false}
      provenance={<>Notebook: {NB} §4. Export: <code>{EXPORTER}</code> → <code>{FOLDER}/</code> ({exp.generated.slice(0, 10)}): {exp.inputs.join('; ')}.</>}>
      <div className="subtitle">
        <p>char_12 asks whether the PTFE filter lot shifts the Addis spectra, as a change of filter manufacturer did in Weakley et al. (2018). Lot 248 is mostly dry-season, so it compares
          the lots over all filters and within the dry season only. This page draws the same comparison for whatever is in view: {f.id.length} filter spectra ({perSite}) with their lot
          (Addis: the ETAD lot map; the SPARTAN sites: the filter list's lot), each in three views — raw absorbance, char_12's second derivative and the AIRSpec-baselined,
          area-normalised view — so every centre, band, box and test below follows your sites, seasons, dates and exclusions. <strong>Filters</strong> on each chart picks the set;
          “{SET_LABEL.notebook}” reproduces char_12's tables.</p>
        <p>Reproduced: on char_12's 239 Addis filters the exported escape scores give its lot × season numbers {agree ? 'exactly' : 'approximately'} (lot 248 vs 251: n {check.lot_effect.map((r) => `${r.n248}/${r.n251}`).join(' and ')},
          medians {check.lot_effect.map((r) => `${fmt(r.median248, 3)} vs ${fmt(r.median251, 3)}`).join(' and ')}, p {check.lot_effect.map((r) => fmt(r.p, 3)).join(' and ')} over all filters and within the dry season),
          and its 04 box groups ({check.lot_season.map((g) => `${g.group} ${g.n}`).join(', ')}).
          {check.airspec_vs_r && <> The AIRSpec view is the repo's Python port of the APRLssb spline, run here for every site; on Addis it matches the R output to a median {fmt(check.airspec_vs_r.median_abs_diff * 1e6, 2)} × 10⁻⁶ absorbance
          ({check.airspec_vs_r.filters} filters).</>} The escape score exists only for Addis in char_12 (and no per-filter table is saved); the exporter recomputes it with char_12's processing and IMPROVE envelope for every filter.</p>
      </div>
    </ChartFrame>
    <LotPanels exp={exp} lots={lots} id="ls-deriv" prefix="ls.d" figure="figure 04 (left)" setPick={pick} defaults={{ view: 'deriv', region: 'ch', band: '5-95' }} />
    <EscapeByLot exp={exp} lots={lots} setPick={pick} />
    <LotPanels exp={exp} lots={lots} id="ls-airspec" prefix="ls.a" figure="figure 04b" setPick={pick} defaults={{ view: 'airspec', region: 'full', band: '10-90' }} />
    <BandMeans exp={exp} lots={lots} setPick={pick} />
  </>
}
