import type { SectionPageProps } from '../../types'
import { PageIntro, Waiting } from './spectralCommon'
import { Exclusion, LoadingCheck, LotControl } from './charcoal/Controls'
import { EXPORTER, SET_LABEL, useCharcoal, useCharcoalExport, useScoreSet } from './charcoal/data'
import { Classes, Feedstocks } from './charcoal/Reference'
import { Populations, RawVsChar, Recurrence, SetPick } from './charcoal/Scores'

/**
 * Charcoal similarity per Addis filter (research/ftir_etad_char char_05–char_09): every filter's best r against
 * McCall's raw and charred feedstock, the dry-season low-similarity population, its lot and loading controls and
 * char_07's exclusion test, recomputed in the browser from the notebooks' per-filter scores joined to the rows in
 * view (gallery/data/export_research_charcoal.py → research/charcoal/scores.json; common/charcoal.ts). The
 * per-feedstock and per-class rankings are the notebooks' tables: no script saves those per filter.
 */
export function CharcoalScores({ rows, meta }: SectionPageProps) {
  const { data, error } = useCharcoalExport()
  const [set, setSet] = useScoreSet()
  const c = useCharcoal(rows, meta, data?.tables.per_filter, set)
  if (!data) return <Waiting error={error} what="the charcoal scores" exporter={EXPORTER} />
  const pick = <SetPick value={set} onChange={setSet} />
  const perFilter = c.total > 0
  return <>
    <PageIntro id="ch-intro" title="Charcoal similarity per Addis filter" data={data}
      notebooks="research/ftir_etad_char char_02, char_03, char_05, char_06, char_07, char_08, char_09 (and char_12 §6)">
      {perFilter
        ? <p>The notebooks scored {c.total} Addis filters (one per physical filter, AIRSpec-baselined spectra). Their ids join the gallery's Addis rows (ETAD-0017-1 → ETAD-0017);
          the filters sampled after the gallery's Addis series ends (ETAD-0249 on) have no gallery row. <strong>Filters</strong> on each chart picks the set:
          “{SET_LABEL.view}” uses the Addis rows in view ({set === 'view' ? c.scored.length : '…'} now), so your seasons, dates and exclusions apply and EC and Fabs are the gallery's;
          “{SET_LABEL.plus}” adds the newer filters, seasoned by your calendar; “{SET_LABEL.notebook}” is every scored filter with the notebooks' own seasons, which reproduces their tables
          (shown beside the page's numbers as “nb”) under Filters → excluded: Include. Filters → excluded acts on every set, the notebooks' too: under Hide it leaves out the scored filters the export excludes (ETAD-0248) and any you exclude by hand. Recomputed here: medians, shares, the monthly rate, the lot and loading Mann–Whitney tests, char_07's exclusions.
          From the notebooks as they ran: the mixture component (a fit), the logistic regressions and every per-feedstock and per-class number. The model-fitting side of these
          notebooks (archive PCA and clustering, the analogue matcher, the burned/unburned classifier) is on <a href="#tab=ml&aep=charcoal">Machine learning → Charcoal archive</a>;
          char_12's envelope escape is on <a href="#tab=research&aep=applicability">Applicability domain</a>.</p>
        : <p>This export was written without the char notebooks' output folder, so it holds only the tables the executed notebooks displayed: the per-filter charts need the run below.</p>}
    </PageIntro>
    {perFilter && <>
      <RawVsChar c={c} data={data} setPick={pick} />
      <Recurrence c={c} data={data} setPick={pick} />
      <Populations c={c} data={data} setPick={pick} />
      <LotControl c={c} data={data} setPick={pick} />
      <LoadingCheck c={c} data={data} setPick={pick} />
      <Exclusion c={c} data={data} setPick={pick} />
    </>}
    <Feedstocks c={c} data={data} />
    <Classes c={c} data={data} />
  </>
}
