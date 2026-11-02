import type { SectionPageProps } from '../../types'
import { SampleControls } from '@/lib/pageSidebar'
import { patchHash, readParams } from '@/lib/url'
import { PageIntro, Waiting } from './spectralCommon'
import { Exclusion, LoadingCheck, LotControl } from './charcoal/Controls'
import { EXPORTER, NOTEBOOK_CALENDAR, useCharcoal, useCharcoalExport, type Charcoal } from './charcoal/data'
import { Classes, Feedstocks } from './charcoal/Reference'
import { Populations, RawVsChar, Recurrence } from './charcoal/Scores'
import { ADDIS } from '../common/ui'

/**
 * The Samples settings that show the char notebooks' own 239 filters: Addis in view, the notebooks' month bins
 * (the shared Ethiopian calendar with February in Dry), every date and season, the export's exclusions included
 * (ETAD-0248) and no PMF pick. Hand exclusions stay as they are. One hash patch, so ⌘Z puts the reader's back.
 */
function applyNotebookSamples() {
  const p = readParams()
  const ss = p.get('ssites')
  const codes = ss && ss !== '*' && ss !== '-' ? ss.split(',') : []
  const sites = p.get('sites')
  patchHash({
    cal: NOTEBOOK_CALENDAR, cs: null, rr: null, seas: null, seasons: null, from: null, to: null, excl: '1', incl: null, pmfon: null, pmfonly: null,
    net: p.get('net') === 'improve' ? null : p.get('net'),
    ssites: ss === '-' ? 'ETAD' : codes.length && !codes.includes('ETAD') ? [...codes, 'ETAD'].join(',') : ss,
    sites: sites && !sites.split('|').includes(ADDIS) ? null : sites,
  })
}

/** the preset at the top of the Samples tab, and whether the filters shown are the notebooks' 239 now */
function NotebookPreset({ c }: { c: Charcoal }) {
  return <SampleControls>
    <div className="row">
      <span className="row-label">preset</span>
      <button type="button" className={`btn${c.notebook ? ' active' : ''}`} aria-pressed={c.notebook} onClick={applyNotebookSamples}
        title="The char notebooks' own 239 Addis filters: Addis in view, their calendar (Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep), every date and season, the export's exclusions included. Hand exclusions are kept. The charts then print the notebooks' numbers beside their own (nb).">
        char notebooks' {c.total}
      </button>
      <span className="method-note" style={{ display: 'block' }}>{c.notebook ? `showing the notebooks' ${c.total}: their numbers appear as “nb”` : `${c.scored.length} of the ${c.total} scored filters shown`}</span>
    </div>
  </SampleControls>
}

/**
 * Charcoal similarity per Addis filter (research/ftir_etad_char char_05–char_09): every filter's best r against
 * McCall's raw and charred feedstock, the dry-season low-similarity population, its lot and loading controls and
 * char_07's exclusion test, recomputed in the browser from the notebooks' per-filter scores joined to the rows in
 * view (gallery/data/export_research_charcoal.py → research/charcoal/scores.json; common/charcoal.ts). The
 * per-feedstock and per-class rankings are the notebooks' tables: no script saves those per filter.
 */
export function CharcoalScores({ rows, meta }: SectionPageProps) {
  const { data, error } = useCharcoalExport()
  const c = useCharcoal(rows, meta, data?.tables.per_filter)
  if (!data) return <Waiting error={error} what="the charcoal scores" exporter={EXPORTER} />
  const perFilter = c.total > 0
  return <>
    <PageIntro id="ch-intro" title="Charcoal similarity per Addis filter" data={data}
      notebooks="research/ftir_etad_char char_02, char_03, char_05, char_06, char_07, char_08, char_09 (and char_12 §6)">
      {perFilter
        ? <p>The notebooks scored {c.total} Addis filters (one per physical filter, AIRSpec-baselined spectra). Their ids join the gallery's Addis rows (ETAD-0017-1 → ETAD-0017);
          the filters sampled after the gallery's Addis series ends (ETAD-0249 on) have no gallery row, and are placed by their date. Every chart uses the scored filters the
          Samples settings leave in view ({c.scored.length} now): your sites, calendar, seasons, dates and exclusions apply, and a gallery row's EC and Fabs are the gallery's.
          Samples → preset “char notebooks' {c.total}” puts their own set in view (their calendar, every date, the export's exclusions included), and the charts then print the notebooks' numbers beside their own as “nb”.
          Recomputed here: medians, shares, the monthly rate, the lot and loading Mann–Whitney tests, char_07's exclusions.
          From the notebooks as they ran: the mixture component (a fit), the logistic regressions and every per-feedstock and per-class number. The model-fitting side of these
          notebooks (archive PCA and clustering, the analogue matcher, the burned/unburned classifier) is on <a href="#tab=ml&aep=charcoal">Machine learning → Charcoal archive</a>;
          char_12's envelope escape is on <a href="#tab=research&aep=applicability">Applicability domain</a>.</p>
        : <p>This export was written without the char notebooks' output folder, so it holds only the tables the executed notebooks displayed: the per-filter charts need the run below.</p>}
    </PageIntro>
    {perFilter && <NotebookPreset c={c} />}
    {perFilter && <>
      <RawVsChar c={c} data={data} />
      <Recurrence c={c} data={data} />
      <Populations c={c} data={data} />
      <LotControl c={c} data={data} />
      <LoadingCheck c={c} data={data} />
      <Exclusion c={c} data={data} />
    </>}
    <Feedstocks c={c} data={data} />
    <Classes c={c} data={data} />
  </>
}
