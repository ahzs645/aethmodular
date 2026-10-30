/**
 * The notebooks' per-feedstock and per-reference-class tables (char_05, char_07, char_08, char_09, char_12 §6). No
 * script saves a filter's r against each feedstock or class, so these are static: the notebooks' 239 filters and
 * seasons. The one per-filter score that spans the classes, char_02's best match against the whole temperature-
 * labelled library, is recomputed on the filters shown beside them.
 */
import { ChartFrame } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { FixedSetNote } from '@/lib/markedRows'
import { INK } from '@/lib/theme'
import { num, type Row } from '@/lib/table'
import { spread } from '../../common/charcoal'
import { DataTable } from '../../common/ui'
import { SetNote } from './Scores'
import { provenance, rows, type SpectralExport } from '../spectralCommon'
import { NB, type Charcoal } from './data'

const s = (v: unknown) => String(v ?? '')

export function Feedstocks({ c, data }: { c: Charcoal; data: SpectralExport }) {
  const ranking = rows(data, 'feedstock_ranking')
  const match = rows(data, 'feedstock_match')
  if (!ranking.length) return null
  const dry = c.colorOf('Dry Season'), wet = INK.accent
  const strip: IntervalRow[] = [...ranking].sort((a, b) => (num(b['dry excluded']) ?? 0) - (num(a['dry excluded']) ?? 0)).map((r) => ({
    key: s(r.feedstock), label: s(r.feedstock), note: `${fmt(num(r.change), 3)}`,
    marks: [[{ mid: num(r['all filters']), color: INK.text }, { mid: num(r['dry excluded']), color: wet, dy: -5 }, { mid: num(r['dry season only']), color: dry, dy: 5, shape: 'hollow' }]],
    tip: [s(r.feedstock), `${r['n ref']} reference spectra`, `all filters ${fmt(num(r['all filters']), 3)} · dry excluded ${fmt(num(r['dry excluded']), 3)} · dry only ${fmt(num(r['dry season only']), 3)}`],
  }))
  const top = ranking.reduce((a, b) => ((num(b['dry excluded']) ?? 0) > (num(a['dry excluded']) ?? 0) ? b : a))
  return (
    <ChartFrame id="ch-feedstocks"
      title={`Removing the dry season reorders the feedstocks: ${s(top.feedstock)} stays closest (mean r ${fmt(num(top['dry excluded']), 3)}), grasses and straw gain, woods and rice husk lose`}
      subtitle="Each raw McCall feedstock's mean r against the Addis filters (the median over filters of the mean over that feedstock's reference spectra), for all 239 filters, without the dry season and in the dry season only; the note is the change on removing the dry season. Static: the notebooks' filters and seasons, because no script saves a filter's r against each feedstock (see the intro)."
      provenance={<>{NB.c05} [14] right; {NB.c07} [9]. {provenance(data, ['feedstock_ranking', 'feedstock_match'])}</>}>
      <IntervalStrip panels={[{ label: 'median mean r against the feedstock' }]} labelWidth={170} noteWidth={60} blocks={[{ key: 'f', rows: strip }]} />
      <Key items={[{ label: 'all filters', color: INK.text }, { label: 'dry season excluded', color: wet }, { label: 'dry season only', color: dry, kind: 'hollow' }]} />
      <FixedSetNote what="each raw feedstock's mean r on the notebooks' 239 filters and seasons (no script saves a filter's r per feedstock)" />
      {match.length > 0 && <DataTable rows={match} rowKey={(r) => s(r.reference)} columns={Object.keys(match[0]).map((key) => ({ key, label: key === 'reference' ? 'char_05 reference (all filters)' : key, num: key !== 'reference', value: (r: Row) => (key === 'reference' ? s(r[key]) : fmt(num(r[key]), key.startsWith('n ') ? 0 : 3)) }))} />}
    </ChartFrame>
  )
}

export function Classes({ c, data }: { c: Charcoal; data: SpectralExport }) {
  const subset = rows(data, 'class_by_subset'), loss = rows(data, 'relative_loss'), fvf = rows(data, 'furnace_fire'), deriv = rows(data, 'derivative_check'), season = rows(data, 'class_by_season')
  if (!subset.length) return null
  const dryC = c.colorOf('Dry Season'), wet = INK.accent
  // char_02's library match per filter, on the filters shown: the one per-filter score across the classes
  const lib = c.scored.filter((x) => x.libraryR !== null)
  const live = (set: typeof lib) => spread(set.map((x) => x.libraryR!))
  const libAll = live(lib), libDry = live(lib.filter(c.isDry)), libWet = live(lib.filter((x) => !c.isDry(x)))
  const lossOf = (cls: string) => num(loss.find((r) => r.class === cls)?.['relative loss (%)'] ?? null)
  const classRows: IntervalRow[] = subset.map((r) => ({
    key: s(r.class), label: s(r.class), note: `${r['n refs']}`,
    marks: [[{ mid: num(r['all filters']), color: INK.text }, { mid: num(r['non-dry only']), color: wet, dy: -5 }, { mid: num(r['dry season only']), color: dryC, dy: 5, shape: 'hollow' }],
      [{ mid: lossOf(s(r.class)), color: dryC }]],
    tip: [s(r.class), `${r['n refs']} reference spectra`, `median best r: all ${fmt(num(r['all filters']), 3)} · non-dry ${fmt(num(r['non-dry only']), 3)} · dry ${fmt(num(r['dry season only']), 3)}`, `relative loss in the dry season ${fmt(lossOf(s(r.class)), 1)} %`],
  }))
  const liveRow: IntervalRow = {
    key: 'live', label: 'char_02 library, filters shown', strong: true, note: `n=${libAll.n}`,
    marks: [[{ mid: libAll.mid, lo: libAll.lo, hi: libAll.hi, color: INK.text, shape: 'diamond' }, { mid: libWet.mid, color: wet, dy: -5, shape: 'diamond' }, { mid: libDry.mid, color: dryC, dy: 5, shape: 'diamond' }],
      [{ mid: libWet.n && libDry.n ? (100 * (libWet.mid - libDry.mid)) / libWet.mid : null, color: dryC, shape: 'diamond' }]],
    tip: ['char_02 best match over the 3,924-spectrum temperature-labelled library (236 of 239 filters match Minatre furnace charcoal best)', `median ${fmt(libAll.mid, 3)} (IQR ${fmt(libAll.lo, 3)}–${fmt(libAll.hi, 3)})`, `${c.dryName} ${fmt(libDry.mid, 3)} (n ${libDry.n}) · rest ${fmt(libWet.mid, 3)} (n ${libWet.n})`],
  }
  const furnace = num(subset[0]?.['all filters']), fire = subset.find((r) => /fire/i.test(s(r.class)))
  return (
    <ChartFrame id="ch-classes"
      title={`Furnace charcoal matches Addis best (median best r ${fmt(furnace, 3)}) and real-fire char worse (${fmt(num(fire?.['all filters'] ?? null), 3)}); the dry season lowers every class together`}
      subtitle={`Each reference class's median best r against the Addis filters: all 239, non-dry and dry season only (char_09), and the relative loss in the dry season. Static, the notebooks' filters and seasons: no script saves each filter's r per class. The diamond row is live: char_02's best match against the whole temperature-labelled library, on the filters shown and your seasons; its winner is Minatre furnace charcoal for 236 of 239 filters, so it tracks the top class.`}
      provenance={<>{NB.c08} [6] left, [16]; {NB.c09} [8] left, [10]. {provenance(data, ['class_by_subset', 'relative_loss', 'per_filter'])}</>}>
      <IntervalStrip panels={[{ label: 'median best r against the class' }, { label: 'relative loss in the dry season (%)', reference: 0 }]} labelWidth={260} noteWidth={50}
        blocks={[{ key: 'live', rows: lib.length ? [liveRow] : [] }, { key: 'classes', title: 'Notebook classes (static)', rows: classRows }]} />
      {/* the live row follows the set and the exclusions; the class rows are the notebooks' */}
      <SetNote c={c} of={(x) => x.libraryR !== null} />
      {c.set !== 'notebook' && <FixedSetNote what="the class rows on the notebooks' 239 filters and seasons (no script saves a filter's r per class)" />}
      <Key items={[{ label: 'all filters', color: INK.text }, { label: 'non-dry', color: wet }, { label: 'dry season', color: dryC, kind: 'hollow' }, { label: 'live: char_02 library match on the filters shown', color: INK.text, kind: 'diamond' }]} />
      {fvf.length > 0 && <>
        <p className="subtitle" style={{ margin: '10px 0 4px' }}>Furnace (Minatre woody) against real-fire char (Minatre combustion facility), paired over the same filters (char_09):</p>
        <DataTable rows={fvf} rowKey={(r) => s(r.subset)} columns={[
          { key: 's', label: 'Subset', value: (r) => s(r.subset) },
          { key: 'n', label: 'n', num: true, value: (r) => fmt(num(r['n filters']), 0) },
          { key: 'f', label: 'Furnace median r', num: true, value: (r) => fmt(num(r['furnace median r']), 3) },
          { key: 'r', label: 'Fire median r', num: true, value: (r) => fmt(num(r['fire median r']), 3) },
          { key: 'g', label: 'Gap', num: true, value: (r) => fmt(num(r.gap), 3) },
          { key: 'p', label: 'Filters favouring fire', num: true, value: (r) => `${fmt(num(r['% favouring fire']), 2)} %` },
        ]} />
      </>}
      {season.length > 0 && <p className="subtitle" style={{ margin: '10px 0 4px' }}>Every class, by season (char_08 [16]): the dry season matches each one worse.</p>}
      {season.length > 0 && <DataTable rows={season} rowKey={(r) => s(r.class)} columns={Object.keys(season[0]).map((key) => ({ key, label: key === 'class' ? 'char_08: median best r by season' : key, num: key !== 'class', value: (r: Row) => (key === 'class' ? s(r[key]) : fmt(num(r[key]), 3)) }))} />}
      {deriv.length > 0 && <>
        <p className="subtitle" style={{ margin: '10px 0 4px' }}>char_12 §6: the same comparison in second-derivative space, with no baseline. Charcoal-adjacency and the dry-season deficit survive; the furnace-over-fire ordering flips, so that ranking depends on the representation:</p>
        <DataTable rows={deriv} rowKey={(r) => s(r.measure)} columns={Object.keys(deriv[0]).map((key) => ({ key, label: key === 'measure' ? 'Measure' : key, num: key !== 'measure', value: (r: Row) => (key === 'measure' ? s(r[key]) : fmt(num(r[key]), 3)) }))} />
      </>}
    </ChartFrame>
  )
}
