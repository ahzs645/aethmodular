import { useMemo, useState } from 'react'
import { CompositionTreemap } from '@/charts/partOfWhole/CompositionTreemap'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { seasonalComposition, type CompositionCalendar } from '@/lib/seasonalComposition'
import { fmt } from '@/lib/stats'
import type { FilterRow, MetaFile } from '@/lib/types'

const CALENDARS = ['Addis local (Belg Feb–May)', 'Meeting bins (Dry Oct–Feb)'] as const

export function SeasonalComposition({ rows, meta }: { rows: FilterRow[]; meta?: MetaFile }) {
  const [calendar, setCalendar] = useState<CompositionCalendar>('local')
  const composition = useMemo(() => meta ? seasonalComposition(rows, meta, calendar) : null, [rows, meta, calendar])
  if (!composition) return <ChartFrame title="Addis seasonal chemistry" id="meeting-seasonal-composition" exportable={false}>
    <Empty>Gallery chemistry metadata is unavailable.</Empty>
  </ChartFrame>
  const { stats, species } = composition
  return <div>
    <div className="frame-controls">
      <Segmented label="Chemistry calendar" options={CALENDARS} value={calendar === 'local' ? CALENDARS[0] : CALENDARS[1]}
        onChange={v => setCalendar(v === CALENDARS[0] ? 'local' : 'dry_feb')} />
    </div>
    <CompositionTreemap rows={composition.rows} meta={composition.meta} species={species}
      stackedOnly initialGrouping="by season" id="meeting-seasonal-composition" title="Addis seasonal chemistry — concentrations and percentages"
      notes={<>
        <p className="chart-note"><strong>{calendar === 'local' ? 'Addis local calendar (belg_feb): Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep.' : 'Meeting month bins (dry_feb): Dry Oct–Feb, Belg Mar–May, Kiremt Jun–Sep.'}</strong> The frozen calibration charts elsewhere on this page retain dry_feb. February changes groups when you change this control; these month bins do not measure weather conditions.</p>
        <p className="chart-note">These are existing SPARTAN filter chemistry records after exclusions, not the new-prediction test set. Carbon uses EC and OC from the existing FTIR product once. Duplicate ChemSpec carbon, optical HIPS BC, organic matter and functional groups are not extra slices.</p>
        <p className="chart-note">Start with concentration to compare absolute seasonal amounts, then switch to composition (%) to compare relative shares. Percentages sum the positive visible species means (or medians), not measured PM2.5; hiding a legend item recalculates that total. Available species can use different filters, so these stacks are descriptive rather than matched-filter mass closure. Missing species are absent, not zero. Signed observations remain in the statistics; non-positive aggregate values have no positive slice.</p>
        <table className="census-table">
          <thead><tr><th>Season / chemistry filters</th><th>Sampling window</th><th>Mean measured PM2.5 (µg/m³)</th></tr></thead>
          <tbody>{stats.map(s => {
            const pm = s.fields.find(f => f.field === 'PM2.5 mass')!
            return <tr key={s.season.name}><td>{s.season.name} · n={s.n}</td><td>{s.start && s.end ? `${s.start} to ${s.end}` : 'No dated filters'}</td><td>{fmt(pm.mean, 2)} · n={pm.n}</td></tr>
          })}</tbody>
        </table>
        {composition.unknownMonths > 0 && <p className="chart-note">{composition.unknownMonths} filters have unknown months; they cannot enter a named seasonal bar.</p>}
        <details><summary>Species coverage by season (available / seasonal filters; negative observations)</summary>
          <div style={{ overflowX: 'auto' }}><table className="census-table">
            <thead><tr><th>Species</th>{stats.map(s => <th key={s.season.name}>{s.season.name}</th>)}</tr></thead>
            <tbody>{species.map(field => <tr key={field}><td>{field}</td>{stats.map(s => {
              const f = s.fields.find(f => f.field === field)!
              return <td key={s.season.name}>{f.n ? `${f.n}/${s.n}${f.negative ? ` · ${f.negative} negative` : ''}` : `Unavailable (0/${s.n})`}</td>
            })}</tr>)}</tbody>
          </table></div>
        </details>
      </>} />
  </div>
}
