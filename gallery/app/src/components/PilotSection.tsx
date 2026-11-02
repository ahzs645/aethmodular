import { Fragment, useMemo } from 'react'
import { fmt } from '@/lib/stats'
import { prettyUnit } from '@/lib/theme'
import { chemText } from '@/lib/format'
import { placements, isFiniteNumber, type ReferenceRow } from '@/lib/referencePopulation'
import { usePilotRecord, type PilotFilters, type PilotRecord } from '@/lib/pilotFilters'
import type { FilterRow } from '@/lib/types'
import { DrawerScroll } from '@/components/DrawerShell'
import { PlacementRow } from '@/components/PlacementRow'

/** the export's units as the drawers write them; ratios and absorbance need no unit beside the number */
const unitText = (u: string | undefined) =>
  u === 'm3' ? 'm³' : u === 'ug' ? 'µg' : u === 'ratio' || u === 'absorbance' ? '' : prettyUnit(u)

const cell = (v: PilotRecord['values'][string]) =>
  v === null || v === undefined || v === '' ? '—' : typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : fmt(v, 3)) : v

/**
 * An Ethiopia pilot filter's record (lib/pilotFilters): its pairing and flags, then each group of
 * measurements, every one the Addis filters also have under the same frozen model placed against
 * them. In the full sample drawer (a Bishoftu filter, which is also a SPARTAN network row) it adds
 * the pilot's values; the measurements the row already shows (`row`) are left out here.
 */
export function PilotSection({ id, row, color }: { id: string; row?: FilterRow; color?: string }) {
  const { data, record } = usePilotRecord(id)
  if (!data || !record) return null
  return <PilotRecordView data={data} record={record} row={row} color={color} />
}

export function PilotRecordView({ data, record, row, color }: { data: PilotFilters; record: PilotRecord; row?: FilterRow; color?: string }) {
  const reference = data.reference
  // the record as a reference row, so the shared placement reads it as it reads a SPARTAN filter
  const asRow = useMemo(() => ({ id: record.id, site: record.site, code: record.code, date: record.date, excluded: false, ...record.values }) as ReferenceRow,
    [record])
  const placed = useMemo(() => placements(asRow, reference.fields, reference.rows.filter((r) => !r.excluded)), [asRow, reference])
  const shown = (f: string) => record.values[f] !== undefined && record.values[f] !== null && !(row && isFiniteNumber(row[f]))
  const campaign = data.campaigns[record.campaign]
  return (
    <section className="drawer-section">
      <h3>{record.campaign === 'Adama' ? 'Adama pilot (CSU AMOD, quartz TOR pair)' : 'Frozen external pilot · Bishoftu'}</h3>
      {(record.partner || record.flags.length > 0 || record.spectrum_map === 'provisional') && <p className="chart-note" style={{ display: 'block', margin: '4px 0 8px' }}>
        {record.flags.map((f) => <span key={f} className="badge danger" style={{ marginRight: 6 }}>{f}</span>)}
        {record.spectrum_map === 'provisional' && <span className="badge" style={{ marginRight: 6 }} title={campaign?.note}>provisional spectrum map</span>}
        {record.partner && <>{record.partner.label}: <strong>{record.partner.id}</strong>.</>}
      </p>}
      {data.groups.map((g) => {
        const fields = g.fields.filter(shown)
        if (!fields.length) return null
        return (
          <Fragment key={g.label}>
            <DrawerScroll label={`${g.label} values`}><table className="placement">
              <thead><tr><th colSpan={4}>{g.label}</th></tr></thead>
              <tbody>{fields.map((f) => {
                const p = placed[f]
                const unit = unitText(data.units[f])
                if (p) return <PlacementRow key={f} label={chemText(f)} p={p} unit={unit} color={color} referenceLabel={reference.label} />
                const v = record.values[f]
                return <tr key={f}><td className="f">{chemText(f)}</td><td className="v" colSpan={3}>{cell(v)}{typeof v === 'number' && unit ? <> <span className="u">{unit}</span></> : null}</td></tr>
              })}</tbody>
            </table></DrawerScroll>
          </Fragment>
        )
      })}
      <p className="chart-note" style={{ display: 'block' }}>
        Bars place this filter against {reference.label.replace(/^SPARTAN Addis/, 'the SPARTAN Addis filters')} ({reference.n}), for the measurements they share; the rest are this filter's values only.
        {' '}{[...record.notes, campaign?.note].filter(Boolean).join(' ')} Frozen run {data.frozen_run_signature.slice(0, 8)}, VIBES background rank {data.background_rank}.
      </p>
    </section>
  )
}
