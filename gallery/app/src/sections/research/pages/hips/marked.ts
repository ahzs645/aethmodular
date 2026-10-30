/**
 * Filters → excluded: Mark (lib/markedRows) for the hips/ views. HipsEnvelope hands them the counted filters only, so
 * each view joins the excluded filters in view to their optics row here, the way the page joins its own (µg/m³
 * whatever the basis), and draws them as × on its crossplots, in no fit, median or count.
 */
import { useMemo } from 'react'
import { useMarkedRows } from '@/lib/markedRows'
import type { MetaFile } from '@/lib/types'
import { readParams } from '@/lib/url'

import { useOptics } from '../../common/optics'
import { useConcRows } from '../../common/ui'
import type { Paired } from './TauViews'

/** the row fields these views read; Filters → units: mass per filter turns them into µg/filter */
const CONC_FIELDS = ['EC (FTIR)', 'OC (FTIR)', 'OM (FTIR)', 'PM2.5 mass', 'Iron']

/** the excluded filters in view with a (non-blank) optics row; empty unless Filters → excluded is Mark */
export function useMarkedPairs(): Paired[] {
  const rows = useMarkedRows()
  const { rows: optics } = useOptics()
  // the views get no meta: the basis is read from the link, as App keeps it there
  const mass = rows.length > 0 && readParams().get('basis') === 'mass'
  const meta = useMemo(() => ({ basis: mass ? 'mass' : 'conc', field_units: Object.fromEntries(CONC_FIELDS.map((f) => [f, 'µg/filter'])) }) as unknown as MetaFile, [mass])
  const conc = useConcRows(rows, meta)
  return useMemo(() => {
    if (!conc.length || !optics) return []
    const byId = new Map(optics.filter((o) => o.type !== 'FB').map((o) => [o.id, o]))
    return conc.flatMap((r) => { const o = byId.get(r.id); return o ? [{ r, o }] : [] })
  }, [conc, optics])
}
