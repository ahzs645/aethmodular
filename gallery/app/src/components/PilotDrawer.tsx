import { useEffect, useState } from 'react'
import type { FilterRecord } from '@/lib/highlight'
import type { PilotRecord } from '@/lib/pilotFilters'
import { usePilotRecord } from '@/lib/pilotFilters'
import { FilterSpectrum } from '@/components/FilterSpectrum'
import { DrawerShell } from '@/components/DrawerShell'
import { RecordSection } from '@/components/RecordDrawer'
import { PilotRecordView } from '@/components/PilotSection'
import { XIcon } from '@/components/Icons'

/**
 * The drawer for an Ethiopia pilot filter that no network export carries (the Adama PTFE filters):
 * what the clicked chart knows, then the filter's pilot record (lib/pilotFilters) with its corrected
 * spectrum. `enter` false: it replaces the lookup's placeholder drawer in place, without sliding in again.
 */
export function PilotDrawer({ record, context, onClose, enter = true }: { record: PilotRecord; context?: FilterRecord | null; onClose: () => void; enter?: boolean }) {
  const { data } = usePilotRecord(record.id)
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const copy = () => navigator.clipboard?.writeText(record.id).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
  return (
    <DrawerShell label={`Filter ${record.id}`} onClose={onClose} enter={enter}>
      <header className="drawer-head">
        <div>
          <div className="drawer-id">{record.id}{record.flags.map((f) => <span key={f} className="badge danger">{f}</span>)}</div>
          <div className="drawer-sub">{record.site} · {record.date} · {record.medium}{record.partner ? ` · quartz ${record.partner.id}` : ''}</div>
        </div>
        <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
      </header>
      <div className="drawer-actions">
        <span className="u">An Ethiopia pilot filter: from the pilot filters export, not a network.</span>
        <span style={{ flex: 1 }} />
        <button type="button" className="btn" onClick={copy}>{copied ? 'Copied' : 'Copy ID'}</button>
      </div>
      {context && <RecordSection rec={context} />}
      <FilterSpectrum id={record.id} />
      {data && <PilotRecordView data={data} record={record} />}
    </DrawerShell>
  )
}
