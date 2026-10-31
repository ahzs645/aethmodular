import { useRef, useState } from 'react'
import { dashParams, tileKeyPrefix, type DashState, type Sheet } from '@/lib/dashboard'
import { loadJson } from '@/lib/data'
import { currentView, applyView, savedViews, saveViews } from '@/lib/views'
import { parseCohort, selectCohort } from '@/lib/dashboardCohort'
import { snapshotDigest, readSnapshot, type DashboardSnapshot } from '@/lib/dashboardSnapshot'
import { downloadBlob, slug } from '@/lib/export'
import { dashboardChart, startFields, type DashboardData } from './registry'

/** Existing saved views hold named checkpoints; the file also carries the exact row populations. */
export function SheetTools({ dash, sheet, data, onSnapshot, replaying = false }: { dash: DashState; sheet: Sheet; data: DashboardData; onSnapshot: (s: DashboardSnapshot) => void; replaying?: boolean }) {
  const [name, setName] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [checkpoints, setCheckpoints] = useState(savedViews)
  const input = useRef<HTMLInputElement>(null)
  const definition = () => {
    const view = currentView()
    for (const [key, value] of Object.entries(dashParams(dash))) { if (value === null) delete view.h[key]; else view.h[key] = value }
    view.h.tab = 'dashboard'
    return view
  }
  const capture = async () => {
    setBusy(true); setNotice('')
    try {
      const view = definition()
      const externalTiles: string[] = []
      const resources = new Set<string>()
      const tiles = sheet.tiles.map(t => {
        const chart = dashboardChart(t.kind)
        if (!chart) throw new Error(`Unknown chart: ${t.kind}`)
        // Capture only the auxiliary exports this sheet declares; unrelated cached datasets stay out.
        for (const file of chart.resources?.(data) ?? []) resources.add(file)
        if (chart.dependencies?.length && !data.calibration) throw new Error('Wait for the calibration tile to load before saving its snapshot.')
        const prefix = tileKeyPrefix(dash.scope, sheet.id, t.id)
        const rows = selectCohort(data.rows, data.referenceRows ?? data.rows, parseCohort(view.q[`${prefix}cohort`] ?? ''))
        const fields = (t.fields.length ? t.fields : startFields(chart, data)).map(data.eff)
        return { id: t.id, kind: t.kind, fields, rows, meta: data.meta, pmf: data.pmf, pmfRows: data.pmfRows }
      })
      const files = Object.fromEntries(await Promise.all([...resources].map(async path => [path, await loadJson(path).catch(() => null)] as const)))
      const usesCalibration = sheet.tiles.some(t => dashboardChart(t.kind)?.dependencies?.includes('calibration_runs.json'))
      const snapshot: DashboardSnapshot = { schema: 'aeth-dashboard', version: 1, created: new Date().toISOString(),
        revision: import.meta.env.VITE_SOURCE_REVISION ?? 'unversioned build', sourceFingerprint: import.meta.env.VITE_SOURCE_FINGERPRINT, view, dash, sheet, provenance: data.provenance ?? {}, tiles, externalTiles, files,
        ...(usesCalibration && data.calibration ? { calibration: { ...data.calibration, knownIds: [...data.calibration.knownIds] } } : {}) }
      const sha256 = await snapshotDigest(snapshot)
      const blob = new Blob([JSON.stringify({ snapshot, sha256 })], { type: 'application/json' })
      if (blob.size > 25 * 1024 * 1024) throw new Error('This snapshot exceeds 25 MiB. Split the sheet into smaller sheets before saving.')
      downloadBlob(blob, `${slug(sheet.name)}-snapshot.json`)
      setNotice('Snapshot saved with filter rows, metadata, auxiliary exports, resolved layout, view settings and checksum. Unavailable optional exports remain unavailable when replayed.')
    } catch (e) { setNotice(String(e)) } finally { setBusy(false) }
  }
  return <section className="dashboard-sheet-tools" aria-label="Dashboard snapshots and checkpoints"><h3>Save this sheet</h3>
    {replaying && <p>Return to current data before saving another snapshot.</p>}
    <p>A view checkpoint reopens against available data. A snapshot also retains the displayed filter rows and their methods. Weather, maps and calibration include their auxiliary exports when available. Snapshot replay keeps saved populations separate from live data.</p>
    <div className="dashboard-data-actions"><button type="button" className="btn" disabled={busy || replaying} onClick={capture}>{busy ? 'Preparing…' : 'Download snapshot'}</button>
      <button type="button" className="btn" disabled={busy} onClick={() => input.current?.click()}>Open snapshot</button>
      <input ref={input} type="file" accept=".json,application/json" aria-label="Open dashboard snapshot file" hidden onChange={async e => {
        const f = e.target.files?.[0]; if (!f) return
        setBusy(true); setNotice('')
        try { if (f.size > 25 * 1024 * 1024) throw new Error('Snapshot exceeds 25 MiB.'); const s = await readSnapshot(await f.text()); onSnapshot(s); applyView(s.view); setNotice('Opened saved populations and auxiliary exports.') }
        catch (err) { setNotice(String(err)) } finally { setBusy(false); e.target.value = '' }
      }} /></div>
    <label>Checkpoint name <input className="select" value={name} maxLength={100} placeholder="Before the seasonal comparison" onChange={e => setName(e.target.value)} /></label>
    <button type="button" className="btn" disabled={!name.trim()} onClick={() => { const label = `Dashboard · ${name.trim()}`; const next = [...savedViews().filter(v => v.name !== label), { name: label, view: definition() }]; saveViews(next); setCheckpoints(next); setName(''); setNotice('Checkpoint saved in this browser.') }}>Save checkpoint</button>
    {checkpoints.filter(v => v.name.startsWith('Dashboard · ')).map(v => <div key={v.name} className="dashboard-data-actions"><button type="button" className="btn quiet" onClick={() => applyView(v.view)}>{v.name}</button>
      <button className="btn quiet" type="button" aria-label={`Delete ${v.name}`} onClick={() => { const next = savedViews().filter(x => x.name !== v.name); saveViews(next); setCheckpoints(next) }}>×</button></div>)}
    {notice && <p role="status">{notice}</p>}
  </section>
}
