import { createContext, useContext } from 'react'
import type { PublicSiteStatus } from '@/lib/spartanPublic'
import { refreshSpartanSource, setSpartanSource, useSpartanSource } from '@/lib/spartanSource'

interface DataStatus { active: boolean; codes: string[]; loading: boolean; error: string; sites: PublicSiteStatus[] }
const Context = createContext<DataStatus | null>(null)
export const SpartanDataProvider = Context.Provider

/** The source label stays visible while the settings menu is closed. */
export function useSpartanSourceLabel() {
  const { source } = useSpartanSource(), status = useContext(Context)
  if (source !== 'live' || !status?.active || !status.codes.length) return ''
  if (status.loading) return 'Loading…'
  const live = status.sites.filter(s => s.state === 'live').length
  return live === status.codes.length ? 'Live' : `Live ${live}/${status.codes.length}`
}

export function SpartanDataSettings() {
  const { source } = useSpartanSource(), status = useContext(Context)
  const count = status?.codes.length ?? 0
  const live = status?.sites.filter(s => s.state === 'live').length ?? 0
  return <section className="spartan-data-settings" aria-label="SPARTAN data settings">
    <fieldset className="display-choice"><legend>SPARTAN public data</legend>
      <div className="seg">
        <button type="button" className={source === 'bundled' ? 'active' : ''} aria-pressed={source === 'bundled'} onClick={() => setSpartanSource('bundled')}>Bundled</button>
        <button type="button" className={source === 'live' ? 'active' : ''} aria-pressed={source === 'live'} onClick={() => setSpartanSource('live')}>Live from SPARTAN</button>
      </div>
    </fieldset>
    <p>Updates public PM2.5 filter chemistry for your selected SPARTAN sites on chart tabs. Lab HIPS, functional groups and spectra use the bundled files.</p>
    <p>{status?.active ? count ? `Selected: ${status.codes.join(', ')}.` : 'Choose SPARTAN sites in Filters.' : 'Select SPARTAN on a chart tab to load live chemistry.'}</p>
    {source === 'live' && status?.active && count > 0 && <>
      <p role="status" aria-live="polite">{status.loading ? 'Loading selected sites…' : status.error ? `Using bundled data: ${status.error}` : `${live} of ${count} sites loaded live${live < count ? '; remaining sites use bundled data' : ''}.`}</p>
      <button type="button" className="btn" disabled={status.loading} onClick={refreshSpartanSource}>Refresh selected sites</button>
      {!status.loading && status.sites.length > 0 && <details><summary>Source and update dates</summary>
        <ul>{status.sites.map(s => <li key={s.code}>
          <a href={s.url} target="_blank" rel="noreferrer">{s.code}</a>: {s.state === 'live'
            ? <>{s.filters} public filters · file updated {s.updated || 'unspecified'}{s.version ? ` · v${s.version}` : ''}<br />Checked {new Date(s.checked).toLocaleString()}</>
            : <>bundled data · {s.message}</>}
        </li>)}</ul>
      </details>}
    </>}
  </section>
}
