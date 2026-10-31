import { useEffect, useSyncExternalStore } from 'react'
import { Combobox } from '@/components/Combobox'
import { Segmented } from '@/components/ChartFrame'
import { useSearchParam } from '@/lib/url'

/**
 * The analysis mode of a page that reads differently for one site than for several (Weather: the
 * comparison table and side-by-side roses, or one site's timeline, correlations and station check).
 * With several sites picked in the Samples panel, the Plot options offer
 *
 *   Analysis  [Compare sites | One site]  [site ▾]
 *
 * so one site's analysis opens without narrowing the pick, and going back is one click. The choice
 * is one URL key for the whole gallery (`sfocus=<site code>`), so a link reopens the same site.
 * It is offered only while the open page asks for it and more than one site is in view; with one
 * site picked the page is in its one-site mode already and the control is hidden.
 *
 *   const focus = useSiteFocus(sites)        // a page: offers the choice, reads it
 *   if (focus.site) …one-site analysis of focus.site…
 *   <SiteFocusControl />                      // wherever the choice is drawn (SubsetBar's page tab)
 */
export interface FocusSite { code: string; name: string; n?: number }

const URL_KEY = 'sfocus'

// the open page's sites (null: the page has no one-site mode), shared with the dock's page tab
let offered: readonly FocusSite[] | null = null
const listeners = new Set<() => void>()
const publish = (next: readonly FocusSite[] | null) => { offered = next; for (const l of listeners) l() }
const useOffered = () => useSyncExternalStore((l) => { listeners.add(l); return () => { listeners.delete(l) } }, () => offered)

// the last site chosen, so Compare → One site reopens it rather than the first in the list
let lastCode: string | null = null

/**
 * For a page: offer the one-site mode over `sites` (the picked sites the page can draw, in the
 * Samples panel's order) and read the choice. `site` is null in the comparison, or when the
 * linked site is not among the picked ones (it was dropped from the pick).
 */
export function useSiteFocus(sites: readonly FocusSite[]) {
  const [code, setCode] = useSearchParam<string>(URL_KEY, '')
  const key = sites.map((s) => s.code).join('|')
  useEffect(() => {
    publish(sites.length > 1 ? sites : null)
    return () => publish(null)
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  const site = sites.length > 1 ? sites.find((s) => s.code === code) ?? null : null
  if (site) lastCode = site.code
  return { site, set: (c: string | null) => setCode(c ?? '') }
}

/** whether the open page offers the choice right now (several sites in view) */
export const useSiteFocusOffered = () => (useOffered()?.length ?? 0) > 1

/** the focused site's code while the choice is offered, for a note on a closed panel's tab */
export function useFocusedSite(): FocusSite | null {
  const sites = useOffered()
  const [code] = useSearchParam<string>(URL_KEY, '')
  return sites?.find((s) => s.code === code) ?? null
}

/** the Analysis control: Compare sites / One site, and which site */
export function SiteFocusControl() {
  const sites = useOffered()
  const [code, setCode] = useSearchParam<string>(URL_KEY, '')
  if (!sites || sites.length < 2) return null
  const site = sites.find((s) => s.code === code) ?? null
  const mode = site ? 'one' : 'compare'
  return (
    <>
      <Segmented value={mode} options={['compare', 'one'] as const}
        optionLabel={(o) => (o === 'one' ? 'One site' : `Compare ${sites.length} sites`)}
        onChange={(v) => setCode(v === 'compare' ? '' : (sites.find((s) => s.code === lastCode) ?? sites[0]).code)}
        title="Compare: the picked sites side by side · One site: the single-site analysis of one of them, without changing the Samples pick" />
      {site && <Combobox label="" title="The site the single-site analysis is for" value={site.code} onChange={setCode}
        options={sites.map((s) => ({ key: s.code, label: s.n != null ? `${s.name} (${s.n})` : s.name }))} />}
    </>
  )
}
