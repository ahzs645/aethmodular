import { useEffect, useSyncExternalStore } from 'react'
import { Combobox } from '@/components/Combobox'
import { Segmented, Select } from '@/components/ChartFrame'
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
/** the gallery-wide key of the site a single-site page is drawn for (usePageSite) */
export const PAGE_SITE_KEY = 'psite'

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
export function useSiteFocus<S extends FocusSite>(sites: readonly S[]) {
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

/**
 * A page with a comparison and a one-site analysis (Weather, AERONET, the Aethalometer's overview and
 * filters), in one hook: `one` is the site the one-site analysis is for (the focused site, or the only
 * site picked), null in the comparison. A click on a site in the comparison opens it (`open`) without
 * changing the Samples pick, and `back` returns to the comparison; `rowsOf` keeps the focused site's
 * rows (all of them otherwise), so the one-site view draws what a pick of that site alone would.
 * Draw the heading with components/SiteModeHead.
 *
 *   const mode = useSiteMode(sites)
 *   const oneRows = mode.rowsOf(rows)
 *   if (mode.one) …one-site analysis… else …comparison, with onSite={mode.open}…
 */
export function useSiteMode<S extends FocusSite>(sites: readonly S[]) {
  const focus = useSiteFocus(sites)
  const one = focus.site ?? (sites.length === 1 ? sites[0] : null)
  const name = focus.site?.name ?? null
  return {
    /** every site the page can draw (the comparison's) */
    sites,
    one,
    /** the one-site view is one of several picked sites (so there is a comparison to go back to) */
    focused: name !== null,
    /** open one site's analysis from the comparison, keeping the pick */
    open: (code: string) => { focus.set(code); window.scrollTo({ top: 0, behavior: 'smooth' }) },
    /** back to the comparison */
    back: () => focus.set(null),
    /** the rows (or pairs) of the site in view: the focused site's, or all of them */
    rowsOf: <T extends { site: string }>(rows: readonly T[]): T[] => (name === null ? [...rows] : rows.filter((r) => r.site === name)),
  }
}
export type SiteMode<S extends FocusSite = FocusSite> = ReturnType<typeof useSiteMode<S>>

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

/** the value of usePageSite's "every site" choice */
export const ALL_SITES = 'all'

/** the site a stored value names (by code, or by name for the older keys that kept names) */
export const findSite = <S extends FocusSite>(sites: readonly S[], v: string): S | null =>
  v ? sites.find((s) => s.code === v) ?? sites.find((s) => s.name === v) ?? null : null

/**
 * The one site a page, or a chart on it, is drawn for when it shows a site at a time (the research pages'
 * Site, the diurnal cycles, the similarity target, the iron test). Kept in one URL key for the gallery
 * (`psite=<site code>`), so moving between such pages keeps the site in view; the per-page keys these
 * pages used before go in `legacy`, so old links still open the same site (a legacy value may be a
 * code or a name). A choice that means something else on a page of its own (the data-quality page's
 * instrument, a chart's pooled-or-one-site split) passes its own `key`.
 *
 * Resolution: the stored site if the page has it; else `prefer` (a code, a name, or ALL_SITES);
 * else the first site. `all` adds an "every site" choice (value ALL_SITES) labelled with it.
 *
 *   const pick = usePageSite(sites, { legacy: ['dn.site'], prefer: 'ETAD' })
 *   pick.site · pick.code · <SitePicker {...pick.picker} />
 */
export function usePageSite<S extends FocusSite>(sites: readonly S[], { key = PAGE_SITE_KEY, legacy, prefer, all }: {
  key?: string
  legacy?: readonly string[]
  prefer?: string
  all?: string
} = {}) {
  // no option list for the URL check: the sites differ from page to page under one key, and arrive with the data
  const [raw, setRaw] = useSearchParam<string>(key, '', undefined, { legacy })
  const resolve = (v: string | undefined) => (v === ALL_SITES && all ? ALL_SITES : v ? findSite(sites, v)?.code ?? null : null)
  const code = resolve(raw) ?? resolve(prefer) ?? sites[0]?.code ?? ''
  const site = code === ALL_SITES ? null : findSite(sites, code)
  const options = all ? [ALL_SITES, ...sites.map((s) => s.code)] : sites.map((s) => s.code)
  const labelOf = (c: string) => (c === ALL_SITES ? all ?? c : findSite(sites, c)?.name ?? c)
  const set = (c: string) => setRaw(c)
  return { code, site, all: code === ALL_SITES, set, options, labelOf,
    /** the site the URL names, if this page has it (null: the page shows its default) */
    stored: findSite(sites, raw),
    /** props for SitePicker */
    picker: { value: code, options, onChange: set, optionLabel: labelOf } }
}

/**
 * The site control: nothing with one site, buttons up to five (`as="select"`: always a dropdown), a
 * dropdown beyond. Values are what the caller keeps (codes from usePageSite, or names).
 */
export function SitePicker({ label = 'Site', value, options, onChange, optionLabel, as, title }: {
  label?: string
  value: string
  options: readonly string[]
  onChange: (v: string) => void
  optionLabel?: (v: string) => string
  as?: 'select'
  title?: string
}) {
  if (options.length <= 1) return null
  return options.length <= 5 && as !== 'select'
    ? <Segmented label={label} value={value} options={options} onChange={onChange} optionLabel={optionLabel ?? ((o) => o)} title={title} />
    : <Select label={label} value={value} options={[...options]} onChange={onChange} optionLabel={optionLabel} title={title} />
}
