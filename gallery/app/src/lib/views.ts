/**
 * Shareable views: the whole gallery state (the hash — tab, network, sites, fields, seasons,
 * dates, hand-excluded filters … — and the per-chart controls in the query string) as one
 * token, made with @firstform/json-url (the codec the webforms builder uses for its share
 * links). A link is `…/#v=<token>`; `?embed=1` adds a chrome-free embed mode for iframes.
 *
 * A view that exactly matches a built-in preset encodes as a reference to it (`{p: key}`),
 * so preset links stay ~30 characters; anything else is compressed (lz) or sent raw,
 * whichever is shorter, behind a version prefix. Older plain-hash links keep working: the
 * token is only read when the hash starts with `v=`.
 */
import { Buffer } from 'buffer'
import createWebShareEngine from '@firstform/json-url/web-share'

// json-url's codecs (raw, lz) use Node's global Buffer; browsers have none, and only the dev
// server happened to supply it, so a production build failed to read any link
const g = globalThis as { Buffer?: typeof Buffer }
g.Buffer ??= Buffer

export interface ViewState {
  /** hash parameters (App's subset state) */
  h: Record<string, string>
  /** query parameters (per-chart controls, lib/url useSearchParam) */
  q: Record<string, string>
}

export interface ViewPreset { key: string; label: string; note: string; view: ViewState }

const v = (h: Record<string, string>, q: Record<string, string> = {}): ViewState => ({ h, q })

/** Built-in starting points. Each is written exactly as the app writes its own URL, so a
 * preset left unchanged is recognised and shared as a short reference. */
export const PRESETS: ViewPreset[] = [
  { key: 'focus', label: 'Focus sites · HIPS BC vs FTIR EC', note: 'The default view: Beijing, Delhi, JPL, Addis Ababa', view: v({ tab: 'correlation' }) },
  { key: 'addis-seasons', label: 'Addis Ababa · FTIR EC by season', note: 'Distributions under the local Bega / Belg / Kiremt calendar', view: v({ tab: 'distribution', ssites: 'ETAD' }) },
  { key: 'addis-weather', label: 'Addis Ababa · weather and the filters', note: 'Rain washout, humidity, wind roses and seasonal correlations, on ERA5 and the airport station',
    view: v({ tab: 'weather', ssites: 'ETAD' }) },
  { key: 'focus-aeronet', label: 'Focus sites · the AERONET column', note: 'AOD, absorption AOD and the column-to-surface scale height at Beijing, Delhi, JPL and Addis Ababa',
    view: v({ tab: 'aeronet' }) },
  { key: 'chile', label: 'Chile · SPARTAN + IMPROVE, PM₂.₅ over time', note: 'Santiago and Talca (SPARTAN) with Antumapu and Las Condes (IMPROVE)',
    view: v({ tab: 'evolution', net: 'both', ssites: 'CLST,CLTA', isites: 'ANTU1,LCON1', y: 'PM2.5 mass', f: 'PM2.5 mass' }) },
  { key: 'chile-improve-tor', label: 'Chile IMPROVE · TOR EC vs HIPS BC', note: 'Thermal EC against absorption at the two new IMPROVE sites',
    view: v({ tab: 'correlation', net: 'improve', isites: 'ANTU1,LCON1', y: 'EC (TOR)', f: 'EC (TOR)' }) },
  { key: 'spartan-pm', label: 'All SPARTAN sites · PM₂.₅ by site', note: 'Every SPARTAN site, ranked and compared',
    view: v({ tab: 'ranking', ssites: '*', f: 'PM2.5 mass' }) },
  // research-hips-met: the HIPS offset and meteorology notebooks, on the focus-site data in the repo
  { key: 'addis-mac-pw', label: 'Addis Ababa · effective MAC vs precipitable water', note: 'HIPS Fabs / FTIR EC against AERONET column water, by season (pw_mac_calibration, the money plot)',
    view: v({ tab: 'correlation', net: 'focus', sites: 'Addis Ababa', fits: 'ols', ratios: 'HIPS%20Fabs:EC%20(FTIR)', x: 'Precipitable water (AERONET)', y: 'HIPS Fabs / EC (FTIR)', f: 'HIPS Fabs / EC (FTIR)', eq: '0' }, { 'scatter.colorBy': 'Season' }) },
  { key: 'addis-fabs-ec-pw-split', label: 'Addis Ababa · Fabs vs FTIR EC, dry vs moist column', note: 'Slope (MAC) and intercept per PW half (pw_mac_calibration §4)',
    view: v({ tab: 'correlation', net: 'focus', sites: 'Addis Ababa', fits: 'ols', x: 'EC (FTIR)', y: 'HIPS Fabs' }, { 'scatter.layout': 'Per group', 'scatter.group': 'nb-pw-median' }) },
  { key: 'addis-hips-ftir-rh45', label: 'Addis Ababa · HIPS vs FTIR EC, dry vs humid (RH 45 %)', note: 'The recursive diagnostic\'s best low-RH split (hips_ftir_intercept_rh_recursive_diagnostics)',
    view: v({ tab: 'correlation', net: 'focus', sites: 'Addis Ababa', fits: 'ols', x: 'EC (FTIR)', y: 'HIPS BC' }, { 'scatter.layout': 'Per group', 'scatter.group': 'nb-rh-45' }) },
  { key: 'focus-fabs-ec-site-fits', label: 'Focus sites · Fabs vs FTIR EC, a fit per site', note: 'Baseline (intercept) and effective MAC (slope) by site (hips_two_problem_decomposition §1)',
    view: v({ tab: 'correlation', net: 'focus', fits: 'ols', x: 'EC (FTIR)', y: 'HIPS Fabs' }, { fitgroup: 'site' }) },
  { key: 'addis-warren-ratio-hips', label: 'Addis Ababa · Warren ratio: Fabs/EC vs Fe/EC', note: 'Iron-associated absorption in HIPS; the intercept is absorption per EC at zero iron (warren_ratio_diagnostics). The aethalometer half needs the Instruments fields, which a link cannot pick before they load',
    view: v({ tab: 'correlation', net: 'focus', sites: 'Addis Ababa', fits: 'ols|deming', ratios: 'HIPS%20Fabs:EC%20(FTIR),Iron:EC%20(FTIR)', x: 'Iron / EC (FTIR)', y: 'HIPS Fabs / EC (FTIR)', f: 'HIPS Fabs / EC (FTIR)', eq: '0' }) },
  { key: 'addis-trange-charcoal', label: 'Addis Ababa · charcoal share vs temperature range', note: 'Cold, stable days and heating (temperature_gradient_analysis, meteorology_source_interaction §3)',
    view: v({ tab: 'correlation', net: 'focus', sites: 'Addis Ababa', fits: 'ols', x: 'Temperature range (station)', y: 'Charcoal share (PMF)', f: 'Charcoal share (PMF)', eq: '0' }, { 'scatter.colorBy': 'Season' }) },
]

const canonical = (view: ViewState) => JSON.stringify([Object.entries(view.h ?? {}).sort(), Object.entries(view.q ?? {}).sort()])
export const sameView = (a: ViewState, b: ViewState) => canonical(a) === canonical(b)

const isRef = (x: unknown): x is { p: string } => !!x && typeof x === 'object' && Object.keys(x as object).length === 1 && typeof (x as { p?: unknown }).p === 'string'
const presetRef = {
  id: 'gallery-preset',
  encode: (value: unknown) => {
    const hit = PRESETS.find((p) => sameView(p.view, value as ViewState))
    return hit ? { p: hit.key } : value
  },
  decode: (value: unknown) => {
    if (!isRef(value)) return value
    const hit = PRESETS.find((p) => p.key === value.p)
    return hit ? JSON.parse(JSON.stringify(hit.view)) : { h: {}, q: {} }
  },
}

const engine = createWebShareEngine<ViewState>({ version: '1', codecs: ['lz', 'raw'], maxLength: 32000, transforms: [presetRef as any] })

const IGNORED_QUERY = new Set(['embed', 'v'])

/** The view the page shows right now. */
export function currentView(): ViewState {
  const url = new URL(window.location.href)
  const h = Object.fromEntries(new URLSearchParams(url.hash.replace(/^#/, '')))
  const q = Object.fromEntries([...url.searchParams].filter(([k]) => !IGNORED_QUERY.has(k)))
  return { h, q }
}

export const encodeView = async (view: ViewState) => (await engine.compressBest(view)).token as string
export const decodeView = (token: string) => engine.decompress(token)

/** A full link to a view; `embed` opens it without the header and filters, for an iframe. */
export async function viewUrl(view: ViewState, embed = false) {
  const base = `${window.location.origin}${window.location.pathname}`
  return `${base}${embed ? '?embed=1' : ''}#v=${await encodeView(view)}`
}

function writeView(view: ViewState) {
  const url = new URL(window.location.href)
  const embed = url.searchParams.get('embed')
  const q = new URLSearchParams(view.q)
  if (embed) q.set('embed', embed)
  const hash = new URLSearchParams(view.h).toString()
  window.history.replaceState(null, '', `${url.pathname}${q.toString() ? `?${q}` : ''}${hash ? `#${hash}` : ''}`)
}

/** Switch the running page to a view: rewrite the URL, then let App and the charts re-read it. */
export function applyView(view: ViewState) {
  writeView(view)
  window.dispatchEvent(new PopStateEvent('popstate'))
  window.dispatchEvent(new HashChangeEvent('hashchange'))
}

/** Before the first render: expand a `#v=<token>` link into the plain URL the app reads. */
export async function restoreSharedView() {
  const hash = window.location.hash.replace(/^#/, '')
  if (!hash.startsWith('v=')) return
  try {
    writeView(await decodeView(decodeURIComponent(hash.slice(2))))
  } catch (e) {
    console.warn('Could not read the shared view; opening the default instead.', e)
    writeView({ h: {}, q: {} })
  }
}

/** A `#v=` link pasted into a page that is already open arrives as a hash change, not a load. */
export function watchSharedViews() {
  window.addEventListener('hashchange', async () => {
    const hash = window.location.hash.replace(/^#/, '')
    if (!hash.startsWith('v=')) return
    try { applyView(await decodeView(decodeURIComponent(hash.slice(2)))) } catch (e) { console.warn('Could not read the shared view.', e) }
  })
}

export const isEmbed = () => new URL(window.location.href).searchParams.get('embed') === '1'

/** The reader's own saved views: a per-browser convenience (localStorage), never shared state. */
const SAVED = 'gallery.views'
export interface SavedView { name: string; view: ViewState }
export function savedViews(): SavedView[] {
  try { return JSON.parse(localStorage.getItem(SAVED) ?? '[]') as SavedView[] } catch { return [] }
}
export function saveViews(list: SavedView[]) {
  try { localStorage.setItem(SAVED, JSON.stringify(list)) } catch { /* private window: nothing kept */ }
}
