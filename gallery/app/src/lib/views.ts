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
  // aeronet: the notebooks' AERONET figures on the window-mean AERONET fields (gallery/data/export_aeronet.py).
  // Ratios travel as `ratios` (lib/viewFields; HIPS BC / EC (FTIR) is a built-in ratio and needs none), groupings as the
  // built-in `nb-aeronet-*` ids (lib/groupings), and the crossplot's layout, colour and grouping as its `scatter.*` keys.
  ...AERONET_PRESETS(),
]

function AERONET_PRESETS(): ViewPreset[] {
  // one `ratios` pair, written as lib/viewFields ratiosParam writes it (each part URI-encoded, joined by ':')
  const ratio = (num: string, den: string) => `${encodeURIComponent(num)}:${encodeURIComponent(den)}`
  const MAC = ratio('HIPS Fabs', 'EC (FTIR)')
  const BC_AOD = ratio('HIPS BC', 'AOD 500 nm (AERONET)')
  const PM_AOD = ratio('PM2.5 mass', 'AOD 500 nm (AERONET)')
  const PW = 'Precipitable water (AERONET)', AOD = 'AOD 500 nm (AERONET)', RATIO = 'HIPS BC / EC (FTIR)'
  const addis = (h: Record<string, string>, q: Record<string, string> = {}) => v({ tab: 'correlation', ssites: 'ETAD', ...h }, q)
  const byGroup = (group: string) => ({ 'scatter.layout': 'Per group', 'scatter.colorBy': 'Group', 'scatter.group': group })
  return [
    { key: 'aeronet-mac-pw', label: 'Addis · effective MAC vs precipitable water', view: addis({ ratios: MAC, x: PW, y: 'HIPS Fabs / EC (FTIR)', f: 'HIPS Fabs / EC (FTIR)', eq: '0' }, { 'scatter.colorBy': 'Season' }),
      note: 'HIPS Fabs / FTIR EC (m²/g) against AERONET precipitable water over each window: the "money plot" of notebooks/analysis/meteorology/pw_mac_calibration.ipynb, and MAC_HIPS vs PW in aeronet_method_agreement.ipynb (A)' },
    { key: 'aeronet-pw-tertiles', label: 'Addis · HIPS Fabs vs FTIR EC by precipitable-water tertile', view: addis({ x: 'EC (FTIR)', y: 'HIPS Fabs', f: 'HIPS Fabs' }, byGroup('nb-aeronet-pw-tertiles')),
      note: 'One panel and one fit per PW third (slope = effective MAC): pw_mac_calibration.ipynb "PW tercile analysis" and aeronet_method_agreement.ipynb section 1' },
    { key: 'aeronet-aod-tertiles', label: 'Addis · HIPS Fabs vs FTIR EC by AOD tertile', view: addis({ x: 'EC (FTIR)', y: 'HIPS Fabs', f: 'HIPS Fabs' }, byGroup('nb-aeronet-aod-tertiles')),
      note: 'The compression test, one fit per AOD third: notebooks/analysis/meteorology/aeronet_method_agreement.ipynb section 3' },
    { key: 'aeronet-hips-ftir-aod', label: 'Addis · HIPS BC vs FTIR EC coloured by AOD 500', view: addis({}, { 'scatter.colorBy': 'Group', 'scatter.group': 'nb-aeronet-aod-quintiles' }),
      note: 'HIPS vs FTIR with each filter coloured by its window\'s AOD 500 (quintiles): notebooks/analysis/surface_columnar_linkage.ipynb section 7 (a)' },
    { key: 'aeronet-ratio-aod', label: 'Addis · HIPS BC / FTIR EC vs AOD 500', view: addis({ x: AOD, y: RATIO, f: RATIO, eq: '0' }, { 'scatter.colorBy': 'Season' }),
      note: 'Does the HIPS–FTIR gap follow the column load? notebooks/analysis/meteorology/aeronet_addis_deep_dive.ipynb section 4 and surface_columnar_linkage.ipynb section 7 (b)' },
    { key: 'aeronet-ratio-coarse', label: 'Addis · HIPS BC / FTIR EC vs coarse-mode AOD (dust test)', view: addis({ x: 'Coarse-mode AOD 500 nm (AERONET)', y: RATIO, f: RATIO, eq: '0' }, { 'scatter.colorBy': 'Season' }),
      note: 'The dust-interference test (the notebook found r ≈ −0.33: dust not the driver): aeronet_addis_deep_dive.ipynb section 4, aeronet_addis_diagnostics.ipynb, aeronet_method_agreement.ipynb section 2' },
    { key: 'aeronet-fine-coarse', label: 'Addis · fine vs coarse AOD by season', view: addis({ x: 'Coarse-mode AOD 500 nm (AERONET)', y: 'Fine-mode AOD 500 nm (AERONET)', f: 'Fine-mode AOD 500 nm (AERONET)' }, { 'scatter.colorBy': 'Season' }),
      note: 'The SDA split over each window; the Kiremt coarse-mode anomaly of aeronet_addis_deep_dive.ipynb section 5' },
    { key: 'aeronet-bc-fmf', label: 'Addis · HIPS BC vs fine-mode fraction, per season', view: addis({ x: 'Fine-mode fraction 500 nm (AERONET)', y: 'HIPS BC', f: 'HIPS BC', eq: '0' }, { 'scatter.layout': 'Per season' }),
      note: 'Surface BC against the column\'s fine fraction, one panel per season: notebooks/analysis/surface_columnar_linkage.ipynb section 5 (the notebook used aethalometer BC)' },
    { key: 'aeronet-bc-aod', label: 'Focus sites · BC/AOD ratio by season', view: v({ tab: 'distribution', ratios: BC_AOD, f: 'HIPS BC / AOD 500 nm (AERONET)' }),
      note: 'HIPS BC / AOD 500, the surface-representativeness proxy of Eom et al. (2025): notebooks/analysis/surface_columnar_linkage.ipynb section 1 and aeronet_addis_deep_dive.ipynb section 2' },
    { key: 'aeronet-pm-aod', label: 'Focus sites · PM₂.₅/AOD ratio by season', view: v({ tab: 'distribution', ratios: PM_AOD, f: 'PM2.5 mass / AOD 500 nm (AERONET)' }),
      note: 'η = PM2.5 / AOD, the quantity Snider et al. (2015) decompose: notebooks/analysis/surface_columnar_linkage.ipynb section 6' },
    { key: 'aeronet-type-hips-ftir', label: 'Focus sites · HIPS BC vs FTIR EC by aerosol type', view: v({ tab: 'correlation' }, byGroup('nb-aeronet-type')),
      note: 'One panel and fit per AERONET aerosol type of the window (after Giles et al. 2012; lib/groupings nb-aeronet-type)' },
    { key: 'aeronet-addis-l2', label: 'Addis Ababa · the AERONET column at Level 2.0', view: v({ tab: 'aeronet', ssites: 'ETAD' }, { aerlev: '20' }),
      note: 'The AERONET tab on final-calibrated Level 2.0: AOD as at Level 1.5, but absorption only at AOD 440 ≥ 0.4, which leaves 9 of 169 Addis windows' },
  ]
}

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
