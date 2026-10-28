/** The live public PM2.5 release, mapped using export_spartan_reference.py's
 * field names, units and repeat-report mean convention. No spectra are fetched. */
import { csvParseRows } from 'd3'
import { baseFilterId } from './filterId.ts'
import { replacePublicFilter } from './filterMerge.ts'
import type { FilterRow, SiteMeta } from './types'

const ROOT = 'https://spartan-cloud.s3.us-west-2.amazonaws.com/GroupByProduct/FilterBased/ChemSpecPM25/'
export const publicSpartanUrl = (code: string) => {
  if (!/^[A-Z]{4}$/.test(code)) throw new Error('Invalid SPARTAN site code')
  return `${ROOT}FilterBased_ChemSpecPM25_${code}.csv`
}

// Keep the mapping aligned with gallery/data/export_spartan_reference.py.
const PARAMETERS: Record<string, string[]> = {
  'EC PM2.5': ['EC (FTIR)', 'EC (ChemSpec FTIR)'],
  'OC PM2.5': ['OC (FTIR)', 'OC (ChemSpec FTIR)'], 'OM PM2.5': ['OM (FTIR)'],
  'Filter PM2.5 mass': ['PM2.5 mass'], 'BC PM2.5': ['BC (public HIPS)', 'BC (public SSR estimate)'],
  'Equivalent BC PM2.5': ['Equivalent BC (SSR)'],
  ...Object.fromEntries(['Sulfate', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Nitrite', 'Phosphate', 'Bromide'].map(f => [`${f} Ion PM2.5`, [f]])),
  ...Object.fromEntries(['Iron', 'Aluminum', 'Titanium', 'Silicon', 'Zinc', 'Copper', 'Lead', 'Chromium', 'Manganese', 'Vanadium', 'Nickel', 'Cobalt', 'Cerium', 'Arsenic', 'Cadmium', 'Selenium', 'Antimony', 'Barium', 'Lithium', 'Phosphorus', 'Rubidium', 'Strontium', 'Tin', 'Sulfur', 'Chlorine'].map(f => [`${f} PM2.5`, [f]])),
  ...Object.fromEntries(['Calcium', 'Potassium', 'Sodium', 'Magnesium'].map(f => [`${f} PM2.5`, [`${f} (element)`]])),
}
export const PUBLIC_SPARTAN_FIELDS = Object.values(PARAMETERS).flat()
export interface PublicSiteStatus {
  code: string; url: string; state: 'live' | 'bundled'; checked: string
  updated?: string; version?: string; filters?: number; message?: string
}
export interface PublicSiteData { rows: FilterRow[]; status: PublicSiteStatus }

function number(value: string): number | null {
  if (!value.trim()) return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}
function sampleDate(y: string, m: string, d: string): string {
  const year = number(y), month = number(m), day = number(d)
  if (year === null || month === null || day === null || year < 1900 || ![year, month, day].every(Number.isInteger)) throw new Error('Invalid sampling date')
  const date = new Date(Date.UTC(year, month - 1, day))
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) throw new Error('Invalid sampling date')
  return date.toISOString().slice(0, 10)
}
function unitFactor(unit: string): number {
  const u = unit.toLowerCase().replace(/μ/g, 'µ')
  if (/ng\/m\^?3|ng m-3|nanograms/.test(u)) return .001
  if (/ug\/m\^?3|µg\/m\^?3|ug m-3|micrograms/.test(u)) return 1
  throw new Error(`Unrecognized concentration unit: ${unit}`)
}

export function parsePublicSpartan(text: string, site: Pick<SiteMeta, 'code' | 'name' | 'lat' | 'lon'>, checked = new Date().toISOString()): PublicSiteData {
  const table = csvParseRows(text.replace(/^\uFEFF/, ''))
  const headerAt = table.findIndex(r => r.some(c => c.trim() === 'Site_Code') && r.some(c => c.trim() === 'Filter_ID'))
  if (headerAt < 0 || headerAt > 20) throw new Error('Missing SPARTAN chemistry CSV header')
  const columns = table[headerAt].map(c => c.trim())
  for (const c of ['Parameter_Name', 'Value', 'Units', 'Analysis_Description', 'Start_Year_local', 'Start_Month_local', 'Start_Day_local']) {
    if (!columns.includes(c)) throw new Error(`Missing CSV column: ${c}`)
  }
  const at = new Map(columns.map((c, i) => [c, i]))
  const prelude = table.slice(0, headerAt).map(r => r[0] ?? '').join('\n')
  const updated = prelude.match(/File Updated:\s*([^\n]+)/i)?.[1]?.trim()
  const version = prelude.match(/Data version\s*:?\s*([^\n]+)/i)?.[1]?.trim()
  const url = publicSpartanUrl(site.code)
  const filters = new Map<string, { row: FilterRow; sums: Map<string, [number, number]>; flags: Set<string>; methods: Set<string>; hours: number[] }>()
  for (const cells of table.slice(headerAt + 1)) {
    if (cells.every(c => !c.trim())) continue
    const get = (c: string) => (cells[at.get(c) ?? -1] ?? '').trim()
    const code = get('Site_Code'), id = baseFilterId(get('Filter_ID'))
    if (code !== site.code || !new RegExp(`^${code}-\\d{4}$`).test(id)) throw new Error(`Unexpected filter identity: ${code} / ${id}`)
    const date = sampleDate(get('Start_Year_local'), get('Start_Month_local'), get('Start_Day_local'))
    const parameter = get('Parameter_Name')
    let fields = PARAMETERS[parameter]
    if (!fields) throw new Error(`Unmapped public parameter: ${parameter}`)
    const analysis = get('Analysis_Description').toUpperCase(), method = get('Method_Code')
    if (['EC PM2.5', 'OC PM2.5', 'OM PM2.5'].includes(parameter) && analysis !== 'FTIR') throw new Error(`Unexpected carbon method: ${analysis}`)
    if (parameter === 'BC PM2.5') {
      if (['219', '221'].includes(method) || analysis === 'HIPS') fields = ['BC (public HIPS)']
      else if (method === '220' || analysis.startsWith('ESTIMATED FROM THE HIPS-SSR BC CURVE')) fields = ['BC (public SSR estimate)']
      else throw new Error(`Unrecognized public BC method: ${method} / ${analysis}`)
    }
    const factor = unitFactor(get('Units')), value = number(get('Value'))
    let entry = filters.get(id)
    if (!entry) {
      entry = { row: { id, code, site: site.name, color: '#7f7f7f', date, year: Number(date.slice(0, 4)), month: Number(date.slice(5, 7)), season: '',
        lat: number(get('Latitude')) ?? (Number.isFinite(site.lat) ? site.lat : null), lon: number(get('Longitude')) ?? (Number.isFinite(site.lon) ? site.lon : null), excluded: false, exclusion_reason: null,
        public_source_url: url, public_file_updated: updated ?? '', public_retrieved_at: checked, public_live_status: 'live' },
      sums: new Map(), flags: new Set(), methods: new Set(), hours: [] }
      filters.set(id, entry)
    }
    if (entry.row.date !== date) throw new Error(`Conflicting sampling dates for ${id}`)
    if (get('Flag')) entry.flags.add(get('Flag'))
    if (method) entry.methods.add(method)
    const hours = number(get('Hours_sampled'))
    if (/SS5i/i.test(get('Collection_Description')) && hours !== null && hours > 0) entry.hours.push(hours)
    if (value === null) continue
    for (const field of fields) {
      const sum = entry.sums.get(field) ?? [0, 0]
      entry.sums.set(field, [sum[0] + value * factor, sum[1] + 1])
    }
  }
  if (!filters.size) throw new Error('The public chemistry file has no filters')
  const rows = [...filters.values()].map(({ row, sums, flags, methods, hours }) => {
    for (const field of PUBLIC_SPARTAN_FIELDS) { const s = sums.get(field); row[field] = s ? s[0] / s[1] : null }
    row.public_quality_flags = [...flags].join('; ')
    row.public_method_codes = [...methods].join(', ')
    if (hours.length) { hours.sort((a, b) => a - b); const i = Math.floor(hours.length / 2); row['Sampled volume'] = .3 * (hours.length % 2 ? hours[i] : (hours[i - 1] + hours[i]) / 2) }
    return row
  }).sort((a, b) => a.date.localeCompare(b.date) || a.id.localeCompare(b.id))
  return { rows, status: { code: site.code, url, state: 'live', checked, updated, version, filters: rows.length } }
}

let generation = -1
const cache = new Map<string, Promise<PublicSiteData>>()
/** A refresh starts a fresh cache; choosing another site reuses completed requests. */
export function loadPublicSpartan(site: Pick<SiteMeta, 'code' | 'name' | 'lat' | 'lon'>, revision = 0): Promise<PublicSiteData> {
  if (generation !== revision) { cache.clear(); generation = revision }
  let request = cache.get(site.code)
  if (!request) {
    request = (async () => {
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 120_000)
      try {
        const response = await fetch(publicSpartanUrl(site.code), { mode: 'cors', credentials: 'omit', cache: 'no-cache', signal: controller.signal })
        if (!response.ok) throw new Error(response.status === 404 ? 'No public chemistry file for this site' : `Public chemistry: HTTP ${response.status}`)
        return parsePublicSpartan(await response.text(), site)
      } catch (e) {
        if (controller.signal.aborted) throw new Error('Public download timed out after two minutes')
        throw e
      } finally { clearTimeout(timer) }
    })()
    const current = request
    request.catch(() => { if (cache.get(site.code) === current) cache.delete(site.code) })
    cache.set(site.code, request)
  }
  return request
}

/** Union physical filters, replace only public chemistry and retain all bundled
 * lab values/flags. A withdrawn public measurement becomes null in live mode. */
export function mergePublicSpartan(bundled: FilterRow[], live: PublicSiteData): FilterRow[] {
  const byId = new Map(live.rows.map(r => [r.id, r]))
  const ids = new Set(bundled.map(r => r.id))
  const rows = bundled.map(r => {
    const current = byId.get(r.id)
    const replacement = current ?? { ...r, ...Object.fromEntries(PUBLIC_SPARTAN_FIELDS.map(f => [f, null])), public_live_status: 'not in current public file',
      public_source_url: live.status.url, public_file_updated: live.status.updated ?? '', public_retrieved_at: live.status.checked }
    return replacePublicFilter(r, replacement, PUBLIC_SPARTAN_FIELDS)
  })
  rows.push(...live.rows.filter(r => !ids.has(r.id)))
  return rows.sort((a, b) => a.date.localeCompare(b.date) || a.id.localeCompare(b.id))
}
