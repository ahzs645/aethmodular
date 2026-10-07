/**
 * Decoders for gallery/data/export_similarity.py. The exporter does all the
 * arithmetic on the full-resolution frozen spectra; the browser only indexes.
 *
 * pairs_{method}_{mask}.bin  int16 upper triangle (row-major, diagonal
 *                            included) of the group × group mean Pearson r,
 *                            r × scale; −32768 = undefined (a one-filter group
 *                            against itself)
 * top_{method}_{mask}.bin    uint16 [G × K] filter indices, then int16
 *                            [G × K] scores; 65535 pads a short list
 * traces_{method}.bin        float32 [F] per-filter scale, then int16 [F × B]
 *                            binned corrected absorbance
 */

export interface SiteInfo {
  site: string; split: Split; n: number; label: string
  name: string | null; region: string | null; lat: number | null; lon: number | null
  elevation?: number | null; land_use?: string | null; setting?: string | null
}

export interface SeasonDef { name: string; months: number[]; color: string }
export type Split = 'train' | 'test' | 'external'

export interface SimilarityMeta {
  schema_version: 1
  source_signature: string
  source_sha256: Record<string, string>
  exporter_sha256: string
  methods: string[]
  masks: { key: string; label: string; co2_low: number | null; upper: number | null; n_channels: number; bins: boolean[] }[]
  default_mask: string
  scale: number
  top_k: number
  n_pairs: number
  bin_wn: number[]
  background_rank?: number
  method_grids?: Record<string, { bin_wn: number[]; masks: SimilarityMeta['masks'] }>
  calendars: Record<string, SeasonDef[]>
  site_detail_sha256: Record<string, string>
  sites: SiteInfo[]
  undated: string[]
  groups: { site: number; season: number | null; n: number }[]
  filters: { sample_id: string[]; site: number[]; date: (string | null)[]; season: number[] }
}

/** "Puget Sound, WA" — the human name for a site code. */
export function siteName(s: SiteInfo): string {
  return [s.name, s.region].filter(Boolean).join(', ') || s.site
}

/** "Puget Sound, WA · 47.57°, −122.31° · 12 m" for tooltips. */
export function siteWhere(s: SiteInfo): string {
  const coords = s.lat != null && s.lon != null ? `${s.lat.toFixed(2)}°, ${s.lon.toFixed(2)}°` : null
  const elev = s.elevation != null ? `${s.elevation} m` : null
  return [siteName(s), coords, elev, s.land_use, s.setting].filter(Boolean).join(' · ')
}

export const BASE = `${import.meta.env.BASE_URL}data/similarity/`
const UNDEFINED = -32768
const PAD = 65535

export const calendarOf = (meta: SimilarityMeta, site: number): SeasonDef[] =>
  meta.calendars[meta.sites[site].site] ?? meta.calendars.default

/** "ETAD · Addis Ababa" or "PHOE1 · Winter (Dec–Feb)". */
export function groupLabel(meta: SimilarityMeta, g: number): string {
  const { site, season } = meta.groups[g]
  const base = meta.sites[site].label
  return season == null ? base : `${meta.sites[site].site} · ${calendarOf(meta, site)[season].name}`
}

export function groupColor(meta: SimilarityMeta, g: number, fallback: string): string {
  const { site, season } = meta.groups[g]
  return season == null ? fallback : calendarOf(meta, site)[season].color
}

/** URL-stable key: "PHOE1" for a site, "PHOE1:0" for its first season. */
export const groupKey = (meta: SimilarityMeta, g: number) => {
  const { site, season } = meta.groups[g]
  return season == null ? meta.sites[site].site : `${meta.sites[site].site}:${season}`
}

export function groupIndex(meta: SimilarityMeta): Map<string, number> {
  return new Map(meta.groups.map((_, g) => [groupKey(meta, g), g]))
}

/** Filter indices belonging to a group. */
export function members(meta: SimilarityMeta, g: number): number[] {
  const { site, season } = meta.groups[g]
  const out: number[] = []
  const f = meta.filters
  for (let i = 0; i < f.site.length; i++) if (f.site[i] === site && (season == null || f.season[i] === season)) out.push(i)
  return out
}

export class PairMatrix {
  constructor(private tri: Int16Array, readonly G: number, private scale: number) {
    if (tri.length !== (G * (G + 1)) / 2) throw new Error('pair matrix size does not match the group table')
  }
  get(i: number, j: number): number | null {
    const [a, b] = i <= j ? [i, j] : [j, i]
    const v = this.tri[a * this.G - (a * (a - 1)) / 2 + (b - a)]
    return v === UNDEFINED ? null : v / this.scale
  }
}

export class TopLists {
  private idx: Uint16Array
  private r: Int16Array
  constructor(buf: ArrayBuffer, readonly G: number, readonly k: number, private scale: number) {
    if (buf.byteLength !== G * k * 4) throw new Error('top-list size does not match the group table')
    this.idx = new Uint16Array(buf, 0, G * k)
    this.r = new Int16Array(buf, G * k * 2, G * k)
  }
  /** Most similar filters from other sites, best first. */
  of(g: number): { filter: number; r: number }[] {
    const out: { filter: number; r: number }[] = []
    for (let i = g * this.k; i < (g + 1) * this.k; i++) {
      if (this.idx[i] === PAD) break
      out.push({ filter: this.idx[i], r: this.r[i] / this.scale })
    }
    return out
  }
}

export class Traces {
  private scale: Float32Array
  private q: Int16Array
  constructor(buf: ArrayBuffer, readonly F: number, readonly B: number) {
    if (buf.byteLength !== F * 4 + F * B * 2) throw new Error('trace file size does not match the filter table')
    this.scale = new Float32Array(buf, 0, F)
    this.q = new Int16Array(buf, F * 4, F * B)
  }
  of(f: number): Float32Array {
    const out = new Float32Array(this.B)
    const s = this.scale[f]
    for (let i = 0; i < this.B; i++) out[i] = this.q[f * this.B + i] * s
    return out
  }
}

const cache = new Map<string, Promise<ArrayBuffer>>()

/** Each binary is fetched once per session, whichever chart asks first. */
export function fetchBinary(name: string): Promise<ArrayBuffer> {
  let p = cache.get(name)
  if (!p) {
    p = fetch(BASE + name).then((r) => {
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`)
      return r.arrayBuffer()
    })
    p.catch(() => cache.delete(name))
    cache.set(name, p)
  }
  return p
}
