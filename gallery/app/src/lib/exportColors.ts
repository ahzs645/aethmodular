import { paletteSeason, seasonExportColor, seasonColorRows, type ExportSeason, type PaletteSeason } from './exportSeasons.ts'
import { color, hcl, lab, schemeDark2, schemeTableau10 } from 'd3'
import { ESTIMATORS, FIT_STYLE, type Estimator } from './regressionMethods.ts'
import { printColour } from './theme.ts'

export interface ExportSite { code: string; name: string; color: string }
export interface PaletteSite { key: string; label: string; color: string }
export interface PaletteFit { key: Estimator; label: string; color: string; mixed: boolean }
export interface ExportPalette { sites: PaletteSite[]; seasons: PaletteSeason[]; fits: PaletteFit[] }
export interface ExportColors { mode: 'standard' | 'custom'; consolidateSeasons: boolean; seasons: Record<string, string>; sameSites: boolean; shared: string; sites: Record<string, string>; fits: Partial<Record<Estimator, string>> }
export const DEFAULT_EXPORT_COLORS: ExportColors = { mode: 'standard', consolidateSeasons: false, seasons: {}, sameSites: false, shared: '#2171b5', sites: {}, fits: {} }

export const hexColor = (value: unknown): string | null => typeof value === 'string' && /^#(?:[\da-f]{3}|[\da-f]{6})$/i.test(value.trim()) ? color(value.trim())!.formatHex() : null
export const paintColor = (value: string | null): string | null => {
  const c = value && color(value)
  return c && c.opacity > 0 && Number.isFinite(lab(c).l) ? c.formatHex() : null
}
export function readExportColors(value: unknown): ExportColors {
  const v = value && typeof value === 'object' ? value as Partial<ExportColors> : {}
  const valid = (o: unknown) => Object.fromEntries(Object.entries(o && typeof o === 'object' && !Array.isArray(o) ? o : {}).flatMap(([k, c]) => hexColor(c) ? [[k, hexColor(c)!]] : []))
  const customized = v.sameSites === true || Object.keys(valid(v.sites)).length > 0 || Object.keys(valid(v.seasons)).length > 0 || Object.keys(valid(v.fits)).length > 0
  return { mode: v.mode === 'standard' ? 'standard' : v.mode === 'custom' || customized ? 'custom' : 'standard', consolidateSeasons: v.consolidateSeasons === true, seasons: valid(v.seasons), sameSites: v.sameSites === true, shared: hexColor(v.shared) ?? DEFAULT_EXPORT_COLORS.shared, sites: valid(v.sites),
    fits: Object.fromEntries(Object.entries(valid(v.fits)).filter(([k]) => ESTIMATORS.includes(k as Estimator))) }
}

/** Standard colors bypass overrides without discarding the reader's custom palette. */
export function effectiveExportColors(prefs: ExportColors): ExportColors {
  return prefs.mode === 'standard' ? { ...prefs, sameSites: false, sites: {}, seasons: {}, fits: {} } : prefs
}

export const siteExportColor = (site: PaletteSite, prefs: ExportColors) => prefs.sameSites ? prefs.shared : prefs.sites[site.key] ?? site.color

/**
 * Site and season colours, which are the same in both themes (lib/theme): a figure composed on the
 * dark page keeps them as drawn, even when a dark colour-scheme step happens to share the hex (a
 * dark ramp stop is exactly Beijing's #e74c3c, which used to print as the light ramp's #df4532).
 */
export const fixedColours = (sites: readonly { color: string }[], seasons: readonly { color: string }[] = [], categories: readonly { color: string }[] = []): Set<string> =>
  new Set([...sites, ...seasons, ...categories].flatMap((s) => paintColor(s.color) ?? []))

/** The light theme's colour for a paint the dark page drew (null: leave it as drawn). */
export function printPaint(paint: string, fixed: ReadonlySet<string> = new Set()): string | null {
  const hex = paintColor(paint)
  return hex && !fixed.has(hex) ? printColour(hex) : null
}

/** Preserve alpha in a paint value; opacity attributes remain untouched too. */
export function replacePaint(paint: string, from: string, to: string): string {
  if (!paintColor(paint) || paintColor(paint) !== paintColor(from)) return paint
  const alpha = color(paint)?.opacity ?? 1
  if (alpha === 1) return to
  const c = color(to)!
  c.opacity = alpha
  return c.formatRgb()
}

export function contrastRatio(a: string, b = '#ffffff'): number {
  const luminance = (s: string) => {
    const c = color(s)!.rgb()
    const channel = (v: number) => (v /= 255) <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4
    return .2126 * channel(c.r) + .7152 * channel(c.g) + .0722 * channel(c.b)
  }
  const x = luminance(a), y = luminance(b)
  return (Math.max(x, y) + .05) / (Math.min(x, y) + .05)
}

/** D3 categorical palettes, extended in HCL for large selections. Greedy Lab spacing
 * separates sites and estimators; all suggestions target a light background.
 * https://d3js.org/d3-scale-chromatic/categorical */
export function recommendExportColors(palette: ExportPalette, sameSites: boolean, consolidateSeasons = false): ExportColors {
  const candidates = [...new Set([...schemeTableau10, ...schemeDark2,
    ...Array.from({ length: 144 }, (_, i) => hcl((i % 36) * 10, i < 72 ? 45 : 65, Math.floor(i / 36) % 2 ? 55 : 45).formatHex())])]
    .filter(c => contrastRatio(c) >= 3).map(c => ({ color: c, lab: lab(c) }))
  const used: typeof candidates = []
  const distance = (a: typeof candidates[number], b: typeof candidates[number]) => (a.lab.l - b.lab.l) ** 2 + (a.lab.a - b.lab.a) ** 2 + (a.lab.b - b.lab.b) ** 2
  const pick = (text = false) => {
    const pool = candidates.filter(c => !used.includes(c) && (!text || contrastRatio(c.color) >= 4.5))
    const next = pool.reduce<(typeof pool)[number] | null>((best, c) => !best || Math.min(...used.map(u => distance(c, u)), Infinity) > Math.min(...used.map(u => distance(best, u)), Infinity) ? c : best, null)
    if (!next) return '#222222'
    used.push(next)
    return next.color
  }
  const prefs: ExportColors = { mode: 'custom', consolidateSeasons, seasons: {}, sameSites, shared: '#2171b5', sites: {}, fits: {} }
  // Fit colors also color their equations, so use the stronger text contrast threshold.
  for (const fit of palette.fits) prefs.fits[fit.key] = pick(true)
  if (sameSites && palette.sites.length) prefs.shared = pick()
  else for (const site of palette.sites) prefs.sites[site.key] = pick()
  for (const season of seasonColorRows(palette.seasons, consolidateSeasons)) prefs.seasons[season.key] = pick()
  return prefs
}

/** Mark only figure elements that actually encode sites. A legend is a fallback for older
 * charts; explicit mark annotations distinguish sites that happen to share a source color. */
export function collectExportPalette(roots: Element[], sites: readonly ExportSite[], seasons: readonly ExportSeason[] = []): ExportPalette {
  const byName = new Map(sites.map(s => [s.name, s]))
  const byCode = new Map(sites.map(s => [s.code, s]))
  const found = new Map<string, PaletteSite>()
  const explicitSites = new Set<string>()
  const bySeason = new Map(seasons.map(s => [s.name, paletteSeason(s)]))
  const foundSeasons = new Map<string, PaletteSeason>()
  const fits = new Map<Estimator, Set<string>>()
  const all = roots.flatMap(root => [root, ...root.querySelectorAll('*')])
  for (const el of all) {
    const name = el.getAttribute('data-export-site') ?? el.getAttribute('data-fit-site')
    const site = name && (byCode.get(name) ?? byName.get(name))
    if (site) {
      const native = paintColor(el.getAttribute('data-export-color')) ?? site.color
      el.setAttribute('data-export-site', site.code)
      el.setAttribute('data-export-color', native)
      found.set(site.code, { key: site.code, label: site.name, color: native })
      if (!el.hasAttribute('data-export-legend-site') && !el.closest('[data-fit-site]')) explicitSites.add(site.code)
    }
    const seasonName = el.getAttribute('data-export-season') ?? el.getAttribute('data-export-series') ?? el.getAttribute('data-fit-site') ?? el.getAttribute('data-export-site')
    const season = seasonName && bySeason.get(seasonName)
    if (season) {
      el.setAttribute('data-export-season', season.key)
      el.setAttribute('data-export-color', paintColor(el.getAttribute('data-export-color')) ?? season.color)
      foundSeasons.set(season.key, season)
    }
    const method = el.getAttribute('data-regression') ?? el.getAttribute('data-regression-color') ?? el.getAttribute('data-regression-stroke') ?? el.getAttribute('data-fit-uncertainty')
    if (method && ESTIMATORS.includes(method as Estimator)) {
      const m = method as Estimator
      if (!fits.has(m)) fits.set(m, new Set())
      const native = paintColor(el.getAttribute(el.hasAttribute('data-regression') || el.hasAttribute('data-regression-stroke') ? 'stroke' : 'fill'))
      if (native) fits.get(m)!.add(native)
    }
  }
  // Legacy charts with a site legend but no mark annotation: recolor matching site paint
  // outside fit/season/other series roles. Ambiguous source colors are never guessed.
  const legacy = all.filter(el => el.hasAttribute('data-export-legend-site') && !explicitSites.has(el.getAttribute('data-export-legend-site')!))
  const colorOwners = new Map<string, Set<string>>()
  for (const el of legacy) {
    const key = el.getAttribute('data-export-legend-site')!, native = paintColor(el.getAttribute('data-export-color'))
    if (native) colorOwners.set(native, new Set([...(colorOwners.get(native) ?? []), key]))
  }
  for (const el of all) {
    if (el.closest('[data-export-site], [data-regression], [data-regression-color], [data-regression-stroke], [data-fit-uncertainty], [data-export-series]')) continue
    for (const attr of ['fill', 'stroke']) {
      const keys = colorOwners.get(paintColor(el.getAttribute(attr)) ?? '')
      if (keys?.size === 1) el.setAttribute(`data-export-${attr}-site`, [...keys][0])
    }
  }
  return { seasons: [...foundSeasons.values()], sites: [...found.values()].sort((a, b) => a.label.localeCompare(b.label)),
    fits: ESTIMATORS.filter(m => fits.has(m)).map(m => ({ key: m, label: FIT_STYLE[m].label, color: [...fits.get(m)!][0] ?? FIT_STYLE[m].color, mixed: fits.get(m)!.size > 1 })) }
}

/** Runs on the laid-out clone, never on the live chart or the captured source figure. */
export function applyExportColors(svg: SVGSVGElement, palette: ExportPalette, prefs: ExportColors) {
  prefs = effectiveExportColors(prefs)
  const sites = new Map(palette.sites.map(s => [s.key, s]))
  const seasons = new Map(palette.seasons.map(s => [s.key, s]))
  for (const el of [svg, ...svg.querySelectorAll('*')]) {
    const scope = el.closest('[data-export-site]')
    const seasonScope = el.closest('[data-export-season]')
    const season = seasons.get(seasonScope?.getAttribute('data-export-season') ?? '')
    for (const attr of ['fill', 'stroke']) {
      const paint = el.getAttribute(attr)
      if (!paint) continue
      const key = el.getAttribute(`data-export-${attr}-site`) ?? scope?.getAttribute('data-export-site')
      const site = key ? sites.get(key) : undefined
      if (season) el.setAttribute(attr, replacePaint(paint, seasonScope?.getAttribute('data-export-color') ?? season.color, seasonExportColor(season, prefs, palette.seasons)))
      else if (site) el.setAttribute(attr, replacePaint(paint, scope?.getAttribute('data-export-color') ?? site.color, siteExportColor(site, prefs)))
    }
    // Explicit estimator overrides take precedence over grouped site colors.
    const strokeMethod = el.getAttribute('data-regression') ?? el.getAttribute('data-regression-stroke')
    const fillMethod = el.getAttribute('data-regression-color') ?? el.getAttribute('data-fit-uncertainty')
    if (strokeMethod && prefs.fits[strokeMethod as Estimator]) el.setAttribute('stroke', prefs.fits[strokeMethod as Estimator]!)
    if (fillMethod && prefs.fits[fillMethod as Estimator]) el.setAttribute('fill', prefs.fits[fillMethod as Estimator]!)
  }
}
