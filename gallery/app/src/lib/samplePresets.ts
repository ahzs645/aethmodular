/**
 * Each investigation page's own Samples (a section page's `samples`, sections/types): the sites, calendar, dates and
 * exclusions the diagnostic is presented with. A visit opens on them, the reader adds to them in the Samples panel,
 * and those additions stay with that page for the session; the rest of the gallery keeps its own set. Hand
 * exclusions (drop / keep) are not part of it: they are the reader's judgement of a filter, so they hold everywhere.
 *
 * A set is the URL's own form (lib/url), so one parser reads a preset, a remembered set and a link.
 */
export const SAMPLE_KEYS = ['net', 'ssites', 'isites', 'sites', 'seasons', 'seas', 'cal', 'cs', 'rr', 'cv', 'pmfg', 'pmf', 'pmfon', 'pmfonly', 'ssr', 'from', 'to', 'excl', 'incl'] as const
export type SampleKey = typeof SAMPLE_KEYS[number]
export type SampleSet = Partial<Record<SampleKey, string>>

export interface SamplePreset {
  /** how the Samples panel names it: what the page is presented on */
  label: string
  samples: SampleSet
}

/** on a page with a preset, marks a link that holds no sample key on purpose: every Samples setting at its default */
export const NO_PRESET_KEY = 'smp'

type Read = { get(key: string): string | null }

/** the sample keys a URL, or App's values for it, sets */
export function samplesIn(from: Read): SampleSet {
  const out: SampleSet = {}
  for (const k of SAMPLE_KEYS) { const v = from.get(k); if (v) out[k] = v }
  return out
}

/** a set as URL parameters, for what reads a link */
export const sampleParams = (set: SampleSet) => new URLSearchParams(set as Record<string, string>)

export const sameSamples = (a: SampleSet, b: SampleSet) =>
  SAMPLE_KEYS.every((k) => (a[k] ?? '') === (b[k] ?? ''))

/** does a URL say which samples it wants, so that a page's preset gives way to it */
export const linkSetsSamples = (p: Read) => p.get(NO_PRESET_KEY) !== null || Object.keys(samplesIn(p)).length > 0

/** the value of the marker: wanted when the view holds no key the page's preset would otherwise fill */
export const noPresetMark = (now: SampleSet, preset: SamplePreset | null | undefined): string | null =>
  preset && Object.keys(preset.samples).length && !Object.keys(now).length ? '1' : null

/** the URL a page opens on: a link that does not say which samples it wants gets the page's preset */
export function withPreset(p: URLSearchParams, preset: SamplePreset | null | undefined): URLSearchParams {
  if (!preset || linkSetsSamples(p)) return p
  const out = new URLSearchParams(p)
  for (const [k, v] of Object.entries(preset.samples)) out.set(k, v)
  return out
}

/**
 * The set a page change lands on. `link`: the samples the URL that caused the change spelled out, which win; else
 * what the reader left the page on; else the page's preset (the gallery-wide set, scope `''`, has none).
 */
export function samplesFor(scope: string, preset: SamplePreset | null | undefined, remembered: ReadonlyMap<string, SampleSet>, link?: SampleSet | null): SampleSet {
  return link ?? remembered.get(scope) ?? preset?.samples ?? {}
}
