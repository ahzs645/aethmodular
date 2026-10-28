/**
 * When a site never measured a field: the documented stand-in for it, or why there is none.
 *
 * SPARTAN changed samplers and methods over its life, so a chart set up on the focus sites' fields (EC (FTIR),
 * HIPS BC) finds nothing at an older site. Where SPARTAN publishes the same quantity another way, the dashboard's
 * starter tiles and the sidebar move to it; where it does not, the tile says so and why. Every rule below cites
 * SPARTAN's own documentation (SOURCES), so a substitution is never a guess:
 *
 * - HIPS BC and FTIR EC/OC are defined only for the AirPhoton SS5i sampler with stretched Teflon filters
 *   (parameters 28202–28204, methods 217/218); equivalent BC by smoke-stain reflectometer (28201) covers every
 *   sampler, the older SS4i and mesh Teflon too (methods 212–216). [params]
 * - The reflectometer's equivalent BC uses the same 10 m²/g as HIPS (σ = 0.1 cm²/µg). [bcSop]
 * - The public BC from HIPS is Fabs / 10 (methods 219/221); method 220 is SPARTAN's estimate of it from the
 *   reflectometer through a HIPS–SSR curve. [pulls]
 * - FTIR EC/OC are calibrated against IMPROVE's thermal-optical (TOR) carbon. [debus]
 * - Silicon is measured only by ED-XRF (methods 515/516), applied to filters from 11/2018; the ICP-MS before it
 *   (methods 412–416) reports no silicon. [params, xrfSop]
 */
export interface Source { label: string; url: string }

export const SOURCES = {
  params: { label: 'SPARTAN Sampling Parameters & Methods v2.3', url: 'https://www.spartan-network.org/_files/ugd/6321a4_219864c4f1454c58a956339bc58b9120.xlsx?dn=Sampling_Parameters_Methods_2.3.xlsx' },
  bcSop: { label: 'SPARTAN Black Carbon SOP rev. 2.0 (23 Oct 2019)', url: 'https://wustl.box.com/shared/static/nnrzksuty2v38rlssq6b3wil7nxwd9p5.pdf' },
  xrfSop: { label: 'SPARTAN XRF SOP rev. 2.3 (5 Dec 2025)', url: 'https://www.spartan-network.org/_files/ugd/6321a4_28af428d5f204d53aecc5f047b6d7b37.pdf' },
  debus: { label: 'Debus et al. 2022, AMT 15, 2685', url: 'https://doi.org/10.5194/amt-15-2685-2022' },
  recreation: { label: 'docs/spartan-ftir-ec-recreation.md', url: 'https://github.com/ahzs645/aethmodular/blob/main/docs/spartan-ftir-ec-recreation.md' },
  pulls: { label: 'docs/spartan-network-data-pulls.md', url: 'https://github.com/ahzs645/aethmodular/blob/main/docs/spartan-network-data-pulls.md' },
} satisfies Record<string, Source>

export interface Equivalent {
  field: string
  /** what makes it the same quantity, for the tile's note */
  basis: string
  source: Source
}

const SSR_ESTIMATE: Equivalent = { field: 'BC (public SSR estimate)', basis: "SPARTAN's public BC estimated from the smoke-stain reflectometer through its HIPS–SSR curve (method 220)", source: SOURCES.pulls }
const SSR_EBC: Equivalent = { field: 'Equivalent BC (SSR)', basis: 'equivalent BC from the smoke-stain reflectometer, at the same 10 m²/g as HIPS; SPARTAN’s BC on every sampler', source: SOURCES.bcSop }
const tor = (kind: 'EC' | 'OC'): Equivalent => ({ field: `${kind} (TOR)`, basis: `IMPROVE’s thermal-optical ${kind}, the reference SPARTAN’s FTIR ${kind} is calibrated to`, source: SOURCES.debus })
const chemspec = (kind: 'EC' | 'OC'): Equivalent => ({ field: `${kind} (ChemSpec FTIR)`, basis: `SPARTAN’s public FTIR ${kind} (the same method)`, source: SOURCES.params })

/** each field's stand-ins, best first: the same instrument, then the same quantity by another documented method */
export const EQUIVALENTS: Record<string, Equivalent[]> = {
  'HIPS BC': [
    { field: 'BC (public HIPS)', basis: 'SPARTAN’s public BC from the same HIPS absorption (Fabs / 10 m²/g)', source: SOURCES.pulls },
    SSR_ESTIMATE, SSR_EBC,
  ],
  'BC (public HIPS)': [SSR_ESTIMATE, SSR_EBC],
  'EC (FTIR)': [
    chemspec('EC'),
    { field: 'EC (FTIR, VIBES)', basis: 'FTIR EC from the same spectra under the VIBES baseline', source: SOURCES.recreation },
    { field: 'EC (FTIR, spline)', basis: 'FTIR EC from the same spectra under the spline baseline', source: SOURCES.recreation },
    tor('EC'),
  ],
  'OC (FTIR)': [chemspec('OC'), tor('OC')],
}

export interface Resolved {
  field: string
  /** set when `field` stands in for the one asked for */
  equivalent?: Equivalent & { for: string }
}

/** the field itself where it has values, else its first stand-in that does, else the field (to say why it is empty) */
export function resolveField(field: string, has: (f: string) => boolean): Resolved {
  if (has(field)) return { field }
  const alt = EQUIVALENTS[field]?.find((e) => has(e.field))
  return alt ? { field: alt.field, equivalent: { ...alt, for: field } } : { field }
}

const FTIR_CARBON = /^(EC|OC|OM) \((FTIR|ChemSpec FTIR)/
const FTIR_GROUPS = new Set(['Alkane CH', 'Alcohol COH', 'Carboxylic COOH', 'Non-acid CO'])
const HIPS = /HIPS/
const XRF_ONLY = new Set(['Silicon', 'Sulfur', 'Chlorine', 'Calcium (element)', 'Potassium (element)', 'Sodium (element)', 'Magnesium (element)'])

export interface SiteSpan { name: string; date_min?: string; date_max?: string }
export interface Missing { text: string; source?: Source }

const span = (s: SiteSpan) => {
  const a = s.date_min?.slice(0, 4), b = s.date_max?.slice(0, 4)
  return a && b ? (a === b ? a : `${a}–${b}`) : ''
}

/**
 * Why a site has no value for a field: SPARTAN's documented method constraint where one explains it, and only
 * what the site's own record shows beyond that (a date claim only where its filters all fall before the change).
 * `network` is the site's network: IMPROVE measures carbon by TOR, not FTIR. `has` says which other fields the
 * site has, to name its own black carbon where that is the reflectometer's.
 */
export function whyMissing(field: string, site: SiteSpan, network: 'spartan' | 'improve' = 'spartan', has: (f: string) => boolean = () => false): Missing {
  const where = `${site.name}${span(site) ? ` (${span(site)})` : ''}`
  const none = `no filter from ${where} in SPARTAN's public release has it`
  const ssr = has('Equivalent BC (SSR)') || has('BC (public SSR estimate)')
  if (network === 'improve' && (FTIR_CARBON.test(field) || FTIR_GROUPS.has(field)))
    return { text: `IMPROVE measures carbon by thermal-optical reflectance, not FTIR: use EC (TOR) / OC (TOR) at ${site.name}.`, source: SOURCES.debus }
  if (FTIR_CARBON.test(field) || FTIR_GROUPS.has(field))
    return { text: `${field} is not available: SPARTAN reports FTIR carbon only for filters from the SS5i sampler on stretched Teflon (methods 217/218), and ${none}.${
      ssr && /^EC/.test(field) ? ' Its black carbon is the reflectometer’s, an optical measurement rather than EC, so it does not stand in here.' : ''}`, source: SOURCES.params }
  if (HIPS.test(field))
    return { text: `${field} is not available: SPARTAN reports HIPS BC only for SS5i stretched-Teflon filters (methods 217/218), and ${none}.${
      ssr ? ' Its black carbon is the smoke-stain reflectometer’s equivalent BC.' : ''}`, source: SOURCES.params }
  if (XRF_ONLY.has(field)) {
    const before = !!site.date_max && site.date_max < XRF_FROM
    return { text: `${field} is not available: SPARTAN measures it only by ED-XRF, applied to filters from November 2018 on (the ICP-MS before it reports no ${field.replace(/ \(element\)$/, '').toLowerCase()}), and ${
      before ? `the filters from ${where} all predate that` : none}.`, source: SOURCES.xrfSop }
  }
  if (/\(ERA5\)|\(station\)|^Temperature|^Relative humidity|^Precipitation|^Boundary-layer|^Wind/.test(field))
    return { text: `No weather for ${where}: the ERA5 export (gallery/data/export_weather.py) does not cover this site yet.` }
  if (/AERONET|AOD|SSA|AAOD|Ångström|Angstrom/.test(field))
    return { text: `No AERONET column for ${where}: no station within 50 km had retrievals in its filter windows.` }
  return { text: `${field} is not available: ${none} yet.` }
}

/** the first filter month the XRF SOP's revision 2.0 applies to */
const XRF_FROM = '2018-11'
