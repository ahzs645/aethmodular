/**
 * Filters whose black carbon came from SPARTAN's smoke-stain reflectometer (SSR) rather than HIPS: the
 * reflectometer's equivalent BC, or the public BC estimated from it through the HIPS–SSR curve (method 220).
 * The Samples panel can hide them (URL `ssr`): only the ones with no HIPS reading, or every one that carries
 * an SSR value (643 of the network's 1,495 also have HIPS, so that choice drops HIPS data too). It is offered
 * only while the sites in view have any.
 */
export type SsrHide = '' | 'only' | 'any'
export const SSR_FIELDS = ['Equivalent BC (SSR)', 'BC (public SSR estimate)'] as const
const HIPS_FIELDS = ['HIPS BC', 'BC (public HIPS)', 'HIPS Fabs'] as const

type Row = Readonly<Record<string, unknown>>
const measured = (r: Row, fields: readonly string[]) => fields.some((f) => { const v = r[f]; return typeof v === 'number' && Number.isFinite(v) })
export const hasSsr = (r: Row) => measured(r, SSR_FIELDS)
export const hasHips = (r: Row) => measured(r, HIPS_FIELDS)
/** an SSR filter with no HIPS reading: its only black carbon is the reflectometer's */
export const ssrOnly = (r: Row) => hasSsr(r) && !hasHips(r)

export const parseSsrHide = (v: string | null): SsrHide => (v === 'only' || v === 'any' ? v : '')
/** does the row stay under the choice */
export const keepUnderSsr = (mode: SsrHide) => (r: Row) => mode === '' || (mode === 'only' ? !ssrOnly(r) : !hasSsr(r))

/** how many of `rows` each choice would hide */
export function ssrCounts(rows: readonly Row[]) {
  let any = 0, only = 0
  for (const r of rows) if (hasSsr(r)) { any++; if (!hasHips(r)) only++ }
  return { any, only }
}
