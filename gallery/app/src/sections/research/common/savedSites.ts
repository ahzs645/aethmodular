/**
 * Saved per-site tables under the Samples sites. A notebook's table names its sites in its own words ("Addis", "Addis
 * Ababa", "Addis (ETAD)", "Pasadena (USPA): screened"); these read that label as a site code, so a page can keep the rows
 * of the sites picked. The tables are saved, so this picks rows and recomputes nothing.
 */
const CODE: Record<string, string> = { Addis: 'ETAD', Bishoftu: 'ETBI', Beijing: 'CHTS', Delhi: 'INDH', JPL: 'USPA', Pasadena: 'USPA' }

/** the site code a saved label names, or null where it names no one site (a pool) */
export const savedSiteCode = (label: unknown): string | null => CODE[String(label).split(/[\s(:]/)[0]] ?? null

/** the site codes in Samples: the pick, or with no pick to follow (net=focus) the sites with rows in view */
export const samplesCodes = (picked: ReadonlySet<string>, rows: readonly { code: string }[]): ReadonlySet<string> =>
  (picked.size ? picked : new Set(rows.map((r) => r.code)))

/** does a saved row's site stay in view? A row that names no one site does; `savedSites` asks whether any site is left */
export const siteInView = (codes: ReadonlySet<string>, label: unknown) => { const c = savedSiteCode(label); return c === null || codes.has(c) }

/** the rows of the sites in view, or none when only rows that name no one site would be left */
export function savedSites<T>(rows: readonly T[], codes: ReadonlySet<string>, labelOf: (r: T) => unknown): T[] {
  const kept = rows.filter((r) => siteInView(codes, labelOf(r)))
  return kept.some((r) => savedSiteCode(labelOf(r)) !== null) ? kept : []
}

/** what a saved per-site chart says when Samples leaves it no site */
export const PICK_SAVED_SITE = 'This chart holds the five calibration sites and none is in Samples: pick Addis Ababa, Bishoftu, Beijing, Delhi or JPL.'
