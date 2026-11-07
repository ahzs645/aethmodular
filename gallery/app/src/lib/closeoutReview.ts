import { newSection, type Section } from './sections.ts'

/** Uses the existing section/share/export path; every view is pinned independently of the reader's current controls. */
export function closeoutReview(generated: string, sourceStatus: string, sourceVerified = false): Section {
  const section = newSection('FTIR transfer: existing-data closeout review')
  const view = (q: Record<string, string> = {}) => ({ h: { tab: 'experimental', xp: 'continuum', net: 'focus' }, q })
  const entries = [
    ['xc-coverage', '1. Coverage and physical-filter identity', 'Counts come from the active export. Repeat scans are not additional independent filters; this cohort differs from the separate 232-filter matched seasonal study.', {}],
    ['xc-source', '2. Source-reference performance', `IMPROVE TOR cross-validation, fixed 18 components. Read median absolute error, signed bias and tail error alongside MAE and RMSE when available; missing metrics are not zero. ${sourceVerified ? 'Verified fold-safe EPO source scores are included in the common physical-filter comparison. This source-domain evaluation does not establish Addis EC accuracy.' : 'EPO source scores remain withheld until a verified fold-safe rerun is attached.'}`, {}],
    ['xc-continuum', '3. Baseline behavior', 'The saved 2–3 November injection study repeats 16 ambient parent filters across shapes and backgrounds; it differs from the later independent-blank control. Broad spectral preservation and numerical convergence answer different questions.', { 'xc-cont.w': '1880' }],
    ['xc-season-aeth', '4. Bega, Belg and Kiremt', 'Calendar: Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep. Contrasts are relative to Belg and use an optical comparator; they do not establish seasonal EC correction factors.', {}],
    ['xc-network', '5. Optical-response sensitivity', 'The response fitted to IMPROVE is a transferred model. Its bend is empirical, not an instrument-certified cutoff; it does not create TOR observations at Addis.', {}],
    ['xc-addis', '6. Candidate tradeoffs and limits', 'Production and anchored VIBES against the MA350 aethalometer. A smaller intercept supports agreement in this comparison; absolute Addis EC accuracy remains unresolved.', { 'xc-addis.x': 'aeth_ir', 'xc-addis.view': 'pair', 'xc-addis.before': 'vibes k18', 'xc-addis.after': 'vibes_anchorhp k18' }],
  ] as const
  return {
    ...section,
    about: `Review candidate · export generated ${generated} · source evaluation: ${sourceStatus}. This section reopens the active data export; it is not a frozen data package. Preserve the source/data release receipt with any exported figures.`,
    items: entries.map(([chart, title, note, q], i) => ({ id: `${section.id}-${i}`, chart, title, note, view: view(q), wide: true })),
  }
}
