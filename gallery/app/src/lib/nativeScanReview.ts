import type { NativeAnalysis } from './spartan'

interface ScanReview {
  weak: string
  preferred?: string
  reason?: string
}

// docs/vibes-rank-check-2026-11-10.json and comparison with the archived repeats.
// These choices affect the display. Source QC and archived analyses stay intact.
const REVIEWS: Partial<Record<string, ScanReview>> = {
  'CHTS-0777-8': { weak: '978', preferred: '1005' },
  'CHTS-0814-5': { weak: '1413' },
  'CHTS-0830-5': { weak: '1929', preferred: '1975', reason: 'has much weaker absorption bands after spline correction.' },
  'ETAD-0037-5': { weak: '406', reason: 'has nearly identical sample and background scans.' },
  'ETAD-0139-3': { weak: '1626', preferred: '1668' },
  'INDH-0209-1': { weak: '4182', preferred: '4199' },
  'INDH-0265-1': { weak: '4528' },
  'KRSE-0136-8': { weak: '470', reason: 'has an unusually small raw absorbance span.' },
  'PRFJ-0073-1': { weak: '530', preferred: '522' },
  'ZAJB-0121-1': { weak: '565', preferred: '564' },
  'ZAJB-0283-3': { weak: '3651', preferred: '3656' },
  'ZAJB-0286-6': { weak: '3655', preferred: '3660' },
}

/** Resolve the displayed scan while preserving explicit picks and archive order. */
export function selectNativeAnalysis(analyses: readonly NativeAnalysis[] = [], choice?: string) {
  const preferred = analyses.find(a => REVIEWS[a.filter_id]?.preferred === a.analysis_id)
  const analysis = analyses.find(a => a.analysis_id === choice) ?? preferred
    ?? analyses.find(a => a.filter_type === 'PM2.5') ?? analyses[0]
  const review = analysis && REVIEWS[analysis.filter_id]
  if (!review) return { analysis }

  const reason = `Analysis ${review.weak} ${review.reason ?? 'is nearly flat before baseline correction.'}`
  if (analysis === preferred) return { analysis, reviewNote: `Default scan. ${reason}` }
  if (analysis.analysis_id !== review.weak) return { analysis }
  if (review.preferred) return { analysis, reviewNote: reason }
  const hasRepeat = analyses.some(a => a.filter_id === analysis.filter_id && a.analysis_id !== review.weak)
  return { analysis, reviewNote: `Lab review. ${reason}${hasRepeat ? '' : ' The archive has no repeat for this filter.'}` }
}
