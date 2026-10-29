/**
 * The pure half of lib/mac.tsx (node-tested in mac.test.mjs): how a statistic exported at one MAC
 * reads at another. See lib/mac.tsx for why slopes, λ and HIPS BC are the only things that move.
 */

/** the MAC the exports use (config.MAC_VALUE) */
export const EXPORT_MAC = 10
/** Deming λ at MAC 10 when an export does not carry its own (calibration.json deming_lambda_mac10) */
export const LAMBDA_10 = 2.96

/** Deming λ for an Fabs/MAC x-axis: λ ∝ MAC² (2.96 at MAC 10); 1 for an axis that is not Fabs/MAC. */
export const demingLambda = (mac: number | null, lambda10: number = LAMBDA_10) => (mac ? lambda10 * (mac / 10) ** 2 : 1)

/** A slope fitted against Fabs / `from` re-expressed against Fabs / `mac` (the intercept does not change). */
export const slopeAt = (slope: number | null | undefined, mac: number, from: number = EXPORT_MAC): number | null =>
  slope == null || !Number.isFinite(slope) ? null : slope * (mac / from)

/** A quantity in HIPS BC units (µg/m³ = Fabs / `from`) re-expressed at `mac`. */
export const bcAt = (bc: number | null | undefined, mac: number, from: number = EXPORT_MAC): number | null =>
  bc == null || !Number.isFinite(bc) ? null : bc * (from / mac)

/** A ratio with HIPS BC in the denominator (EC ÷ HIPS BC) re-expressed at `mac`: it scales like a slope. */
export const perBcAt = slopeAt

export interface MacView {
  /** the reader's MAC (m²/g) */
  mac: number
  /** the MAC the exports were written with */
  exportMac: number
  /** the reader moved it off the export's */
  changed: boolean
  /** Deming λ at MAC 10 (calibration.json) and at the reader's MAC */
  lambda10: number
  lambda: number
  slope: (s: number | null | undefined, from?: number) => number | null
  bc: (v: number | null | undefined, from?: number) => number | null
  /** HIPS BC from Fabs at the reader's MAC */
  fromFabs: (fabs: number | null | undefined) => number | null
  /** "MAC 10", for titles and axis labels */
  label: string
}

export function macView(mac: number, exportMac: number = EXPORT_MAC, lambda10: number = LAMBDA_10): MacView {
  return {
    mac, exportMac, changed: mac !== exportMac, lambda10, lambda: demingLambda(mac, lambda10),
    slope: (s, from = exportMac) => slopeAt(s, mac, from),
    bc: (v, from = exportMac) => bcAt(v, mac, from),
    fromFabs: (fabs) => (fabs == null || !Number.isFinite(fabs) ? null : fabs / mac),
    label: `MAC ${+mac.toFixed(2)}`,
  }
}
