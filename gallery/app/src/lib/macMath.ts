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

/*
 * RMSE and mean residual against the 1:1 line (prediction − Fabs/MAC) do not scale like a slope: they move
 * with the MAC through the evaluation set's HIPS BC, but only through its mean and variance. With x = HIPS BC
 * at the export's MAC, c = exportMac / mac, and an OLS fit y = a + b·x on the same filters (so ȳ = a + b·x̄):
 *
 *   mean residual(mac) = ȳ − c·x̄
 *   RMSE(mac)²         = RMSE₁₀² + 2b·Var x·(1 − c) + Var x·(c² − 1) + (ȳ − c·x̄)² − (ȳ − x̄)²
 *
 * The RMSE form anchors on the saved RMSE, so it is exact at the export's MAC and never reads R² (which a
 * near-zero fit rounds to nothing). x̄ and Var x are not exported, but every configuration scored on the
 * same filters shares them, so `solveHipsMoments` recovers them from the saved fits themselves (the
 * explorer's MAC 6 and 17 readouts check it: mac.test.mjs).
 */

/** HIPS BC on an evaluation set, at the export's MAC: what RMSE and mean residual need to follow the MAC */
export interface HipsMoments {
  mean: number
  variance: number
  /** fits the solve used, and the median |RMSE rebuilt − saved| over them at the export's MAC (µg/m³) */
  rows: number
  miss: number
}

/** a saved OLS fit of predictions on HIPS BC at the export's MAC, with its RMSE against the 1:1 line */
export interface SavedFit { slope: number | null | undefined; intercept: number | null | undefined; r2?: number | null; rmse: number | null | undefined }

const ok = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)

/** RMSE rebuilt from slope, intercept and R² (the solve's model; R² sets Var y = b²·Var x / R²) */
const rmseFromR2 = (b: number, a: number, r2: number, mean: number, variance: number) =>
  Math.sqrt(Math.max(variance * (b * b / r2 - 2 * b + 1) + (a + (b - 1) * mean) ** 2, 0))

const median = (v: number[]) => {
  const s = [...v].sort((p, q) => p - q), h = s.length >> 1
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2
}

/** solve a small dense linear system (Gaussian elimination, partial pivoting); null when singular */
function solveLinear(A: number[][], y: number[]): number[] | null {
  const n = y.length, M = A.map((r, i) => [...r, y[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    if (Math.abs(M[p][c]) < 1e-12 * Math.max(1, ...M.map((r) => Math.abs(r[c])))) return null
    ;[M[c], M[p]] = [M[p], M[c]]
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k] }
  }
  return M.map((r, i) => r[n] / r[i])
}

/**
 * x̄ and Var x of HIPS BC from fits that share one evaluation set. Each fit gives
 * RMSE² − a² = K·Var x + 2a(b − 1)·x̄ + (b − 1)²·x̄², K = b²/R² − 2b + 1: linear in (Var x, x̄, x̄²), so a least-squares
 * start, then Gauss–Newton on RMSE itself, once more without fits that miss by far more than the rest. Fits with
 * R² under `minR2` are left out (their rounded R² says little). Null when too few fits, a singular system, or the
 * fits do not agree on one set (median miss over `maxMiss`): then RMSE stays at the export's MAC.
 */
export function solveHipsMoments(fits: readonly SavedFit[], { minR2 = 0.05, maxMiss = 0.005, minRows = 4 } = {}): HipsMoments | null {
  const use = fits.filter((f) => ok(f.slope) && ok(f.intercept) && ok(f.rmse) && ok(f.r2) && f.r2 >= minR2 && f.r2 <= 1) as
    { slope: number; intercept: number; r2: number; rmse: number }[]
  if (use.length < minRows) return null
  // linear start
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], y = [0, 0, 0]
  for (const { slope: b, intercept: a, r2, rmse } of use) {
    const row = [b * b / r2 - 2 * b + 1, 2 * a * (b - 1), (b - 1) ** 2], t = rmse * rmse - a * a
    for (let i = 0; i < 3; i++) { y[i] += row[i] * t; for (let j = 0; j < 3; j++) A[i][j] += row[i] * row[j] }
  }
  const lin = solveLinear(A, y)
  if (!lin) return null
  let variance = lin[0], mean = lin[1]
  const fit = (rows: typeof use) => {
    for (let it = 0; it < 30; it++) {
      const J = [[0, 0], [0, 0]], g = [0, 0]
      for (const { slope: b, intercept: a, r2, rmse } of rows) {
        const K = b * b / r2 - 2 * b + 1, e = a + (b - 1) * mean, f = Math.max(rmseFromR2(b, a, r2, mean, variance), 1e-9)
        const d = [(b - 1) * e / f, K / (2 * f)], r = f - rmse
        for (let i = 0; i < 2; i++) { g[i] += d[i] * r; for (let j = 0; j < 2; j++) J[i][j] += d[i] * d[j] }
      }
      const step = solveLinear(J, g)
      if (!step) return
      mean -= step[0]; variance -= step[1]
      if (Math.abs(step[0]) < 1e-10 && Math.abs(step[1]) < 1e-10) return
    }
  }
  const misses = (rows: typeof use) => rows.map((f) => Math.abs(rmseFromR2(f.slope, f.intercept, f.r2, mean, variance) - f.rmse))
  fit(use)
  const m0 = misses(use), cut = Math.max(0.01, 5 * median(m0))
  const kept = use.filter((_, i) => m0[i] <= cut)
  if (kept.length >= minRows && kept.length < use.length) fit(kept)
  if (!(variance > 0) || !Number.isFinite(mean)) return null
  const miss = median(misses(kept))
  return miss <= maxMiss ? { mean, variance, rows: kept.length, miss } : null
}

/** RMSE against Fabs/`mac` from a fit saved at `from` (OLS slope and intercept, RMSE) and its set's HIPS BC moments */
export function rmseAtMac(fit: SavedFit, m: Pick<HipsMoments, 'mean' | 'variance'> | null | undefined, mac: number, from: number = EXPORT_MAC): number | null {
  const { slope: b, intercept: a, rmse } = fit
  if (!ok(b) || !ok(a) || !ok(rmse) || !m) return null
  if (mac === from) return rmse
  const c = from / mac, ym = a + b * m.mean
  return Math.sqrt(Math.max(rmse * rmse + 2 * b * m.variance * (1 - c) + m.variance * (c * c - 1) + (ym - c * m.mean) ** 2 - (ym - m.mean) ** 2, 0))
}

/** mean residual (prediction − Fabs/`mac`) from an OLS fit on HIPS BC at `from` and the set's mean HIPS BC there */
export function meanResidualAtMac(slope: number | null | undefined, intercept: number | null | undefined, meanBc: number | null | undefined, mac: number, from: number = EXPORT_MAC): number | null {
  if (!ok(slope) || !ok(intercept) || !ok(meanBc)) return null
  return intercept + slope * meanBc - (from / mac) * meanBc
}

/**
 * The set's mean HIPS BC from one saved OLS fit and its mean residual at the same MAC: OLS passes through the means,
 * so residual = a + (b − 1)·x̄. Null when the slope is too near 1 for that to pin x̄ down.
 */
export function meanBcFromResidual(slope: number | null | undefined, intercept: number | null | undefined, residual: number | null | undefined): number | null {
  if (!ok(slope) || !ok(intercept) || !ok(residual) || Math.abs(slope - 1) < 0.05) return null
  return (residual - intercept) / (slope - 1)
}
