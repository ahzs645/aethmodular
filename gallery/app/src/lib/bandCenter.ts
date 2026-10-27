/**
 * Where a spectrum's ~1600 cm⁻¹ band peaks: ftir_12's `band_center`
 * (research/ftir_ec_phase3/ftir_12_band_1600_identity.ipynb), for one spectrum.
 *
 * A straight continuum is drawn from the mean of the left anchor window
 * (1750–1800 cm⁻¹, at that window's mean wavenumber) to the mean of the right
 * one (1500–1530); the peak is the wavenumber of the largest absorbance above
 * that line inside 1550–1680. A maximum on the first two or last two channels
 * of the window is a neighbouring band leaking in (the carbonyl shoulder at the
 * high edge), not a resolved peak, and is flagged `edge`.
 */
export interface BandCenter { center: number; height: number; edge: boolean }

export const BAND_1600 = { peak: [1550, 1680], left: [1750, 1800], right: [1500, 1530] } as const

const inside = (w: number, [lo, hi]: readonly [number, number]) => w >= lo && w <= hi
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

export function bandCenter(wn: ArrayLike<number>, y: ArrayLike<number>, win: { peak: readonly [number, number]; left: readonly [number, number]; right: readonly [number, number] } = BAND_1600): BandCenter | null {
  const idx = (range: readonly [number, number]) => Array.from(wn, (w, i) => (inside(w, range) && Number.isFinite(y[i]) ? i : -1)).filter((i) => i >= 0)
  const peak = idx(win.peak), left = idx(win.left), right = idx(win.right)
  if (!peak.length || !left.length || !right.length) return null
  const lx = mean(left.map((i) => wn[i])), rx = mean(right.map((i) => wn[i]))
  const ly = mean(left.map((i) => y[i])), ry = mean(right.map((i) => y[i]))
  const slope = (ry - ly) / (rx - lx)
  // the peak window in the input's order (ftir_12 keeps the grid's order: high to low wavenumber)
  let best = 0
  const rel = peak.map((i) => y[i] - (ly + slope * (wn[i] - lx)))
  rel.forEach((v, j) => { if (v > rel[best]) best = j })
  return { center: wn[peak[best]], height: rel[best], edge: best <= 1 || best >= peak.length - 2 }
}
