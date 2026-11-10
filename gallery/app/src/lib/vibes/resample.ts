/**
 * A spectrum onto the VIBES background's grid: the mean of its linear interpolant over each channel's
 * width, with the ends extended flat. The same rule as `resample` in
 * research/ftir_hips_chem/workflows/network_baselines_20261019.py, which made the network run's inputs.
 */
export interface UniformGrid { first: number; last: number; n: number }

export interface Resampled {
  values: Float64Array
  /** channels whose centre lies outside the spectrum's own range; they take the end value */
  extended: number
}

export function resampleToGrid(wn: ArrayLike<number>, y: ArrayLike<number>, grid: UniformGrid): Resampled {
  const m = wn.length
  if (m < 2 || y.length !== m) throw new Error('a spectrum needs at least two points, and as many values as wavenumbers')
  const rising = wn[m - 1] > wn[0]
  // one flat point beyond each end, so every channel is covered
  const xe = new Float64Array(m + 2), ve = new Float64Array(m + 2)
  for (let i = 0; i < m; i++) {
    const k = rising ? i : m - 1 - i
    if (!Number.isFinite(wn[k]) || !Number.isFinite(y[k])) throw new Error('the spectrum has a value that is not a number')
    xe[i + 1] = wn[k]; ve[i + 1] = y[k]
    if (i > 0 && !(xe[i + 1] > xe[i])) throw new Error('wavenumbers must run strictly one way')
  }
  const n = grid.n, lowG = Math.min(grid.first, grid.last), highG = Math.max(grid.first, grid.last)
  const h = (highG - lowG) / (n - 1)
  const edge = (j: number) => lowG + (j - 0.5) * h            // n + 1 edges around n rising centres
  xe[0] = Math.min(edge(0), xe[1]) - h; ve[0] = ve[1]
  xe[m + 1] = Math.max(edge(n), xe[m]) + h; ve[m + 1] = ve[m]
  const cum = new Float64Array(m + 2)
  for (let i = 0; i < m + 1; i++) cum[i + 1] = cum[i] + (xe[i + 1] - xe[i]) * (ve[i + 1] + ve[i]) / 2
  // the integral of the piecewise-linear curve up to each edge
  const at = new Float64Array(n + 1)
  let k = 0
  for (let j = 0; j <= n; j++) {
    const e = edge(j)
    while (k < m && xe[k + 1] < e) k++
    const t = e - xe[k], slope = (ve[k + 1] - ve[k]) / (xe[k + 1] - xe[k])
    at[j] = cum[k] + ve[k] * t + slope * t * t / 2
  }
  const values = new Float64Array(n)
  const falling = grid.first > grid.last
  let extended = 0
  for (let j = 0; j < n; j++) {
    const centre = lowG + j * h
    if (centre < xe[1] || centre > xe[m]) extended++
    values[falling ? n - 1 - j : j] = (at[j + 1] - at[j]) / h
  }
  return { values, extended }
}

/** The wavenumber of each channel, in the grid's own order. */
export const gridWavenumbers = (g: UniformGrid): number[] => Array.from({ length: g.n }, (_, i) => g.first + (g.last - g.first) * i / (g.n - 1))
