/**
 * The VIBES baseline solver compiled to WebAssembly (research/ftir_hips_chem/wasm_demo/src/lib.rs), one
 * spectrum per call: the noise level by Newton on the ELBO, then the baseline by a certified interior
 * point on the pinball MAP. Everything is float64. No DOM here, so it runs in a worker and in Node.
 * vibes_wasm.wasm is a build product; `python research/ftir_hips_chem/workflows/wasm_demo.py build --gallery`
 * rebuilds it and records its source in vibes_wasm.build.json.
 */
export const NEWTON_STATUS = ['converged', 'iteration limit', 'line search failed', 'not finite', 'factorisation failed'] as const

export interface VibesFit {
  /** the noise level the baseline was solved at (the scale of what the baseline leaves behind) */
  sigma: number
  /** the noise level was fitted here; false when the caller supplied it */
  fitted: boolean
  /** NaN when the noise level was supplied */
  elbo: number
  iterations: number
  status: (typeof NEWTON_STATUS)[number]
  mapIterations: number
  /** the duality gap of the baseline problem closed; the baseline is NaN otherwise */
  certified: boolean
  gap: number
  baseline: Float64Array
}

interface Exports {
  memory: WebAssembly.Memory
  vibes_alloc(length: number): number
  vibes_fit(w: number, w2: number, mean: number, y: number, p: number, c: number, tau: number, baseline: number, out: number): void
  vibes_baseline_at(w: number, mean: number, y: number, p: number, c: number, tau: number, sigma: number, baseline: number, out: number): void
}

/** The first `rank` columns of a p × c row-major matrix. */
export function truncate(components: Float64Array, p: number, c: number, rank: number): Float64Array {
  if (!Number.isInteger(rank) || rank < 1 || rank > c) throw new Error(`rank must be 1 to ${c}`)
  if (rank === c) return components
  const out = new Float64Array(p * rank)
  for (let j = 0; j < p; j++) out.set(components.subarray(j * c, j * c + rank), j * rank)
  return out
}

export class VibesSolver {
  x!: Exports
  p = 0
  c = 0
  ptr = { w: 0, w2: 0, mean: 0, y: 0, base: 0, out: 0 }

  /** `components` is p × c row-major, `mean` has length p. */
  static async create(wasm: BufferSource, components: Float64Array, mean: Float64Array, p: number, c: number): Promise<VibesSolver> {
    if (components.length !== p * c || mean.length !== p) throw new Error('background shape does not match its grid')
    const { instance } = await WebAssembly.instantiate(wasm, {})
    const s = new VibesSolver()
    s.x = instance.exports as unknown as Exports
    s.p = p
    s.c = c
    s.ptr = { w: s.put(components), w2: s.put(Float64Array.from(components, (v) => v * v)), mean: s.put(mean),
      y: s.x.vibes_alloc(p), base: s.x.vibes_alloc(p), out: s.x.vibes_alloc(8) }
    return s
  }

  // made anew at each use: the module's memory can grow
  view(ptr: number, n: number): Float64Array { return new Float64Array(this.x.memory.buffer, ptr, n) }
  put(a: Float64Array): number { const ptr = this.x.vibes_alloc(a.length); this.view(ptr, a.length).set(a); return ptr }

  /**
   * `y`: one spectrum on the background's grid, every value finite. With `sigma`, the noise level is not
   * fitted: the baseline is solved at that value, which is how a saved run is repeated from its record.
   */
  fit(y: ArrayLike<number>, tau = 0.1, sigma?: number): VibesFit {
    const { x, p, c, ptr } = this
    if (y.length !== p) throw new Error(`spectrum has ${y.length} points; the background has ${p}`)
    if (!(tau > 0 && tau < 1)) throw new Error('tau must lie between 0 and 1')
    if (sigma !== undefined && !(sigma > 0 && Number.isFinite(sigma))) throw new Error('the noise level must be a positive number')
    this.view(ptr.y, p).set(y)
    if (sigma === undefined) x.vibes_fit(ptr.w, ptr.w2, ptr.mean, ptr.y, p, c, tau, ptr.base, ptr.out)
    else x.vibes_baseline_at(ptr.w, ptr.mean, ptr.y, p, c, tau, sigma, ptr.base, ptr.out)
    const o = this.view(ptr.out, 8)
    return { sigma: o[0], fitted: sigma === undefined, elbo: o[1], iterations: o[2], status: NEWTON_STATUS[o[3]] ?? 'not finite', mapIterations: o[4],
      certified: o[5] === 1, gap: o[6], baseline: this.view(ptr.base, p).slice() }
  }
}
