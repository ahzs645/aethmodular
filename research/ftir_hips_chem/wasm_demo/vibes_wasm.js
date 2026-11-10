// The VIBES baseline solver compiled to WebAssembly (src/lib.rs), wrapped for one spectrum at a time.
// It runs unchanged in a browser, a Web Worker or Node. Everything is float64.
export const NEWTON_STATUS = ["converged", "iteration limit", "line search failed", "not finite", "factorisation failed"];

export class VibesWasm {
  // wasm: the module bytes. W: p x c row-major background components. mean: length p.
  static async create(wasm, W, mean, p, c, tau = 0.1) {
    const { instance } = await WebAssembly.instantiate(wasm, {});
    const s = new VibesWasm();
    Object.assign(s, { x: instance.exports, p, c, tau });
    s.pW = s.put(W);
    s.pW2 = s.put(Float64Array.from(W, (v) => v * v));
    s.pMean = s.put(mean);
    s.pY = s.x.vibes_alloc(p);
    s.pBase = s.x.vibes_alloc(p);
    s.pOut = s.x.vibes_alloc(8);
    return s;
  }
  view(ptr, n) { return new Float64Array(this.x.memory.buffer, ptr, n); } // re-made each call: memory can grow
  put(a) { const ptr = this.x.vibes_alloc(a.length); this.view(ptr, a.length).set(a); return ptr; }
  // y: one spectrum on the model grid. The baseline is NaN unless the fit is certified.
  fit(y) {
    const { x, p, c, tau } = this;
    this.view(this.pY, p).set(y);
    x.vibes_fit(this.pW, this.pW2, this.pMean, this.pY, p, c, tau, this.pBase, this.pOut);
    const o = this.view(this.pOut, 8);
    return { sigma: o[0], elbo: o[1], iterations: o[2], status: NEWTON_STATUS[o[3]], mapIterations: o[4], certified: o[5] === 1,
      gap: o[6], hessians: o[7], baseline: this.view(this.pBase, p).slice() };
  }
}
