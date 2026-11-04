// Batched VIBES (PB loss, fixed tau) in the browser: damped Newton ELBO + interior-point MAP.
// Same algorithms as scripts/vibes_batch.py (fit_elbo_newton, solve_map_batch). JavaScript numbers are
// float64, so objective, gradient, linear solves and the certificate are float64; only the Khatri-Rao
// product runs on the GPU, in float32, because WGSL has no float64.
const SQRT_2PI = Math.sqrt(2 * Math.PI);

export function ndtr(x) { // West (2005), Hart 5666: double-precision normal CDF
  const a = Math.abs(x); let c;
  if (a > 37) c = 0;
  else {
    const e = Math.exp(-a * a / 2);
    if (a < 7.07106781186547) {
      let b = 3.52624965998911e-02 * a + 0.700383064443688;
      b = b * a + 6.37396220353165; b = b * a + 33.912866078383; b = b * a + 112.079291497871;
      b = b * a + 221.213596169931; b = b * a + 220.206867912376; c = e * b;
      b = 8.83883476483184e-02 * a + 1.75566716318264;
      b = b * a + 16.064177579207; b = b * a + 86.7807322029461; b = b * a + 296.564248779674;
      b = b * a + 637.333633378831; b = b * a + 793.826512519948; b = b * a + 440.413735824752;
      c = c / b;
    } else {
      let b = a + 0.65; b = a + 4 / b; b = a + 3 / b; b = a + 2 / b; b = a + 1 / b;
      c = e / b / 2.506628274631;
    }
  }
  return x > 0 ? 1 - c : c;
}

const WGSL = `
struct Params { rows: u32, pv: u32, q: u32, first_q: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x + P.first_q;
  if (c >= P.q) { return; }
  let r = id.y;
  let wo = r * P.pv;
  let ko = c * P.pv;
  var s = 0.0;
  for (var j = 0u; j < P.pv; j++) { s += dot(w[wo + j], K[ko + j]); }
  out[r * P.q + c] = s;
}`;

export function khatriRao(W, p, c) { // column-major (each q column contiguous) float32
  const tri = c * (c + 1) / 2, ww = new Float32Array(tri * p), w22 = new Float32Array(tri * p), wx = new Float32Array(c * c * p);
  let q = 0;
  for (let i = 0; i < c; i++) for (let k = i; k < c; k++) {
    for (let j = 0; j < p; j++) {
      const a = W[j * c + i], b = W[j * c + k];
      ww[q * p + j] = a * b; w22[q * p + j] = a * a * b * b;
    }
    q++;
  }
  for (let i = 0; i < c; i++) for (let k = 0; k < c; k++) for (let j = 0; j < p; j++) {
    const a = W[j * c + i], b = W[j * c + k];
    wx[(i * c + k) * p + j] = a * b * b;
  }
  return { WW: { m: ww, q: tri }, WW2: { m: wx, q: c * c }, W2W2: { m: w22, q: tri } };
}

export class JsGemm { // no GPU: the same product in plain JavaScript
  constructor(K, p) { this.K = K; this.p = p; this.seconds = 0; this.name = "javascript"; }
  async run(name, w, rows) {
    const t = performance.now(), { m, q } = this.K[name], p = this.p, out = new Float64Array(rows * q);
    for (let r = 0; r < rows; r++) { const wo = r * p;
      for (let c = 0; c < q; c++) { const ko = c * p; let s = 0;
        for (let j = 0; j < p; j++) s += w[wo + j] * m[ko + j];
        out[r * q + c] = s; } }
    this.seconds += (performance.now() - t) / 1e3;
    return out;
  }
}

export class WebGpuGemm {
  static async create(K, p, rowsPerDispatch = 64, colsPerDispatch = 2048) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    const g = new WebGpuGemm();
    Object.assign(g, { device, p, rowsPerDispatch, colsPerDispatch, seconds: 0, name: "webgpu", pv: Math.ceil(p / 4), K: {}, limits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    let info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    g.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
    g.lost = null; device.lost.then(i => { g.lost = i.message; });
    g.pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: "main" } });
    for (const [n, v] of Object.entries(K)) {
      const stride = g.pv * 4, packed = new Float32Array(v.q * stride);
      for (let c = 0; c < v.q; c++) packed.set(v.m.subarray(c * p, (c + 1) * p), c * stride);
      const buf = device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, packed);
      g.K[n] = { buf, q: v.q };
    }
    g.ubuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    g.wcap = g.ocap = 0;
    return g;
  }
  ensure(wbytes, obytes) {
    const d = this.device;
    if (wbytes > this.wcap) { this.wbuf = d.createBuffer({ size: wbytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); this.wcap = wbytes; }
    if (obytes > this.ocap) {
      this.obuf = d.createBuffer({ size: obytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.rbuf = d.createBuffer({ size: obytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); this.ocap = obytes;
    }
  }
  async chunk(name, w, rows) {
    const d = this.device, { buf, q } = this.K[name], stride = this.pv * 4, p = this.p;
    this.ensure(rows * stride * 4, rows * q * 4);
    const packed = new Float32Array(rows * stride);
    for (let r = 0; r < rows; r++) packed.set(w.subarray(r * p, (r + 1) * p), r * stride);
    d.queue.writeBuffer(this.wbuf, 0, packed);
    const bg = d.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.ubuf } }, { binding: 1, resource: { buffer: this.wbuf } },
      { binding: 2, resource: { buffer: buf } }, { binding: 3, resource: { buffer: this.obuf } }] });
    for (let first = 0; first < q; first += this.colsPerDispatch) { // bounded work per submit: the GPU watchdog
      d.queue.writeBuffer(this.ubuf, 0, new Uint32Array([rows, this.pv, q, first]));
      const e = d.createCommandEncoder(), cp = e.beginComputePass();
      cp.setPipeline(this.pipeline); cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(Math.ceil(Math.min(this.colsPerDispatch, q - first) / 32), rows, 1); cp.end();
      d.queue.submit([e.finish()]);
    }
    const e = d.createCommandEncoder(); e.copyBufferToBuffer(this.obuf, 0, this.rbuf, 0, rows * q * 4); d.queue.submit([e.finish()]);
    await this.rbuf.mapAsync(GPUMapMode.READ, 0, rows * q * 4);
    const out = new Float32Array(this.rbuf.getMappedRange(0, rows * q * 4)).slice();
    this.rbuf.unmap();
    return out;
  }
  async run(name, w, rows) {
    const t = performance.now(), q = this.K[name].q, p = this.p, w32 = Float32Array.from(w), out = new Float32Array(rows * q);
    for (let i = 0; i < rows; i += this.rowsPerDispatch) {
      const n = Math.min(this.rowsPerDispatch, rows - i);
      out.set(await this.chunk(name, w32.subarray(i * p, (i + n) * p), n), i * q);
    }
    this.seconds += (performance.now() - t) / 1e3;
    if (this.lost) throw new Error("device lost: " + this.lost);
    return out;
  }
}

function cholesky(A, n) { // in place, lower; returns false if not positive definite
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j]; const jo = j * n;
    for (let k = 0; k < j; k++) d -= A[jo + k] * A[jo + k];
    if (!(d > 0)) return false;
    d = Math.sqrt(d); A[jo + j] = d;
    for (let i = j + 1; i < n; i++) { const io = i * n; let s = A[io + j];
      for (let k = 0; k < j; k++) s -= A[io + k] * A[jo + k];
      A[io + j] = s / d; }
  }
  return true;
}
function cholSolve(L, n, b, x) {
  for (let i = 0; i < n; i++) { let s = b[i]; const o = i * n; for (let k = 0; k < i; k++) s -= L[o + k] * x[k]; x[i] = s / L[o + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

export class Model {
  constructor(W, mu, p, c, tau = 0.1) {
    Object.assign(this, { W, mu, p, c, tau }); this.W2 = new Float64Array(W.length);
    for (let i = 0; i < W.length; i++) this.W2[i] = W[i] * W[i];
    this.iu = []; for (let i = 0; i < c; i++) for (let k = i; k < c; k++) this.iu.push([i, k]);
    this.th = new Float64Array(p); this.de = new Float64Array(p); this.PhiB = new Float64Array(p); this.phiB = new Float64Array(p);
  }
  // -ELBO at x=[nu,d] for spectrum v (y - mu). With g set: gradient + per-channel weights.
  eval(v, x, want) {
    const { p, c, W, W2, tau, th, de, PhiB, phiB } = this; let S = 0;
    for (let j = 0; j < p; j++) { const o = j * c; let a = 0, b = 0;
      for (let k = 0; k < c; k++) { a += x[k] * W[o + k]; const dk = x[c + k]; b += dk * dk * W2[o + k]; }
      const t = v[j] - a, dl = Math.sqrt(b), z = t / dl, P = ndtr(z), ph = Math.exp(-0.5 * z * z) / SQRT_2PI;
      th[j] = t; de[j] = dl; PhiB[j] = P; phiB[j] = ph; S += t * P + dl * ph + (tau - 1) * t; }
    const s = S / p; let sn = 0, sd = 0, ld = 0;
    for (let k = 0; k < c; k++) { sn += x[k] * x[k]; const dk = x[c + k]; sd += dk * dk; ld += Math.log(dk); }
    const elbo = p * Math.log(tau) + p * Math.log(1 - tau) - p * Math.log(s) - 0.5 * sn - 0.5 * sd + ld + (c / 2 - p);
    return { f: -elbo, s };
  }
  grad(x, s) { // uses buffers from eval; returns grad, G (grad S), bW2 and weight arrays
    const { p, c, W, W2, tau, th, de, PhiB, phiB } = this, g = new Float64Array(2 * c), G = new Float64Array(2 * c), bW2 = new Float64Array(c);
    for (let j = 0; j < p; j++) { const o = j * c, a = (1 - tau) - PhiB[j], b = phiB[j] / de[j];
      for (let k = 0; k < c; k++) { G[k] += a * W[o + k]; bW2[k] += b * W2[o + k]; } }
    for (let k = 0; k < c; k++) { const d = x[c + k]; G[c + k] = bW2[k] * d; g[k] = G[k] / s + x[k]; g[c + k] = G[c + k] / s + d - 1 / d; }
    return { g, G, bW2 };
  }
}

export async function newton(model, V, rows, gemm, opts = {}) {
  const { p, c } = model, n = 2 * c, maxiter = opts.maxiter ?? 300, gtol = 1e-5, decr = 1e-13, kmax = 0.99;
  const X = new Float64Array(rows * n); for (let r = 0; r < rows; r++) for (let k = 0; k < c; k++) X[r * n + c + k] = 1;
  const res = { sigma: new Float64Array(rows), elbo: new Float64Array(rows), iterations: new Int32Array(rows), status: new Array(rows).fill("") };
  let active = [...Array(rows).keys()];
  const nt = model.iu.length;
  for (let it = 0; it <= maxiter && active.length; it++) {
    const A_ = active.length, w1 = new Float32Array(A_ * p), w2 = new Float32Array(A_ * p), w3 = new Float32Array(A_ * p), st = [];
    for (let a = 0; a < A_; a++) { const r = active[a], v = V.subarray(r * p, (r + 1) * p), x = X.subarray(r * n, (r + 1) * n), e = model.eval(v, x), gr = model.grad(x, e.s);
      for (let j = 0; j < p; j++) { const ph = model.phiB[j], dl = model.de[j], b = ph / dl, z = model.th[j] / dl; w1[a * p + j] = b; w2[a * p + j] = z * b / dl; w3[a * p + j] = b * (z * z - 1) / (dl * dl); }
      st.push({ r, x, f: e.f, s: e.s, ...gr }); }
    const h1 = await gemm.run("WW", w1, A_), h2 = await gemm.run("WW2", w2, A_), h3 = await gemm.run("W2W2", w3, A_);
    const next = [];
    for (let a = 0; a < A_; a++) { const q = st[a], A = new Float64Array(n * n), { x, s, g, G, bW2 } = q, sq = Math.sqrt(p) * s;
      for (let t = 0; t < nt; t++) { const [i, k] = model.iu[t], u1 = h1[a * nt + t] / s, u3 = h3[a * nt + t] * x[c + i] * x[c + k] / s;
        A[i * n + k] = A[k * n + i] = u1; A[(c + i) * n + c + k] = A[(c + k) * n + c + i] = u3; }
      for (let i = 0; i < c; i++) for (let k = 0; k < c; k++) { const h = h2[a * c * c + i * c + k] * x[c + k] / s; A[i * n + c + k] = h; A[(c + k) * n + i] = h; }
      for (let i = 0; i < c; i++) { A[i * n + i] += 1; const d = x[c + i]; A[(c + i) * n + c + i] += 1 + bW2[i] / s + 1 / (d * d); }
      if (!cholesky(A, n)) { q.bad = true; res.status[q.r] = "cholesky"; res.sigma[q.r] = s; res.elbo[q.r] = -q.f; continue; }
      const mg = new Float64Array(n), u = new Float64Array(n), qq = new Float64Array(n), w = new Float64Array(n);
      for (let i = 0; i < n; i++) { mg[i] = -g[i]; u[i] = G[i] / sq; }
      cholSolve(A, n, mg, qq); cholSolve(A, n, u, w);
      let kap = 0, uq = 0; for (let i = 0; i < n; i++) { kap += u[i] * w[i]; uq += u[i] * qq[i]; }
      let exact = kap < kmax; const step = new Float64Array(n), fac = exact ? uq / (1 - kap) : 0; let gd = 0;
      for (let i = 0; i < n; i++) { step[i] = qq[i] + fac * w[i]; gd += g[i] * step[i]; }
      if (!(gd < 0)) { exact = false; gd = 0; for (let i = 0; i < n; i++) { step[i] = qq[i]; gd += g[i] * step[i]; } }
      let pg = 0; for (let i = 0; i < n; i++) pg = Math.max(pg, Math.abs(g[i]));
      if ((exact && -gd <= decr * Math.max(1, Math.abs(q.f))) || pg <= gtol || it === maxiter) { res.sigma[q.r] = s; res.elbo[q.r] = -q.f; res.status[q.r] = pg <= gtol || it < maxiter ? "ok" : "maxiter"; continue; }
      let t = 1; for (let k = 0; k < c; k++) { const dk = step[c + k]; if (dk < 0) t = Math.min(t, 0.95 * x[c + k] / -dk); }
      const cand = new Float64Array(n), v = V.subarray(q.r * p, (q.r + 1) * p); let accepted = false;
      for (let b = 0; b < 50; b++) { for (let i = 0; i < n; i++) cand[i] = x[i] + t * step[i];
        const fv = model.eval(v, cand).f; if (Number.isFinite(fv) && fv <= q.f + 1e-4 * t * gd) { accepted = true; break; } t *= 0.5; }
      if (!accepted) { res.sigma[q.r] = s; res.elbo[q.r] = -q.f; res.status[q.r] = exact && -gd <= 1e-9 * Math.max(1, Math.abs(q.f)) ? "ok" : "linesearch"; continue; }
      x.set(cand); res.iterations[q.r]++; next.push(q.r);
    }
    active = next; if (opts.progress) await opts.progress(`newton iteration ${it}: ${active.length} active`);
  }
  return res;
}

export async function mapSolve(model, V, rows, sigma, gemm, opts = {}) {
  // Mehrotra primal-dual interior point, one spectrum's state per row; float32 M preconditions float64 CG.
  const { p, c, W } = model, tau = model.tau, lo = tau - 1, hi = tau, maxiter = opts.maxiter ?? 100, nt = model.iu.length;
  const S = []; for (let r = 0; r < rows; r++) { const v = V.subarray(r * p, (r + 1) * p); let sc = 0; for (let j = 0; j < p; j++) sc += Math.abs(v[j]); sc = Math.max(sc / p, 1e-12);
    const z1 = new Float64Array(p), z2 = new Float64Array(p), a = new Float64Array(p).fill(tau - 0.5);
    for (let j = 0; j < p; j++) { z1[j] = Math.max(-v[j], 0) + sc; z2[j] = Math.max(v[j], 0) + sc; }
    S.push({ r, v, sg: sigma[r], x: new Float64Array(c), a, z1, z2, done: false, its: maxiter, wx: new Float64Array(p) }); }
  const WT = (x, out) => { for (let j = 0; j < p; j++) { const o = j * c; let s = 0; for (let k = 0; k < c; k++) s += W[o + k] * x[k]; out[j] = s; } return out; };
  const Wv = (y, out) => { out.fill(0); for (let j = 0; j < p; j++) { const o = j * c, yj = y[j]; for (let k = 0; k < c; k++) out[k] += yj * W[o + k]; } return out; };
  const cert = (s, x, a) => { const r = WT(x, new Float64Array(p)), ac = new Float64Array(p), wa = new Float64Array(c); let P = 0, xx = 0, D = 0, ww = 0;
    for (let j = 0; j < p; j++) { const t = s.v[j] - r[j]; P += Math.max(tau * t, (tau - 1) * t); ac[j] = Math.min(hi, Math.max(lo, a[j])); D += ac[j] * s.v[j]; }
    Wv(ac, wa); for (let k = 0; k < c; k++) { xx += x[k] * x[k]; ww += wa[k] * wa[k]; }
    P += 0.5 * s.sg * xx; D -= ww / (2 * s.sg); return { P, D, gap: P - D, tol: 1e-9 + 1e-8 * Math.max(1, Math.abs(P), Math.abs(D)) }; };
  let cgIters = 0;
  for (let it = 1; it <= maxiter; it++) {
    const act = S.filter(s => !s.done); if (!act.length) break;
    const w = new Float32Array(act.length * p);
    for (let i = 0; i < act.length; i++) { const s = act[i]; s.s1 = new Float64Array(p); s.s2 = new Float64Array(p); s.Dinv = new Float64Array(p);
      for (let j = 0; j < p; j++) { s.s1[j] = Math.max(s.a[j] - lo, 1e-15); s.s2[j] = Math.max(hi - s.a[j], 1e-15); s.Dinv[j] = 1 / (s.z1[j] / s.s1[j] + s.z2[j] / s.s2[j]); w[i * p + j] = s.Dinv[j]; } }
    const up = await gemm.run("WW", w, act.length);
    for (let i = 0; i < act.length; i++) { const s = act[i], M = new Float64Array(c * c), Dinv = s.Dinv, wxt = new Float64Array(p), tmp = new Float64Array(c), si = new Float64Array(c);
      for (let t = 0; t < nt; t++) { const [a_, k] = model.iu[t]; M[a_ * c + k] = M[k * c + a_] = up[i * nt + t]; }
      for (let k = 0; k < c; k++) { M[k * c + k] += s.sg; si[k] = 1 / Math.sqrt(M[k * c + k]); }
      for (let a_ = 0; a_ < c; a_++) for (let k = 0; k < c; k++) M[a_ * c + k] *= si[a_] * si[k];
      let jitter = 1e-5, L; for (;;) { L = M.slice(); for (let k = 0; k < c; k++) L[k * c + k] += jitter; if (cholesky(L, c)) break; jitter *= 10; if (jitter > 1) { L = null; break; } }
      if (!L) { s.its = it; s.failed = true; s.done = true; continue; }
      const prec = (r, o) => { const b = new Float64Array(c); for (let k = 0; k < c; k++) b[k] = si[k] * r[k]; cholSolve(L, c, b, o); for (let k = 0; k < c; k++) o[k] *= si[k]; return o; };
      const op = (q, o) => { WT(q, wxt); for (let j = 0; j < p; j++) wxt[j] *= Dinv[j]; Wv(wxt, o); for (let k = 0; k < c; k++) o[k] += s.sg * q[k]; return o; };
      const solve = (rhs) => { const xx = prec(rhs, new Float64Array(c)), Ax = op(xx, new Float64Array(c)), r_ = new Float64Array(c), z_ = new Float64Array(c), Ad = new Float64Array(c); let nr = 0;
        for (let k = 0; k < c; k++) { r_[k] = rhs[k] - Ax[k]; nr += rhs[k] * rhs[k]; } nr = Math.sqrt(nr); prec(r_, z_); const d_ = z_.slice(); let rz = 0; for (let k = 0; k < c; k++) rz += r_[k] * z_[k];
        for (let q = 0; q < 25; q++) { let rn = 0; for (let k = 0; k < c; k++) rn += r_[k] * r_[k]; if (Math.sqrt(rn) <= 1e-12 * nr) break; cgIters++;
          op(d_, Ad); let dA = 0; for (let k = 0; k < c; k++) dA += d_[k] * Ad[k]; const al = rz / (dA || 1); for (let k = 0; k < c; k++) { xx[k] += al * d_[k]; r_[k] -= al * Ad[k]; }
          prec(r_, z_); let rz2 = 0; for (let k = 0; k < c; k++) rz2 += r_[k] * z_[k]; const be = rz2 / (rz || 1); for (let k = 0; k < c; k++) d_[k] = z_[k] + be * d_[k]; rz = rz2; }
        return xx; };
      const xW = WT(s.x, new Float64Array(p)), aW = Wv(s.a, new Float64Array(c)), r_a = new Float64Array(p), r_x = new Float64Array(c); let mu_c = 0;
      for (let j = 0; j < p; j++) { r_a[j] = xW[j] - s.v[j] - s.z1[j] + s.z2[j]; mu_c += s.s1[j] * s.z1[j] + s.s2[j] * s.z2[j]; } mu_c /= 2 * p;
      for (let k = 0; k < c; k++) r_x[k] = s.sg * s.x[k] - aW[k];
      const direction = (t1, t2) => { const h = new Float64Array(p), dh = new Float64Array(p); for (let j = 0; j < p; j++) { h[j] = -r_a[j] + t1[j] / s.s1[j] - t2[j] / s.s2[j]; dh[j] = Dinv[j] * h[j]; }
        const rhs = Wv(dh, new Float64Array(c)); for (let k = 0; k < c; k++) rhs[k] -= r_x[k];
        const dx = solve(rhs), xd = WT(dx, new Float64Array(p)), da = new Float64Array(p), dz1 = new Float64Array(p), dz2 = new Float64Array(p);
        for (let j = 0; j < p; j++) { da[j] = Dinv[j] * (h[j] - xd[j]); dz1[j] = (t1[j] - s.z1[j] * da[j]) / s.s1[j]; dz2[j] = (t2[j] + s.z2[j] * da[j]) / s.s2[j]; } return { dx, da, dz1, dz2 }; };
      const maxStep = (d) => { let al = 1; for (let j = 0; j < p; j++) { if (d.da[j] < 0) al = Math.min(al, s.s1[j] / -d.da[j]); if (d.da[j] > 0) al = Math.min(al, s.s2[j] / d.da[j]); if (d.dz1[j] < 0) al = Math.min(al, s.z1[j] / -d.dz1[j]); if (d.dz2[j] < 0) al = Math.min(al, s.z2[j] / -d.dz2[j]); } return al; };
      const t1 = new Float64Array(p), t2 = new Float64Array(p); for (let j = 0; j < p; j++) { t1[j] = -s.s1[j] * s.z1[j]; t2[j] = -s.s2[j] * s.z2[j]; }
      let d = direction(t1, t2), al = maxStep(d), mu_aff = 0;
      for (let j = 0; j < p; j++) mu_aff += (s.s1[j] + al * d.da[j]) * (s.z1[j] + al * d.dz1[j]) + (s.s2[j] - al * d.da[j]) * (s.z2[j] + al * d.dz2[j]); mu_aff /= 2 * p;
      const sm = Math.pow(mu_aff / mu_c, 3) * mu_c; for (let j = 0; j < p; j++) { t1[j] = -s.s1[j] * s.z1[j] + sm - d.da[j] * d.dz1[j]; t2[j] = -s.s2[j] * s.z2[j] + sm + d.da[j] * d.dz2[j]; }
      d = direction(t1, t2); al = Math.min(1, 0.99 * maxStep(d));
      if (!Number.isFinite(al)) { s.its = it; s.done = true; s.failed = true; continue; }
      for (let k = 0; k < c; k++) s.x[k] += al * d.dx[k];
      for (let j = 0; j < p; j++) { s.a[j] += al * d.da[j]; s.z1[j] += al * d.dz1[j]; s.z2[j] += al * d.dz2[j]; }
      const q = cert(s, s.x, s.a); s.cert = q;
      if (Number.isFinite(q.gap) && q.gap <= q.tol && q.gap >= -q.tol) { s.done = true; s.its = it; s.ok = true; }
    }
    if (opts.progress) await opts.progress(`map iteration ${it}: ${S.filter(s => !s.done).length} active`);
  }
  const base = new Float64Array(rows * p); S.forEach((s, r) => { const b = WT(s.x, new Float64Array(p)); for (let j = 0; j < p; j++) base[r * p + j] = s.ok ? model.mu[j] + b[j] : NaN; });
  return { baseline: base, converged: S.map(s => !!s.ok), iterations: S.map(s => s.its), cgIters };
}
