// Run the WebAssembly solver in Node on a prepared folder and report agreement with the Python results.
//   node check.mjs <folder with meta.json, W.f64, mu.f64, Y.f64, B.f64, vibes_wasm.wasm>
import { readFileSync } from "node:fs";
import { VibesWasm } from "./vibes_wasm.js";
const dir = process.argv[2];
const meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8")), { p, c, tau, rows, ref } = meta;
const f64 = (name) => { const b = readFileSync(`${dir}/${name}`); return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const Y = f64("Y.f64"), B = f64("B.f64");
const solver = await VibesWasm.create(readFileSync(`${dir}/vibes_wasm.wasm`), f64("W.f64"), f64("mu.f64"), p, c, tau);
let sigma = 0, elbo = 0, base = 0, converged = 0, certified = 0, seconds = 0;
for (let i = 0; i < rows; i++) {
  const t = performance.now(), r = solver.fit(Y.subarray(i * p, (i + 1) * p));
  seconds += (performance.now() - t) / 1e3;
  converged += r.status === "converged"; certified += r.certified;
  sigma = Math.max(sigma, Math.abs(r.sigma / ref.sigma[i] - 1));
  elbo = Math.max(elbo, Math.abs(r.elbo - ref.elbo[i]) / Math.max(1, Math.abs(ref.elbo[i])));
  for (let j = 0; j < p; j++) base = Math.max(base, Math.abs(r.baseline[j] - B[i * p + j]));
}
console.log(JSON.stringify({ rows, converged, certified, sigma_rel_max: sigma, elbo_rel_max: elbo, baseline_abs_max: base, s_per_spectrum: seconds / rows }));
