// One solver instance per worker. Workers share nothing, so the page needs no cross-origin isolation headers.
import { VibesWasm } from "./vibes_wasm.js";
let solver;
self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    solver = await VibesWasm.create(data.wasm, data.W, data.mean, data.p, data.c, data.tau);
    self.postMessage({ type: "ready" });
  } else {
    const t = performance.now(), r = solver.fit(data.y);
    self.postMessage({ type: "fit", row: data.row, ms: performance.now() - t, ...r }, [r.baseline.buffer]);
  }
};
