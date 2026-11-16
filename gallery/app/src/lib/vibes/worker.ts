// The solver off the main thread: one background, a solver per rank (made when first asked for).
import wasmUrl from './vibes_wasm.wasm?url'
import { VibesSolver, truncate } from './solver.ts'

interface Init { type: 'init'; p: number; c: number; mean: Float64Array; components: Float64Array }
interface Fit { type: 'fit'; id: number; y: Float64Array; rank: number; tau: number }

// the worker's own global, typed here so the page's DOM types stay the project's only globals
const scope = self as unknown as { onmessage: ((e: MessageEvent<Init | Fit>) => void) | null; postMessage(message: unknown, transfer?: Transferable[]): void }

let model: Init | null = null
let wasm: Promise<ArrayBuffer> | null = null
const solvers = new Map<number, Promise<VibesSolver>>()

function solver(rank: number): Promise<VibesSolver> {
  if (!model) throw new Error('the background has not been loaded')
  const m = model
  let s = solvers.get(rank)
  if (!s) {
    wasm ??= fetch(wasmUrl).then((r) => { if (!r.ok) throw new Error(`solver module: HTTP ${r.status}`); return r.arrayBuffer() })
    s = wasm.then((bytes) => VibesSolver.create(bytes, truncate(m.components, m.p, m.c, rank), m.mean, m.p, rank))
    solvers.set(rank, s)
    s.catch(() => solvers.delete(rank))
  }
  return s
}

scope.onmessage = async ({ data }) => {
  if (data.type === 'init') { model = data; solvers.clear(); return }
  try {
    const s = await solver(data.rank)
    const started = performance.now()
    const fit = s.fit(data.y, data.tau)
    scope.postMessage({ id: data.id, fit: { ...fit, ms: performance.now() - started } }, [fit.baseline.buffer])
  } catch (e) {
    scope.postMessage({ id: data.id, error: String(e instanceof Error ? e.message : e) })
  }
}
