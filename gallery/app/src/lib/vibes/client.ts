/**
 * The page's side of the VIBES worker: load the exported background once, then ask for fits.
 * One worker for the session, started the first time a page needs it.
 */
import { loadBinary, loadDerived, loadJson } from '../data.ts'
import type { VibesFit } from './solver.ts'
import type { UniformGrid } from './resample.ts'

/** public/data/experimental/vibes_background.json (gallery/data/export_vibes_background.py) */
export interface VibesBackgroundMeta {
  schema_version: number; generated: string; source: string; file: string; bytes: number; sha256: string
  grid: UniformGrid; rank: number; tau: number; blanks: Record<string, number>; component_rms: number[]
}
export interface VibesBackground { meta: VibesBackgroundMeta; mean: Float64Array; components: Float64Array }
export interface TimedFit extends VibesFit { /** time inside the solver, in milliseconds */ ms: number }

export const BACKGROUND_META = 'experimental/vibes_background.json'

export const loadVibesBackground = () => loadDerived<VibesBackground>('vibes-background', async () => {
  const meta = await loadJson<VibesBackgroundMeta>(BACKGROUND_META)
  if (meta.schema_version !== 1) throw new Error('unsupported data version')
  const buffer = await loadBinary(meta.file)
  const p = meta.grid.n, c = meta.rank
  if (buffer.byteLength !== 8 * p * (c + 1)) throw new Error('the background file does not match its description')
  return { meta, mean: new Float64Array(buffer, 0, p), components: new Float64Array(buffer, 8 * p, p * c) }
})

let worker: Worker | null = null
let next = 0
const waiting = new Map<number, { resolve: (fit: TimedFit) => void; reject: (e: Error) => void }>()

function start(background: VibesBackground): Worker {
  const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  w.onmessage = ({ data }: MessageEvent<{ id: number; fit?: TimedFit; error?: string }>) => {
    const job = waiting.get(data.id)
    if (!job) return
    waiting.delete(data.id)
    if (data.fit) job.resolve(data.fit)
    else job.reject(new Error(data.error ?? 'the solver failed'))
  }
  w.onerror = (e) => {
    for (const job of waiting.values()) job.reject(new Error(e.message || 'the solver could not start'))
    waiting.clear()
    worker = null
  }
  const { meta, mean, components } = background
  w.postMessage({ type: 'init', p: meta.grid.n, c: meta.rank, mean, components })
  return w
}

/**
 * The baseline of one spectrum on the background's grid, at this rank (the first `rank` components) and tau.
 * The noise level is fitted to convergence unless `sigma` gives it (a saved run's recorded value).
 */
export async function fitVibes(y: Float64Array, rank: number, tau: number, sigma?: number): Promise<TimedFit> {
  const background = await loadVibesBackground()
  worker ??= start(background)
  const id = next++
  return new Promise<TimedFit>((resolve, reject) => {
    waiting.set(id, { resolve, reject })
    worker!.postMessage({ type: 'fit', id, y, rank, tau, sigma })
  })
}

/** public/data/experimental/vibes_saved_fits.json: what the saved network run recorded for each scan */
export interface SavedFit {
  /** which blank background the saved trace used; the page's own is b87 */
  model: string
  /** the saved run's noise level; null where its fit failed */
  sigma: number | null
  /** steps its optimiser took, and whether that counts as an early stop (under 30) */
  iterations: number
  early: boolean
  /** the noise level of the 3 Nov converged run, for the same background */
  converged: number | null
}
interface SavedFitsFile { schema_version: number; run: string; columns: { analysis_id: string[]; model: string[]; sigma: (number | null)[]; iterations: number[]; early_stop: boolean[]; sigma_converged: (number | null)[] } }

export const loadSavedFits = () => loadDerived<Map<string, SavedFit>>('vibes-saved-fits', async () => {
  const file = await loadJson<SavedFitsFile>('experimental/vibes_saved_fits.json')
  if (file.schema_version !== 1) throw new Error('unsupported data version')
  const c = file.columns
  return new Map(c.analysis_id.map((id, i) => [String(id), { model: c.model[i], sigma: c.sigma[i], iterations: c.iterations[i], early: c.early_stop[i], converged: c.sigma_converged[i] }]))
})
