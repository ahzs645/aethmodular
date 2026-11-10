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

/** The baseline of one spectrum on the background's grid, at this rank (the first `rank` components) and tau. */
export async function fitVibes(y: Float64Array, rank: number, tau: number): Promise<TimedFit> {
  const background = await loadVibesBackground()
  worker ??= start(background)
  const id = next++
  return new Promise<TimedFit>((resolve, reject) => {
    waiting.set(id, { resolve, reject })
    worker!.postMessage({ type: 'fit', id, y, rank, tau })
  })
}
