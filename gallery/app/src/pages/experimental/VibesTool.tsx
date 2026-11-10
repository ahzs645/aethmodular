import { useMemo, useState, type ChangeEvent } from 'react'
import { ChartFrame, Note, Segmented } from '@/components/ChartFrame'
import { Combobox } from '@/components/Combobox'
import { ExportState, LoadState } from '@/components/ExportState'
import { SpectrumPlot } from '@/components/FilterSpectrumPlot'
import { TraceChips } from '@/components/TraceChips'
import { useLoaded } from '@/lib/data'
import { downloadCsv } from '@/lib/export'
import { sci } from '@/lib/format'
import { loadNativeSite, loadNativeTraces, loadSpartanIndex, type NativeAnalysis } from '@/lib/spartan'
import { BASELINE_COLOR, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { fitVibes, loadSavedFits, loadVibesBackground, type SavedFit, type TimedFit } from '@/lib/vibes/client'
import { parseSpectrumText } from '@/lib/vibes/parse'
import { gridWavenumbers, resampleToGrid } from '@/lib/vibes/resample'
import { sourceOf } from './shared'

/**
 * VIBES on one spectrum, solved in the browser: the converged solver (Newton on the ELBO, certified MAP)
 * compiled to WebAssembly (lib/vibes), with the network run's 87-blank background. Pick any scan of the
 * SPARTAN network or read a file, change the rank or tau, and compare with the saved 19 Oct trace. The
 * saved run's own noise level per scan (vibes_saved_fits.json) makes the one setting that differs visible,
 * and the "19 Oct run" preset reuses it to repeat the saved trace.
 * (docs/vibes-rank-and-wasm-2026-11-10.md; background: gallery/data/export_vibes_background.py)
 */

const source = sourceOf('VibesTool')
const SOURCES = ['Network scan', 'Your file'] as const
const TAUS = ['0.05', '0.1', '0.2'] as const
const RANKS = [10, 20, 40, 70]
const VIEWS = ['Corrected', 'Raw and baseline'] as const
const NOISES = ['Solved here', 'As saved'] as const
/** the background this page solves with, as the saved runs name it; a scan saved under another one cannot be repeated here */
const BANK = 'b87'
const NETWORK_TAU = '0.1'
type TraceKey = 'live' | 'saved' | 'raw' | 'baseline'
const LABEL: Record<TraceKey, string> = { live: 'Solved here', saved: 'Saved 19 Oct run', raw: 'Raw', baseline: 'Baseline' }
const NAME: Record<TraceKey, string> = {
  live: 'Corrected spectrum, VIBES baseline solved in this browser', saved: 'Corrected spectrum saved by the 19 Oct network run',
  raw: 'Raw absorbance on the background\'s grid', baseline: 'VIBES baseline solved in this browser',
}
const COLORS: Record<TraceKey, string> = {
  get live() { return BASELINE_COLOR['VIBES-full-r70'] }, get saved() { return INK.muted }, get raw() { return INK.text }, get baseline() { return BASELINE_COLOR['VIBES-full-r70'] },
}
const TYPE_NAME: Record<string, string> = { FB: 'field blank' }

/** what goes to the solver: a spectrum on its own grid, and the saved trace to compare with where there is one */
interface Input { key: string; label: string; wn: ArrayLike<number>; y: ArrayLike<number>; note?: string
  /** a network scan's analysis id, which finds what the saved run recorded for it */
  analysisId?: string
  /** the saved run's corrected trace, and why it may not be the same calculation */
  saved?: ArrayLike<number>; savedNote?: string }

export function VibesTool() {
  const background = useLoaded('vibes-background', loadVibesBackground)
  const [from, setFrom] = useSearchParam<(typeof SOURCES)[number]>('vt-from', 'Network scan', SOURCES)
  const [rank, setRank] = useSearchParam<string>('vt-rank', '70')
  const [tau, setTau] = useSearchParam<(typeof TAUS)[number]>('vt-tau', '0.1', TAUS)
  const [view, setView] = useSearchParam<(typeof VIEWS)[number]>('vt-view', 'Corrected', VIEWS)
  const [hidden, setHidden] = useState<TraceKey[]>([])
  const [file, setFile] = useState<{ input: Input | null; error: string }>({ input: null, error: '' })
  const scan = useNetworkScan(from === 'Network scan')
  const input = from === 'Network scan' ? scan.input : file.input
  const bg = background.data
  const ranks = bg ? RANKS.filter((r) => r <= bg.meta.rank).map(String) : []
  const full = bg?.meta.rank ?? 0
  const k = ranks.includes(rank) ? +rank : full
  // what the saved run recorded for this scan; with its noise level the page can repeat its trace
  const [noise, setNoise] = useSearchParam<(typeof NOISES)[number]>('vt-noise', 'Solved here', NOISES)
  const fits = useLoaded('vibes-saved-fits', loadSavedFits)
  const record = input?.analysisId ? fits.data?.get(input.analysisId) : undefined
  const repeatable = !!record && record.sigma != null && record.model === BANK
  const asSaved = noise === 'As saved' && repeatable
  // a link that asks for the saved noise level waits for the records instead of solving twice
  const waiting = noise === 'As saved' && !!input?.analysisId && fits.loading
  const preset = k === full && tau === NETWORK_TAU ? (asSaved ? '19 Oct run' : 'Converged') : 'Custom'
  const presets = ['Converged', ...(repeatable ? ['19 Oct run'] : []), ...(preset === 'Custom' ? ['Custom'] : [])]
  const setPreset = (name: string) => {
    if (name === 'Custom') return
    setRank(String(full)); setTau(NETWORK_TAU); setNoise(name === '19 Oct run' ? 'As saved' : 'Solved here')
  }

  // the spectrum on the background's grid, as the network run prepared its inputs
  const prepared = useMemo(() => {
    if (!bg || !input) return null
    try { return { ...resampleToGrid(input.wn, input.y, bg.meta.grid), error: '' } }
    catch (e) { return { values: null, extended: 0, error: String(e instanceof Error ? e.message : e) } }
  }, [bg, input])
  // a result carries the spectrum it was solved for, so a kept result never pairs with a newer spectrum
  const sigma = asSaved ? record!.sigma! : undefined
  const result = useLoaded(prepared?.values && input && !waiting ? `${input.key}|${k}|${tau}|${sigma ?? 'fit'}` : null, async () => {
    const y = prepared!.values!
    const { saved, savedNote, label } = input!
    return { y, extended: prepared!.extended, rank: k, tau: +tau, saved, savedNote, label, record, fit: await fitVibes(y, k, +tau, sigma) }
  }, { keep: true })

  const grid = bg?.meta.grid
  const wn = useMemo(() => (grid ? gridWavenumbers(grid) : []), [grid])

  if (!bg) return <ExportState title="VIBES on one spectrum" what="the VIBES background" error={background.error} command="uv run python gallery/data/export_vibes_background.py" />
  const r = result.data
  const corrected = r?.fit.certified ? Float64Array.from(r.y, (v, i) => v - r.fit.baseline[i]) : null
  const series = r ? {
    raw: { wn, y: r.y }, baseline: r.fit.certified ? { wn, y: r.fit.baseline } : undefined,
    live: corrected ? { wn, y: corrected } : undefined, saved: r.saved && r.saved.length === wn.length ? { wn, y: r.saved } : undefined,
  } : {}
  const options = (view === 'Corrected' ? ['saved', 'live'] as const : ['raw', 'baseline'] as const).filter((t) => series[t])
  const on = options.filter((t) => !hidden.includes(t))
  const shown = on.length ? on : options.slice(-1)
  const toggle = (t: TraceKey) => setHidden((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]))
  const blanks = Object.entries(bg.meta.blanks).map(([name, n]) => `${n} ${name === 'etad' ? 'Addis field' : name === 'improve' ? 'IMPROVE' : name} blanks`).join(' and ')
  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    try {
      const s = parseSpectrumText(await f.text())
      setFile({ error: '', input: { key: `file:${f.name}:${f.size}:${f.lastModified}`, label: f.name, wn: s.wn, y: s.y,
        note: `${s.wn.length.toLocaleString()} points, ${Math.round(s.wn[0])}–${Math.round(s.wn[s.wn.length - 1])} cm⁻¹${s.skipped ? `; ${s.skipped} line${s.skipped === 1 ? '' : 's'} without two numbers skipped` : ''}` } })
    } catch (err) { setFile({ input: null, error: `${f.name}: ${String(err instanceof Error ? err.message : err)}` }) }
  }
  const save = () => r && corrected && downloadCsv(['wavenumber_cm-1', 'absorbance', 'vibes_baseline', 'corrected'],
    wn.map((w, i) => [w, r.y[i], r.fit.baseline[i], corrected[i]]), `vibes ${r.label} rank ${r.rank} tau ${r.tau}`)

  return (
    <ChartFrame id="xt-tool" source={source('VibesTool')} exportable={false}
      title="VIBES on one spectrum — solved in your browser"
      subtitle={`The converged VIBES solver, compiled to WebAssembly, with the network run's background (${blanks}). It fits the noise level, then the baseline, and accepts the baseline only when its optimality check closes. Nothing leaves this page.`}
      provenance="research/ftir_hips_chem/wasm_demo (solver) · gallery/data/export_vibes_background.py (background) · docs/vibes-rank-and-wasm-2026-11-10.md"
      controls={<>
        <Segmented label="Spectrum" value={from} options={SOURCES} onChange={setFrom} />
        {from === 'Network scan' ? scan.pickers : <label className="btn" title="Two columns: wavenumber (cm⁻¹), then absorbance. Commas, tabs or spaces. The file is read here and is not uploaded.">
          Choose a file…<input type="file" accept=".csv,.txt,.tsv,.dpt,text/plain,text/csv" hidden onChange={onFile} />
        </label>}
        <Segmented label="Preset" value={preset} options={presets} onChange={setPreset}
          title="Converged: rank 70, tau 0.1 and the noise level solved to convergence here (the 3 Nov method). 19 Oct run: the same rank and tau with the noise level the saved run's optimiser stopped at, which repeats the saved trace. Changing a setting below makes it Custom." />
        <Segmented label="Rank" value={String(k)} options={ranks} onChange={setRank} title="How many of the background's components the baseline may use, largest first. The network run used all 70." />
        <Segmented label="tau" value={tau} options={TAUS} onChange={setTau} title="The quantile the baseline aims for under the spectrum. The network run used 0.1." />
        {repeatable && <Segmented label="Noise level" value={asSaved ? 'As saved' : 'Solved here'} options={NOISES} onChange={setNoise}
          title="The noise level sets how stiff the baseline is. Solved here: fitted to convergence in this browser. As saved: the value the 19 Oct run's optimiser stopped at for this scan." />}
      </>}>
      {from === 'Your file' && <Note>{file.error ? <span role="alert">{file.error}</span> : file.input ? <>{file.input.label}: {file.input.note}.</>
        : <>Choose a text file with two columns, wavenumber (cm⁻¹) then absorbance. It is read in this browser and is not uploaded.</>}</Note>}
      {from === 'Network scan' && scan.state}
      {prepared?.error ? <Note><span role="alert">This spectrum cannot be corrected: {prepared.error}.</span></Note>
        : !input ? null
        : !r ? <LoadState what="the baseline" error={result.error} />
        : <>
          <Facts fit={r.fit} record={r.record} standard={r.rank === full && r.tau === +NETWORK_TAU} extended={r.extended} n={wn.length} stale={result.loading} />
          {r.fit.certified ? <>
            <div className="frame-controls">
              <Segmented value={view} options={VIEWS} onChange={setView} />
              <TraceChips options={options} shown={shown} labels={LABEL} names={NAME} colors={COLORS} onToggle={toggle} />
              <button type="button" className="btn quiet" onClick={save} title="Wavenumber, absorbance on the background's grid, baseline and corrected spectrum">Download CSV</button>
            </div>
            <SpectrumPlot spec={{ series }} shown={shown} names={NAME} colors={COLORS} dashed={['baseline']} expanded roundWindow={false}
              plotLabel={`${shown.map((t) => LABEL[t]).join(' and ')}: ${r.label}`} />
            {view === 'Corrected' && series.saved && r.record && <Settings fit={r.fit} record={r.record} rank={r.rank} full={full} tau={r.tau} />}
            {view === 'Corrected' && series.saved && corrected && <Agreement live={corrected} saved={series.saved.y} fit={r.fit} record={r.record} rank={r.rank} full={full} tau={r.tau} note={r.savedNote} />}
          </> : <Note><span role="alert">The solver did not reach a certified baseline for this spectrum ({r.fit.status}), so none is drawn.</span></Note>}
        </>}
      <Note explain>The rank is the number of background components the baseline may use, largest first; components beyond about 40 are tiny and only matter on blanks.
        Narrow bands survive at any rank, and a feature about 1,500 cm⁻¹ wide is removed at every rank (the rank check, 10 Nov).
        A spectrum on another grid is averaged onto the background's {wn.length.toLocaleString()} channels the way the network run did it, and channels outside its range take the end value.</Note>
    </ChartFrame>
  )
}

const steps = (n: number) => `${n.toLocaleString()} step${n === 1 ? '' : 's'}`

/** the solver's own account of a fit, in one line */
function Facts({ fit, record, standard, extended, n, stale }: { fit: TimedFit; record?: SavedFit; standard: boolean; extended: number; n: number; stale: boolean }) {
  // at the network run's rank and tau, the 3 Nov converged run solved the same problem: say when this page lands on its value
  const same = fit.fitted && standard && record?.converged != null && Math.abs(fit.sigma / record.converged - 1) < 1e-4
  return <p className="chart-note" style={{ display: 'block', opacity: stale ? 0.55 : 1 }} aria-live="polite">
    Noise level <strong>{sci(fit.sigma)}</strong>{fit.fitted ? <>, solved here in {steps(fit.iterations)}{same && <> (the 3 Nov converged run has the same value)</>}</>
      : <>, taken from the saved 19 Oct run</>} · baseline in {steps(fit.mapIterations)},{' '}
    {fit.certified ? 'certified' : 'not certified'} · {fit.ms < 999.5 ? `${Math.round(fit.ms)} ms` : `${(fit.ms / 1000).toFixed(1)} s`} in the solver
    {/* a network scan stops one channel short of the background's first; say so only when a file covers less */}
    {extended > 1 && <> · {extended.toLocaleString()} of {n.toLocaleString()} channels lie outside the spectrum and take its end value</>}
    {stale && <> · solving…</>}
  </p>
}

const RIGHT = { textAlign: 'right' } as const

/** what each line was made with, side by side; the settings that differ are in bold */
function Settings({ fit, record, rank, full, tau }: { fit: TimedFit; record: SavedFit; rank: number; full: number; tau: number }) {
  const other = record.model !== BANK
  const differs = { rank: rank !== full, tau: tau !== +NETWORK_TAU, sigma: !other && (record.sigma == null || Math.abs(fit.sigma / record.sigma - 1) > 0.01) }
  const mark = (on: boolean, text: string) => (on ? <strong>{text}</strong> : text)
  return <table className="placement" aria-label="What each line was made with" style={{ maxWidth: 760 }}>
    <thead><tr><th>Line</th><th>Background</th><th style={RIGHT}>Rank</th><th style={RIGHT}>tau</th><th style={RIGHT}>Noise level</th><th>How the noise level was set</th></tr></thead>
    <tbody>
      <tr><td className="f">{LABEL.saved}</td><td className="f">{mark(other, other ? 'another blank bank' : '87 blanks')}</td><td className="v">{mark(differs.rank, String(full))}</td><td className="v">{mark(differs.tau, NETWORK_TAU)}</td>
        {/* a scan saved under its lot's own bank was refitted later; the record here is of the 87-blank fit, not of the line drawn */}
        <td className="v">{other || record.sigma == null ? '—' : mark(differs.sigma, sci(record.sigma))}</td>
        <td className="f">{other ? 'fitted with its lot\'s own blank bank (26 Oct)' : record.sigma == null ? 'the saved fit failed'
          : <>original optimiser, {steps(record.iterations)}{record.early && <strong> (an early stop)</strong>}</>}</td></tr>
      <tr><td className="f">{LABEL.live}</td><td className="f">{mark(other, '87 blanks')}</td><td className="v">{mark(differs.rank, String(rank))}</td><td className="v">{mark(differs.tau, String(tau))}</td>
        <td className="v">{mark(differs.sigma, sci(fit.sigma))}</td>
        <td className="f">{fit.fitted ? <>solved to convergence here, {steps(fit.iterations)}</> : 'taken from the saved run'}</td></tr>
    </tbody>
  </table>
}

/** how far the baseline solved here sits from the saved run's, and which setting accounts for it */
function Agreement({ live, saved, fit, record, rank, full, tau, note }: { live: ArrayLike<number>; saved: ArrayLike<number>; fit: TimedFit; record?: SavedFit; rank: number; full: number; tau: number; note?: string }) {
  let worst = 0, square = 0, count = 0
  for (let i = 0; i < live.length; i++) {
    if (!Number.isFinite(saved[i])) continue
    worst = Math.max(worst, Math.abs(live[i] - saved[i])); square += live[i] * live[i]; count++
  }
  if (!count) return null
  const share = worst / Math.sqrt(square / count)
  const why = note ?? (rank !== full || tau !== +NETWORK_TAU ? 'The saved run used rank 70 and tau 0.1, so a difference here is the setting you changed.'
    : !fit.fitted ? `This repeats the saved run: the same background, rank, tau and noise level. ${worst <= 1e-5 ? 'What remains is within the tolerance of the two baseline solvers.' : 'The traces still differ, which is not expected.'}`
      : share < 0.01 ? 'The two agree: the saved run\'s optimiser reached the same noise level.'
        : `Only the noise level differs. The saved run's optimiser stopped ${record?.early ? 'early' : 'short of the optimum'}${record?.sigma != null ? ` at ${sci(record.sigma)}` : ''}; solved to convergence it is ${sci(fit.sigma)}, and a lower noise level lets the baseline follow this filter's own broad shape. Choose the 19 Oct run preset to repeat the saved trace.`)
  return <Note>Largest difference from the saved 19 Oct trace: <strong>{sci(worst)}</strong> absorbance, {share < 0.001 ? 'under 0.1' : (share * 100).toFixed(share < 0.1 ? 1 : 0)} % of this corrected spectrum's RMS. {why}</Note>
}

/** site, filter and scan pickers over the gallery's native SPARTAN archive, and the chosen scan as solver input */
function useNetworkScan(active: boolean) {
  const [code, setCode] = useSearchParam<string>('vt-site', 'ETAD')
  const [filter, setFilter] = useSearchParam<string>('vt-filter', '')
  const [scanId, setScanId] = useSearchParam<string>('vt-scan', '')
  const index = useLoaded(active ? 'spartan-index' : null, loadSpartanIndex)
  const site = index.data?.sites.find((s) => s.code === code) ?? index.data?.sites[0]
  const native = useLoaded(site ? site.code : null, () => loadNativeSite(site!))
  const spectra = native.data?.spectra
  const filters = useMemo(() => (spectra ? Object.keys(spectra).sort() : []), [spectra])
  // a link's filter if the site has it, else the site's first PM2.5 filter
  const base = filters.includes(filter) ? filter : filters.find((f) => spectra![f].analyses.some((a) => a.filter_type === 'PM2.5')) ?? filters[0]
  const entry = base ? spectra![base] : undefined
  const analysis = entry?.analyses.find((a) => String(a.analysis_id) === scanId) ?? entry?.analyses.find((a) => a.filter_type === 'PM2.5') ?? entry?.analyses[0]
  const grid = index.data?.baselines?.vibes?.grid ?? null
  const traces = useLoaded(entry && analysis ? `${base}:${analysis.analysis_id}:${entry.file}` : null, () => loadNativeTraces(entry!.file, analysis!, grid), { deps: [grid] })
  const scanLabel = (a: NativeAnalysis) => `Analysis ${a.analysis_id} · ${TYPE_NAME[a.filter_type] ?? a.filter_type}${a.qc ? ` · QC ${a.qc}` : ''}`
  const input = useMemo<Input | null>(() => {
    const t = traces.data
    if (!t || !analysis) return null
    const model = grid?.models?.[analysis.vibes_model ?? '']
    return { key: `scan:${analysis.analysis_id}`, analysisId: String(analysis.analysis_id), label: `${analysis.filter_id} analysis ${analysis.analysis_id}`, wn: t.raw.wn, y: t.raw.y, saved: t.vibes?.y,
      savedNote: analysis.vibes_model && analysis.vibes_model !== 'b87' ? `The saved trace for this filter used another background (${model?.label ?? analysis.vibes_model}), so the two are not the same calculation.` : undefined }
  }, [traces.data, analysis, grid])
  const pickers = <>
    <Combobox label="Site" value={site?.code ?? code} onChange={(v) => { setCode(v); setFilter(''); setScanId('') }} minWidth={150}
      options={(index.data?.sites ?? []).map((s) => ({ key: s.code, label: `${s.code} · ${s.name}` }))} />
    <Combobox label="Filter" value={base ?? ''} onChange={(v) => { setFilter(v); setScanId('') }} minWidth={130} search
      options={filters.map((f) => ({ key: f, label: f, hint: [...new Set(spectra![f].analyses.map((a) => a.filter_type))].join(', ') }))} />
    {entry && entry.analyses.length > 1 && analysis && <Combobox label="Scan" value={String(analysis.analysis_id)} onChange={setScanId} minWidth={190}
      options={entry.analyses.map((a) => ({ key: String(a.analysis_id), label: scanLabel(a) }))} />}
  </>
  const error = index.error || native.error || traces.error
  const state = error ? <LoadState what="the network's native spectra" error={error} command="uv run python gallery/data/export_spartan_network.py" />
    : !input ? <LoadState what="the spectrum" /> : null
  return { input: active ? input : null, pickers, state }
}
