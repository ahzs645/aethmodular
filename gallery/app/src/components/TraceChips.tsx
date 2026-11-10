/**
 * One chip per spectrum, coloured as its trace, so the picker is also the legend: any combination,
 * and the last one on is locked so at least one spectrum stays shown. The full name is the tooltip.
 */
export function TraceChips<K extends string>({ options, shown, labels, names, colors, onToggle, suffix }: {
  /** the traces on offer, in order */
  options: readonly K[]
  shown: readonly K[]
  /** the short word on the chip */
  labels: Record<K, string>
  /** the full name, for the tooltip */
  names: Record<K, string>
  colors: Record<K, string>
  onToggle: (key: K) => void
  /** text after a chip's label (a trace still loading) */
  suffix?: (key: K, on: boolean) => string
}) {
  return <div className="trace-chips" role="group" aria-label="Spectra shown">
    {options.map(k => {
      const on = shown.includes(k)
      const locked = on && shown.length === 1
      return <button key={k} type="button" className={`chip trace${on ? ' active' : ''}${locked ? ' locked' : ''}`}
        style={{ ['--chip' as string]: colors[k] }} aria-pressed={on} aria-disabled={locked || undefined}
        title={`${names[k]}${locked ? ' (at least one spectrum stays shown)' : ''}`} onClick={() => { if (!locked) onToggle(k) }}>
        <i aria-hidden="true" />{labels[k]}{suffix?.(k, on) ?? ''}
      </button>
    })}
  </div>
}
