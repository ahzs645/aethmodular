import { useEffect, useRef, useState, type ReactNode } from 'react'
import { downloadPng, downloadSvg } from '@/lib/export'
import { usePrefs } from '@/lib/prefs'

/**
 * The panel every chart sits in. Quiet by default: the title's first clause,
 * the controls, the chart. The explanation (the rest of the title, the
 * subtitle, the provenance line) is behind the "?" button, or shown on every
 * chart at once with the header's "explain" toggle.
 */
export function ChartFrame({
  title,
  subtitle,
  provenance,
  controls,
  children,
  exportable = true,
  id,
}: {
  /** "Short name — longer clause": only the short name shows unless explaining */
  title: string
  subtitle?: string
  /** which notebooks this chart family stands in for */
  provenance?: ReactNode
  controls?: ReactNode
  children: ReactNode
  /** show the SVG / PNG download buttons (off for text panels) */
  exportable?: boolean
  /** anchor id so a URL can jump straight to this chart */
  id?: string
}) {
  const ref = useRef<HTMLElement>(null)
  const { explain } = usePrefs()
  const [info, setInfo] = useState(explain)
  useEffect(() => setInfo(explain), [explain])
  const firstSvg = () => ref.current?.querySelector('svg') ?? null

  const dash = title.indexOf(' — ')
  const short = dash > 0 ? title.slice(0, dash) : title
  const rest = dash > 0 ? title.slice(dash + 3) : ''
  const hasInfo = !!(rest || subtitle || provenance)

  return (
    <section className="panel" ref={ref} id={id}>
      <header className="frame-head">
        <div>
          <h2>
            {short}
            {info && rest && <span className="title-rest"> — {rest}</span>}
          </h2>
          {info && subtitle && <p className="subtitle">{subtitle}</p>}
          {info && provenance && <p className="provenance">{provenance}</p>}
        </div>
        <div className="frame-actions">
          {hasInfo && (
            <button type="button" className={`btn quiet ${info ? 'on' : ''}`} title={info ? 'Hide the explanation' : 'What this chart shows and where it comes from'} aria-pressed={info} onClick={() => setInfo((v) => !v)}>
              ?
            </button>
          )}
          {exportable && (
            <>
              <button type="button" className="btn quiet" title="Download this chart as SVG (editable, for Illustrator / Inkscape)" onClick={() => { const s = firstSvg(); if (s) downloadSvg(s, short) }}>
                SVG
              </button>
              <button type="button" className="btn quiet" title="Download this chart as a 2× PNG (for a slide)" onClick={() => { const s = firstSvg(); if (s) downloadPng(s, short) }}>
                PNG
              </button>
            </>
          )}
        </div>
      </header>
      {controls && <div className="frame-controls">{controls}</div>}
      {children}
    </section>
  )
}

export function Select({
  label,
  value,
  options,
  onChange,
  title,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
  title?: string
}) {
  return (
    <label className="control" title={title}>
      {label}
      <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )
}

/** Segmented control — for 2–4 mutually exclusive options that deserve to be visible at once. */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  title,
}: {
  label?: string
  value: T
  options: readonly T[]
  onChange: (v: T) => void
  title?: string
}) {
  return (
    <span className="control" title={title}>
      {label}
      <span className="seg" role="radiogroup">
        {options.map((o) => (
          <button key={o} type="button" role="radio" aria-checked={o === value} className={o === value ? 'active' : ''} onClick={() => onChange(o)}>
            {o}
          </button>
        ))}
      </span>
    </span>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
  title,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  title?: string
}) {
  return (
    <label className="toggle" title={title}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

/** A data caveat under a chart (values not drawn, rows pinned to an edge). Always shown: it is about the data, not the reader. */
export function Note({ children }: { children: ReactNode }) {
  return <p className="chart-note">{children}</p>
}

/** A hint for the reader; shown only when explanations are on. */
export function Hint({ children }: { children: ReactNode }) {
  const { explain } = usePrefs()
  return explain ? <span className="control hint">{children}</span> : null
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="frame-empty">{children}</p>
}

/**
 * Swaps the two axis fields.
 *
 * Not cosmetic: OLS of y-on-x is not symmetric, so swapping changes the
 * reported slope (the ETAD Fabs/MAC-vs-EC pair reads 1.898 one way and 0.402
 * the other — the same fit, two different questions). Deming is symmetric at
 * lambda=1, so watching which number moves tells you which one you meant.
 */
export function SwapButton({ onClick, title }: { onClick: () => void; title?: string }) {
  return (
    <button type="button" className="btn" onClick={onClick} title={title ?? 'Swap the x and y axes'} aria-label="Swap x and y axes">
      ⇅ swap
    </button>
  )
}
