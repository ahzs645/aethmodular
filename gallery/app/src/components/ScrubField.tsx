import { useLayoutEffect, useRef, type CSSProperties, type ReactNode } from 'react'
import { ScrubNumberField, type ScrubNumberFieldProps } from '@pgmaps/geo-toolkit/ui'
import '@pgmaps/geo-toolkit/styles.css'

/** A ratio unit such as "m²/g" or "m/s" drawn as a small stacked fraction, so it takes one narrow column. */
function stackedUnit(unit: ReactNode): ReactNode {
  if (typeof unit !== 'string') return unit
  const cut = unit.indexOf('/')
  if (cut <= 0 || cut === unit.length - 1) return unit
  return <span className="scrub-unit-frac"><span>{unit.slice(0, cut)}</span><span>{unit.slice(cut + 1)}</span></span>
}

/**
 * A number to drag or type (geo-toolkit's ScrubNumberField in the gallery's ink, styles.css .scrub-field):
 * drag sideways to scrub across min–max, click to type a value, arrow keys to step (Shift ×10). Replaces a
 * range slider and the number box that used to sit beside it. A ratio unit ("m²/g") is stacked numerator
 * over denominator; the screen-reader value text still reads the unit in full.
 */
export function ScrubField({ width = 96, style, className, unit, ...props }: ScrubNumberFieldProps & { width?: number }) {
  const host = useRef<HTMLSpanElement>(null)
  const shown = stackedUnit(unit)
  // geo-toolkit only puts the unit into aria-valuetext when it is a string; restore it for stacked units.
  useLayoutEffect(() => {
    if (shown === unit || typeof unit !== 'string') return
    const field = host.current?.querySelector('.gt-scrub-field')
    const text = field?.getAttribute('aria-valuetext')
    if (field && text && !text.endsWith(unit)) field.setAttribute('aria-valuetext', `${text} ${unit}`)
  })
  return <span ref={host} style={{ display: 'contents' }}>
    <ScrubNumberField {...props} unit={shown} className={`scrub-field ${className ?? ''}`}
      style={{ ['--gt-scrub-w' as string]: `${width}px`, ...style } as CSSProperties} />
  </span>
}
