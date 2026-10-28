import type { CSSProperties } from 'react'
import { ScrubNumberField, type ScrubNumberFieldProps } from '@pgmaps/geo-toolkit/ui'
import '@pgmaps/geo-toolkit/styles.css'

/**
 * A number to drag or type (geo-toolkit's ScrubNumberField in the gallery's ink, styles.css .scrub-field):
 * drag sideways to scrub across min–max, click to type a value, arrow keys to step (Shift ×10). Replaces a
 * range slider and the number box that used to sit beside it.
 */
export function ScrubField({ width = 96, style, className, ...props }: ScrubNumberFieldProps & { width?: number }) {
  return <ScrubNumberField {...props} className={`scrub-field ${className ?? ''}`}
    style={{ ['--gt-scrub-w' as string]: `${width}px`, ...style } as CSSProperties} />
}
