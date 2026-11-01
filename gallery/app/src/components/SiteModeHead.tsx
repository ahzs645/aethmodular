import type { ReactNode } from 'react'
import type { SiteMode } from '@/lib/siteFocus'

/**
 * The heading of a page with a comparison and a one-site analysis (lib/siteFocus useSiteMode): the
 * title, what is drawn (children), then in the one-site view how to get to the others: back to the
 * comparison when the site is one of several picked, or "pick more sites" when it is the only one.
 * `controls` sit on the right, beside the way back; `extra` goes under the text (a caveat).
 *
 *   <SiteModeHead mode={mode} title={`${mode.one.name} · weather and the filters`}>{n} filters …</SiteModeHead>
 */
export function SiteModeHead({ mode, title, children, extra, controls, panel = 'Samples' }: {
  mode: SiteMode
  title?: ReactNode
  children?: ReactNode
  extra?: ReactNode
  controls?: ReactNode
  /** the panel that holds the site pick ('Samples', or 'Filters' on the Aethalometer pages) */
  panel?: string
}) {
  const n = mode.sites.length
  const note = !mode.one ? null : mode.focused
    ? `One of the ${n} picked sites: switch site or go back to the comparison under Page › analysis.`
    : `Pick more sites in the ${panel} panel to compare them.`
  const text = children || note
  return (
    <div className="panel weather-head">
      <div>
        {title && <h2 style={{ margin: 0 }}>{title}</h2>}
        {text && <p className="subtitle" style={{ margin: title ? '4px 0 0' : 0 }}>{children}{children && note ? ' ' : ''}{note}</p>}
        {extra}
      </div>
      {(controls || mode.focused) && <div className="weather-head-controls">
        {controls}
        {mode.focused && <button type="button" className="btn" onClick={mode.back}
          title="Back to the picked sites side by side (Page › analysis)">← Back to comparing {n} sites</button>}
      </div>}
    </div>
  )
}
