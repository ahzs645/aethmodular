import type { AreaPage } from '@/components/AreaHeader'

export interface InvestigationArea {
  tab: string
  label: string
  /** what the area holds, a sentence or two */
  about: string
  pages: readonly AreaPage[]
}

/** how many of an area's pages its card lists before "+ N more" */
const LISTED = 6

/**
 * The Investigations tab: one card per area of work beyond the chart tabs (Experimental, Research
 * diagnostics, Aethalometer, Data availability). A card opens its area; each page listed on it opens
 * that page. The header menu's Investigations group holds the same pages.
 */
export function InvestigationsPage({ areas, onOpen }: { areas: readonly InvestigationArea[]; onOpen: (tab: string, page?: string) => void }) {
  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>Investigations</h2>
          <p className="subtitle" style={{ margin: '4px 0 0' }}>The analyses beyond the chart tabs: open an area, or go straight to one of its pages</p>
        </div>
      </div>
      <div className="area-cards">
        {areas.map((a) => {
          const rest = a.pages.length - LISTED
          return (
            <section key={a.tab} className="panel area-card">
              <button type="button" className="area-card-head" onClick={() => onOpen(a.tab)} title={`Open ${a.label}`}>
                <strong>{a.label}</strong>
                <span className="area-card-n">{a.pages.length} {a.pages.length === 1 ? 'page' : 'pages'}</span>
              </button>
              <p className="area-card-about">{a.about}</p>
              <ul className="area-card-pages">
                {a.pages.slice(0, LISTED).map((p) => (
                  <li key={p.key}><button type="button" className="area-card-link" title={p.title} onClick={() => onOpen(a.tab, p.key)}>{p.label}</button></li>
                ))}
              </ul>
              <button type="button" className="btn area-card-open" onClick={() => onOpen(a.tab)}>
                {rest > 0 ? `Open · ${rest} more ${rest === 1 ? 'page' : 'pages'}` : 'Open'} ›
              </button>
            </section>
          )
        })}
      </div>
    </>
  )
}
