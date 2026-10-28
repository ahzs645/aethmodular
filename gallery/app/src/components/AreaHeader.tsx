import { Combobox } from '@/components/Combobox'
import { useHeaderNav } from '@/lib/headerNav'
import { usePrefs } from '@/lib/prefs'

/** the tab that lists the investigation areas as cards (App's INVESTIGATIONS) */
export const HUB_TAB = 'investigations'

export interface AreaPage { key: string; label: string; title?: string }

/**
 * The heading of every investigation area (Experimental, Research diagnostics, Aethalometer, Data
 * availability): a way back to the Investigations cards, "Area · Page", the page's one-line
 * description, and one Page dropdown, whatever the number of pages, so the areas look alike.
 * The header menu's Investigations group lists the same pages, so this shows only with Explain on.
 */
export function AreaHeader<K extends string>({ area, pages, value, onPage }: {
  area: string
  pages: readonly (AreaPage & { key: K })[]
  value: K
  onPage: (key: K) => void
}) {
  const nav = useHeaderNav()
  const { explain } = usePrefs()
  const current = pages.find((p) => p.key === value) ?? pages[0]
  if (!explain) return null
  return (
    <div className="panel weather-head area-head">
      <div>
        {nav?.tabs.some((t) => t.key === HUB_TAB) && <button type="button" className="area-back" onClick={() => nav.goTo({ tab: HUB_TAB })}>‹ Investigations</button>}
        <h2 style={{ margin: 0 }}>{area} · {current.label}</h2>
        {current.title && <p className="subtitle" style={{ margin: '4px 0 0' }}>{current.title}</p>}
      </div>
      {pages.length > 1 && <div className="weather-head-controls">
        <Combobox label="Page" value={current.key} onChange={onPage} minWidth={220}
          options={pages.map((p) => ({ key: p.key, label: p.label, title: p.title }))} />
      </div>}
    </div>
  )
}
