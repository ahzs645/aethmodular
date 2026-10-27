import { Segmented } from '@/components/ChartFrame'
import type { SectionPageProps } from '../types'
import { AETH_PAGES } from './pages'

/**
 * The Aethalometer tab: one sub-page per family of notebooks (./pages.ts), picked here or from the
 * header menu (URL `aep`). Every sub-page gets the same rows, measurement and site pick as the Weather
 * tab, with the aethalometer fields already on the rows (lib.ts).
 */
export function AethalometerPage(props: SectionPageProps) {
  const current = AETH_PAGES.find((p) => p.key === props.page) ?? AETH_PAGES[0]
  const Page = current.Page
  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>Aethalometer · {current.label}</h2>
          {current.title && <p className="subtitle" style={{ margin: '4px 0 0' }}>{current.title}</p>}
        </div>
        <div className="weather-head-controls">
          <Segmented label="Page" value={current.key} options={AETH_PAGES.map((p) => p.key)} onChange={props.onPage}
            optionLabel={(k) => AETH_PAGES.find((p) => p.key === k)?.label ?? k} />
        </div>
      </div>
      <Page {...props} />
    </>
  )
}
