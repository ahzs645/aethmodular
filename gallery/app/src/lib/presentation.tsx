import { createContext, useContext, type ReactNode } from 'react'
import { Segmented } from '@/components/ChartFrame'

/**
 * Meeting pages communicate a result, so by default they show it as presented: the exported
 * figures at MAC 10 in µg/m³, the saved fits, and none of the exclusions made elsewhere in the
 * gallery. Exploring lets the page follow the gallery (hand exclusions, the MAC and units under
 * Filters → units, ratios). Either way the page's own controls (baseline, axes, fit method,
 * hover and pin) keep working. Outside a meeting page there is no provider: everything explores.
 *
 * Read by the shared pieces, not by each chart: lib/filterEdits useHandEdits ignores the
 * exclusions when presented, and App hands the meeting page the presented rows and units.
 */
export type Presentation = 'presented' | 'exploring'
const Ctx = createContext<{ mode: Presentation; set: (m: Presentation) => void } | null>(null)

export function PresentationProvider({ mode, set, children }: { mode: Presentation; set: (m: Presentation) => void; children: ReactNode }) {
  return <Ctx.Provider value={{ mode, set }}>{children}</Ctx.Provider>
}

export const usePresentation = (): Presentation => useContext(Ctx)?.mode ?? 'exploring'

/** the page's switch; `ignored` counts the gallery's hand exclusions a presented page leaves out */
export function PresentationSwitch({ ignored = 0, viewChanged = false }: { ignored?: number; viewChanged?: boolean }) {
  const c = useContext(Ctx)
  if (!c) return null
  const presented = c.mode === 'presented'
  const held = [ignored ? `${ignored} filter${ignored === 1 ? '' : 's'} excluded elsewhere` : '', viewChanged ? 'the MAC and units set under Filters' : ''].filter(Boolean)
  return <>
    <Segmented label="Show" value={presented ? 'As presented' : 'Exploring'} options={['As presented', 'Exploring'] as const}
      onChange={(v) => c.set(v === 'As presented' ? 'presented' : 'exploring')}
      title="As presented: the figures as exported (MAC 10, µg/m³, saved fits), untouched by exclusions or units chosen elsewhere in the gallery. Exploring: the page follows them. The page's own controls work either way." />
    {presented && held.length > 0 && <p className="chart-note" style={{ margin: 0 }}>Not applied here: {held.join(' and ')}. Switch to Exploring to include them.</p>}
    {!presented && <p className="chart-note" style={{ margin: 0 }}>Following the gallery's exclusions, MAC and units.</p>}
  </>
}
