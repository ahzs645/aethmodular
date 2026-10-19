import { createContext, useContext, type ReactNode } from 'react'
import type { ExportSeason } from './exportSeasons'
import type { ExportSite } from './exportColors'
const Sites = createContext<readonly ExportSite[]>([])
const Seasons = createContext<readonly ExportSeason[]>([])
export const useExportSeasons = () => useContext(Seasons)
export const ExportSitesProvider = ({ sites, seasons, children }: { sites: readonly ExportSite[]; seasons: readonly ExportSeason[]; children: ReactNode }) => <Sites.Provider value={sites}><Seasons.Provider value={seasons}>{children}</Seasons.Provider></Sites.Provider>
export const useExportSites = () => useContext(Sites)
