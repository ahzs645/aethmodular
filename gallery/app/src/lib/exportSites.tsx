import { createContext, useContext, type ReactNode } from 'react'
import type { ExportSite } from './exportColors'
const Sites = createContext<readonly ExportSite[]>([])
export const ExportSitesProvider = ({ sites, children }: { sites: readonly ExportSite[]; children: ReactNode }) => <Sites.Provider value={sites}>{children}</Sites.Provider>
export const useExportSites = () => useContext(Sites)
