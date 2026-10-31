import { createContext, useContext } from 'react'
import type { FilterRow } from './types'

export interface DetailRequest { title: string; rows: FilterRow[]; fields: string[]; note?: string }
/** Aggregate marks pass their actual contributing rows, rather than reconstructing them in a dialog. */
export const DashboardDetails = createContext<((request: DetailRequest) => void) | null>(null)
export const useDashboardDetails = () => useContext(DashboardDetails)
