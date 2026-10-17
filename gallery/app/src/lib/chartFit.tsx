import { createContext, useContext, type ReactNode } from 'react'

/**
 * The height a chart may draw at, when its host has a fixed size (a Dashboard tile that
 * fits the window). Each chart keeps its own height everywhere else: `useFitHeight()` is
 * null outside a tile, so a chart reads it as
 *
 *   const fit = useFitHeight()
 *   const height = fit ?? 440
 *
 * The tile adjusts the number until the chart's frame fills the tile without overflow
 * (pages/DashboardPage), so a chart only has to draw taller for a larger number. A chart
 * with several panels (per-site grids) may ignore it; its tile then scrolls inside itself.
 */
const FitCtx = createContext<number | null>(null)

export const ChartFit = ({ height, children }: { height: number | null; children: ReactNode }) =>
  <FitCtx.Provider value={height}>{children}</FitCtx.Provider>

export const useFitHeight = () => useContext(FitCtx)

/** an inner plot height from a fitted outer one, never below `min` */
export const fitInner = (fit: number | null, margins: number, min = 80) =>
  fit === null ? null : Math.max(min, fit - margins)
