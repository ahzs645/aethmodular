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
 * with several panels divides the height between them; minimum legible sizes can still scroll.
 */
const FitCtx = createContext<number | null>(null)

export const ChartFit = ({ height, children }: { height: number | null; children: ReactNode }) =>
  <FitCtx.Provider value={height}>{children}</FitCtx.Provider>

export const useFitHeight = () => useContext(FitCtx)

/** an inner plot height from a fitted outer one, never below `min` */
export const fitInner = (fit: number | null, margins: number, min = 80) =>
  fit === null ? null : Math.max(min, fit - margins)

/** Display context only: data keys and legend interactions retain their full site names. */
const SiteCtx = createContext<string | null>(null)
export const ChartSite = ({ name, children }: { name: string | null; children: ReactNode }) =>
  <SiteCtx.Provider value={name}>{children}</SiteCtx.Provider>
export const useChartLabel = () => {
  const site = useContext(SiteCtx)
  return (label: string) => site && label === site ? '' : site && label.startsWith(`${site} · `) ? label.slice(site.length + 3) : label
}

/**
 * Where a tile puts its chart's legends (the tile's ⋯ menu): auto (the first few, "+N more"),
 * bottom (every item), right (a column beside the chart: the chart measures that much narrower,
 * hooks/useGalleryData useDimensions), or off. Outside a tile: null, legends as everywhere.
 */
export type LegendMode = 'auto' | 'bottom' | 'right' | 'off'
export const LEGEND_MODES: readonly LegendMode[] = ['auto', 'bottom', 'right', 'off']
const LegendCtx = createContext<{ mode: LegendMode; reserve: number } | null>(null)
export const TileLegend = ({ mode, reserve, children }: { mode: LegendMode; reserve: number; children: ReactNode }) =>
  <LegendCtx.Provider value={{ mode, reserve }}>{children}</LegendCtx.Provider>
export const useTileLegend = () => useContext(LegendCtx)
