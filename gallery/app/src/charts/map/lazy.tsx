import { lazy, Suspense, type ComponentProps } from 'react'
import { Empty } from '@/components/ChartFrame'

/**
 * The maps load MapLibre (via @pgmaps/geo-toolkit), about as large as the rest of the gallery, so they are
 * fetched when a map is first shown rather than with every page.
 */
const Bubble = lazy(() => import('@/charts/map/SiteBubbleMap').then((m) => ({ default: m.SiteBubbleMap })))
const Composition = lazy(() => import('@/charts/map/SpartanCompositionMap').then((m) => ({ default: m.SpartanCompositionMap })))

const loading = <Empty>Loading the map…</Empty>
export const SiteBubbleMap = (props: ComponentProps<typeof Bubble>) => <Suspense fallback={loading}><Bubble {...props} /></Suspense>
export const SpartanCompositionMap = () => <Suspense fallback={loading}><Composition /></Suspense>
