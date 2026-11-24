import { createContext, useContext } from 'react'
import type { Network } from '@/lib/networks'
import type { SamplePreset } from '@/lib/samplePresets'

/**
 * The gallery's Samples (network and SPARTAN sites), for pages without the Samples panel (the meeting pages) so a
 * chart there follows the one setting rather than keeping its own. App provides it; outside App (a test, an
 * embed of one chart) the hooks fall back to the defaults below.
 *
 *   const { network, setNetwork } = useGalleryNetwork()       // 'focus' | 'spartan' | 'improve' | 'both'
 *   const first = useFirstSampleSite()                          // the first SPARTAN site picked, or null
 *
 * A chart with a network split reads `network`; a chart that draws one site reads `psite` (lib/siteFocus
 * usePageSite) with `prefer: useFirstSampleSite() ?? …`. On a page with a preset (lib/samplePresets) both are the
 * page's own, so a choice made there does not move the Correlation tab.
 */
export interface GallerySamples {
  network: Network
  setNetwork: (n: Network) => void
  /** SPARTAN sites by code: null = the four focus sites (the default), 'all', or a list */
  spartanSites: string[] | 'all' | null
}

const Ctx = createContext<GallerySamples | null>(null)
export const GallerySamplesProvider = Ctx.Provider

export function useGalleryNetwork(): Pick<GallerySamples, 'network' | 'setNetwork'> {
  const ctx = useContext(Ctx)
  return ctx ?? { network: 'spartan', setNetwork: () => {} }
}

/** the first SPARTAN site the Samples pick names (a code), or null for the default four, all sites or none */
export function useFirstSampleSite(): string | null {
  const sel = useContext(Ctx)?.spartanSites
  return Array.isArray(sel) && sel.length ? sel[0] : null
}

/**
 * The Samples the HIPS meeting pages open on: the IMPROVE curve and the Addis filters. Their charts with a network
 * split (the bend) and a site (psite) follow these, and a change there stays with the page.
 */
export const HIPS_PAGE_SAMPLES: SamplePreset = { label: 'IMPROVE curve, Addis filters', samples: { net: 'improve', ssites: 'ETAD' } }

/** meeting pages with their own Samples (lib/samplePresets), by meeting key */
export const MEETING_PRESETS: Partial<Record<string, SamplePreset>> = {
  'hips-parts': HIPS_PAGE_SAMPLES,
  'hips-linearise': HIPS_PAGE_SAMPLES,
  'hips-story': HIPS_PAGE_SAMPLES,
  'hips-warren': HIPS_PAGE_SAMPLES,
  'hips-checks': { label: 'IMPROVE curve, Delhi and Addis filters', samples: { net: 'improve', ssites: 'INDH,ETAD' } },
}

/** the bend's three views and the gallery network each one is (focus counts as SPARTAN) */
export type NetworkView = 'IMPROVE' | 'SPARTAN' | 'Both'
export const NETWORK_VIEWS: readonly NetworkView[] = ['IMPROVE', 'SPARTAN', 'Both']
export const viewOfNetwork = (n: Network): NetworkView => (n === 'improve' ? 'IMPROVE' : n === 'both' ? 'Both' : 'SPARTAN')
export const networkOfView = (v: NetworkView): Network => (v === 'IMPROVE' ? 'improve' : v === 'Both' ? 'both' : 'spartan')
