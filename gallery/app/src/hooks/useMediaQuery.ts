import { useEffect, useState } from 'react'

/** whether a CSS media query matches, kept current as the window changes */
export function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const m = window.matchMedia(query)
    const on = () => setMatches(m.matches)
    on()
    m.addEventListener('change', on)
    return () => m.removeEventListener('change', on)
  }, [query])
  return matches
}

/** a phone-width window: charts move in-plot readouts (fit equations) below the plot */
export const useNarrowScreen = () => useMediaQuery('(max-width: 640px)')
