import { createElement, lazy, Suspense, type ComponentType, type FunctionComponent } from 'react'

/** Share the same loading boundary for pages and chart families loaded on demand. */
export function lazyView<P extends object>(load: () => Promise<FunctionComponent<P>>) {
  const attempt = async () => {
    try { return { default: await load() } }
    catch (error) { View = lazy(attempt); throw error }
  }
  let View = lazy(attempt)
  return function LazyView(props: P) {
    // React.lazy retains the loader’s props; the cast bridges React 18’s generic ref mapping.
    const Loaded = View as unknown as ComponentType<P>
    return <Suspense fallback={<p className="frame-empty" role="status">Loading chart…</p>}>{createElement(Loaded, props)}</Suspense>
  }
}
