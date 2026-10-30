import { Component, lazy, Suspense, type ReactNode } from 'react'
import type { FilterRow, MetaFile } from '@/lib/types'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { setChartBuilderEnabled } from '@/lib/chartBuilderPrefs'

// No Graphic Walker runtime or styles enter the ordinary gallery's import graph.
const GraphicWalkerWorkspace = lazy(() => import('@/components/chartBuilder/GraphicWalkerWorkspace'))

export function ChartBuilderPage({ rows, meta, enabled }: { rows: FilterRow[]; meta: MetaFile; enabled: boolean }) {
  return <ChartFrame title="Chart builder" exportable={false} id="chart-builder">
    <div className="chart-builder-intro">
      <p>{enabled && <><strong>{rows.length.toLocaleString()} filters</strong> · </>}Drag measures onto the axes; add site or season to create facets.</p>
      <details className="chart-builder-help"><summary>How to build and keep a chart</summary>
        <p>The gallery’s Filters panel controls the population. Units, ratios, MAC and site/season colours follow your current gallery settings.</p>
        <p>Drag fields from the field list onto X-Axis, Y-Axis or Color. Add multiple measures or a site/season dimension to an axis for small multiples. Aggregation starts off; use mean or median as appropriate when grouping concentrations.</p>
        <p>Drafts survive tab switches during this session. Download before reloading. Import applies a draft to the current selection; gallery share links carry filters, not chart definitions. Use the standard gallery charts for regression, linked point selection and publication export.</p>
      </details>
    </div>
    {!enabled ? <div className="chart-builder-intro">
      <p>This optional interface is off. Enable it here or in Display and data settings → Interface.</p>
      <button type="button" className="btn" onClick={() => setChartBuilderEnabled(true)}>Enable chart builder</button>
    </div> : <BuilderBoundary>
      <Suspense fallback={<Empty>Loading chart builder…</Empty>}>
        <GraphicWalkerWorkspace rows={rows} meta={meta} />
      </Suspense>
    </BuilderBoundary>}
    <p className="chart-builder-credit">Built with <a href="https://github.com/Kanaries/graphic-walker" target="_blank" rel="noreferrer">Graphic Walker</a> · <a href={`${import.meta.env.BASE_URL}licenses/graphic-walker.txt`} target="_blank" rel="noreferrer">License and attribution</a></p>
  </ChartFrame>
}

class BuilderBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    return this.state.failed ? <div role="alert" className="chart-builder-intro">
      <p>The chart builder could not open. Your gallery filters are still available. Reload the page to retry.</p>
      <button type="button" className="btn" onClick={() => window.location.reload()}>Reload</button>
    </div> : this.props.children
  }
}
