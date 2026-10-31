import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { GraphicWalker, type IChart, type IChannelScales, type IDefaultConfig, type IMutField, type IRow, type VizSpecStore } from '@kanaries/graphic-walker'
import { WALKER_PLOT_THEME, WALKER_SHADOW_CSS, WALKER_UI_THEME } from './graphicWalkerTheme'
import contract from '@/lib/graphicWalkerContract.json'

/** The gallery talks to this interface; only this adapter knows Graphic Walker's store API. */
export interface ChartBuilderEngine {
  exportCharts(): IChart[]
  importCharts(charts: IChart[]): void
}
interface Props {
  data: IRow[]
  fields: IMutField[]
  charts?: IChart[]
  theme: 'light' | 'dark'
  scales: IChannelScales
  onError: (error: Error) => void
  onDispose: (charts: IChart[]) => void
}
const DEFAULT_CONFIG: IDefaultConfig = { config: { defaultAggregated: false }, layout: { size: { mode: 'auto', width: 640, height: 420 }, useSvg: true } }

export const GraphicWalkerAdapter = forwardRef<ChartBuilderEngine, Props>(function GraphicWalkerAdapter(props, ref) {
  const container = useRef<HTMLDivElement>(null)
  const store = useRef<VizSpecStore | null>(null)
  const dispose = useRef(props.onDispose)
  dispose.current = props.onDispose
  const [shadowRoot, setShadowRoot] = useState<ShadowRoot | null>(null)

  useImperativeHandle(ref, () => ({
    exportCharts() {
      if (!store.current) throw new Error('The chart builder is still opening. Try again in a moment.')
      return store.current.exportCode()
    },
    importCharts(charts) {
      if (!store.current) throw new Error('The chart builder is still opening. Try again in a moment.')
      store.current.importCode(charts)
    },
  }), [])

  // Layout cleanup runs before the child's passive cleanup clears storeRef. This uses
  // public APIs, with no dependency on upstream private edit events or instance IDs.
  useLayoutEffect(() => () => {
    if (store.current) dispose.current(JSON.parse(JSON.stringify(store.current.exportCode())) as IChart[])
  }, [])

  useEffect(() => {
    const host = container.current!
    const findRoot = () => {
      const root = host.querySelector('.gallery-walker')?.shadowRoot
      if (!root) return false
      setShadowRoot(root)
      return true
    }
    if (findRoot()) return
    const observer = new MutationObserver(() => { if (findRoot()) observer.disconnect() })
    observer.observe(host, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [])

  return <div ref={container} className="chart-builder-engine" role="region" aria-label="Chart construction workspace"
    data-walker-version={contract.testedVersion} data-walker-ready={shadowRoot ? 'true' : 'false'}>
    <GraphicWalker className="gallery-walker" style={{ height: '100%', width: '100%' }}
      data={props.data} fields={props.fields} chart={props.charts} storeRef={store}
      appearance={props.theme} uiTheme={WALKER_UI_THEME} vizThemeConfig={WALKER_PLOT_THEME}
      i18nLang="en-US" defaultConfig={DEFAULT_CONFIG} scales={props.scales}
      fieldKeyGuard={false} hideSegmentNav hideProfiling onError={props.onError} />
    {shadowRoot && createPortal(<style data-gallery-walker-style>{WALKER_SHADOW_CSS}</style>, shadowRoot)}
  </div>
})
