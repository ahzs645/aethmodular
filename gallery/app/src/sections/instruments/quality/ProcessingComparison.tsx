import { Figure } from '@/components/Figure'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { ExportState } from '@/components/ExportState'
import { useSpectral, rows } from '../../research/pages/spectralCommon'
import { PROCESSING_COLOR } from '@/lib/theme'

/** The same Figure/SweepPlot/ScatterPlot machinery as the ML and research views. */
export function ProcessingComparison() {
  const { data, error } = useSpectral('processing', 'aethalometer/quality')
  if (!data) return <ExportState what="the ONA/CMA/DEMA comparison" error={error}
    command="uv run python gallery/data/export_processing_comparison.py --minute PATH" />
  const settings = rows(data, 'settings')[0]
  const channel = [{ column: 'channel', label: 'Channel', default: 'IR' }]
  return <>
    <ChartFrame id="processing-scope" title={`ONA / CMA / DEMA · ${settings?.scope ?? 'minute record'}`} exportable={false}>
      <p className="subtitle">{data.text.scope}</p>
    </ChartFrame>
    <Figure kind="table" id="processing-coverage" data={rows(data, 'coverage')}
      title="Input coverage and BCc ranges (µg/m³)" />
    <Figure kind="sweep" id="processing-trace" data={rows(data, 'trace')} x="time" y="bc" series="method" choose={channel} colors={PROCESSING_COLOR}
      xScale="time" xLabel="Addis local time" yLabel="BCc (µg/m³)" title="Same minutes, four processing methods" height={300} />
    <Figure kind="ecdf" id="processing-windows" data={rows(data, 'windows')} value="minutes" count="count" choose={channel}
      xLabel="Minutes per ONA window" title="Adaptive averaging windows" subtitle="One observation per ONA window, including the final truncated window in each segment." />
    <Figure kind="sweep" id="processing-lag" data={rows(data, 'lag')} x="lag_minutes" y="correlation" series="method" choose={channel} colors={PROCESSING_COLOR}
      xLabel="Lag (minutes; positive means delay)" yLabel="Pearson r with raw BCc" title="Signal preservation and lag" />
    <Figure kind="table" id="processing-summary" data={rows(data, 'summary')} choose={channel}
      title="Processing differences on the same finite minutes" />
    {rows(data, 'daily').length ? <>
      <Figure kind="sweep" id="processing-daily" data={rows(data, 'daily')} x="date" y="bc" series="method" choose={channel} colors={PROCESSING_COLOR}
        xScale="time" xLabel="09:00 local day" yLabel="Mean BCc (µg/m³)" title="Daily means on common minutes" />
      <Figure kind="scatter" id="processing-agreement" data={rows(data, 'agreement')} x="mean" y="difference"
        choose={[...channel, { column: 'pair', label: 'Methods' }]} refY={[0]}
        xLabel="Mean of the two methods (µg/m³)" yLabel="Difference (µg/m³)" title="Daily method agreement" />
    </> : <ChartFrame id="processing-daily" title="Daily method agreement" exportable={false}>
      <Empty>This record has no 09:00 local days with at least 720 common valid minutes.</Empty>
    </ChartFrame>}
  </>
}
