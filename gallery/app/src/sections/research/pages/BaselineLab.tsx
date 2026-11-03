import { dataUrl } from '@/lib/data'
import { fmt } from '@/lib/format'
import { Grid } from '../common/ui'
import { LAB_FILE } from '../common/baselineLab'
import { SEASONAL_ML_FILE } from '../common/seasonalML'
import type { SectionPageProps } from '../../types'
import { BaselineFilterScatter, BaselineSelectedSpectrum } from './baselineLab/Filters'
import { BaselineAssociations, BaselineChronology, BaselineControls, BaselineRecalibration } from './baselineLab/Experiments'
import { BaselineEthiopia } from './baselineLab/Ethiopia'
import { SeasonalMLPerformance, SeasonalMLRegions } from './baselineLab/SeasonalML'
import { SeasonalMLChecks } from './baselineLab/SeasonalChecks'
import { SEASONAL_CHECKS_FILE } from '../common/seasonalChecks'
import { useLabExport } from './baselineLab/data'

/** Graph-first review of executed controls; registered only through the existing private Research section. */
export function BaselineLab(props: SectionPageProps) {
  const { data } = useLabExport()
  return <>
    <div className="row" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
      <span className="chart-note">Existing-data experiments · FTIR − optical BC · MAC {fmt(data?.mac_m2_g, 1)} m²/g · optical agreement is not EC accuracy</span>
      {props.onPick && <span className="row"><span className="row-label">Samples</span>
        <button className="btn" type="button" onClick={() => props.onPick?.(['ETAD', 'ETBI'])}>Ethiopia</button>
        <button className="btn" type="button" onClick={() => props.onPick?.(data?.sites.map(s => s.code) ?? ['ETAD', 'ETBI', 'INDH', 'CHTS', 'USPA'])}>Five spectral sites</button>
      </span>}
    </div>
    <Grid min={430}>
      <BaselineFilterScatter {...props} />
      <BaselineSelectedSpectrum {...props} />
    </Grid>
    <SeasonalMLPerformance {...props} />
    <SeasonalMLRegions {...props} />
    <SeasonalMLChecks {...props} />
    <BaselineControls {...props} />
    <Grid min={430}>
      <BaselineAssociations {...props} />
      <BaselineChronology {...props} />
    </Grid>
    <Grid min={430}>
      <BaselineEthiopia {...props} />
      <BaselineRecalibration {...props} />
    </Grid>
    <details className="panel" style={{ padding: '12px 16px' }}>
      <summary>Methods, provenance and downloads</summary>
      <div className="subtitle">
        <p>Live filter views use Samples and shared exclusions. Saved experiment scores retain their stated physical-filter cohorts. Every panel has its detailed methods under the information control.</p>
        {data?.notes?.map((note, i) => <p key={i}>{note}</p>)}
        {data?.methods.map(method => <p key={method.id}><strong>{method.label}</strong>{method.description ? `: ${method.description}` : ''}</p>)}
        <p>{data?.generated ? `Export: ${data.generated}. ` : ''}{data?.source_revision ? `Source revision: ${data.source_revision}.` : ''}</p>
        <ul>
          <li><a href={dataUrl(LAB_FILE)} download>Complete diagnostic data (JSON)</a></li>
          <li><a href={dataUrl(SEASONAL_ML_FILE)} download>Seasonal ML scores, predictions and region tests (JSON)</a></li>
          <li><a href={dataUrl(SEASONAL_CHECKS_FILE)} download>Seasonal ML checks: lot, FTIR EC line and residual regions (JSON)</a></li>
          {data?.downloads?.map(file => <li key={file.path}><a href={dataUrl(file.path)} download>{file.label}</a></li>)}
        </ul>
      </div>
    </details>
  </>
}
