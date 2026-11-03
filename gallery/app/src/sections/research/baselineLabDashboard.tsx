/**
 * Private Research charts register through the shared Dashboard registry. This module is imported by
 * sections/research/index.tsx, so VITE_PUBLIC_BUILD still omits their code together with Research.
 */
import type { ComponentType } from 'react'
import { registerDashboardChart } from '@/components/dashboard/registry'
import { LAB_FILE } from './common/baselineLab'
import { SEASONAL_ML_FILE } from './common/seasonalML'
import { BaselineFilterScatter, BaselineSelectedSpectrum } from './pages/baselineLab/Filters'
import { BaselineAssociations, BaselineChronology, BaselineControls, BaselineRecalibration } from './pages/baselineLab/Experiments'
import { BaselineEthiopia } from './pages/baselineLab/Ethiopia'
import { SeasonalMLPerformance, SeasonalMLRegions } from './pages/baselineLab/SeasonalML'
import type { LabChartProps } from './pages/baselineLab/data'

const charts: { kind: string; label: string; blurb: string; Chart: ComponentType<LabChartProps> }[] = [
  { kind: 'baseline-lab-filters', label: 'Baseline: filter diagnostics', blurb: 'physical-filter spectral diagnostics against current optical disagreement', Chart: BaselineFilterScatter },
  { kind: 'baseline-lab-spectrum', label: 'Baseline: signed spectra', blurb: 'raw, removed and corrected spectra; original scans and physical-filter means', Chart: BaselineSelectedSpectrum },
  { kind: 'baseline-lab-ml-performance', label: 'Seasonal ML: held-out performance', blurb: 'future-cycle classification and FTIR–optical disagreement prediction, with exact test coverage', Chart: SeasonalMLPerformance },
  { kind: 'baseline-lab-ml-regions', label: 'Seasonal ML: spectra and regions', blurb: 'signed seasonal profiles, held-out grouped permutation and refitted leave-region-out tests', Chart: SeasonalMLRegions },
  { kind: 'baseline-lab-controls', label: 'Baseline: experimental controls', blurb: 'blank residuals, injected-signal recovery and original repeat acquisitions', Chart: BaselineControls },
  { kind: 'baseline-lab-associations', label: 'Baseline: spectral associations', blurb: 'saved squared partial rank correlations with optical disagreement', Chart: BaselineAssociations },
  { kind: 'baseline-lab-chronology', label: 'Baseline: forward evaluation', blurb: 'later-quarter disagreement prediction with training-only scaling and tuning', Chart: BaselineChronology },
  { kind: 'baseline-lab-ethiopia', label: 'Baseline: Ethiopia comparisons', blurb: 'dates, lots, seasons and existing matched Addis/Bishoftu cohorts', Chart: BaselineEthiopia },
  { kind: 'baseline-lab-recalibration', label: 'Baseline: matched recalibration', blurb: 'existing independent source-reference scores for matched preprocessing and calibration', Chart: BaselineRecalibration },
]

const spectrumResources = () => {
  const ids = new Set(['ETAD-0037'])
  // Snapshot all selected spectrum tiles, including their scoped URL settings, without bundling unrelated spectra.
  for (const [key, value] of new URL(window.location.href).searchParams) if (key.endsWith('blab.filter')) ids.add(value)
  return [LAB_FILE, ...[...ids].map(id => `research/baseline_lab/spectra/${encodeURIComponent(id)}.json`)]
}

for (const { kind, label, blurb, Chart } of charts) registerDashboardChart({
  kind, label, blurb, group: 'Research · Baseline lab', fields: 'none', size: [6, 6],
  resources: kind === 'baseline-lab-spectrum' ? spectrumResources : () => [kind.startsWith('baseline-lab-ml-') ? SEASONAL_ML_FILE : LAB_FILE],
  render: data => <Chart rows={data.rows} meta={data.meta} />,
})
