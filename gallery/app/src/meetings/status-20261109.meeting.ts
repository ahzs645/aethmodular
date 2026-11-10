import type { MeetingSpec } from './types'

// The status page for the supervisor's end-of-December check-in: where the existing-data investigation stands
// after the 7 Nov closeout. Not a recorded meeting; every table is read from an existing export by
// gallery/data/export_meeting_20261109.py. Titles carry the message, since section text shows only with Explain on.
const HIPS = 'HIPS optical BC (Fabs / MAC 10)'

export default {
  id: 'status-20261109',
  label: 'Nov 9, 2026',
  date: '2026-11-09',
  title: 'Investigation status: the existing-data closeout, what it showed, and the path to the December check-in',
  command: 'uv run python gallery/data/export_meeting_20261109.py',
  intro: '@intro',
  sections: [
    {
      id: 'domain',
      title: '1. Addis is outside the IMPROVE calibration library',
      text: '@domain',
      columns: 2,
      figures: [
        {
          kind: 'bars', id: 's1109-ec-share', table: 'ec_share', label: 'population', value: 'ec_share', color: 'network', format: 'f2',
          xLabel: 'EC share of PM2.5 (%)',
          title: 'EC is about a fifth of Addis PM2.5, against about 4 % across IMPROVE',
          subtitle: 'Median EC / PM2.5 by population (28 Oct speciation table). Addis EC is SPARTAN FTIR EC; IMPROVE EC is TOR.',
          noteAlways: true,
          note: 'Addis EC here is FTIR-derived, not thermal; IMPROVE EC is thermal (TOR).',
        },
        {
          kind: 'table', id: 's1109-loading', table: 'loading',
          title: 'Where the Addis median sits in the IMPROVE distribution',
          subtitle: 'HIPS Fabs / MAC 10, as concentration and as mass on the filter (9 Oct meeting).',
          columns: [
            { key: 'measure', label: 'Measure' }, { key: 'IMPROVE median', format: 'f2' }, { key: 'IMPROVE p99', format: 'f2' },
            { key: 'Addis median', format: 'f2' }, { key: '% of Addis above IMPROVE p99', label: 'Addis above IMPROVE p99', format: 'percent' },
          ],
        },
      ],
    },
    {
      id: 'preprocess',
      title: '2. Preprocessing on the Addis cross-plots',
      text: '@preprocess',
      columns: 2,
      figures: [
        {
          kind: 'bars', id: 's1109-addis-intercept', table: 'addis', label: 'candidate', value: 'intercept', color: 'candidate', reference: 0, format: 'f2',
          choose: [{ column: 'comparator', label: 'Against', default: HIPS }],
          xLabel: 'Addis intercept (µg/m³)',
          title: 'The spline baseline moves the HIPS intercept from about −6 to about −1.4, the VIBES variants stay between −3.4 and −6.4; against the aethalometer most candidates are near zero',
          subtitle: 'Deming FTIR EC against the chosen optical comparator, k 18 (closeout saved fits). The published product is on 189 filters, the recalibrated candidates on 233 (HIPS) or 172 (aethalometer).',
          noteAlways: true,
          note: 'Optical comparators, not thermal EC: an intercept near zero is agreement between methods, not validated Addis EC.',
        },
        {
          kind: 'bars', id: 's1109-addis-slope', table: 'addis', label: 'candidate', value: 'slope', color: 'candidate', reference: 1, format: 'f2',
          choose: [{ column: 'comparator', label: 'Against', default: HIPS }],
          xLabel: 'Addis slope',
          title: 'No candidate gives a 1:1 slope against HIPS',
          subtitle: 'Same fits. The slope scales with the assumed MAC; the intercept does not.',
          noteAlways: true,
          note: 'HIPS is Fabs / MAC 10, an optical equivalent. The 10 Nov page shows HIPS is compressed at Addis loadings, which steepens every slope against it.',
        },
      ],
    },
    {
      id: 'seasons',
      title: '3. Seasons and the choice of calibration filters',
      text: '@seasons',
      figures: [
        {
          kind: 'intervals', id: 's1109-seasons', table: 'cals', row: 'season', series: 'calibration',
          title: 'Season-specific look-alikes fix Kiremt’s slope; Dry and Belg stay between 0.3 and 0.75 whatever the calibration filters',
          subtitle: 'Addis slope and intercept by season for five ways of choosing the IMPROVE calibration filters (28 Oct meeting). Baseline-corrected spectra, Deming FTIR EC vs HIPS Fabs / MAC 10, the 189 poster filters.',
          noteAlways: true,
          note: 'Deming fits against HIPS Fabs / MAC 10 (optical, not thermal EC), Dry Oct–Feb calendar. The 10 Nov page re-reads these against the aethalometer.',
          panels: [
            { value: 'slope', label: 'Slope', reference: 1 },
            { value: 'intercept', label: 'Intercept (µg/m³)', reference: 0 },
            { value: 'R2', label: 'R²' },
          ],
          labelWidth: 90,
        },
      ],
    },
    {
      id: 'scores',
      title: '4. The source scoreboard: six preprocessing pipelines on the same IMPROVE filters',
      text: '@scores',
      columns: 2,
      figures: [
        {
          kind: 'bars', id: 's1109-scores', table: 'scores_long', label: 'candidate', value: 'value', color: 'candidate', format: 'f2',
          choose: [{ column: 'metric', label: 'Metric', default: 'Mean absolute error' }],
          xLabel: 'µg per filter, vs IMPROVE TOR EC',
          title: 'No universal winner: each error metric names a different best pipeline',
          subtitle: 'Pooled out-of-fold errors on 12,386 common IMPROVE filters (157 sites), 18 PLS components, five site-grouped folds, every eligible filter predicted (closeout 7 Nov).',
        },
        {
          kind: 'bars', id: 's1109-heavy', table: 'heavy', label: 'candidate', value: 'ratio', color: 'candidate', reference: 1, format: 'f2', domain: [0, 1.1],
          xLabel: 'median FTIR / TOR',
          title: 'EPO brings the heavy, low-OC/EC filters nearest 1:1',
          subtitle: 'The 163 IMPROVE filters with 15 ≤ TOR EC < 150 µg and TOR OC/EC < 2.5: the IMPROVE filters most like Addis.',
        },
        {
          kind: 'table', id: 's1109-scores-table', table: 'scores', title: 'The scoreboard as a table',
          columns: [
            { key: 'candidate', label: 'Pipeline' }, { key: 'MedAE', format: 'f3' }, { key: 'MAE', format: 'f3' }, { key: 'RMSE', format: 'f3' },
            { key: 'mean bias', format: 'f3' }, { key: 'P95 abs error', format: 'f3' }, { key: 'heavy median FTIR / TOR', format: 'f3' }, { key: 'n_common', label: 'filters', format: 'int' },
          ],
          note: 'All errors in µg per filter. Descriptive differences on one population; no significance test. Fixed settings include earlier exploratory choices, so this is a retrospective comparison rather than independent model selection.',
        },
      ],
    },
    {
      id: 'time',
      title: '5. Where the time went',
      text: '@time',
      figures: [
        {
          kind: 'bars', id: 's1109-commits', table: 'commits', label: 'week', value: 'commits', format: 'int',
          xLabel: 'commits to aethmodular (week starting)',
          title: 'Analysis and tooling peaked in late October; no manuscript draft yet',
          subtitle: 'Commits per week since 31 Aug 2026, from the repository history. A proxy for where effort went, not a measure of progress.',
        },
      ],
    },
    {
      id: 'schedule',
      title: '6. Schedule to the December check-in and the January decision',
      text: '@next',
      figures: [
        {
          kind: 'table', id: 's1109-milestones', table: 'milestones', title: 'Milestones',
          subtitle: 'Commitments from the 1 Sep committee meeting and the 7 Oct supervisory check-in; proposed writing dates.',
          columns: [{ key: 'milestone', label: 'Milestone' }, { key: 'due', label: 'Due' }, { key: 'status', label: 'Status' }],
        },
        {
          kind: 'table', id: 's1109-claims', table: 'claims', title: 'Claims the paper should not make',
          columns: [{ key: 'do not claim', label: 'Do not claim' }, { key: 'why', label: 'Why' }],
        },
      ],
    },
  ],
} satisfies MeetingSpec
