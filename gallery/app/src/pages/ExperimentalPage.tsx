import { PageToc } from '@/components/PageToc'
import { ChartFrame, Note, Select } from '@/components/ChartFrame'
import { SidebarControls } from '@/lib/pageSidebar'
import { useExport } from '@/lib/data'
import { EXPERIMENTS } from '@/experiments'
import { INK } from '@/lib/theme'
import { FlagStudy } from '@/pages/experimental/FlagStudy'
import { AnalogsStudy } from '@/pages/experimental/AnalogsStudy'
import { AttributionStudy } from '@/pages/experimental/AttributionStudy'
import { SelectorStudy } from '@/pages/experimental/SelectorStudy'
import { ForwardStudy } from '@/pages/experimental/ForwardStudy'
import { FixesStudy } from '@/pages/experimental/FixesStudy'
import { TorStudy } from '@/pages/experimental/TorStudy'
import { LotStudy } from '@/pages/experimental/LotStudy'

/**
 * Experimental: analyses still being worked out, one subpage per test of the 22 Oct
 * protocol review (the header's Experimental group, like Meetings). The flag study
 * comes from the decision workbook (Section 6b, gallery/data/export_experimental.py);
 * the rest from experiments_20261022.ipynb (gallery/data/export_experiments.py).
 */
export function ExperimentalPage({ experiment, onPick }: { experiment: string; onPick: (key: string) => void }) {
  return (
    <PageToc>
      <SidebarControls title="Experiment">
        <Select label="" value={experiment} options={EXPERIMENTS.map((e) => e.key)} onChange={onPick}
          optionLabel={(k) => EXPERIMENTS.find((e) => e.key === k)?.label ?? k} />
      </SidebarControls>
      {experiment === 'plan' && <TestPlan onPick={onPick} />}
      {experiment === 'flag' && <FlagStudy />}
      {experiment === 'analogs' && <AnalogsStudy />}
      {experiment === 'attribution' && <AttributionStudy />}
      {experiment === 'selector' && <SelectorStudy />}
      {experiment === 'forward' && <ForwardStudy />}
      {experiment === 'fixes' && <FixesStudy />}
      {experiment === 'lots' && <LotStudy />}
      {experiment === 'tor' && <TorStudy />}
    </PageToc>
  )
}

interface PlanData { schema_version: number; generated: string; items: Record<string, { status: 'done' | 'partial' | 'waiting'; headline: string }> }

const PLAN: { key: string; n: string; question: string; method: string }[] = [
  { key: 'flag', n: '1–2', question: 'Does spectral distance add anything beyond loading, and which failures does the flag miss?',
    method: 'Flag refitted on growing input sets from weighed PM2.5 alone; error against coverage; missed failures at a 10 % flag rate; Addis and SPARTAN on the same map.' },
  { key: 'analogs', n: '3', question: 'Does matching calibration neighbours on deposited mass as well as shape move Addis toward HIPS while IMPROVE holds?',
    method: 'Local PLS with shape, shape + loading and loading-only neighbours on 600 held-out IMPROVE filters (vs TOR) and every Addis filter (vs HIPS).' },
  { key: 'attribution', n: '4', question: 'How much do preprocessing, window, calibration pool and k each move the result, and do they interact?',
    method: 'Factorial grid on the same filters: IMPROVE out-of-site error against TOR and the Addis intercept against HIPS, with main effects and interactions.' },
  { key: 'selector', n: '5', question: 'Does predicting each model\'s error beat picking a winner, and does abstaining help?',
    method: 'Nested by site fold on 1,500 IMPROVE filters: full spline, full raw and local raw; error regressors; winner classifier; oracle.' },
  { key: 'forward', n: '6', question: 'Does the calibration carry forward to a later year as well as to a new site?',
    method: 'Train 2020–22, test 2023; equal-sized new-site, random and new-site-in-the-later-year splits.' },
  { key: 'fixes', n: '–', question: 'Is ETAD-0241 a real PM2.5 filter, and do the Addis seasonal slopes survive moving February into Belg?',
    method: 'Exclusion matched by filter ID; seasonal Deming slopes under both calendars with bootstrap intervals.' },
  { key: 'tor', n: 'D', question: 'Where do the Adama TOR pilot pairs sit against IMPROVE, the SPARTAN network and SPARTAN Addis, and where would they need to sit to test the Addis problem?',
    method: 'Axis-by-axis position in SPARTAN Addis\'s range; loading = concentration × volume; season; nearest TOR sites; precision by number of pairs; a draft pass contract for the Addis city pairs.' },
]
const STATUS: Record<string, { label: string; color: string }> = {
  done: { label: 'run', get color() { return INK.good } }, partial: { label: 'partly run', get color() { return INK.warn } }, waiting: { label: 'waits for data', get color() { return INK.muted } },
}

function TestPlan({ onPick }: { onPick: (key: string) => void }) {
  const { data } = useExport<PlanData>('experimental/plan.json', (d) => d.schema_version === 1 ? null : 'unsupported data version')
  return (
    <ChartFrame id="xp-plan" title="Test plan — the 22 Oct protocol review, test by test" exportable={false}
      subtitle="Every fair test of a calibration runs inside IMPROVE against TOR, holding out whole sites (or years). SPARTAN Addis is read against HIPS only as a cross-check, because HIPS ÷ 10 assumes the MAC. Click a test to open its page.">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12, marginTop: 6 }}>
        {PLAN.map((p) => {
          const it = data?.items?.[p.key], st = it ? STATUS[it.status] : null
          return (
            <button key={p.key} type="button" onClick={() => onPick(p.key)}
              style={{ textAlign: 'left', border: `1px solid ${INK.border}`, borderRadius: 8, padding: '10px 12px', background: INK.panel, cursor: 'pointer', font: 'inherit', color: INK.text }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
                <strong style={{ fontSize: 13 }}>{p.n !== '–' ? `${p.n} · ` : ''}{EXPERIMENTS.find((e) => e.key === p.key)?.label.replace(/^[\d–]+ · /, '')}</strong>
                {st && <span style={{ fontSize: 11, color: st.color, whiteSpace: 'nowrap' }}>● {st.label}</span>}
              </div>
              <div style={{ fontSize: 12, marginTop: 4 }}>{p.question}</div>
              <div style={{ fontSize: 11, color: INK.muted, marginTop: 4 }}>{p.method}</div>
              {it?.headline && <div style={{ fontSize: 12, marginTop: 6, paddingTop: 6, borderTop: `1px dashed ${INK.border}` }}>{it.headline}</div>}
            </button>
          )
        })}
      </div>
      <Note>Status and headlines come from the latest runs (gallery/data/export_experiments.py). Headlines are statements about that run's settings; each page carries its caveats.</Note>
    </ChartFrame>
  )
}
