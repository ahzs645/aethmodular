import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { FixedSetNote } from '@/lib/markedRows'
import { IMPROVE_GREY } from '@/lib/theme'
import type { SectionPageProps } from '../../types'
import { DataTable } from '../common/ui'

const NB = 'research/spartan/spartan_vs_improve_comparison.ipynb'
const FOLLOW = 'research/spartan/follow_up_mass_and_ratio_plots.ipynb · research/spartan/site_ratios_mass_space.ipynb'
const COMMAND = 'uv run python gallery/data/export_research_spartan_improve.py'
const SPARTAN = '#c0392b'
type Row = Record<string, string | number | null>
interface Tables { generated: string; tables: Record<'parameter_map' | 'coverage' | 'distribution' | 'ratios' | 'fits' | 'analogs', Row[]> }
const n = (v: unknown) => (typeof v === 'number' ? v : null)
const color = (net: unknown) => (net === 'SPARTAN' ? SPARTAN : IMPROVE_GREY)

/**
 * SPARTAN against IMPROVE on the quantities both networks carry (EC, OC, fAbs, iron, PM2.5) as one set of
 * panels: distributions, the volume-free ratios (OC/EC, fAbs/EC ≈ MAC, Fe/PM2.5), per-network fits, and the
 * IMPROVE sites whose fAbs/EC and OC/EC sit closest to Addis. From the notebooks' saved tables.
 */
export function SpartanImprove(_props: SectionPageProps) {
  const { data, error } = useExport<Tables>('research/spartan_improve/tables.json')
  if (error || !data) return <ChartFrame id="si-missing" title="SPARTAN against IMPROVE" provenance={NB}><Empty>{error ? <>Not exported ({error}). Run <code>{COMMAND}</code>.</> : 'Loading…'}</Empty></ChartFrame>
  const t = data.tables
  const quantities = [...new Set(t.distribution.map((r) => String(r.quantity)))]
  const nets = ['IMPROVE', 'SPARTAN']
  const distRows: IntervalRow[] = nets.map((net) => ({ key: net, label: net, labelColor: color(net),
    marks: quantities.map((q) => { const r = t.distribution.find((d) => d.network === net && d.quantity === q)
      return r ? [{ mid: n(r.median), lo: n(r.p05), hi: n(r.p95), color: color(net) }, { mid: n(r.median), lo: n(r.p25), hi: n(r.p75), color: color(net), r: 4 }] : [] }),
    tip: [net, ...quantities.map((q) => { const r = t.distribution.find((d) => d.network === net && d.quantity === q); return r ? `${q}: median ${fmt(r.median, 3)} ${r.units} (p05 ${fmt(r.p05, 3)}, p95 ${fmt(r.p95, 3)}, n ${n(r.n)?.toLocaleString()})` : `${q}: —` })] }))
  const ratioNames = [...new Set(t.ratios.map((r) => String(r.ratio)))]
  const ratioRows: IntervalRow[] = nets.map((net) => ({ key: net, label: net, labelColor: color(net),
    marks: ratioNames.map((q) => { const r = t.ratios.find((d) => d.network === net && d.ratio === q); return r ? [{ mid: n(r.median), lo: n(r.p25), hi: n(r.p75), color: color(net) }] : [] }) }))
  const unitOf = (q: string) => String(t.distribution.find((d) => d.quantity === q)?.units ?? '')
  return (
    <>
      <ChartFrame id="si-dist" title="The shared parameters, network against network — median, IQR and p05–p95"
        subtitle="Only what both networks carry: thermal EC and OC (SPARTAN ChemSpec), HIPS fAbs, iron and fine mass, in µg/m³ (fAbs in Mm⁻¹). The networks share no sites or dates, so this is a distribution comparison, not a row-level join. In concentration a SPARTAN filter samples ~7 m³ and an IMPROVE filter ~33 m³; the Addis vs IMPROVE loading page repeats the set in mass on the filter."
        provenance={`${NB} (fig 01–03)`}>
        <IntervalStrip blocks={[{ key: 'd', rows: distRows }]} panels={quantities.map((q) => ({ label: `${q} (${unitOf(q)})` }))} labelWidth={80} minPanelWidth={150} />
        <Key items={[{ label: 'IMPROVE', color: IMPROVE_GREY }, { label: 'SPARTAN (four focus sites)', color: SPARTAN }]} />
        <FixedSetNote what="the notebook's saved SPARTAN and IMPROVE summaries" />
      </ChartFrame>

      <ChartFrame id="si-ratios" title="Volume-free ratios — OC/EC, fAbs/EC (≈ MAC) and Fe/PM2.5"
        subtitle="Median and interquartile range per network. SPARTAN's fAbs/EC uses FTIR EC, the only EC measured on the HIPS filter; read the MAC comparison with the FTIR-vs-thermal difference in mind."
        provenance={`${NB} (fig 05)`}>
        <IntervalStrip blocks={[{ key: 'r', rows: ratioRows }]} panels={ratioNames.map((q) => ({ label: `${q} (${t.ratios.find((r) => r.ratio === q)?.unit ?? ''})` }))} labelWidth={80} minPanelWidth={180} />
        <DataTable rows={t.ratios} rowKey={(r) => `${r.network}-${r.ratio}`} columns={[
          { key: 'n', label: 'Network', value: (r) => <span style={{ color: color(r.network), fontWeight: 600 }}>{String(r.network)}</span> }, { key: 'r', label: 'Ratio', value: (r) => String(r.ratio) },
          { key: 'u', label: 'Unit', value: (r) => String(r.unit) }, { key: 'c', label: 'n', num: true, value: (r) => n(r.n)?.toLocaleString() ?? '—' },
          { key: 'm', label: 'Median', num: true, value: (r) => fmt(r.median, 3) }, { key: 'i', label: 'IQR', num: true, value: (r) => `${fmt(r.p25, 3)}–${fmt(r.p75, 3)}` }]} />
        <FixedSetNote what="the notebook's saved SPARTAN and IMPROVE ratios" />
      </ChartFrame>

      <ChartFrame id="si-fits" title="The bridging relationships, fitted per network — fAbs ~ EC, OC ~ EC, Fe ~ PM2.5"
        subtitle="OLS per network over every filter that carries both. SPARTAN's fAbs ~ EC intercept (Addis dominated) against IMPROVE's near-zero one is the offset question in one line."
        provenance={`${NB} (fig 04)`} exportable={false}>
        <DataTable rows={t.fits} rowKey={(r) => `${r.relationship}-${r.network}`} columns={[
          { key: 'r', label: 'Relationship', value: (r) => String(r.relationship).replace(/_/g, ' ').replace('~', ' ~ ') },
          { key: 'n', label: 'Network', value: (r) => <span style={{ color: color(r.network), fontWeight: 600 }}>{String(r.network)}</span> },
          { key: 'c', label: 'n', num: true, value: (r) => n(r.n)?.toLocaleString() ?? '—' }, { key: 's', label: 'Slope', num: true, value: (r) => fmt(r.slope, 4) },
          { key: 'i', label: 'Intercept', num: true, value: (r) => fmt(r.intercept, 4) }, { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.r_squared, 3) }]} />
        <p className="subtitle" style={{ margin: '14px 0 4px' }}>Coverage, and which parameter was mapped to which</p>
        <DataTable rows={t.coverage} rowKey={(r) => `${r.network}-${r.quantity}`} columns={[
          { key: 'n', label: 'Network', value: (r) => String(r.network) }, { key: 'q', label: 'Quantity', value: (r) => String(r.quantity) },
          { key: 'c', label: 'n', num: true, value: (r) => n(r.n)?.toLocaleString() ?? '—' }, { key: 's', label: 'Sites', num: true, value: (r) => String(r.n_sites) },
          { key: 'd', label: 'Dates', value: (r) => `${r.date_min} – ${r.date_max}` }, { key: 'm', label: 'Median', num: true, value: (r) => fmt(r.median, 3) }]} />
        <div style={{ height: 14 }} />
        <DataTable rows={t.parameter_map} rowKey={(r) => String(r.harmonized_col)} columns={[
          { key: 'q', label: 'Quantity', value: (r) => String(r.quantity) }, { key: 's', label: 'SPARTAN parameter', value: (r) => <code>{String(r.spartan_param)}</code> },
          { key: 'i', label: 'IMPROVE column', value: (r) => <code>{String(r.improve_col)}</code> }, { key: 'n', label: 'Note', value: (r) => String(r.note) }]} />
        <FixedSetNote what="the notebook's saved per-network fits and coverage" />
      </ChartFrame>

      <ChartFrame id="si-analogs" title="IMPROVE sites that look like Addis — fAbs/EC and OC/EC by site, nearest first"
        subtitle="Each IMPROVE site's median fAbs/EC and OC/EC, and its distance to Addis in those two (scaled) coordinates. Mostly urban eastern sites; several were discontinued before the FED export carries flow, so they drop out of the mass-on-filter figures."
        provenance={FOLLOW} exportable={false}>
        <DataTable rows={t.analogs} rowKey={(r) => String(r.SiteCode)} columns={[
          { key: 'c', label: 'Site', value: (r) => `${r.SiteName ?? ''} (${r.SiteCode}${r.State ? `, ${r.State}` : ''})` }, { key: 'n', label: 'n', num: true, value: (r) => n(r.n)?.toLocaleString() ?? '—' },
          { key: 'f', label: 'Median fAbs / EC', num: true, value: (r) => fmt(r.Fabs_EC_med, 2) }, { key: 'o', label: 'Median OC / EC', num: true, value: (r) => fmt(r.OC_EC_med, 2) },
          { key: 'd', label: 'Distance to Addis', num: true, value: (r) => fmt(r.dist_to_ETAD, 2) }]} />
        <Note>The per-site ratios are identical in concentration and in mass on the filter (the same volume divides numerator and denominator), which is what site_ratios_mass_space.ipynb shows.</Note>
      </ChartFrame>
    </>
  )
}
