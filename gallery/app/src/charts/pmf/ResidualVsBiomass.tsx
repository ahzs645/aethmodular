import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Crossplot, sharedCrossDomains, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { RefLine } from '@/components/RefLines'
import { useGalleryData } from '@/hooks/useGalleryData'
import { coef, f2, fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { biomassPct, BIOMASS_SOURCES } from '@/lib/pmfCoverage'
import { regression } from '@/lib/stats'
import { INK } from '@/lib/theme'
import type { MetaFile, PmfFile, PmfRow } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import instruments from '@/sections/instruments'

const AETH = 'IR BCc (aethalometer)'
/** the method pairs, minuend first: the notebook's residual is FTIR EC − aethalometer BC */
const PAIRS = {
  'FTIR EC − IR BCc': { a: 'EC (FTIR)', b: AETH },
  'FTIR EC − HIPS BC': { a: 'EC (FTIR)', b: 'HIPS BC' },
  'IR BCc − HIPS BC': { a: AETH, b: 'HIPS BC' },
} as const
type Pair = keyof typeof PAIRS
const PAIR_KEYS = Object.keys(PAIRS) as Pair[]
const FORMS = ['Absolute', 'Percent'] as const
const LAYOUTS = ['One plot', 'Per season'] as const

interface Day { p: PmfRow; bm: number; res: number; season: string }

/**
 * ETAD_Factor_Analysis c37–38: does biomass burning move the methods apart? Each PMF day's difference between two
 * BC/EC measurements (FTIR EC − aethalometer IR BCc in the notebook) against the biomass share of its PMF solution
 * (Wood Burning + Charcoal, % of the explained sources), with a fit per season (c38's "by Ethiopian season") or a
 * panel per season. The seasons are the Seasons choice in the sidebar (the notebook used the Belg-owns-February
 * calendar); HIPS BC is Fabs ÷ the reader's MAC.
 */
export function ResidualVsBiomass({ pmf, rows, meta }: { pmf: PmfFile; rows: PmfRow[]; meta: MetaFile }) {
  const { data } = useGalleryData()
  const mac = useMac()
  const aeth = instruments.useFields([pmf.site])
  const [pairParam, setPair] = useSearchParam<Pair>('pmfbm.pair', PAIR_KEYS[0], PAIR_KEYS)
  const [form, setForm] = useSearchParam<(typeof FORMS)[number]>('pmfbm.form', 'Absolute', FORMS)
  const [layout, setLayout] = useSearchParam<(typeof LAYOUTS)[number]>('pmfbm.layout', 'One plot', LAYOUTS)
  const hasAeth = !!aeth?.fields.includes(AETH)
  const pairs = PAIR_KEYS.filter((k) => hasAeth || !Object.values(PAIRS[k]).includes(AETH))
  const pair = pairs.includes(pairParam) ? pairParam : pairs[0]
  const { a, b } = PAIRS[pair]
  const pct = form === 'Percent'

  const days = useMemo<Day[]>(() => {
    const byId = new Map((data?.filters.rows ?? []).filter((r) => r.code === pmf.site).map((r) => [r.id, r]))
    const aIdx = aeth ? aeth.fields.indexOf(AETH) : -1
    const value = (id: string, f: string): number | null => {
      if (f === AETH) { const v = aIdx >= 0 ? aeth?.rows[id]?.[aIdx] : null; return typeof v === 'number' ? v : null }
      const r = byId.get(id)
      if (!r) return null
      // HIPS BC at the reader's MAC, from the absorption (filters.json carries it at the export's)
      if (f === 'HIPS BC') return mac.fromFabs(typeof r['HIPS Fabs'] === 'number' ? r['HIPS Fabs'] : null)
      return typeof r[f] === 'number' ? (r[f] as number) : null
    }
    const labels = pmf.sources.map((s) => s.label)
    return rows.flatMap((p) => {
      const va = value(p.id, a), vb = value(p.id, b), bm = biomassPct(p.fraction, labels)
      if (va === null || vb === null || bm === null || (pct && va === 0)) return []
      const d = va - vb
      return [{ p, bm, res: pct ? (100 * d) / va : d, season: p.season }]
    })
  }, [data, aeth, rows, pmf, a, b, pct, mac])

  const seasonColor = (s: string, i: number) => meta.seasons.find((x) => x.name === s)?.color ?? d3.schemeTableau10[i % 10]
  const seasons = useMemo(() => [...new Set(days.map((d) => d.season))].sort((x, y) => {
    const ix = meta.seasons.findIndex((s) => s.name === x), iy = meta.seasons.findIndex((s) => s.name === y)
    return (ix < 0 ? 99 : ix) - (iy < 0 ? 99 : iy) || x.localeCompare(y)
  }), [days, meta.seasons])
  const colorOf = new Map(seasons.map((s, i) => [s, seasonColor(s, i)]))
  const yLabel = pct ? `(${a} − ${b}) / ${a} (%)` : `${a} − ${b} (µg/m³)`
  const point = (d: Day): CrossPoint => ({
    x: d.bm, y: d.res, color: colorOf.get(d.season) ?? INK.neutral, id: d.p.id, site: pmf.site_name, season: d.season, seasonColor: colorOf.get(d.season),
    tip: [d.p.id, `${pmf.site_name} · ${d.p.date}`, `biomass: ${d.bm.toFixed(1)} % of the sources`, `${pct ? 'difference' : `${a} − ${b}`}: ${f2(d.res)}${pct ? ' %' : ' µg/m³'}`, d.season],
    rec: { id: d.p.id, source: 'PMF sources · residual against biomass', site: pmf.site_name, date: d.p.date, fields: [['biomass (% of sources)', +d.bm.toFixed(2)], [yLabel, +d.res.toFixed(3)], ['season', d.season]] },
  })
  const all = days.map(point)
  const panels = seasons.map((s) => ({ season: s, points: days.filter((d) => d.season === s).map(point) }))
  const shared = sharedCrossDomains(panels.map((p) => p.points), false)
  // per season, the notebook's table: n, R², OLS slope, mean biomass %
  const table = [...panels.map((p) => ({ name: p.season, ds: days.filter((d) => d.season === p.season) })), { name: 'All seasons', ds: days }].map(({ name, ds }) => {
    const st = regression(ds.map((d) => d.bm), ds.map((d) => d.res))
    return { name, n: ds.length, st, bm: d3.mean(ds, (d) => d.bm) ?? NaN }
  })
  // the fits are OLS of the difference on the biomass share, as the notebook's linregress: a trend against a
  // share, not a method comparison, so the sidebar's errors-in-variables fits do not apply
  const olsLine = (name: string): CrossLine[] => {
    const t = table.find((x) => x.name === name)
    return t?.st ? [{ slope: t.st.slope, intercept: t.st.intercept, color: colorOf.get(name) ?? INK.text, label: `${name} OLS`, method: 'ols', stats: t.st }] : []
  }
  const zero = (x: d3.ScaleLinear<number, number>, y: d3.ScaleLinear<number, number>) => <RefLine orient="h" at={y(0)} from={0} to={x.range()[1]} />

  return (
    <ChartFrame id="pmf-residual-biomass"
      title={`Method difference against the biomass share, by season — ${pair}${pct ? ` as % of ${a}` : ''}, per ${pmf.site_name} PMF day`}
      subtitle={`Biomass is ${BIOMASS_SOURCES.join(' + ')} as a % of the explained sources (the normalised fractions); each point is one PMF day in view with both measurements (${days.length} of ${rows.length}). ${layout === 'One plot' ? 'One plot with a fit per season, as the notebook drew it' : 'One panel per season, on shared axes'}; lines and table are OLS of the difference on the share, as the notebook fitted them (a trend against a share, so the sidebar's errors-in-variables fits do not apply). Seasons follow the sidebar's Seasons choice (the notebook used the Belg-owns-February calendar). ${hasAeth ? 'IR BCc is the aethalometer averaged over each filter\'s sampling window.' : 'The aethalometer pairs need the Instruments section\'s per-filter export, which this build does not have.'} HIPS BC = HIPS Fabs ÷ MAC ${mac.mac}.`}
      provenance={`research/ftir_hips_chem/ETAD_Factor_Analysis.ipynb c37 (absolute and % residual vs biomass %), c38 (by Ethiopian season) · from pmf.json + filters.json${hasAeth ? ' + aethalometer/filters' : ''}`}
      controls={<>
        <Segmented label="Pair" value={pair} options={pairs} onChange={setPair} title="The two measurements whose difference is plotted (first minus second)" />
        <Segmented label="Difference" value={form} options={FORMS} onChange={setForm} optionLabel={(o) => (o === 'Percent' ? `% of ${a.replace(' (aethalometer)', '')}` : 'µg/m³')} title="The difference in µg/m³, or as a % of the first measurement (c37's right panel)" />
        {seasons.length > 1 && <Segmented label="Layout" value={layout} options={LAYOUTS} onChange={setLayout} title="One plot with a fit per season, or a panel per season" />}
      </>}>
      {days.length < 3 ? <Empty>{data ? `Too few PMF days in view with ${a} and ${b} (${days.length}).` : 'Loading the filters…'}</Empty> : <>
        {layout === 'One plot' || seasons.length < 2
          ? <Crossplot points={all} title={`${pair} against biomass %`} xLabel="Biomass (% of explained sources)" yLabel={yLabel} sameScale={false}
              grouping="none" maxSide={520} background={zero} extraLines={seasons.length > 1 ? seasons.flatMap(olsLine) : olsLine('All seasons')} />
          : <FacetGrid items={panels} minCell={260} maxCols={4} maxCell={380} render={(p, w) => (
              <Crossplot points={p.points} title={`${p.season} · n = ${p.points.length}`} xLabel="Biomass (% of explained sources)" yLabel={yLabel} sameScale={false}
                grouping="none" maxSide={w - 70} domain={shared.domain} yDomain={shared.yDomain} background={zero} extraLines={olsLine(p.season)} />)} />}
        <div style={{ overflowX: 'auto', marginTop: 8 }}>
          <table className="census-table">
            <thead><tr><th>Season</th><th>n</th><th>R²</th><th>slope (per % biomass)</th><th>intercept</th><th>mean biomass %</th></tr></thead>
            <tbody>{table.map((t) => (
              <tr key={t.name}><td><span className="swatch" style={{ background: colorOf.get(t.name) ?? 'transparent' }} /> {t.name}</td><td>{t.n}</td>
                <td>{t.st ? coef(t.st.r2) : '—'}</td><td>{t.st ? fmt(t.st.slope, 4) : '—'}</td><td>{t.st ? coef(t.st.intercept) : '—'}</td><td>{Number.isFinite(t.bm) ? t.bm.toFixed(1) : '—'}</td></tr>
            ))}</tbody>
          </table>
          <p className="chart-note">OLS of the difference on the biomass share (scipy.stats.linregress, as ETAD_Factor_Analysis printed it); a season with fewer than three days has no fit.</p>
        </div>
      </>}
    </ChartFrame>
  )
}
