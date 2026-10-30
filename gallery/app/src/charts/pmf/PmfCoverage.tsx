import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Toggle } from '@/components/ChartFrame'
import { XAxis } from '@/components/Axes'
import { Key } from '@/components/Key'
import { useDimensions, useGalleryData } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fieldCoverage, matchPmf } from '@/lib/pmfCoverage'
import { pickerName } from '@/lib/pmfGroups'
import { FONT, INK } from '@/lib/theme'
import type { FilterRow, MetaFile, PmfFile, PmfRow } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { HIDDEN_DUPLICATES } from '@/fieldLineage'
import { FixedSetNote } from '@/lib/markedRows'

type Status = 'shown' | 'hidden' | 'none'
const NO_PMF_COLOR = '#bbbbbb'
const LANE = 18
const LABEL_W = 150
const COUNT_W = 76

/**
 * ETAD_Factor_Analysis c11–13: which of the PMF site's filters the PMF solution covers. One lane of every filter
 * the gallery has at the site, then one lane per measurement it carries, each filter a tick coloured by whether it
 * has a PMF day (in this view's seasons and sources, outside them, or none), over the PMF's date range shaded.
 * The counts at the right are "with PMF / all". Computed here from filters.json and pmf.json (lib/pmfCoverage), so
 * a new export moves it.
 */
export function PmfCoverage({ pmf, rows, meta }: { pmf: PmfFile; rows: PmfRow[]; meta: MetaFile }) {
  const { data } = useGalleryData()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [lanes, setLanes] = useSearchParam<'0' | '1'>('pmfcov.fields', '1', ['0', '1'])
  const color = meta.sites.find((s) => s.code === pmf.site)?.color ?? INK.accent

  const view = useMemo(() => {
    const filters = (data?.filters.rows ?? []).filter((r) => r.code === pmf.site).sort((a, b) => a.date.localeCompare(b.date))
    const { of, unmatched } = matchPmf(filters, pmf.rows)
    const shownIds = new Set(rows.map((r) => r.id))
    const status = (f: FilterRow): Status => { const p = of.get(f.id); return !p ? 'none' : shownIds.has(p.id) ? 'shown' : 'hidden' }
    const fields = (data?.meta.fields ?? meta.fields).filter((f) => !HIDDEN_DUPLICATES.includes(f))
    const cov = fieldCoverage(filters, fields, (f) => of.has(f.id)).filter((c) => c.all > 0)
    const dates = pmf.rows.map((r) => r.date).sort()
    return { filters, status, cov, unmatched, range: dates.length ? [dates[0], dates[dates.length - 1]] as const : null,
      n: { all: filters.length, pmf: of.size, shown: filters.filter((f) => status(f) === 'shown').length, excluded: filters.filter((f) => f.excluded).length } }
  }, [data, pmf, rows, meta.fields])

  const site = pmf.site_name
  const laneList = [{ field: null as string | null, label: `All ${site} filters`, all: view.n.all, pmf: view.n.pmf }, ...(lanes === '1' ? view.cov.map((c) => ({ ...c, label: pickerName(c.field) })) : [])]
  const iw = Math.max(240, width - LABEL_W - COUNT_W - 8)
  const ext = d3.extent(view.filters, (f) => new Date(`${f.date.slice(0, 10)}T00:00:00Z`)) as [Date, Date] | [undefined, undefined]
  const x = d3.scaleUtc().domain(ext[0] ? [d3.utcMonth.floor(ext[0]), d3.utcMonth.ceil(ext[1]!)] : [new Date(0), new Date(1)]).range([0, iw])
  const H = laneList.length * LANE + 8
  const xOf = (d: string) => x(new Date(`${d.slice(0, 10)}T00:00:00Z`))
  const fill: Record<Status, string> = { shown: color, hidden: d3.color(color)?.copy({ opacity: 0.35 })?.formatRgb() ?? color, none: NO_PMF_COLOR }
  const label: Record<Status, string> = { shown: 'PMF day, in this view', hidden: 'PMF day, outside the chosen seasons or sources', none: 'no PMF day' }
  // a lane's filters in date order, for the nearest-filter readout
  const laneFilters = (field: string | null) => (field ? view.filters.filter((f) => typeof f[field] === 'number') : view.filters)
  const bisect = d3.bisector((f: FilterRow) => xOf(f.date)).center

  return (
    <ChartFrame id="pmf-coverage"
      title={`PMF coverage — the ${site} filters the PMF solution analysed, among all of them`}
      subtitle={`${view.n.all} ${site} filters in the gallery${view.n.excluded ? ` (${view.n.excluded} excluded by the notebooks' list, kept here)` : ''}; ${view.n.pmf} have a PMF day (${view.n.all ? ((100 * view.n.pmf) / view.n.all).toFixed(1) : '0'} %), ${view.n.shown} of them in this view's seasons and sources; ${view.n.all - view.n.pmf} have none.${view.range ? ` The PMF days run ${view.range[0]} to ${view.range[1]} (shaded).` : ''}${view.unmatched.length ? ` ${view.unmatched.length} PMF days match no filter.` : ' Every PMF day matches a filter.'} A filter has a PMF day when the PMF row carries its id or, failing that, is dated within a day of it (the notebook's rule). The lanes below repeat it per measurement, so the parameters the PMF subset can and cannot speak to show at a glance.`}
      provenance="research/ftir_hips_chem/ETAD_Factor_Analysis.ipynb c11 (PMF coverage), c12 (timeline, all vs PMF-analysed), c13 (parameter coverage timeline) · from filters.json + pmf.json"
      controls={<Toggle label="A lane per measurement" checked={lanes === '1'} onChange={(v) => setLanes(v ? '1' : '0')} title="One lane per measurement the site's filters carry (c13), or only the all-filters lane (c12)" />}>
      <div ref={wrapRef} className="chart-wrap">
        {!data ? <Empty>Loading the filters…</Empty> : !view.filters.length ? <Empty>No {site} filters in the export.</Empty> : <>
          <svg width={width} height={H + 34} fontFamily={FONT.family}>
            <g transform={`translate(${LABEL_W},4)`}>
              {view.range && <rect x={xOf(view.range[0])} width={Math.max(1, xOf(view.range[1]) - xOf(view.range[0]))} y={0} height={H - 8} fill={color} opacity={0.08} />}
              {laneList.map((l, i) => {
                const fs = laneFilters(l.field)
                const y0 = i * LANE
                return (
                  <g key={l.label} transform={`translate(0,${y0})`}>
                    <text x={-8} y={LANE / 2 + 4} textAnchor="end" fontSize={i === 0 ? 11.5 : 11} fontWeight={i === 0 ? 600 : 400} fill={INK.text}>{l.label}</text>
                    <line x1={0} x2={iw} y1={LANE / 2} y2={LANE / 2} stroke={INK.grid} />
                    {(['none', 'hidden', 'shown'] as const).map((st) => (
                      <path key={st} d={fs.filter((f) => view.status(f) === st).map((f) => `M${xOf(f.date).toFixed(1)},${3}V${LANE - 3}`).join('')}
                        stroke={fill[st]} strokeWidth={st === 'none' ? 1.2 : 1.6} />
                    ))}
                    <text x={iw + 8} y={LANE / 2 + 4} fontSize={11} fontFamily={FONT.mono} fill={INK.muted}>{l.pmf}/{l.all}</text>
                    {/* the lane's pointer layer: the nearest filter's date, id and PMF status */}
                    <rect width={iw} height={LANE} fill="transparent"
                      onMouseMove={(e) => {
                        if (!fs.length) return
                        const f = fs[bisect(fs, d3.pointer(e.nativeEvent, e.currentTarget)[0])]
                        tip.show(e, [f.id, `${f.site} · ${f.date.slice(0, 10)}`, l.field ? `${l.field}: measured` : `${l.pmf} of ${l.all} filters with a PMF day`, label[view.status(f)]])
                      }}
                      onMouseLeave={tip.hide} />
                  </g>
                )
              })}
              <XAxis scale={x} y={H - 4} tickCount={8} format={(d: Date) => d3.utcFormat(d.getUTCMonth() === 0 ? '%Y' : '%b')(d)} />
            </g>
          </svg>
          <Key items={[{ label: `PMF day, in this view (${view.n.shown})`, color: fill.shown, kind: 'bar' }, ...(view.n.pmf > view.n.shown ? [{ label: `PMF day, outside the chosen seasons or sources (${view.n.pmf - view.n.shown})`, color: fill.hidden, kind: 'bar' as const }] : []),
            { label: `No PMF day (${view.n.all - view.n.pmf})`, color: NO_PMF_COLOR, kind: 'bar' }, { label: 'PMF date range', color, kind: 'faded' }]} />
        </>}
        {tip.node}
      </div>
      <FixedSetNote what="every filter of the PMF site and the PMF solution's days" />
    </ChartFrame>
  )
}
