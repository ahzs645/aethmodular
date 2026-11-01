import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useChartOptions } from '@/lib/chartOptions'
import { HeatMatrix } from '@/components/HeatMatrix'
import { INK } from '@/lib/theme'
import type { CalibFile, CalibRow } from '@/lib/types'
import { configKey, configTipLines, passes, sameConfig, score, shortConfig, type Config } from './common'
import { METRIC } from '@/lib/labels'
import { fmt } from '@/lib/format'
import { useColourScheme } from '@/lib/colourSchemes'
import { FixedSetNote } from '@/lib/markedRows'

const CELLS = ['Deming intercept', 'Deming slope', 'R²'] as const
type Cell = (typeof CELLS)[number]

/**
 * Cross-site heatmap — is the answer Addis-specific? Rows are configurations
 * (the top N at Addis by the explorer's score inside the guardrails, plus
 * whatever is selected above), columns are the five SPARTAN evaluation sites.
 * CROSS_SITE_EVALUATION_2026-08-22 turned "the offset is Addis-specific" into
 * an ordering: large negative intercepts at Addis and Delhi, near zero at
 * Beijing, JPL and Bishoftu — a compositional signature, not one city's
 * quirk. Quote OFFSET_ADJUDICATION_2026-08-23 for the weighted numbers.
 */
export function CrossSiteHeatmap({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const o = useChartOptions('cross-site', [
    { key: 'cell', label: 'Cell', choices: CELLS, default: 'Deming intercept' },
    { key: 'topN', label: 'Top N at Addis', choices: ['5', '10', '15', '25', '40'], default: '15' },
  ])
  const cell = o.values.cell as Cell
  const topN = o.values.topN as string
  // the rows are site-grouped cross-validation (the only protocol every evaluation site has)
  const mode = 'site_heldout'

  const targets = useMemo(() => {
    const present = [...new Set(calib.grid.filter((r) => r.mo === mode).map((r) => r.tg))]
    const order = ['addis', 'etbi', 'indh', 'chts', 'uspa']
    return present.sort((a, b) => (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 99 : order.indexOf(b)))
  }, [calib.grid, mode])

  const { rows, byKey } = useMemo(() => {
    const base = calib.grid.filter((r) => r.mo === mode && r.lot === 'all' && r.el === 'all')
    const byKey = new Map<string, CalibRow>()
    for (const r of base) byKey.set(`${configKey({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })}|${r.tg}`, r)
    const addis = base.filter((r) => r.tg === 'addis' && passes(r, calib)).sort((a, b) => (score(a) ?? 1e9) - (score(b) ?? 1e9))
    const cfgs: Config[] = []
    const seen = new Set<string>()
    const push = (c: Config) => {
      const k = configKey(c)
      if (!seen.has(k)) { seen.add(k); cfgs.push(c) }
    }
    push(selected)
    for (const r of addis.slice(0, Number(topN))) push({ co: r.co, cut: r.cut, sel: r.sel, sp: r.sp })
    return { rows: cfgs, byKey }
  }, [calib, mode, topN, selected])

  const valueOf = (r: CalibRow | undefined): number | null => {
    if (!r) return null
    if (cell === 'Deming intercept') return r.db
    if (cell === 'Deming slope') return r.dm
    return r.r2
  }
  // intercept and slope diverge around agreement (0, 1); R² is a sequential scale: each its own palette
  const divColours = useColourScheme('cross-site', 'diverging')
  const seqColours = useColourScheme('cross-site-r2', 'sequential')
  const colours = cell === 'R²' ? seqColours : divColours
  const color = useMemo(() => {
    if (cell === 'Deming intercept') return d3.scaleLinear<string>().domain([-6, -4, -2, 0, 2, 4, 6]).range(divColours.stops).clamp(true)
    if (cell === 'Deming slope') return d3.scaleLinear<string>().domain([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]).range(divColours.stops).clamp(true)
    return d3.scaleQuantize<string>().domain([0, 1]).range(seqColours.stops)
  }, [cell, divColours.stops.join(), seqColours.stops.join()]) // eslint-disable-line react-hooks/exhaustive-deps

  const rowItems = useMemo(() => rows.map((cfg) => {
    const on = sameConfig(cfg, selected)
    return { key: configKey(cfg), label: shortConfig(cfg), mono: true, color: on ? INK.text : INK.muted, weight: on ? 600 : 400 }
  }), [rows, selected])
  const colItems = useMemo(() => targets.map((t) => ({ key: t, label: calib.targets[t]?.site ?? t, color: calib.targets[t]?.color ?? INK.text })), [targets, calib.targets])
  const selIdx = rows.findIndex((c) => sameConfig(c, selected))

  return (
    <ChartFrame
      id="cross-site"
      title="Cross-site heatmap — does the calibration travel?"
      subtitle="The best Addis configurations (by the explorer's score, inside the slope box and above the IMPROVE cross-validation R² floor) read out at every SPARTAN site with FTIR spectra and HIPS Fabs; each site is a test set the model never saw. A row that is blue at Addis and Delhi but white at Beijing and JPL is the compositional-offset signature. The first row is whatever is selected above."
      provenance={`CROSS_SITE_EVALUATION_2026-08-22.md · FIVE_SITE_GRID_2026-08-23.md · site-grouped CV · MAC ${calib.mac_value} · fixed set`}
      controls={o.controls()}
    >
      {rows.length === 0 || targets.length === 0 ? (
        <Empty>No cross-site rows in the export.</Empty>
      ) : (
        // the cells (not cells + row labels) sit in the middle of the panel when there is room
        <HeatMatrix rows={rowItems} columns={colItems} center cellWidth={[70, 140]} cellHeight={26} gap={2} radius={2} maxLabelWidth={250}
          colLabels="horizontal" colLabelSize={11.5} colLabelWeight={600} labelSize={10.5} color={color} format={(v) => fmt(v, 2)}
          onRow={(i) => onSelect(rows[i])}
          cell={(i, j) => {
            const cfg = rows[i], t = targets[j], r = byKey.get(`${configKey(cfg)}|${t}`), v = valueOf(r)
            return { value: v, outline: i === selIdx, stroke: INK.panel,
              tip: r ? [`${shortConfig(cfg)} @ ${calib.targets[t]?.site ?? t}`, `${METRIC.k}: ${r.k}`, ...configTipLines(r)]
                : [`${shortConfig(cfg)} @ ${calib.targets[t]?.site ?? t}`, 'not evaluated at that site'] }
          }}
          legend={{ scheme: colours.scheme, label: cell === 'Deming intercept' ? 'Test set Deming intercept (µg/m³)' : `Test set ${cell}`, width: 200, ticks: cell === 'R²' ? 5 : 6,
            format: (v) => (cell === 'Deming intercept' ? (v > 0 ? `+${v}` : String(v)) : String(v)),
            note: cell === 'Deming intercept' ? 'white = 0 · clamped at ±6' : cell === 'Deming slope' ? 'white = slope 1' : undefined }}
          legendExtra={<span className="legend-note">grey cell = not evaluated at that site</span>} />
      )}
      <FixedSetNote what="the explorer's saved readouts at every site, on its own test sets" />
    </ChartFrame>
  )
}
