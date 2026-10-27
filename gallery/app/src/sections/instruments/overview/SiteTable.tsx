import { useMemo, useRef } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { SCREEN_TEXT, filterField, instColor, type Screen, type Series } from './data'
import { summarize } from './stats'

const COLS = ['bins', 'first', 'last', 'IR days', 'mean', 'median', 'SD', 'max', 'coverage %', 'filters', 'with aeth']

/**
 * One row per instrument: its daily record in the span and its IR BCc (Multi_Site_Analysis §7's
 * summary table and _Fixed §8's BC statistics). Clicking a site narrows the Filters panel's pick to it.
 */
export function SiteTable({ series, meta, screen, onSite }: { series: Series[]; meta: MetaFile; screen: Screen; onSite?: (code: string) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const table = useMemo(() => series.map((s) => {
    const ir = summarize(s.days.map((d) => d.bc.IR))
    const cov = summarize(s.days.map((d) => d.coverage))
    const field = filterField('IR')
    return {
      s, ir, cov,
      cells: [String(s.days.length), s.days[0]?.date ?? '—', s.days[s.days.length - 1]?.date ?? '—', String(ir?.n ?? 0),
        fmt(ir?.mean, 2), fmt(ir?.median, 2), fmt(ir?.sd, 2), fmt(ir?.max, 2), cov ? fmt(cov.mean, 0) : '—',
        String(s.rows.length), s.inst.code === s.inst.siteCode ? String(s.rows.filter((r) => typeof r[field] === 'number').length) : '—'],
    }
  }), [series])
  const labelW = 200, numW = Math.max(52, Math.min(80, (width - labelW - 2 * 86) / (COLS.length - 2)))
  const colW = COLS.map((c) => (c === 'first' || c === 'last' ? 86 : numW))
  const colX = colW.map((_, j) => labelW + colW.slice(0, j).reduce((a, b) => a + b, 0))
  const rowH = 24, head = 30
  const w = colX[colX.length - 1] + colW[colW.length - 1], h = head + table.length * rowH + 4
  return (
    <ChartFrame id="aeth-overview-sites" title="Instruments compared — each daily record and its IR black carbon"
      subtitle={`Per instrument at the sites in view: its daily bins in the span (first and last bin start), the days with IR BCc and their mean, median, SD and maximum (µg/m³), the mean minute coverage (absent at Addis), the site's filters in view and how many carry an aethalometer value (the bins cover at least 12 h of the filter's window). Screen: ${SCREEN_TEXT[screen]}.${onSite ? ' Click a site to look at it alone.' : ''}`}
      provenance="Multi_Site_Analysis.ipynb §7 (summary statistics table); Multi_Site_Analysis_Fixed.ipynb §8 (BC statistics by site) · gallery/data/export_aethalometer.py">
      <div ref={wrapRef} className="chart-wrap" style={{ overflowX: 'auto' }}>
        {!table.length ? <Empty>No instruments at the sites in view.</Empty> : (
          <svg width={w} height={h} fontFamily={FONT.family}>
            {COLS.map((c, j) => <text key={c} x={colX[j] + 4} y={head - 10} fontSize={10.5} fill={INK.muted}>{c}</text>)}
            {table.map((t, i) => (
              <g key={t.s.inst.code} transform={`translate(0,${head + i * rowH})`} style={{ cursor: onSite ? 'pointer' : undefined }} onClick={() => onSite?.(t.s.inst.siteCode)}
                onMouseEnter={(e) => tip.show(e, [t.s.inst.label, `${t.s.inst.info.file}: ${t.s.bins} bins, ${t.s.inst.info.first} to ${t.s.inst.info.last}`,
                  `bin start hours: ${Object.entries(t.s.inst.info.bin_start_hours).map(([hh, n]) => `${hh}:00 ×${n}`).join(', ')}`,
                  ...(t.ir ? [`IR BCc 10–90 %: ${fmt(t.ir.q10, 2)}–${fmt(t.ir.q90, 2)} µg/m³`] : []), onSite ? 'click to look at this site alone' : ''].filter(Boolean))}
                onMouseLeave={tip.hide}>
                <rect width={w} height={rowH - 2} fill={i % 2 ? INK.zebra : 'transparent'} />
                <circle cx={10} cy={rowH / 2 - 1} r={5} fill={instColor(meta, t.s.inst)} />
                <text x={22} y={rowH / 2} dy="0.1em" fontSize={12} fill={INK.text}>{t.s.inst.code === t.s.inst.siteCode ? t.s.inst.site : `${t.s.inst.info.site} (Addis, 2nd)`}</text>
                {t.cells.map((v, k) => <text key={k} x={colX[k] + 4} y={rowH / 2} dy="0.1em" fontSize={11} fontFamily={FONT.mono} fill={INK.muted}>{v}</text>)}
              </g>
            ))}
          </svg>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
