import { fmt } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { stripPosition, type Placement } from '@/lib/referencePopulation'

/** English ordinal suffix: 1st, 2nd, 3rd, 11th, 51st. */
export const ordinal = (n: number) => (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th')

/**
 * One measurement (or ratio) of a filter against a reference population, as a drawer table row: the value,
 * the reference's range as a strip with its interquartile band and median, the filter's dot, and its
 * percentile. A value outside the range sits at the strip's edge and says so. `table.placement` rows.
 */
export function PlacementRow({ label, p, unit, color, referenceLabel, hot = false, title }: {
  label: string; p: Placement; unit: string; color?: string; referenceLabel: string; hot?: boolean; title?: string
}) {
  const s = (value: number) => stripPosition(value, p)
  const outside = p.v < p.min ? 'below' : p.v > p.max ? 'above' : null
  return (
    <tr className={hot ? 'hot' : ''}>
      <td className="f" title={title}>{label}</td>
      <td className="v">{fmt(p.v, 3)} <span className="u">{unit}</span></td>
      <td className="strip" title={`${referenceLabel}: min ${fmt(p.min, 2)} · q1 ${fmt(p.q1, 2)} · median ${fmt(p.med, 2)} · q3 ${fmt(p.q3, 2)} · max ${fmt(p.max, 2)} · n=${p.n}${outside ? ` · filter is ${outside} the reference range; dot shown at the edge` : ''}`}>
        <span className="track" />
        <span className="iqr" style={{ left: `${s(p.q1)}%`, width: `${Math.max(1, s(p.q3) - s(p.q1))}%` }} />
        <span className="med" style={{ left: `${s(p.med)}%` }} />
        <span className={`dot${outside ? ' outside' : ''}`} data-outside-range={outside ?? undefined} style={{ left: `${s(p.v)}%`, background: color ?? INK.text }} />
      </td>
      <td className="pct">{fmt(p.pct, 0)}<span className="u">{ordinal(Math.round(p.pct))}</span>{outside && <span className="u" style={{ display: 'block' }}>{outside} range</span>}</td>
    </tr>
  )
}
