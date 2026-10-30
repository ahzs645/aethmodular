import { useMemo } from 'react'
import { ChartFrame, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Key } from '@/components/Key'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { useSearchParam } from '@/lib/url'
import { INK } from '@/lib/theme'
import type { CalibFile } from '@/lib/types'
import { cohortName, foldedPlane, scoreOf, spectraName, W } from './search'
import { FixedSetNote } from '@/lib/markedRows'

const XMAX = 2, YMAX = 8
const CONTOURS = [1, 2, 3, 4]
const SETS = ['all pairs', 'fixed 190'] as const
/** the configurations ftir_55 starred, as the grid has them (one row per configuration, at the rule k) */
const STARS = [
  { co: 'ocec', cut: 440, sp: 'airspec', label: 'ftir_55’s vetted cohort: Lowest OC/EC 440 × spline', color: () => INK.negative },
  { co: 'ocec', cut: 800, sp: 'airspec', label: 'the locked Lowest OC/EC 800 × spline', color: () => INK.good },
]

/**
 * ftir_55 §2a, the 27 Aug deck's screen, from calibration.json: every site-held-out Addis configuration folded onto
 * |slope − 1| and |intercept|, grey below the held-out TOR R² floor, with the objective's diamond contours.
 */
export function FoldedPlane({ calib, provenance }: { calib: CalibFile; provenance: React.ReactNode }) {
  const mac = useMac()
  const [set, setSet] = useSearchParam<(typeof SETS)[number]>('ml-model-form-folded.set', 'all pairs', SETS)
  const allPairs = set === 'all pairs'
  const pts = useMemo(() => foldedPlane(calib, mac.mac, allPairs), [calib, mac.mac, allPairs])
  const shown = pts.filter((p) => p.dx <= XMAX && p.dy <= YMAX)
  const above = pts.filter((p) => p.floor).length
  const stars = STARS.flatMap((s) => {
    const p = pts.find((q) => q.r.co === s.co && q.r.cut === s.cut && q.r.sp === s.sp && (q.r.sel ?? 'raw') === 'raw')
    return p ? [{ ...s, p }] : []
  })
  const best = pts.filter((p) => p.floor).reduce<(typeof pts)[number] | null>((b, p) => (!b || p.dx * W + p.dy < b.dx * W + b.dy ? p : b), null)
  const tip = (p: (typeof pts)[number]) => [
    `${cohortName(p.r.co)}${p.r.cut != null ? ` ${p.r.cut}` : ''}${p.r.sel && p.r.sel !== 'raw' ? ` (selected on ${spectraName(p.r.sel).toLowerCase()})` : ''} × ${spectraName(p.r.sp)}, k = ${p.r.k}`,
    `Deming ${f2(p.slope)}x ${p.intercept < 0 ? '−' : '+'} ${f2(Math.abs(p.intercept))} (${mac.label}, ${set})`,
    `score ${f2(scoreOf(p.slope, p.intercept))} · held-out TOR R² ${f2(p.r.ho)}`]
  const points: CrossPoint[] = [
    ...shown.filter((p) => !p.floor).map((p) => ({ x: p.dx, y: p.dy, color: INK.neutral, tip: tip(p) })),
    ...shown.filter((p) => p.floor).map((p) => ({ x: p.dx, y: p.dy, color: INK.accent, opacity: 0.55, tip: tip(p) })),
    ...stars.map((s) => ({ x: Math.min(s.p.dx, XMAX), y: Math.min(s.p.dy, YMAX), color: s.color(), tip: [s.label, ...tip(s.p)] })),
  ]
  return (
    <ChartFrame id="ml-model-form-folded" title={`Only ${above.toLocaleString()} of ${pts.length.toLocaleString()} site-held-out configurations keep the held-out floor${best ? `, and the best of them scores ${f2(best.dx * W + best.dy)}, not 0` : ''} — ftir_55 [7], the deck’s folded plane`}
      subtitle={`Every site-held-out Addis configuration in calibration.json (training lot all, at the rule k), folded onto |Deming slope − 1| and |Deming intercept| against Fabs/${mac.mac} (${mac.label}: slope × MAC/10, intercept unchanged), on ${allPairs ? 'all 239 pairs' : 'the fixed 190 filters'}. Blue keeps the held-out TOR R² ≥ ${calib.heldout_floor} on untouched IMPROVE sites; grey falls below it. Dashed lines are the search's objective |intercept| + ${W}·|slope − 1| at ${CONTOURS.join(', ')}: one unit of slope costs as much as ${W} µg/m³ of intercept.${best ? ` Best above the floor: ${cohortName(best.r.co)} ${best.r.cut ?? ''} × ${spectraName(best.r.sp)}, k = ${best.r.k}.` : ''} ftir_55 drew the same screen on every k in the cache (14,834 configurations, 1,480 above the floor, axes to 3.2 and 14); this grid holds one k per configuration.${pts.length - shown.length ? ` ${pts.length - shown.length} configurations lie beyond the axes.` : ''}`}
      provenance={provenance}
      controls={<Segmented label="Evaluation" value={set} options={SETS} onChange={setSet} />}>
      <Crossplot title="" points={points} fit={false} grouping="none" sameScale={false} domain={[0, XMAX]} yDomain={[0, YMAX]} maxSide={520}
        xLabel={`|Deming slope − 1| (Addis, ${mac.label})`} yLabel="|Deming intercept| (µg/m³)"
        background={(x, y) => <g pointerEvents="none">
          {CONTOURS.map((c) => <g key={c}>
            <line x1={x(0)} y1={y(c)} x2={x(c / W)} y2={y(0)} stroke={INK.text} strokeOpacity={0.55} strokeDasharray="4 3" />
            <text x={x(0) + 4} y={y(c) - 4} fontSize={10} fill={INK.muted}>score {c}</text>
          </g>)}
          {stars.map((s) => <circle key={s.label} cx={x(Math.min(s.p.dx, XMAX))} cy={y(Math.min(s.p.dy, YMAX))} r={8} fill="none" stroke={s.color()} strokeWidth={1.6} />)}
        </g>} />
      <Key items={[{ label: `held-out TOR R² ≥ ${calib.heldout_floor} (n = ${above.toLocaleString()})`, color: INK.accent },
        { label: `below the floor (n = ${(pts.length - above).toLocaleString()})`, color: INK.neutral },
        ...stars.map((s) => ({ label: `${s.label} (rule k = ${s.p.r.k})`, color: s.color(), kind: 'hollow' as const })),
        { label: 'objective contours', color: INK.text, kind: 'dash' as const }]} />
      <FixedSetNote what="the explorer's saved fits of every configuration on its own Addis test set" />
    </ChartFrame>
  )
}
