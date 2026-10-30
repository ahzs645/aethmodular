import { useMemo, type ReactNode } from 'react'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { SSA_BY_WAVELENGTH } from '@/lib/aeronet'
import { fmt } from '@/lib/format'
import { ExcludedNote } from '@/lib/markedRows'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'

import { envelopeOf, type Envelope } from '../../common/optics'
import { ols } from '../../common/stats'
import { line, point, shortSeason } from '../../common/ui'
import { useMarkedPairs } from './marked'

const ENVELOPE_NB = 'research/ftir_hips_chem/addis_06_hips_ptfe_operating_envelope.ipynb'

/** addis_06 c14's six panels, in its order */
const PANELS: { key: string; label: string; value: (e: Envelope) => number | null }[] = [
  { key: 'ec', label: 'EC loading (µg/cm²)', value: (e) => e.ecLoad },
  { key: 'oc', label: 'OC loading (µg/cm²)', value: (e) => e.ocLoad },
  { key: 'ocec', label: 'OC / EC', value: (e) => e.ocEc },
  { key: 'pm', label: 'PM2.5 loading (µg/cm²)', value: (e) => e.pmLoad },
  { key: 'tau', label: 'HIPS tau', value: (e) => e.o.tau },
  { key: 't', label: 'HIPS t', value: (e) => e.o.t },
]

interface Common {
  env: Envelope[]; site: string; relative: boolean; mac: number
  seasons: { name: string; color: string }[]
  /** the page's site and residual pickers */
  controls: ReactNode
}

const residOf = (e: Envelope, relative: boolean) => (relative ? e.relResid : e.resid)
const residLabel = (relative: boolean) => (relative ? '(HIPS BC − FTIR EC) / FTIR EC' : 'HIPS BC − FTIR EC (µg/m³)')
const finite = (v: number | null | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v)

/** Filters → excluded: Mark: the site's excluded filters as envelopes at the page's MAC, for the × only */
function useMarkedEnv(site: string, mac: number): Envelope[] {
  const marked = useMarkedPairs()
  return useMemo(() => marked.flatMap((p) => { const e = p.r.site === site ? envelopeOf(p.r, p.o, mac) : null; return e ? [e] : [] }), [marked, site, mac])
}

/** addis_06 c14: the residual against loading, composition and optical depth, one panel each with its OLS line */
export function ResidualCrossplots({ env, site, relative, mac, seasons, controls }: Common) {
  const colorOf = (e: Envelope) => seasons.find((s) => s.name === e.row.season)?.color ?? INK.muted
  const markedEnv = useMarkedEnv(site, mac)
  const xy = (es: Envelope[], p: (typeof PANELS)[number]) => es.flatMap((e) => { const x = p.value(e), y = residOf(e, relative); return finite(x) && finite(y) ? [{ e, x, y }] : [] })
  const panels = PANELS.map((p) => {
    const pts = xy(env, p)
    return { ...p, pts, marked: xy(markedEnv, p), fit: pts.length >= 5 ? ols(pts.map((q) => q.x), pts.map((q) => q.y)) : null }
  })
  return (
    <ChartFrame id="he-resid-cross" title={`The residual against loading, composition and optical depth — ${site}`}
      subtitle={`Each filter's ${relative ? 'relative' : 'absolute'} residual, HIPS BC (Fabs / ${mac}) − FTIR EC, against six candidate drivers, coloured by season, with the OLS line (scipy linregress) and its r. Optical loading should structure the residual through tau, t or EC loading; a composition artefact (Pandey, Lack, Cappa) through OC / EC or OC loading. The ρ² screen above ranks the same pairs.`}
      provenance={`${ENVELOPE_NB} (c14, Addis residual cross-plots)`} controls={controls}>
      {env.length < 5 ? <Empty>Too few filters at {site} with both HIPS and FTIR EC.</Empty> : <FacetGrid items={panels} minCell={260} render={(p, w) => (
        <div style={{ width: w }}>
          <Crossplot sameScale={false} fit={false} maxSide={w - 72} xLabel={p.label} yLabel={relative ? 'Relative residual' : 'Residual (µg/m³)'}
            title={p.fit ? `n ${p.fit.n} · r ${fmt(p.fit.r, 2)}` : `n ${p.pts.length}`}
            extraLines={line(p.fit, INK.text, `${p.key} OLS`)}
            points={p.pts.map((q) => point(q.e.row, q.x, q.y, colorOf(q.e), p.label, 'residual', 'Research · HIPS envelope', [`tau ${fmt(q.e.o.tau, 3)}`]))}
            marked={p.marked.map((q) => point(q.e.row, q.x, q.y, colorOf(q.e), p.label, 'residual', 'Research · HIPS envelope', [`tau ${fmt(q.e.o.tau, 3)}`]))} />
        </div>
      )} />}
      <Key items={[...seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: 'OLS', color: INK.text, kind: 'line' }]} />
      <ExcludedNote count={markedEnv.length} drawn />
    </ChartFrame>
  )
}

/** addis_06 c22: the residual against the AERONET inversion's SSA on the filter's day, 675 nm by default (nearest HIPS's 633 nm) */
export function ResidualSsa({ env, site, relative, mac, seasons, controls }: Common) {
  const [wl, setWl] = useSearchParam<string>('he.ssa', '675', SSA_BY_WAVELENGTH.map(([w]) => String(w)))
  const field = SSA_BY_WAVELENGTH.find(([w]) => String(w) === wl)?.[1] ?? SSA_BY_WAVELENGTH[1][1]
  const colorOf = (e: Envelope) => seasons.find((s) => s.name === e.row.season)?.color ?? INK.muted
  const ssaOf = (es: Envelope[]) => es.flatMap((e) => { const x = e.row[field], y = residOf(e, relative); return typeof x === 'number' && finite(x) && finite(y) ? [{ e, x, y }] : [] })
  const pts = ssaOf(env)
  const marked = ssaOf(useMarkedEnv(site, mac))
  const fit = pts.length >= 5 ? ols(pts.map((q) => q.x), pts.map((q) => q.y)) : null
  // SSA sits near 1: the x range from just below the lowest value to 1, not from 0
  const lo = pts.length ? Math.floor((Math.min(...pts.map((q) => q.x)) - 0.01) * 20) / 20 : 0
  return (
    <ChartFrame id="he-resid-ssa" title={`The residual against AERONET single-scattering albedo — ${site}`}
      subtitle={`${residLabel(relative)} at MAC ${mac} against the AERONET inversion's SSA at ${wl} nm matched to the filter (Level 1.5 unless the AERONET tab says otherwise), coloured by season, with the OLS line. Pandey et al. (2019) motivate the test: a filter artefact that depends on the aerosol's scattering would make the residual track SSA. The inversion is uncertain below AOD 440 = 0.4, so few days qualify.`}
      provenance={`${ENVELOPE_NB} (c22, AERONET SSA hook) · AERONET fields as on the AERONET tab`}
      controls={<>{controls}<Select label="SSA wavelength (nm)" value={wl} options={SSA_BY_WAVELENGTH.map(([w]) => String(w))} onChange={setWl} /></>}>
      {pts.length < 3 ? <Empty>{pts.length} {site} filters in view have an AERONET SSA at {wl} nm. The AERONET fields load with the sites' stations; try another wavelength or site.</Empty> : <>
        <Crossplot sameScale={false} fit={false} maxSide={420} domain={[lo, Math.max(1, ...pts.map((q) => q.x))]} xLabel={`SSA ${wl} nm (AERONET)`} yLabel={relative ? 'Relative residual' : 'Residual (µg/m³)'}
          title={fit ? `n ${fit.n} · r ${fmt(fit.r, 2)} · slope ${fmt(fit.slope, 2)} per unit SSA` : `n ${pts.length}`}
          extraLines={line(fit, INK.text, 'SSA OLS')}
          points={pts.map((q) => point(q.e.row, q.x, q.y, colorOf(q.e), `SSA ${wl}`, 'residual', 'Research · HIPS envelope', [`tau ${fmt(q.e.o.tau, 3)}`, `OC / EC ${fmt(q.e.ocEc, 2)}`]))}
          marked={marked.map((q) => point(q.e.row, q.x, q.y, colorOf(q.e), `SSA ${wl}`, 'residual', 'Research · HIPS envelope', [`tau ${fmt(q.e.o.tau, 3)}`, `OC / EC ${fmt(q.e.ocEc, 2)}`]))} />
        <Key items={[...seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: 'OLS', color: INK.text, kind: 'line' }]} />
        <ExcludedNote count={marked.length} drawn />
      </>}
    </ChartFrame>
  )
}
