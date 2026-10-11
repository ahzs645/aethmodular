import { useId } from 'react'
import type { ScaleLinear } from 'd3'
import { OCT1_HATCH, SPECTRAL_REGION } from '@/lib/theme'

/**
 * Background shading for spectral regions on every spectrum plot (reversed
 * wavenumber x axis), clipped to the visible range, from one set of region
 * definitions (lib/theme SPECTRAL_REGION). `hatch` marks a region dropped only
 * by the 1 Oct similarity window, so it reads differently from a region every
 * selection drops.
 */
export interface Region { lo: number; hi: number; fill?: string; label?: string; hatch?: boolean }

export const ANALOG_SELECTION_EXCLUDED: Region[] = [SPECTRAL_REGION.co2, SPECTRAL_REGION.aboveCH]
export const OCT1_ALSO_EXCLUDED: Region[] = [{ ...SPECTRAL_REGION.ptfe, hatch: true }, { ...SPECTRAL_REGION.belowOct1, hatch: true }]

export function SpectralRegions({ x, height, regions }: { x: ScaleLinear<number, number>; height: number; regions: readonly Region[] }) {
  const hatchId = useId()
  const [hiWn, loWn] = x.domain() as [number, number]
  const shown = regions.filter((r) => r.hi > loWn && r.lo < hiWn)
  return <g pointerEvents="none">
    {shown.some((r) => r.hatch) && <defs>
      <pattern id={hatchId} width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width={6} height={6} fill={OCT1_HATCH.fill} /><line x1={0} y1={0} x2={0} y2={6} stroke={OCT1_HATCH.stroke} strokeWidth={2} />
      </pattern>
    </defs>}
    {shown.map((r) => {
      const left = x(Math.min(r.hi, hiWn)), right = x(Math.max(r.lo, loWn))
      return <rect key={`${r.lo}-${r.hi}`} x={left} width={Math.max(0, right - left)} height={height}
        fill={r.hatch ? `url(#${CSS.escape(hatchId)})` : r.fill ?? SPECTRAL_REGION.co2.fill}>{r.label && <title>{r.label}</title>}</rect>
    })}
  </g>
}
