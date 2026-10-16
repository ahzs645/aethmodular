import { useCallback, useMemo, type ReactNode } from 'react'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import type { MetaFile } from '@/lib/types'

export interface AxisOption { key: string; group: string; unit?: string; hint?: string }

/**
 * A chart's own x and y choices (principal components, a calibration's prediction, the
 * gallery's measurements …) behind the same clickable axis titles as the gallery's shared x / y
 * (components/AxisFieldPopover). Wrap the plot; give its axes `pick="x"` / `pick="y"`. As
 * everywhere, choosing the other axis's value swaps the two.
 */
export function AxisChoices({ options, x, y, noun = 'choices', children }: {
  options: AxisOption[]
  x: [string, (v: string) => void]
  y: [string, (v: string) => void]
  noun?: string
  children: ReactNode
}) {
  const meta = useMemo(() => {
    const groups: { label: string; fields: string[] }[] = []
    for (const o of options) {
      const g = groups.find((x) => x.label === o.group)
      if (g) g.fields.push(o.key)
      else groups.push({ label: o.group, fields: [o.key] })
    }
    return {
      fields: options.map((o) => o.key), field_groups: groups,
      field_units: Object.fromEntries(options.flatMap((o) => (o.unit ? [[o.key, o.unit]] : []))),
      coverage: Object.fromEntries(options.flatMap((o) => (o.hint ? [[o.key, o.hint]] : []))),
    } as unknown as MetaFile
  }, [options])
  const [xv, setX] = x, [yv, setY] = y
  const value = useMemo(() => ({ x: xv, y: yv, field: '' }), [xv, yv])
  const set = useCallback((role: AxisRole, v: string) => {
    if (role === 'x') { if (v === yv) setY(xv); setX(v) }
    else if (role === 'y') { if (v === xv) setX(yv); setY(v) }
  }, [xv, yv, setX, setY])
  return <AxisFieldsProvider meta={meta} value={value} set={set} ratios={false} noun={noun}>{children}</AxisFieldsProvider>
}
