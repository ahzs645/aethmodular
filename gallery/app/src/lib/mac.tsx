import { createContext, useContext, useMemo, type ReactNode } from 'react'
import { EXPORT_MAC, macView, type MacView } from './macMath'

export * from './macMath'

/**
 * The MAC every chart uses, from one place. The reader sets it under Filters → units (App's `mac`);
 * App provides it here, so a page or chart reads `useMac()` instead of threading meta.mac_value.
 * Meeting pages shown "As presented" get the export's MAC (lib/presentation).
 *
 * What a MAC change does to a crossplot against HIPS BC = Fabs / MAC: x rescales by exportMac / mac,
 * so a fit of y on x keeps its intercept and R² while its slope scales by mac / exportMac. That holds
 * for OLS exactly and for Deming when λ scales with MAC² (the explorer's convention,
 * calibration_modes.deming_lambda), which every exporter follows: checked on calibration_runs.json,
 * whose fits at MAC 6, 10 and 17 obey it to the stored digits. So a statistic exported at MAC 10 can
 * follow the reader's MAC in the browser: slopes through `slope`, λ through `lambda`, HIPS BC through `bc`.
 * The pure helpers live in lib/macMath.ts (node-tested in mac.test.mjs).
 */

const Ctx = createContext<MacView>(macView(EXPORT_MAC))

export function MacProvider({ mac, exportMac, lambda10, children }: { mac: number; exportMac: number; lambda10: number; children: ReactNode }) {
  const value = useMemo(() => macView(mac, exportMac, lambda10), [mac, exportMac, lambda10])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

/** The reader's MAC and what follows from it; MAC 10 outside a provider. */
export const useMac = (): MacView => useContext(Ctx)
