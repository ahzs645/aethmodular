/**
 * A spectrum from a text file: two numeric columns, wavenumber (cm⁻¹) then absorbance, separated by
 * commas, tabs, semicolons or spaces. Header and comment lines are skipped. Rows come back sorted by
 * falling wavenumber, as the gallery draws them.
 */
export interface ParsedSpectrum { wn: number[]; y: number[]; skipped: number }

export const MIN_POINTS = 50

export function parseSpectrumText(text: string): ParsedSpectrum {
  const rows: [number, number][] = []
  let skipped = 0
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue
    const cells = line.trim().split(/[,;\t]+|\s+/).filter(Boolean)
    const a = Number(cells[0]), b = Number(cells[1])
    if (cells.length < 2 || !Number.isFinite(a) || !Number.isFinite(b)) { skipped++; continue }
    rows.push([a, b])
  }
  if (rows.length < MIN_POINTS) throw new Error(`found ${rows.length} rows of two numbers; a spectrum needs at least ${MIN_POINTS}`)
  rows.sort((p, q) => q[0] - p[0])
  for (let i = 1; i < rows.length; i++) if (rows[i][0] === rows[i - 1][0]) throw new Error(`wavenumber ${rows[i][0]} appears twice`)
  return { wn: rows.map((r) => r[0]), y: rows.map((r) => r[1]), skipped }
}
