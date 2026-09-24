/** Pairs that restate one physical measurement rather than validate it. */
export const RESTATES: Record<string, string> = {
  'HIPS BC': 'HIPS Fabs', // divide by MAC
  'EC (ChemSpec FTIR)': 'EC (FTIR)', // public report of the same FTIR product
  'OC (ChemSpec FTIR)': 'OC (FTIR)', // public report of the same FTIR product
}

export function sameProduct(a: string, b: string): boolean {
  return RESTATES[a] === b || RESTATES[b] === a
}

/** Uncertainties and detection limits are bookkeeping columns, not species to correlate or rank. */
export function isBookkeeping(field: string): boolean {
  return field.includes('uncertainty') || field.includes('MDL')
}
