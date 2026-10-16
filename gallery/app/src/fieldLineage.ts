/** Pairs that restate one physical measurement rather than validate it. */
export const RESTATES: Record<string, string> = {
  'HIPS BC': 'HIPS Fabs', // divide by MAC
  'EC (ChemSpec FTIR)': 'EC (FTIR)', // public report of the same FTIR product
  'OC (ChemSpec FTIR)': 'OC (FTIR)', // public report of the same FTIR product
  'BC (public HIPS)': 'HIPS BC', // public Fabs / 10 of the same HIPS measurement
}

/**
 * Restatements kept in the rows but never offered in a picker: the public ChemSpec report
 * of the FTIR carbon is the same product as EC / OC (FTIR) (identical on every SPARTAN
 * network filter; within 0.3 µg/m³ on the focus sites' shared filters), so a crossplot of
 * one against the other only shows R² = 1. Charts that need the public report on purpose
 * (the method Sankey, the ChemSpec composition) still read the row values.
 */
export const HIDDEN_DUPLICATES = ['EC (ChemSpec FTIR)', 'OC (ChemSpec FTIR)']

/** a reader's ratio (lib/viewFields, "num / den") shares a term with its numerator and its denominator */
const ratioParts = (f: string) => { const i = f.indexOf(' / '); return i > 0 ? [f.slice(0, i), f.slice(i + 3)] : [] }

export function sameProduct(a: string, b: string): boolean {
  return RESTATES[a] === b || RESTATES[b] === a || ratioParts(a).includes(b) || ratioParts(b).includes(a)
}

/** Uncertainties and detection limits are bookkeeping columns, not species to correlate or rank. */
export function isBookkeeping(field: string): boolean {
  return field.includes('uncertainty') || field.includes('MDL') || field === 'Sampled volume'
}
