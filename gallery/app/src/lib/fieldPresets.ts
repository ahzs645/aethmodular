import type { FieldPreset } from '@/components/FieldMultiSelect'
import { isPmfField } from '@/lib/pmfGroups'

/**
 * The named field sets every multi-field chart offers (components/FieldMultiSelect): the
 * correlogram's species, the lollipop's candidates. Each is a function of the fields in view, so
 * a set only ever names fields the chart can draw; a chart prepends its own (a "Top 12 by
 * coverage", an "Everything" that means its candidates) and keeps these in this order.
 */
export const listed = (names: string[]) => (av: string[]) => names.filter((f) => av.includes(f))
export const isWeather = (f: string) => /\((ERA5|station)\)$/.test(f)
export const isAeronet = (f: string) => /\(AERONET\)$/.test(f)

export const FIELD_PRESETS: FieldPreset[] = [
  { key: 'carbon', label: 'Carbon & optical', fields: listed(['EC (FTIR)', 'OC (FTIR)', 'OM (FTIR)', 'EC (TOR)', 'OC (TOR)', 'HIPS BC', 'HIPS Fabs', 'BC (public HIPS)', 'Equivalent BC (SSR)', 'PM2.5 mass']) },
  { key: 'ions', label: 'Ions & mass', fields: listed(['PM2.5 mass', 'Sulfate', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Nitrite', 'Phosphate', 'Bromide']) },
  { key: 'dust', label: 'Dust & metals', fields: listed(['Silicon', 'Aluminum', 'Iron', 'Titanium', 'Calcium (element)', 'Potassium (element)', 'Manganese', 'Soil', 'Zinc', 'Copper', 'Lead']) },
  { key: 'combustion', label: 'Combustion tracers', note: 'EC, OC and BC with potassium, zinc, lead and the combustion PMF factors',
    fields: listed(['EC (FTIR)', 'OC (FTIR)', 'HIPS BC', 'EC (TOR)', 'Potassium', 'Zinc', 'Lead', 'Wood Burning share (PMF)', 'Charcoal share (PMF)', 'Fossil Fuel Combustion share (PMF)']) },
  { key: 'groups', label: 'FTIR functional groups', fields: listed(['Alkane CH', 'Alcohol COH', 'Carboxylic COOH', 'Non-acid CO', 'OM (FTIR)', 'OC (FTIR)', 'EC (FTIR)']) },
  { key: 'pmf', label: 'PMF sources', note: 'the PMF factor shares with EC and BC', fields: (av) => [...listed(['EC (FTIR)', 'HIPS BC'])(av), ...av.filter(isPmfField)] },
  { key: 'weather', label: 'Weather', note: 'EC, BC and mass with the weather over each filter', fields: (av) => [...listed(['EC (FTIR)', 'HIPS BC', 'PM2.5 mass'])(av), ...av.filter(isWeather)].slice(0, 14) },
  { key: 'aeronet', label: 'AERONET column', note: 'BC, EC and mass with the sun photometer over each filter', fields: (av) => [...listed(['HIPS Fabs', 'HIPS BC', 'BC (public HIPS)', 'EC (FTIR)', 'PM2.5 mass'])(av), ...av.filter(isAeronet)].slice(0, 16) },
  { key: 'chem', label: 'Filter chemistry only', note: 'every filter measurement, without the PMF shares, weather and AERONET',
    fields: (av) => av.filter((f) => !isPmfField(f) && !isWeather(f) && !isAeronet(f)) },
  { key: 'all', label: 'Everything', note: 'every field with enough values', fields: (av) => av },
]
