/**
 * Plain-language definitions for readers outside the FTIR group (15 Oct 2026 practice talk:
 * "most people, even at AAAR, don't know what HIPS is"). The Glossary dialog lists TERMS; axis
 * titles and field pickers show describeField() on hover, so a chart says what its axes are.
 */
export interface Term { term: string; long?: string; text: string }

export const TERMS: Term[] = [
  { term: 'Addis Ababa', text: 'Capital of Ethiopia, about 2,400 m above sea level; the SPARTAN site ETAD. Three seasons: Bega (dry), Belg (short rains) and Kiremt (long rains, the most polluted). Homes burn a lot of wood and charcoal.' },
  { term: 'SPARTAN', long: 'Surface PARTiculate mAtter Network', text: 'A global network of filter samplers, mostly in cities, collecting PM2.5 on Teflon filters at about 5 L/min (about 7 m³ a day).' },
  { term: 'IMPROVE', long: 'Interagency Monitoring of PROtected Visual Environments', text: 'A US network, mostly at remote and rural sites, sampling at about 22 L/min (about 33 m³ a day) on Teflon and quartz filters. Its collocated FTIR and TOR filters are the only library large enough to calibrate FTIR EC anywhere; there is no Addis or urban equivalent.' },
  { term: 'CSN', long: 'Chemical Speciation Network', text: 'The US urban network. Its filters are larger and thicker and sampled at a lower flow, so they have not been usable for FTIR calibration.' },
  { term: 'PM2.5', text: 'Particles smaller than 2.5 µm. "PM2.5 mass" is the filter\'s weighed mass per cubic metre of air.' },
  { term: 'Concentration vs mass per filter', text: 'Concentration is µg per cubic metre of air; mass per filter is µg on the filter (concentration × sampled volume). FTIR only sees what is on the filter. IMPROVE samples about 4× more air per filter than SPARTAN, so equal concentrations mean very different loadings.' },
  { term: 'EC', long: 'elemental carbon', text: 'The soot-like, light-absorbing carbon. Reference values come from TOR on quartz filters; SPARTAN reports EC predicted from FTIR spectra.' },
  { term: 'OC', long: 'organic carbon', text: 'The carbon in organic compounds. OC/EC is low where combustion is strong; Addis has one of the lowest OC/EC ratios of any SPARTAN or IMPROVE site.' },
  { term: 'OM', long: 'organic matter', text: 'OC plus the hydrogen, oxygen and other atoms bonded to it. FTIR estimates it from functional groups; in Addis OM and EC are each about a quarter of PM2.5.' },
  { term: 'BC', long: 'black carbon', text: 'Light-absorbing carbon measured optically. Here HIPS BC = HIPS Fabs / MAC.' },
  { term: 'FTIR', long: 'Fourier-transform infrared spectroscopy', text: 'Shines infrared light through the Teflon filter without destroying it and records how much is absorbed at each wavenumber: the filter\'s spectrum. A calibration turns a spectrum into EC, OC or OM.' },
  { term: 'Spectrum (spectra)', text: 'Absorbance against wavenumber (cm⁻¹) for one filter. Chemical bonds absorb at characteristic wavenumbers (C-H near 2900, Si-O near 1000-1300).' },
  { term: 'Baseline correction of the spectra', text: 'Removing the smooth background (scattering by the filter and particles) from each spectrum before calibrating. The spline baseline (AIRSpec) is the established method; VIBES is an experimental one from Satoshi Takahama\'s group.' },
  { term: 'Calibration (calibration set)', text: 'A PLS model trained on IMPROVE filters that have both an FTIR spectrum and a TOR EC value, then applied to other spectra. The calibration set (cohort) is which IMPROVE filters it learns from.' },
  { term: 'Analog spectra', text: 'IMPROVE spectra that most resemble the Addis spectra (highest correlation across wavenumbers, excluding chosen regions such as CO₂ and PTFE). Used to pick a calibration set that looks like Addis.' },
  { term: 'PLS', long: 'partial least squares regression', text: 'The regression from spectrum to concentration. It compresses the spectrum into a few factors; the number of PLS factors (k) is chosen by cross-validation.' },
  { term: 'Cross-validation', text: 'Testing a calibration on filters it did not learn from. Site-grouped: whole sites are held out together, the harder and more honest test. Interleaved: filters are held out in turn regardless of site.' },
  { term: 'TOR', long: 'thermal/optical reflectance', text: 'The reference method for EC and OC on quartz filters: carbon is evolved by heating in steps (OC1-OC4, EC1-EC3) while a laser watches the filter darken and recover.' },
  { term: 'OP', long: 'optically detected pyrolysis', text: 'Char formed from organic carbon during TOR, subtracted so EC = EC1 + EC2 + EC3 - OP. On very dark filters the laser can saturate and OP can read zero, which may push TOR EC up.' },
  { term: 'HIPS', long: 'Hybrid Integrating Plate and Sphere', text: 'Measures how much red light (633 nm) the Teflon filter absorbs, giving Fabs. Not a carbon measurement; dust and iron also absorb.' },
  { term: 'Fabs', text: 'The absorption coefficient from HIPS, in Mm⁻¹ (per million metres).' },
  { term: 'MAC', long: 'mass absorption cross-section', text: 'm²/g of absorption per mass of BC: BC = Fabs / MAC. The gallery uses 10, a convention that puts Fabs in µg/m³, not a value measured in Addis. A different MAC scales the slope of a fit against HIPS BC; the intercept stays (exactly for OLS, nearly for Deming). Choose it under Filters → units.' },
  { term: 'Slope and intercept', text: 'For predicted against reference, a slope near 1 and an intercept near 0 mean agreement. A high R² with a slope or intercept far off means the two are related, but not in a physically sensible way.' },
  { term: 'Deming, OLS, York', text: 'Fit methods. OLS puts all error in y; Deming allows error in both axes (the default here); York uses each point\'s own uncertainties.' },
  { term: 'Ratio', text: 'A measurement divided by another on the same filter, e.g. EC / PM2.5 mass (the EC fraction of PM2.5) or OC / EC. Presets and your own ratios sit in every field picker under Ratios.' },
  { term: 'XRF, ICP-MS, IC', text: 'How SPARTAN measures elements and ions: X-ray fluorescence and inductively coupled plasma mass spectrometry for elements, ion chromatography for ions (sulfate, nitrate …). Fields marked "(element)" are the XRF element where an ion of the same name also exists.' },
  { term: 'MDL', long: 'method detection limit', text: 'The smallest value the method can tell apart from zero.' },
  { term: 'PMF', long: 'positive matrix factorization', text: 'A statistical split of each filter\'s chemistry into source factors (sea salt, wood, charcoal, fossil fuel …). Available for Addis.' },
  { term: 'ERA5', text: 'A global weather reanalysis; the gallery averages it over each filter\'s sampling window.' },
]

const FIELDS: Record<string, string> = {
  'EC (FTIR)': 'Elemental carbon predicted from the filter\'s FTIR spectrum by the IMPROVE-trained calibration (SPARTAN\'s reported EC).',
  'OC (FTIR)': 'Organic carbon predicted from the FTIR spectrum.',
  'OM (FTIR)': 'Organic matter from the FTIR functional groups.',
  'HIPS BC': 'Light absorption measured by HIPS, divided by the MAC to express it as black carbon (µg/m³).',
  'HIPS Fabs': 'Light absorption (633 nm) of the Teflon filter measured by HIPS, Mm⁻¹.',
  'BC (public HIPS)': 'SPARTAN\'s published BC where HIPS measured it (Fabs / 10).',
  'EC (TOR)': 'Elemental carbon by thermal/optical reflectance on the IMPROVE quartz filter: the reference value.',
  'OC (TOR)': 'Organic carbon by thermal/optical reflectance on the IMPROVE quartz filter.',
  'PM2.5 mass': 'The filter\'s weighed PM2.5 mass per cubic metre of air.',
  'Sampled volume': 'Air drawn through the filter, m³ (SPARTAN about 7, IMPROVE about 33 a day).',
  'OP': 'TOR optically detected pyrolysis: char formed during analysis, subtracted from EC1 + EC2 + EC3.',
  'Alkane CH': 'FTIR functional group: aliphatic C-H bonds.',
  'Alcohol COH': 'FTIR functional group: alcohol C-OH.',
  'Carboxylic COOH': 'FTIR functional group: carboxylic acid COOH.',
  'Non-acid CO': 'FTIR functional group: non-acid carbonyl C=O.',
}

/** one line on what a field is, for axis titles and pickers */
export function describeField(f: string): string {
  if (FIELDS[f]) return `${f}: ${FIELDS[f]}`
  const i = f.indexOf(' / ')
  if (i > 0) return `${f}: ${f.slice(0, i)} divided by ${f.slice(i + 3)} on the same filter.`
  if (/\((ERA5|station)\)$/.test(f)) return `${f}: weather averaged over the filter's sampling window.`
  if (/^(EC[123]|OC[1-4])$/.test(f)) return `${f}: a TOR thermal fraction (carbon evolved at one temperature step).`
  if (f.endsWith('(element)')) return `${f}: the element by XRF/ICP-MS, distinct from the ion of the same name.`
  if (/uncertainty|MDL/.test(f)) return `${f}: bookkeeping for the measurement it names.`
  return f
}
