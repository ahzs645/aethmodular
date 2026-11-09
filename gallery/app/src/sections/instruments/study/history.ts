import type { InvestigationHistory } from '@/lib/investigationHistory'

/** Curated public evidence for the study page; observations remain in its generated export. */
export const ADDIS_STUDY_HISTORY: InvestigationHistory = {
  schema: 2,
  coverage: 'The Addis study comparison uses the gallery’s saved Jacros hourly IR-BC record and the Shlipak et al. (2026) study profile. Open its graphs to inspect the observation window, coverage and processing limits.',
  chronology: 'The source date records the gallery integration. Observation dates and the saved export’s generation date are reported on the study page. This diagnostic does not establish an exact reproduction of the paper.',
  families: [{
    id: 'addis_study',
    title: 'Addis study comparison',
    purpose: 'Compare the saved Jacros observations with the study’s temporal frame and published context, with explicit measurement and coverage limits.',
  }],
  records: [{
    id: 'shlipak_2026_saved_export',
    title: 'Shlipak et al. (2026) · the saved Jacros record',
    family: 'addis_study',
    source: 'docs/aethalometer-addis-study-gallery.md',
    sourceAvailable: true,
    sourceDate: '2026-11-09',
    tracked: true,
    evidence: 'diagnostic',
    evidenceLabel: 'Saved-export comparison',
    question: 'What temporal patterns can the existing Jacros hourly export show within the paper’s April 2022–March 2025 study window?',
    summary: 'The page summarizes available hourly optical BC into daily, monthly, seasonal and local-hour views, with coverage and festival windows kept visible. Published Central and Jacros values provide descriptive context. The saved observations cover only part of the paper’s period, so these results are not a matched reproduction or independent validation.',
    caveats: [
      'The saved Jacros hourly record runs from 12 April 2022 to 20 August 2024. Coverage is shown both within that saved span and against the full April 2022–March 2025 study window; missing hours do not become zero concentrations.',
      'Central observations are absent. The ETAD and ETAD-JACROS daily files describe the same MA350-0238 over their shared dates and cannot serve as two independent deployments.',
      'The selected saved hourly export contains IR BC only. It cannot recover the paper’s fossil/biomass split; AAE class frequencies count observations and are not source fractions of BC mass.',
      'The paper’s MA350 operated without a size-selective inlet. Its optical BC represents the sampled aerosol stream and does not establish PM2.5 thermal EC or independent validation of the Addis FTIR product.',
      'The page declares its season calendar and festival windows. Its available-hour summaries retain the saved exporter’s processing; the paper’s detailed QC, completeness denominator and climatology exclusions remain unresolved.',
    ],
    charts: [{
      tab: 'aethalometer',
      page: 'addis-study',
      label: 'Addis study comparison',
      primaryExports: ['aethalometer/addis-study.json'],
    }],
  }],
  edges: [],
}
