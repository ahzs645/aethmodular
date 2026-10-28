/**
 * The Experimental tab's pages, in menu order (the header's Experimental group, like Meetings).
 * Each page reads its own export under public/data/experimental/ (gallery/data/export_experimental.py
 * for the flag study, gallery/data/export_experiments.py for the rest).
 */
export interface Experiment { key: string; label: string; title: string
  /** left out of the header menu, the Investigations card and the page picker; opens only from a link (#tab=experimental&xp=<key>) */
  hidden?: boolean }

export const EXPERIMENTS: Experiment[] = [
  { key: 'plan', label: 'Test plan', title: 'The 22 Oct protocol review: every test, what it asks and where it stands', hidden: true },
  { key: 'flag', label: '1–2 · What the flag sees', title: 'Ablation ladder, coverage curves and missed failures for the "impacted" flag' },
  { key: 'analogs', label: '3 · Loading-matched analogs', title: 'Calibration neighbours chosen by deposited mass as well as spectral shape' },
  { key: 'attribution', label: '4 · Attribution grid', title: 'Preprocessing × window × pool × k on the same filters, with interactions' },
  { key: 'selector', label: '5 · Error-predicting selector', title: 'Pick the model with the smallest predicted error, or abstain' },
  { key: 'forward', label: '6 · Earlier → later years', title: 'Train on 2020–22, test on 2023, against new-site and random splits' },
  { key: 'fixes', label: 'Quick fixes', title: 'ETAD-0241/0248 identity and the February season calendar' },
  { key: 'lots', label: 'Lot-aware VIBES blanks', title: 'Filter lots in the VIBES blank background: held-out blanks by lot, added peaks, and the lot 241/241a fix (26 Oct)' },
  { key: 'tor', label: 'Adama TOR study', title: 'The Adama TOR pilot against IMPROVE and SPARTAN: where it sits, where it would need to sit, and what counts as a pass' },
]
export const EXPERIMENT_KEYS = EXPERIMENTS.map((e) => e.key)
/** the pages the menu, the Investigations card and the page picker list */
export const LISTED_EXPERIMENTS = EXPERIMENTS.filter((e) => !e.hidden)
/** the page #tab=experimental opens on */
export const DEFAULT_EXPERIMENT = LISTED_EXPERIMENTS[0].key
