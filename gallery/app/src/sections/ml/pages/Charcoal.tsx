import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** Charcoal archive: structure, analog matching and classifiers (char_02, char_03, char_04, char_06, char_11): being built. */
export function Charcoal(_props: SectionPageProps) {
  return <PagePending id="ml-charcoal" title="Charcoal archive: structure, analog matching and classifiers" notebooks="char_02, char_03, char_04, char_06, char_11"
    plan="PCA, KMeans and Ward structure of the charcoal reference archive, Addis analog matching and the feedstock classifiers." />
}
