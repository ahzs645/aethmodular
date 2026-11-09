import { isText, itemTab, type Section, type SectionItem } from './sections.ts'

/**
 * A section as slides or pages, in the house deck style (docs/deck-style-prompt.md): one title, one figure, nothing
 * else on the slide, and speaker notes with SAY and NOTES. What the reader wrote stays theirs: a chart's note is what
 * is said over it, and a block of text is a divider, or a text slide if it has a body. Pure, so it is tested without a browser.
 */
export interface Slide {
  kind: 'title' | 'divider' | 'text' | 'chart'
  title: string
  body?: string
  notes: string
  /** the item whose drawing goes on it */
  item?: SectionItem
}

/** No em dashes on a slide (a standing rule for this repo's decks): a chart's "name — clause" reads as "name: clause". */
export const deckText = (s: string) => s.replace(/\s*—\s*/g, ': ').trim()

export function deckSlides(section: Section, link?: string): Slide[] {
  const charts = section.items.filter((i) => !isText(i))
  const roadmap = section.items.filter((i) => i.title).map((i) => deckText(i.title)).join('; ')
  const slides: Slide[] = [{
    kind: 'title', title: deckText(section.name), ...(section.about ? { body: deckText(section.about) } : {}),
    notes: `SAY: ${roadmap ? `Roadmap: ${roadmap}.` : ''}\nNOTES: ${charts.length} chart${charts.length === 1 ? '' : 's'} from the gallery.${link ? ` The section, live: ${link}` : ''}`,
  }]
  for (const item of section.items) {
    const title = deckText(item.title)
    const body = item.note ? deckText(item.note) : ''
    if (isText(item)) {
      if (title || body) slides.push({ kind: body ? 'text' : 'divider', title, ...(body ? { body } : {}), notes: `SAY: ${body || title}\nNOTES:` })
    } else slides.push({ kind: 'chart', title, item, notes: `SAY: ${body}\nNOTES: From the gallery’s ${itemTab(item)} page, at the settings shown.` })
  }
  return slides
}
