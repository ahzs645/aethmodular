/**
 * One cue for every chart or table that scrolls sideways inside its card (a dense chart on a phone):
 * the side with more content to see fades out (styles.css `[data-more]`). Charts mark nothing
 * themselves; any element inside a `.panel` whose content is wider than its box is picked up,
 * whether it scrolls through `.chart-scroll`, an inline `overflow-x`, or a phone table.
 */
const SCROLLERS = '.panel .chart-scroll, .panel .chart-wrap, .panel table, .panel [style*="overflow-x"], .panel [style*="overflow: auto"]'

function mark(el: HTMLElement) {
  const over = el.scrollWidth - el.clientWidth
  const style = getComputedStyle(el)
  if (over <= 2 || !/(auto|scroll)/.test(style.overflowX)) { if (el.dataset.more) delete el.dataset.more; return }
  const left = el.scrollLeft > 2, right = el.scrollLeft < over - 2
  const more = left && right ? 'both' : left ? 'left' : right ? 'right' : ''
  if (more) el.dataset.more = more
  else delete el.dataset.more
}

export function watchScrollEdges() {
  let frame = 0
  const scan = () => { frame = 0; document.querySelectorAll<HTMLElement>(SCROLLERS).forEach(mark) }
  const later = () => { if (!frame) frame = requestAnimationFrame(scan) }
  new MutationObserver(later).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['width', 'style', 'class'] })
  window.addEventListener('resize', later)
  // scrolling one region only changes that region's edges
  document.addEventListener('scroll', (e) => { const t = e.target; if (t instanceof HTMLElement && t.matches(SCROLLERS)) mark(t) }, { capture: true, passive: true })
  later()
}
