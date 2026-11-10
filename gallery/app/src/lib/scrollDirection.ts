/**
 * Marks the page while the reader scrolls down (`html[data-scroll="down"]`), so the phone's floating
 * Samples / Plot / Page buttons can step aside instead of covering the chart being read (styles.css).
 * Scrolling up, stopping near the top or reaching the bottom of the page brings them back.
 */
export function watchScrollDirection() {
  let last = window.scrollY
  let frame = 0
  const root = document.documentElement
  const update = () => {
    frame = 0
    const y = window.scrollY
    const atEnd = y + window.innerHeight >= root.scrollHeight - 40
    if (y < 120 || atEnd || y < last - 6) delete root.dataset.scroll
    else if (y > last + 6) root.dataset.scroll = 'down'
    if (Math.abs(y - last) > 6 || y < 120 || atEnd) last = y
  }
  window.addEventListener('scroll', () => { if (!frame) frame = requestAnimationFrame(update) }, { passive: true })
}
