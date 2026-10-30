/** Playwright locators pierce open shadow roots; frameLocator and body.innerText do not. */
export async function walkerWorkspace(page) {
  const editor = page.getByRole('region', { name: 'Chart construction workspace', exact: true })
  await editor.locator('[data-rbd-droppable-id="columns"]').waitFor({ timeout: 60000 })
  if (await editor.locator('iframe').count()) throw new Error('The builder unexpectedly moved into an iframe.')
  const host = editor.locator('.gallery-walker')
  await host.evaluate(el => {
    if (!el.shadowRoot) throw new Error('Graphic Walker no longer exposes the expected open shadow root.')
  })
  return editor
}

/** Drag through the real controls, including scrolling long field lists inside the shadow root. */
export async function dragWalkerField(page, editor, label, channel, position = 'end') {
  const catalog = editor.locator('[data-rbd-droppable-id="dimensions"], [data-rbd-droppable-id="measures"]')
  const name = catalog.getByText(label, { exact: true }).first()
  const id = await name.evaluate(el => el.closest('[data-rbd-draggable-id]')?.getAttribute('data-rbd-draggable-id'))
  if (!id) throw new Error(`Cannot find a drag handle for ${label}. Check the upgraded field-list markup.`)
  const source = catalog.locator(`[data-rbd-draggable-id="${id}"]`)
  const target = editor.locator(`[data-rbd-droppable-id="${channel}"]`)
  await target.scrollIntoViewIfNeeded()
  await source.scrollIntoViewIfNeeded()
  const from = await source.boundingBox(), to = await target.boundingBox()
  if (!from || !to) throw new Error(`Cannot reach ${label} → ${channel}.`)
  await page.mouse.move(from.x + Math.min(60, from.width / 2), from.y + from.height / 2)
  await page.mouse.down()
  await page.mouse.move(from.x + 70, from.y + from.height / 2, { steps: 5 })
  await page.mouse.move(position === 'start' ? to.x + 20 : to.x + to.width - 25, to.y + to.height / 2, { steps: 24 })
  await page.mouse.up()
  await target.getByText(label, { exact: true }).waitFor()
}

export const settleLayout = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
