# Graphic Walker integration and upgrades

The optional **Chart builder** uses the npm React component in the gallery's
page. Graphic Walker creates an open shadow root for its styles; there is no
iframe, separate application server, or copied upstream source. Enable it in
**Display and data settings → Interface → Free-form chart builder**, then open it.
The editor bundle loads only when the enabled builder is opened.

## Integration boundaries

| File | Responsibility |
| --- | --- |
| `src/components/chartBuilder/GraphicWalkerAdapter.tsx` | The only runtime import of Graphic Walker; public component props and `storeRef.exportCode()` / `importCode()` APIs; gallery-facing `exportCharts()` / `importCharts()` handle. |
| `src/components/chartBuilder/graphicWalkerTheme.ts` | Gallery palette and typography via public theme props, plus scoped CSS inside the shadow root for framing and mobile field lists. This CSS is the deliberate upstream DOM dependency. |
| `src/components/chartBuilder/GraphicWalkerWorkspace.tsx` | Download/import actions and a bounded session cache of chart definitions, separated by field/unit schema. |
| `src/lib/chartBuilder.ts` | Conversion of already-filtered gallery rows and metadata; stable field identifiers, units, missing values, category colours and versioned draft validation. |
| `src/lib/graphicWalkerContract.json` | Exact reviewed dependency version and the upstream draft versions the adapter can read. |
| `src/pages/ChartBuilderPage.tsx`, `src/lib/chartBuilderPrefs.ts` | Lazy loading, error boundary and the optional interface setting. |

The adapter saves charts during layout cleanup through the public store API,
before upstream unmount cleanup clears the store reference. It does not listen
to private Graphic Walker edit events. React owns the scoped style element
inside the shadow root, so theme integration also cleans up on unmount.

The gallery still owns network/site/date/season/exclusion filters and unit/MAC
transformations. The builder receives those resulting rows and declared fields.
Drafts contain chart definitions and selection provenance, including filter IDs,
but no measurement rows. Importing a draft applies it to the current selection;
it does not restore measurements or filters. No server computation or AI endpoint
is configured. Upstream branding and licenses remain visible. The upstream
package requests a Leaflet stylesheet even when map charts are not used.

## Checks

Run from `gallery/app` with Node 22.18+ or Node 24 (the unit tests load TypeScript
directly using Node's type stripping):

```bash
npm run check:walker
npm run typecheck
npm run test:walker
npm run build
```

`check:walker` runs offline and also runs before the production build. It checks
the exact version in the manifest, lockfile, installed package and reviewed
contract, installed peer dependencies, and the presence of a frozen chart for
every accepted draft version. A package update alone intentionally fails this
check; changing the contract is a declaration that the adapter has been reviewed,
not proof of compatibility. Run the browser check as well.

In one terminal, serve the production build:

```bash
npm run preview -- --host 127.0.0.1 --port 4173
```

In another terminal:

```bash
npx playwright install chromium
npm run test:walker:browser -- --base-url http://localhost:4173 --output /tmp/walker-check
```

The install command is unnecessary if Playwright Chromium is already installed,
or if `/usr/bin/chromium` or `/usr/bin/chromium-browser` exists. For another
installation, pass `--browser-path /absolute/path/to/chromium` to the browser
check. Omit `--output` to skip screenshots and traces. `--base-url` also accepts a
running Vite development server.

The browser check uses bundled focus-network data and exercises opt-in/lazy
loading, real field drags, two measures with site facets and season colour,
rendered SVG marks, tab retention, draft import/export and rejection, old saved
charts, gallery filtering, light/dark appearance, responsive widths, zero rows,
and disabling/reloading. `scripts/graphic-walker-browser.mjs` centralizes the
upstream control selectors. Playwright locators cross open shadow roots;
`frameLocator`, XPath and reading only `document.body.innerText` do not operate
this editor correctly. The check fails explicitly if the expected shadow root
or framing changes.

## Updating upstream

1. Review the candidate's release notes, peer dependencies, component props,
   public store API, chart serialization and licenses. **0.4.82** is the reviewed
   version for this React 18 gallery. Releases from 0.4.83 require React 19; treat
   that framework migration as separate work and do not bypass peer checks with
   `--force` or `--legacy-peer-deps`.
2. Install the candidate with an exact version and commit the resulting lockfile
   with the adapter changes. Keep upstream-specific changes in the adapter and
   theme files. If field IDs or unit handling change, provide a draft migration
   rather than silently reinterpreting existing charts.
3. Update `testedVersion` in `graphicWalkerContract.json` and include it in
   `readableDraftVersions`. Keep previously accepted versions while their frozen
   charts still pass. If an old format needs migration, implement and test that
   migration before declaring it readable; removing a version explicitly stops
   accepting drafts from that version.
4. Preserve `scripts/fixtures/graphic-walker-<old-version>.json` unchanged. Add a
   fixture for the candidate using the same site-faceted, two-measure, season
   colour chart built in its UI. Download a draft and save only
   `{ "graphicWalkerVersion": "<version>", "chart": <draft.charts[0]> }` in the
   new fixture. These fixtures contain no measurements or filter IDs. Recreating
   an old fixture with the new release would conceal serialization regressions.
5. Run all checks above against the production build. Inspect the desktop and
   mobile screenshots and trace. Confirm the controls remain usable, the gallery
   theme and upstream attribution remain visible, and saved chart encodings and
   rendering survive. Update the isolated selectors/CSS when markup changes.
6. Update `public/licenses/graphic-walker.txt` from the candidate's pinned source
   if its license terms changed, and document any remaining limitations here.

This is a tested upgrade boundary, not a promise that arbitrary upstream releases
are interchangeable. The optional setting keeps the existing gallery usable
while a new editor release is being adapted.
