# Dashboard review Site

[Open the published private Site](https://aeth-dashboard-review.ahzs6455.chatgpt.site).
It hosts the working dashboard, 63 verification screenshots, the test report and a downloadable review package.

The landing page, stylesheet and rendered verification report are retained here. Build the static
bundle from the gallery source and committed evidence with:

```sh
cd gallery/app
npm ci
cd ../..
python gallery/site/build.py
python -m http.server 4190 --directory gallery/site/dist
```

The output has the overview at `/`, the working app at `/dashboard/`, and evidence at `/review/`.
The dashboard starts on its real bundled focus-site data. The app is built with
`VITE_PUBLIC_BUILD=1`; data passes through the existing publisher's scrubbing and private-section
exclusions. Generated bundles stay ignored. `--out` selects another output directory.

`.openai/hosting.json` records the already-created Site identity. It contains no credentials; reuse
that ID for future Sites updates. Building does not publish. The live Site uses its separate Sites
source repository and remains owner-private; this PR does not change its audience or add deployment automation.

The rendered report and screenshot gallery document the verified implementation. Re-run the gallery's
browser suite and update the evidence/report together when changing their claims or screenshots.
