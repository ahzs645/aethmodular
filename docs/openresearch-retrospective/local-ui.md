# Local research-history interface

Current as of 2026-09-22. The local extension is based on OpenResearch **0.2.8**.
The live dashboard uses `~/.cargo/bin/orx-research`; the official
`~/.cargo/bin/orx` remains separate. Start the custom dashboard with:

```sh
orx-research --no-telemetry up --no-browser --no-agent
```

Stop the previous dashboard before switching, and check for active runs first.
The installed development build reads the built `ui/dist` directory in
`.local-checkouts/openresearch-ui`; retain that checkout.
An official CLI update cannot overwrite the separate custom executable, but
future upstream releases still need to be merged and tested in the custom fork.

Historical evidence badges are separate from native run status. Catalogs with
family headings have a Relationships view with a searchable record browser and
bounded direct-neighbor maps. Dependencies, related work, qualifications and
revisions retain distinct meanings. Family membership and Git ancestry do not
establish scientific data dependencies.

A Linked native experiments panel connects the historical AIRSpec contract to
its execution project. It reads actual run status from the local API. The
historical record retains its evidence identity; no historical run is fabricated.
The map refits when its panel is resized.

Verification: 199 UI tests, 9 development-slot tests and 897 Rust tests passed
(3 Rust tests ignored), along with formatting, Clippy, localization, typecheck,
style checks and the production UI build. Browser checks covered the map,
evidence labels and native-run link. The 110 retrospective nodes, its completed
release run, and the separate AIRSpec experiment/run matched before/after API
snapshots. Release/Windows builds were not tested.

The consistent database backup, official executable and comparison snapshots are
in `~/.local/share/openresearch/ui-backup-2026-09-22/`. For a UI rollback, stop the
custom dashboard and start the official CLI. Do not restore the database merely
to change the interface: that could discard subsequent work.

Detailed rebuild/update instructions are maintained in
`.local-checkouts/openresearch-ui/LOCAL_HISTORY_UI.md`.
The UI commits are `ea16760` (0.2.8 migration/native links) and `0f88bc6` (map sizing).
The current catalog counts are in [the retrospective overview](README.md), and
the current scientific interpretation is in [the research summary](../current-research-summary.md).
