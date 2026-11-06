# Retained scientific dependencies

`airspec_baseline.py` is the unmodified recovered Python port of the APRLssb
segmented smoothing-spline procedure. It was recovered from the user's archived
Python PMF analysis and is the same implementation previously used in aethmodular's
FTIR research. APRLssb's GPL-3.0 license is retained in `APRLssb_LICENSE`.
Do not remove its attribution or treat that component as permissively licensed.
The module is deliberately pinned so an upstream/research change cannot silently
change a released input snapshot. Promote a replacement through a new protocol
version and a numerical comparison, not a silent import of changing research code.

`esat_source` contains the original Apache-2.0-licensed ESAT Python SA controller,
LS-NMF and WS-NMF update modules and necessary utilities from the pinned Leah
branch. `PINNED_SOURCES.json` verifies their exact bytes before loading. The
upstream files are unchanged; `esat_backend.py` implements a separate, explicitly
selected repair. Neither native Rust nor the complete upstream ESAT distribution
is vendored or claimed to have been validated.

Runtime source snapshots include the license texts and pinned source receipts.
