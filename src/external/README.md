# Vendored calibration module

`calibration.py` is vendored third-party aethalometer calibration and
data-processing code. It was copied from the local external file
`PKL data cleaning pipeline/calibration.py` into `src/external/` in git history
commit `8df009f` (`pkl updater`, 2025-07-13). The integration record described
the source only as external and did not record an upstream repository or
version. The file imports and embeds pieces identified as `aethpy`, but an
exact aethpy upstream and version cannot be established from this repository.

Do not edit `calibration.py` locally except for documented patches. Preserving
its original formatting and structure keeps future comparison with an upstream
copy possible. The known retained local patch is its top-level
`from tqdm import tqdm` import.

The file is intentionally excluded from Ruff and retains targeted per-file
ignores because it mixes tabs and spaces, uses compact legacy statements and
global import guards, and carries duplicate definitions. Reformatting or
lint-driven rewrites would destroy upstream diffability; lint applies normally
to this package's local documentation and wrapper code.
