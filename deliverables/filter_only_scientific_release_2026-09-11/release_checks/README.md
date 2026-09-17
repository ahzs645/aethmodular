# Release checks

The frozen filter-only analyses were run in a newly created Python 3.13.9
virtual environment outside the original checkout, from `/tmp`, after installing
only the pinned release requirements. The CLI entry point and the complete
portable notebook both passed. The notebook used an explicitly isolated kernel
with its working directory inside the detached release notebook folder.

- [Numerical reproduction](fresh_environment.json): original predictions agreed
  within 1e-10 tolerance; split identifiers and paired test memberships matched.
- [Notebook execution](notebook_execution.json): seven executed code cells,
  five embedded graphs and zero saved errors.
- [Tested files](tested_numerical_files.json): the final numerical sources,
  frozen inputs and environment pins match the detached copy used in the check.
- [Reader links](reader_links.json): release-relative report and notebook links
  resolve without the original checkout.
- [Word visual review](visual_review.json): all nine rendered pages inspected.

This repeats the specified downstream analyses from frozen inputs. It is not an
independent validation of the scientific measurements or a reconstruction of
upstream FTIR predictions from spectra. Draft layout changes after execution
changed only authoring code; the tested numerical files remain identical.
