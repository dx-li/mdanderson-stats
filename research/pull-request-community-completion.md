# Recover mTPI final estimation and audit community completion

Proposed base: `main`. Head: `codex/community-completion-2026-10-12`.

This branch preserves the full unmerged `codex/native-software-coverage`
checkpoint. Its earlier [prepared description](pull-request-native-software-coverage.md)
records the FLECS90, Multc99/Multc Lean, interaction-index and BayesFactorTTE work.

The additional change exposes the official mTPI R final estimator as a separate
Python API. It uses the recovered Beta(.005,.005) prior, inverse posterior-variance
isotonic weights, historical candidate limits and dose-index offsets. The existing
paper-based trial simulation remains unchanged. Original-R synthetic references
and independent exact-rational isotonic checks validate the new calculation.
Degenerate native indexing is corrected explicitly; native trial-controller and
Excel/XLL equivalence are not claimed.

Live catalog membership still matches all 138 preserved entries. Coverage remains
93 implemented, 39 partial and six pending. Updated guides, 19 repaired guide
links, contributor instructions and a per-application report make those boundaries
reviewable. A separate licensing review maps explicitly restricted adaptations
and packaging dependencies; upstream restrictions remain in force.

See the [continuation report](continuation-2026-10-12.md#validation) for the final
regression, original-reference regeneration, lint/types/build and clean-wheel
checks. Local checks do not imply GitHub CI for this head. This file is a draft
description, not a created PR or authorization to merge or publish a release.
