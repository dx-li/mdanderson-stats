# PerfectMatch quantile-profile source audit

The cached PerfectMatch Manual 2.3 is pinned in `docs/perfectmatch-sources.json`
(SHA-256 `e84bd6578e0bf95ac60b1d32361ce7ec86764a133983502913a38d2b186b95b0`).
Section 2, printed page 2, specifies a Quantile File with per-CEL sorted
intensity values at 2%, 25%, 50%, 75%, and 98%, intended for visual comparison
of chips. It does not give a percentile interpolation/rank rule or a numerical
threshold for labeling an array erroneous. The Python profile therefore uses
linear interpolation explicitly and returns values without classifying arrays.

Existing `quantile_normalize` already implements average-rank quantile
normalization and a supplied reference profile. This addition supplies the
manual's separate per-array QC summary on decoded intensities. It does not
parse CEL/binCEL files, reproduce the native Quantile File format, or infer
the documented-broken `ScalingFactor` and `Absent genes` fields.
