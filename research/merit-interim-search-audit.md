# MERIT interim sample-size search audit

The implementation follows the source scope recorded in
[`docs/merit-sources.json`](../docs/merit-sources.json): Yang et al. v2 sections
2.2–2.5, Table 2, and the public interim help already summarized by
[`docs/merit.md`](../docs/merit.md). It does not claim parity with the native
application's current implementation.

## Source-defined pieces

- The fixed-size design uses final integer toxicity and efficacy boundaries.
- The paper's ordered null corners have no truly admissible dose; alternative
  corners define an acceptable interval. Power I is selection of at least one
  acceptable dose with no unacceptable dose selected. Power II is selection of
  at least one acceptable dose.
- Interim monitoring uses posterior tail probabilities at explicit patient
  counts. `MERITInterims.boundaries` derives the equivalent integer event-count
  thresholds from the Beta posterior. Its existing policy uses raw arm counts,
  explicit prior/targets/cutoffs, and strict probability comparisons.
- The fixed-size search already evaluates all integer final-boundary pairs by
  inclusion-exclusion, retaining no trial-by-boundary tensor.

## Python search and conduct policies

- The caller supplies `MERITInterims`; the search requires its toxicity and
  efficacy targets to equal the corresponding acceptable alternative rates
  within `1e-12`. This makes the interim posterior thresholds and corner model
  use the same target values.
- Candidate n begins at one above the latest interim look. The first n meeting
  estimated constraints is returned. Final boundaries range over every integer
  pair from zero through n.
- An arm stops permanently after either interim event. Enrollment is not
  reallocated. Stopped arms are excluded from final isotonic pooling; each
  surviving arm can reach the candidate n. This matches the package's existing
  interim trial convention.
- Candidate ties use the existing fixed-size search ordering: greater chosen
  global power, lower global type-I error, smaller toxicity maximum, greater
  efficacy minimum. Corner simulation uses common latent-normal draws across
  scenarios and candidate n to reduce comparison noise.
- Reported corner MCSEs are plug-in binomial errors at the selected candidate;
  they do not account for selecting n and boundaries using the same simulated
  data. Independent validation remains appropriate for consequential designs.

## Bounded computation

Each scenario keeps only cumulative patient/toxicity/efficacy counts and an
active-arm mask of shape `(trials, doses)`. At each candidate n, current
survivor patterns are grouped and the existing inclusion-exclusion histogram
evaluator computes all final boundary probabilities. The search retains the
candidate's corner-by-boundary summaries, not patient histories or a
trial-by-boundary tensor. Preflight estimates state bytes, summary and
evaluation scratch, trial/scenario updates, histogram scans, and boundary-grid
cumulative work before drawing random outcomes.

The fixed-size Power I evaluator also now subtracts integer event counts
before dividing by the replicate count. This removes last-bit subtraction
differences between empirical probabilities at constraint and tie boundaries.
The mathematical event and corner definitions are unchanged.

The work preflight includes trial-level histogram scans and the sum of
nonempty subset counts across survivor patterns (`3**d - 2**d`), plus grid
cumulative passes. Its default 500-million-unit ceiling accommodates the
two-dose default search horizon and trial count while bounding larger requests.
The estimate measures computational work, not seconds; the scratch estimate
bounds working arrays, not total process RSS.

## Independent integration checks

Eleven affected MERIT tests pass, including boundary-event enumeration and
no-interim reduction. The worker run took 2.17 seconds, peaked at 146.86 MiB
and reported zero swaps; targeted Ruff and mypy pass.

A separate comparison against published `78d319f` reproduces its no-interim
1,000-trial search result `(n,mT,mE)=(30,7,9)` and all reported corner rates
within 1e-15. A nonempty interim schedule selects `(21,6,7)` in a separate
2,000-trial scenario. Replaying each of its six null and three alternative
corners through the existing trial runner reproduces all selected-boundary
probabilities exactly, and matches mean enrollment and its MCSE. These are
checks of empirical event accounting, not fresh-data guarantees of the
selected design's underlying error or power. The comparison takes 0.223 seconds,
peaks at 127.23 MiB and reports zero swaps.

After the final work-budget accounting and real-input guard, the four focused
interim-search checks pass again with warnings treated as errors: 2.22 seconds,
157.17 MiB peak RSS and zero swaps.
