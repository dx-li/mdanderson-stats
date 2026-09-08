# SEQBIN sequential binomial designs

Catalog entry 54 is **partial**. Bayesian boundary construction and exact
operating characteristics, prior mean/effective-sample-size input and frequentist
calibration are implemented, along with compact/verbose boundary tables, complete
numerical reports, study revision and JSON replay. The entry remains partial
pending the final source/manual coverage audit.

Source: [SEQBIN 1.5](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/54),
Barry W. Brown, distributed in `SEQBIN_V1.5.zip`. Original files remain local
research inputs. The Python implementation independently expresses posterior
beta probabilities and forward Bernoulli propagation.

```python
from mdanderson_stats import SeqBinDesign

design = SeqBinDesign(
    50,
    prior=[1, 1],
    null_probability=0.2,
    alternative="greater",
    tail_probability=0.025,
    looks=[10, 20, 30, 40, 50],
)
properties = design.operating_characteristics([0.1, 0.2, 0.4])
print(design.continue_high)
print(properties.rejection_probability, properties.expected_subjects)
```

The beta prior has positive shape parameters `(a, b)`. After `k` events among
`n` subjects, the posterior is Beta(a+k, b+n−k). A greater alternative stops when
Pr(theta ≤ p0 | data) is **strictly less** than `tail_probability`; a less
alternative uses Pr(theta ≥ p0 | data). A two-sided design uses either a common cutoff or a pair
`tail_probability=[low_tail, high_tail]`. Equality continues. Cutoffs and p0
are in (0, 1). If unusually large cutoffs make the stopping regions overlap,
the low side takes precedence, matching the source. It is a posterior tail threshold, **not** the trial's frequentist
significance level; obtain the latter by evaluating rejection probability at p0.

`max_subjects` supports 2–10,000, matching the source interface. Without `looks`,
the trial checks after every subject. Explicit looks must be strictly increasing
positive integers ending at the maximum. Returned read-only continuation bounds
are inclusive: stop low for `k < continue_low`, stop high for
`k > continue_high`. At the final look, paths within those bounds complete
without rejecting. Bounds may include outcomes made unreachable by earlier
stopping; operating characteristics account for that earlier stopping exactly.

The boundary search uses vectorized monotone integer bisection across all looks,
requiring O(L log N) beta evaluations for L looks and maximum N. Direct beta
complement evaluation avoids subtracting a tail from one. Operating
characteristics propagate surviving Bernoulli paths and broadcast over true
probabilities. They use O(B N²) arithmetic and O(B(N+L)) working/output storage
for B probabilities; they do not allocate a subject-by-event-by-probability cube.
This is a deterministic probability recursion, not simulation.

`quit_low` and `quit_high` have the probability input shape followed by the look
axis. They contain unconditional probabilities of stopping at that look on that
side. `complete` is the probability of finishing without rejection.
`expected_subjects` includes both early stops and maximum-size completions.
Conditional expected subject counts are provided separately for each rejection
side; a zero-probability event has undefined conditional expectation (`NaN`).

## Original behavior and validation

The original `bayesian_design` clamps a continuation boundary to [0,n], even if
no event count meets the posterior continuation criterion. That forces an
endpoint to continue under sufficiently strong priors. The default Python
behavior represents an empty continuation set with n+1 on the low side or −1
on the high side. Use `legacy_bounds=True` for the original clamping.

The original `calculate_properties` substitutes the maximum sample size for a
conditional rejection expectation when the rejection probability is at most
1e-8, and uses −1 for the unused side. The Python method computes the conditional
expectation whenever the probability is positive and returns NaN only when it
is zero. Neither method forces final nonrejecting outcomes to reject.

`tools/reference_seqbin.py` builds unchanged boundary and forward-probability
routines with their original beta dependencies in an independent driver. Eighteen
native cases cover three beta priors, all alternatives, sequential/group looks,
continuation boundaries, stopping mass and expected sample size. Fixture hashes
record the source, extracted module, driver and compiler. Comparisons of native
conditional expectations exclude its documented <=1e-8 fallback region.

Independent tests check integer-beta/binomial identities at every count,
threshold equality, strong-prior empty continuation, zero/one true probabilities,
all paths of six-subject trials, batched probabilities, rare-event conditional
expectations, input snapshots, and the 10,000-subject boundary limit. The original full executable's input/report flow is not covered by these core
tests. Calibration validation is described below.


## Prior inputs

Pass `(0.5, 0.5)` to `SeqBinDesign` for the source's noninformative prior,
`(1, 1)` for its uniform prior, or any two positive beta shape parameters.
`seqbin_prior(mean, effective_subjects)` provides the alternative mean/size
entry convention. By default it returns a=m·N and b=(1−m)·N, so a/(a+b)=m
and a+b=N.

`conversion="legacy"` instead uses the executable's a=m·(N+1)−0.5 and b=N−a.
For example, m=0.2 and N=10 gives (1.7,8.3), whose actual beta mean is 0.17.
This differs from the source prompt's description of its mean input. Some small
N values produce a zero or negative shape; Python rejects them instead of
passing an improper beta distribution to the probability routines. Tests verify
both formulas, the resulting beta mean and mass, and invalid boundary cases.

## Frequentist calibration

```python
from mdanderson_stats import seqbin_calibrate, seqbin_calibrate_tails

calibration = seqbin_calibrate(
    50,
    0.05,
    prior=[0.5, 0.5],
    null_probability=0.2,
    alternative="greater",
    selection="conservative",
)
print(calibration.chosen.significance)
design = calibration.chosen.design
print(design.tail_probability)

separate = seqbin_calibrate_tails(50, [0.025, 0.025], prior=[0.5, 0.5])
print(separate.null_properties.quit_low.sum(), separate.null_properties.quit_high.sum())
```

`seqbin_calibrate` varies a common posterior tail cutoff and evaluates exact
forward-recursion rejection probability at the null event probability.
`selection="conservative"` chooses the largest computed attainable level at most
the target; `"nearest"` chooses the closest, selecting the smaller level on a
tie. A two-sided call calibrates total rejection probability, not equal tail errors.

The result contains `chosen`, `lower` and `upper` calibration points, each with
its complete design and achieved `significance`, plus the requested target,
selection rule and number of distinct design evaluations. If the allowed cutoff
range lies entirely below the target, `upper` is None. If it lies entirely above,
`lower` is None; conservative selection then raises, while nearest chooses the
least-rejecting endpoint. `tail_bounds` defaults to (1e-6, 1−1e-6), the original
FIND_STOP search range, and may be changed explicitly.

Unlike the original continuous root finder with finite absolute/relative stopping
tolerances, the Python search bisects ordered binary64 cutoff bit patterns until
the two bracketing cutoffs are adjacent. Thus it does not interpolate an
unattainable significance inside a jump. Repeated integer boundary configurations
reuse their evaluated rejection probabilities. This is a discrete search over the
computed floating-point posterior rules; it is not a claim of exact real-number
beta evaluation. For original endpoint clamping, pass `legacy_bounds=True`.

`seqbin_calibrate_tails` accepts [low, high] target levels. It separately calibrates
the two one-sided designs, combines their cutoffs, and returns both calibration
results, the combined design, and its actual null operating characteristics.
Opposite-boundary stopping can reduce the achieved error on each side below its
one-sided value. The source's separate-tail dialog assigns `cstop_lo` on both
sides even after requesting nearest selection. Python honors the requested
selection; choose conservative to reproduce that source behavior.

Twelve native FIND_STOP fixtures cover all alternatives, sequential/group looks,
and conservative/nearest choices for 50 subjects, a Beta(0.5,0.5) prior and null
probability 0.2. Achieved bracket levels agree; cutoffs are compared at the source
root finder's precision. Separate exhaustive six-subject tests enumerate rational
beta/binomial tail thresholds and all outcome paths to verify every attainable
level independently. Additional tests check out-of-range targets, ties, separate
tail composition, invalid inputs and prior conversion.


## Boundary tables and complete studies

`seqbin_boundary_table(design, compact=False)` returns a read-only table with
columns exposed as `columns` and numerical values in `rows`. The nine columns
are subject count, low continuation boundary, low stopping probability,
cumulative low probability, high continuation boundary, high stopping
probability, cumulative high probability, total stopping probability, and
cumulative total probability. All probabilities assume the null event rate.
Continuation bounds are inclusive; these are not inclusive rejection cutoffs.

Verbose tables include every subject count up to the maximum. At unscheduled
looks, both continuation bounds permit all outcomes, stopping probability is
zero, and cumulative probability stays unchanged. Compact tables omit rows
with exactly zero computed null stopping probability. An entirely nonrejecting
design has a valid empty compact table. Compaction never changes the design.

```python
from mdanderson_stats import SeqBinStudySpecification

study = SeqBinStudySpecification(
    50,
    prior=[0.5, 0.5],
    alternative="two-sided",
    significance=[0.025, 0.025],
    probabilities=[0.1, 0.2, 0.4],
).run()
print(study.report(compact=True))
study.write_report("seqbin.tsv", digits=17, compact=True)
study.write_specification("seqbin.json")
revised = study.revise(max_subjects=60)
```

Specify either `tail_probability` or `significance`; with neither, the tail
cutoff is 0.05. A scalar significance requests common-cutoff calibration, and a
[low, high] pair requests separate calibration with `alternative="two-sided"`.
To change a posterior-threshold study into a calibrated one, explicitly clear
the stored cutoff, e.g. `study.revise(tail_probability=None, significance=0.05)`.
When changing an explicit schedule, supply updated `looks`; an implicit
sequential schedule follows the new `max_subjects` automatically.

The specification stores actual beta shapes, requested calibration settings or
posterior cutoffs, the schedule and up to 101 alternative probabilities. It does
not infer whether beta shapes came from raw input or either mean/size conversion.
`run()` snapshots input arrays independently. `from_json(text).run()` reconstructs
the study; unknown fields and invalid numerical settings raise. JSON is input
data, not executable code. Revision returns a new study and preserves the original.

The report contains all specification settings, actual tail cutoffs, calibration
chosen/lower/upper levels where applicable, the null boundary table, and an
operating-characteristic row for the null and each alternative in input order.
Rows include rejection/completion probabilities, unconditional expected subject
count, side-specific rejection probabilities and conditional expected subject
counts. Undefined conditional expectations print as `NA`. An empty alternative
list produces a null-only study. `properties` has the same probability axis:
null first, followed by the specified alternatives. All scheduled-look stopping
probabilities remain accessible there even when the report is compact.

Reports support 1–17 significant digits; 17 preserves binary64 table values on
numeric parsing. Settings and JSON inputs retain full precision. File methods
write UTF-8, replace the named destination and propagate I/O errors.

### Corrected original report state

The original main program calls COMPACT_TABLE before recomputing the null
operating characteristics, and restores its boundary arrays only before the
alternative-probability loop. The recomputation therefore indexes compacted
boundaries as if they still referred to all subject counts. The Python report
uses display-only compaction and retains the correct null calculations.

Six native cases execute unchanged COMPACT_TABLE and probability routines in an
independent driver. They cover all alternatives and sequential/group schedules;
the compact table rows match. Each also records the correct pre-compaction null
level and the different value obtained by the source's subsequent recomputation.
Regression tests ensure reporting cannot introduce that mutation. The source's
two-sided table also leaves its first per-look total uninitialized; Python
calculates low+high for every row, tested with a design stopping at its first look.
The fixture is a numerical table comparison, not a byte-for-byte terminal-format
comparison. Additional tests parse complete reports, replay all design modes,
revise studies, preserve snapshots, handle empty compact tables, verify unscheduled
verbose rows and exercise serialization/file failures.
