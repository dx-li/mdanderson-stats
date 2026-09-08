# SEQBIN sequential binomial designs

Catalog entry 54 is **partial**. Bayesian boundary construction and exact
operating characteristics are implemented. Frequentist calibration to a desired
significance level, prior mean/effective-sample-size input, compact boundary
tables, full study reports and interactive
revision equivalents remain pending.

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
expectations, input snapshots, and the 10,000-subject boundary limit. Calibration
and the original full executable's input/report flow are not covered by this
increment.
