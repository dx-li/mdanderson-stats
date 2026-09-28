# STPLAN exact discrete significance planning

STPLAN 4.5 exposes inverse significance calculations through `QBIN1` and
`QPOI1` with `IWHICH=4`. The archived source and provenance are recorded in
[stplan-sources.json](../docs/stplan-sources.json). `CRBIN1` and `CRPOI1`
construct one-sided, nonrandomized critical regions. `SBIN1` and `SPOI1`
calculate their attained significance; the corresponding P routines compute
power under the alternative.

When the alternative is below the null, the rejection region is `X<=k`.
When it is above the null, the region is `X>=k`. For a fixed sample size or
exposure, selecting the smallest attained significance meeting a requested
power therefore means selecting the smallest qualifying lower-tail bound,
or largest qualifying upper-tail bound. The discrete region is the essential
result; a nominal alpha lying between attainable tail probabilities does not
identify an additional test. Equality of null and alternative does not specify
a direction and must not silently become a meaningful alternative.

The native inverse uses a transformed normal approximation to initialize a
bounded monotone solve, then increases significance when necessary and
recomputes attained significance and power. Its tolerance and subsequent
quantile rounding are numerical conventions, not a reason to apply a continuous
root solver to a stepwise power function in Python. Direct integer-boundary
selection can preserve the statistical test while making attainment explicit.

## Executed native reference

`tools/reference_stplan_discrete_significance.f90` calls the unchanged original
routines. It was compiled serially with the existing gfortran installation,
using the same legacy flags and dependency sources as the earlier discrete
probe. Build command, source list and output are retained under ignored
`research/raw/STPLAN/probe-discrete-significance`.

`tests/fixtures/stplan-discrete-significance.csv` has eight cases: four binomial
and four Poisson, covering both effect directions, high required significance
and an unattainable binomial case. Seven native solves returned success. For
example, n=40 with binomial probabilities .2 versus .4 and target power .8 gives
`X>=13`, attained alpha .0432416223763 and power .8714903219293. Poisson rates
2 versus 1 over 12.5 units give `X<=15`, alpha .0222930213074 and power
.8060290010444.

With n=2, binomial probabilities .2 versus .4 and target power .8, the native
routine reports failure at its upper significance bound .99999999. Attaining
the target would require rejecting the whole support, with alpha one. The
failed row retains native output arguments for audit; its `power` field is
the unchanged requested input, **not achieved power**, and its `critical` is
only the forward boundary at the failed search limit. Consumers must check
`ok` and `status` before interpreting solved quantities.

These references verify the native statistical planning contract. They do not
establish automatic branch discovery for other STPLAN inverses, K-group integer
allocation or native session/report compatibility.

## Python integration

The public binomial and Poisson planners now select critical regions in integer
space and return actual significance, power, the source significance limit and
an explicit attainment flag. All eight native reference rows were checked;
the failed binomial case returns the actual best allowed region (alpha .36,
power .64), rather than the native stale power field.

Review found two Poisson edge cases: a fixed support bracket truncated tiny
requested powers, and positive rate/exposure products could underflow to zero.
Adaptive CDF/SF bracketing and an explicit arithmetic failure address these.
A subsequent review caught exclusion of the feasible one-event upper region;
the bracket now starts at the always-infeasible zero-event cutoff. Its regression
uses the independent identity `P(X>=1)=-expm1(-mean)`.

Six focused significance tests pass, including all implemented regression cases.
Both public guide examples execute, and additional targets 1e-30 and 1e-100
attain the requested power while the next cutoff does not. Focused Ruff,
formatting and mypy checks pass. Checks use one numerical thread and small
scalar calculations; no large simulation or CI expansion was introduced.
