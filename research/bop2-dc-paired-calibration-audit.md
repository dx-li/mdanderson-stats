# Paired-endpoint BOP2-DC exact-grid calibration

This extension calibrates the existing two-endpoint BOP2-DC monitor over
explicit finite grids. It uses the model and rules in the pinned primary paper,
Zhao et al., “Bayesian optimal phase II designs with dual-criterion decision
making,” Sections 2.1.4, 2.2, and 2.3 (`research/raw/BOP2-DC/paper.txt`).

## Joint truth scenarios

Each supplied scenario is a four-cell joint probability vector in the same
order as the paired monitor: `(both, first only, second only, neither)`. The
calibration retains these joint vectors and derives their two margins for
inspection. It never replaces the joint law with independent margins. For
efficacy/toxicity, the first event is efficacy and the second is toxicity.

The futile scenario is explicitly caller-declared. The paper notes that a
futile treatment effect may equal the LRV but need not, so this API does not
impose a further relationship between futile margins and LRVs. Effective truth
must match the design's clinical-go composition: at least one endpoint at or
above CMV for multiple efficacy, and efficacy at or above CMV together with
toxicity at or below CMV for efficacy/toxicity.

## Objectives and constraints

The source defines FGR as terminal go at futile truth, FNGR as no-go at
effective truth, CGR as terminal go at effective truth, and FCR as the maximum
terminal consider probability across the two truths. Interim no-go is included
in FNGR. The optimizer either maximizes CGR or minimizes expected sample size
at futile truth, subject to caller-supplied FGR/FNGR limits and optional FCR
limit. Remaining ties use product/input grid order after the secondary
objective criterion.

The four cutoff/power grids accept either a one-dimensional list of symmetric
scalar controls or rows of two endpoint-specific controls. A grid row with two
different values is the way to specify one asymmetric candidate.

## Computation and limits

Every candidate is evaluated by the existing exact paired forward recursion
under both supplied joint scenarios. The per-call recursion bound remains
`2 * (N + 1)**3 <= 5,000,000`; the total candidate work is also bounded before
enumeration, as are retained per-candidate decision and sample-size summaries.
There is no simulation, random-number stream, Monte Carlo error, or claim of
continuous-grid optimization. The procedure assumes fully observed binary
endpoints at the configured enrollment looks; it does not add calendar-time
accrual or delayed outcome ascertainment.

## Independent verification

`tools/reference_bop2_dc_paired_calibration.R` independently enumerates all
256 four-patient paths under each joint truth. It calculates marginal Beta
tails directly in base R and applies the strict decision rules to each path,
including early absorption. The 48-row fixture covers both endpoint modes,
three joint associations with identical marginal rates, and eight asymmetric
cutoff/exponent combinations at looks 2 and 4.

False-go, false-no-go and false-consider bounds are .35, .5 and .3 in this
reference. Each scenario includes feasible and infeasible candidates. Both
objectives, selected parameter rows, all outcome probabilities, stopping and
sample-size distributions, expected enrollment and derived metrics agree in
`tools/check_bop2_dc_paired_calibration.py`: 1,920 numeric summaries across
both objectives. Efficacy/toxicity selects candidate 1 for CGR and candidate 0
for futile expected sample size. Changing association while preserving the
two marginal rates changes operating characteristics, as it should.

The Python reference comparison took .0971 seconds after imports, with
120.63 MiB peak resident memory and no reported swaps. The R generator and
Python comparison were run sequentially. Three focused worker tests also
passed, including infeasibility and preflight work rejection. The retained
allocation bound accounts for both working arrays and returned owned copies.
No new CI workflow or native optimizer parity is claimed.
