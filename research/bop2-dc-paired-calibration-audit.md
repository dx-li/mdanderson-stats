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
