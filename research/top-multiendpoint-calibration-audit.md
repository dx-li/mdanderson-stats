# TOP two-endpoint calibration contract

The published endpoint combination rules are recorded in the
[monitoring audit](top-endpoints-audit.md). A finite-grid Python calibration
must preserve those rules and the supplied association between binary outcomes.
Every scenario therefore specifies all four joint probabilities in order
`(1,1), (1,0), (0,1), (0,0)`, rather than only endpoint margins.

For co-primary efficacy, the null region has both efficacy probabilities at or
below their null thresholds; an alternative has at least one above threshold.
For efficacy/toxicity, the null region has efficacy at or below its threshold
**or** toxicity at or above its threshold. An alternative has efficacy above
and toxicity below its threshold. Calibration at one both-boundary scenario
does not establish control over either composite null region.

## Exact partial-null example

`tools/reference_top_multiendpoint_null_grid.R` enumerates all 455 four-cell
count states for a final-only 12-patient efficacy/toxicity design. Its prior
probabilities are (.05,.15,.25,.55), with concentration one, giving null efficacy
.2 and toxicity .3. Gamma has no effect at the final look.

| Truth scenario | Joint probabilities | Success at C=.8 | Success at C=.95 |
| --- | --- | ---: | ---: |
| Both at boundary | (.05,.15,.25,.55) | .057319368045 | .002122218968 |
| Efficacy at null, toxicity safe | (.02,.18,.08,.72) | .182654914272 | .012788122710 |
| Efficacy effective, toxicity at null | (.15,.35,.15,.35) | .234360321242 | .052102752776 |
| Effective and safe alternative | (.05,.45,.05,.45) | .824225267212 | .403831946287 |

With a .1 error target, C=.8 passes the both-boundary scenario but fails both
partial-null scenarios. C=.95 passes these three null scenarios at the cost
of power. Neither result establishes control over unexamined scenarios or
endpoint associations. The fixture also includes C=.6, for 12 reference rows.
Base R computes the Beta tails and multinomial probabilities independently;
Python enumeration using the public monitoring kernel and SciPy multinomial
probabilities agreed within 1e-13 for all rows.

## Required search and validation behavior

The search accepts an explicit collection of null scenarios and one explicit
alternative. Feasibility uses the maximum estimated success probability across
the supplied nulls. Among feasible candidates, select the largest estimated
alternative success probability; ties prefer smaller worst-null mean enrollment,
then candidate input order. Preserve analysis priors, timing assumptions,
assessment windows, looks and suspension convention while varying C and gamma.

Candidates and scenarios share underlying arrival, joint-cell and event-time
random draws to reduce Monte Carlo noise in comparisons. An independent seed
evaluates the selected design without changing selection. Report probabilities,
Monte Carlo standard errors, enrollment and duration for each scenario at both
stages, with the selected parameters and reproducible seeds. Independent
validation can exceed the target; do not redraw until it passes.

Any result describes only the explicitly evaluated finite scenario grid.
Native optimizer equivalence, arbitrary composite-null error control and
clinical suitability are not established by this calculation. The calendar's
conditional event-time independence and accrual conventions are specified in
the [simulation guide](../docs/top-endpoints-simulation.md).

## Python calibration validation

`tests/test_top_multi_calibration.py` checks the final-only 12-patient example
against the independent exact rows above with a five-Monte-Carlo-standard-error
tolerance. At C=.8, the both-boundary null alone is below .1, but the two
partial-null scenarios are above .1; the optimizer therefore marks C=.8
infeasible and selects C=.95 on the supplied grid. The same test replays the
alternative holdout using `simulate_top_multiendpoint(..., rng=result.validation_seed)`
and matches its probability and mean enrollment exactly.

The public guide's 200-trial example ran with seed 134 and selected
`[C, gamma]=[.95,.5]`; its four holdout success estimates were
`[.005,.015,.055,.395]`, with appreciable Monte Carlo uncertainty. A separate
runtime-budget check sets `max_work` exactly at the one-analysis-per-look
preflight estimate and confirms that repeated final suspension analyses exhaust
the shared scan budget with an explicit error. The focused TOP endpoint,
calendar, and calibration tests passed (10 tests); Ruff, mypy, JSON parsing and
format checks passed. These small checks do not establish composite-null
control beyond the supplied scenarios or native optimizer parity.
