# BOP2-DC survival finite-grid calibration

This module implements a bounded Monte Carlo finite-grid selector for the
existing time-to-event BOP2-DC monitor. It does not claim continuous
optimization or control of the true error rates from finite simulation.

## Source contract

The pinned primary source is Zhao, Li, Liu, and Yuan, “Bayesian optimal phase
II designs with dual-criterion decision making,” cached as
`research/raw/BOP2-DC/paper.txt`. Section 2.1.3 defines event time as
exponential with mean `theta/log(2)`, where `theta` is median survival, and
places an inverse-gamma prior on the exponential mean. With `d` events and
total observed time `T`, the posterior is
`IG(prior_shape + d, prior_scale + T)`. The implementation reuses
`BOP2DCSurvivalDesign.monitor`, so posterior tails and strict equality behavior
come from the already source-checked monitor rather than a second rule copy.

Section 2.2.3 (printed p. 11) defines FGR as final go at the futile truth,
FNGR as no-go at the effective truth, CGR as final go at the effective truth,
and FCR as the larger final-consider probability at either truth. The calibrated
objectives are (i) maximize CGR controlling FGR and FNGR, and (ii) minimize
expected sample size under futile truth controlling the same errors. Section
3.1 also uses an upper FCR limit in its simulation example. The caller supplies
the two truths, all four cutoff/power grids, prior, looks, calendar, and limits;
no truth or clinical threshold is silently inferred. The effective median is
required to be at least CMV, matching the source's definition of effective
truth; the futile median remains explicit and must be smaller.

The implementation counts interim `stop_no_go` together with terminal
`final_no_go` in FNGR, while FGR and CGR are probabilities of terminal
`final_go`; FCR uses terminal `final_consider` only. This follows the paper's
decision definitions. The optional FCR constraint is applied to the maximum
consider rate across the two truth scenarios.

## Simulation and selection

For each truth, one set of paths is generated and posterior tails are evaluated
at each configured look. Every grid candidate is then evaluated against those
same paths in bounded batches using the exact strict BOP2-DC decision
comparisons. The effective and futile truths use common standardized draws
within calibration; the held-out stage uses a separate seed and is never used
to reselect. Tie resolution is declared: CGR maximization breaks ties by lower
futile expected sample size, ESS minimization breaks ties by higher CGR, and
remaining ties select the first candidate in product/input order.

Calibration constraints are empirical Monte Carlo estimates and do not
guarantee true FGR/FNGR/FCR control. The result retains per-candidate scenario
decision probabilities and MCSEs, error/objective metrics, feasibility,
selected index/design, and independent held-out scenario summaries and
feasibility. The returned master/calibration/validation seeds replay the entire
selection/validation procedure.

Calendar paths use the existing fixed or Poisson accrual implementations and
administrative censoring. They are explicit Python simulation choices; the
paper provides an accrual rate and final follow-up example but does not specify
a gap distribution. No native software RNG or optimizer parity is claimed.

## Resource bounds

The implementation preflights candidate-by-look-by-trial work, path chunk
cells, posterior-probability storage, candidate batch memory, and all retained
candidate/validation arrays before consuming the RNG. The optimizer is serial;
it retains only the current posterior path surface and candidate batch during
selection, then holds compact per-candidate summaries and two validation
summaries. `max_work` may lower the hard work ceiling but cannot raise it.

## Independent verification

`tools/check_bop2_dc_survival_calibration.py` reconstructs small input paths
from reported calibration and validation seeds. The companion base-R oracle
independently observes every patient at each look, computes Gamma probabilities,
applies absorbing decisions, and selects both objectives from all 18 candidates.
All 760 probability, enrollment and Monte Carlo error summaries agree; exact
grid objectives and feasibility flags agree too. CGR selects candidate 0 and
futile-ESS selects candidate 11. The former passes the 4% calibration FGR bound
at `2/64` but fails held-out validation at `3/48`; the selected design is
retained and `validation_feasible` is false. The latter passes its holdout.

The check takes 0.6084 seconds after Python imports. Parent peak resident memory
is 120.50 MiB and the R child peaks at 83.41 MiB; summing their separate peaks
gives a conservative combined upper bound of 203.91 MiB. No Python swaps were
reported. Numerical work is serial. Two focused worker tests also pass,
including a nondegenerate analytic one-patient event/censoring partition and
pre-RNG validation/work rejection. Ruff, formatting and mypy pass. There is no
new CI job or full-suite run for this addition.
