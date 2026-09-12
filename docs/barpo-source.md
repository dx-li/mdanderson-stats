# BARPO core

`barpo_posterior` updates independent beta priors with the observed success
and failure counts and delegates the multi-arm best-posterior probabilities to
`arand_best_probability`.  Counts in `assigned` include pending assignments;
they affect allocation methods but not the posterior.  The allocation API
rejects an assigned count below the observed count.

`barpo_monitor` reports posterior lower tails for futility and upper tails for
early and final efficacy.  In no-control mode these compare each arm with the
provided fixed response-rate reference.  In control mode arm zero is the
control and the reported probabilities are direct beta posterior ordering
probabilities for each experimental arm.  Control monitoring takes only the
confidence cutoffs; fixed response-rate thresholds are invalid there.  The
futility decision uses `>` and efficacy decisions use `>=`, matching BARPO.

`barpo_allocation` implements the documented methods:

* BARCP uses `pbest**tau`.
* BARN2N uses `pbest**(n/(2*max_n))`.
* BARMTV uses `sqrt(pbest * posterior_variance / (assigned + 1))`.
* DBCD uses `(y * (y/x)**tau)**tau1`, where `y` is the required explicit
  `target_probability` and `x` is the assigned-count proportion.

Weights are evaluated in log space and centered over eligible arms.  Stopped
arms receive zero probability.  Optional per-arm minimum probabilities are
applied by iterative water filling: fixed arms receive their floor and the
remaining mass is distributed among other eligible arms proportional to their
weights.  Floors must be feasible and stopped arms must have zero floors.
DBCD does not smooth zero assigned proportions; it raises an error instead.

The source BARPO guide does not specify how its application constructs DBCD's
desired target vector, so this library requires that vector explicitly.
