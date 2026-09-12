# BARPO posterior monitoring and allocation

BARPO uses independent Beta priors for binary response rates. The Python
interface separates posterior inference, monitoring criteria and allocation
weights so the posterior calculation can be reused across allocation methods.

```python
from mdanderson_stats import barpo_posterior, barpo_allocation, barpo_monitor

posterior = barpo_posterior([1, 4, 8], [3, 2, 1], prior=[[1, 1]] * 3)
allocation = barpo_allocation(
    posterior, assigned=[5, 8, 12], method="barn2n", max_n=50,
    minimum_probability=[0.2, 0, 0],
)
monitor = barpo_monitor(
    [1, 4, 8], [3, 2, 1], prior=[[1, 1]] * 3,
    theta_fut=0.25, pfut=0.45,
    theta_eff=0.65, peff=0.9,
    theta_final=0.5, pfinal=0.95,
)
print(allocation)
print(monitor.futile, monitor.efficacious, monitor.final_efficacious)
```

Success and failure counts include observed outcomes only. Assigned counts
include pending outcomes and must cover the observed counts. Posterior
best-arm probabilities use deterministic integration and return absolute
error estimates. Allocation raises an error when a needed probability tail
is unresolved at that precision.

The four methods are:

- `barcp`: normalize best-arm probabilities raised to `tau`.
- `barn2n`: use power `n/(2*max_n)`, with `n` the total assigned count.
- `barmtv`: normalize the square root of best-arm probability times posterior
  response-rate variance divided by assigned count plus one.
- `dbcd`: use explicit desired proportions `target_probability` and current
  assigned proportions, with powers `tau` and `tau1`. Positive assigned
  proportions are required; the implementation does not add pseudo-assignments.

`minimum_probability` specifies per-arm allocation floors. Setting only its
first entry enforces a control-arm minimum using the guide's proportional
rescaling. Simultaneous floors use proportional redistribution among the
remaining arms. Floors must be feasible. `stopped` excludes arms from allocation
and their floors must be zero. The allocation uses the supplied best-arm
probabilities with stopped arms removed; it does not recompute best-arm
probabilities over a reduced set of competitors.

Without a control, `barpo_monitor` compares each posterior to fixed response-rate
thresholds. Futility uses a strict `>` confidence comparison; early and final
efficacy use `>=`. With `control=True`, arm zero is the control and experimental
arms are compared to its response rate. Supply `pfut`, `peff` and/or `pfinal`
without fixed response-rate thresholds. The control is excluded from treatment
decisions: its probability-array entries are zero placeholders and its flags
are false. Monitoring returns separate criteria; it does not resolve conflicting
criteria into a trial-level stopping policy.

Trial scheduling, equal-randomization burn-in, stopping-state management,
simulation and generated reports remain pending. DBCD target construction and
simultaneous-floor parity with the original app are unverified. See
[source notes](barpo-source.md) and [independent numerical references](barpo-reference.md).
