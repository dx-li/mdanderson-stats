# BOP2-DC

`bop2_dc_design` creates a binary efficacy design with separate statistical
and clinical reference values. It returns posterior decisions and exact
binomial operating characteristics, including a distinct final **consider**
outcome. Supplied cutoff parameters are not automatically calibrated.

```python
from mdanderson_stats import bop2_dc_design

design = bop2_dc_design(
    40,
    lrv=0.2,
    cmv=0.3,
    prior=(0.1, 0.1),
    looks=[10, 20, 30, 40],
    lambda_lrv=0.9,
    lambda_cmv=0.5,
    gamma_lrv=0.5,
    gamma_cmv=0.5,
)
decision = design.monitor(responses=12, sample_size=40)
print(decision.decision)  # final_consider
oc = design.operating_characteristics([0.2, 0.3, 0.4])
print(oc.final_go, oc.final_consider, oc.no_go_probability)
```

Interim stopping applies only at the configured looks. Counts between looks
return posterior probabilities and `continue`. At the final look, both
posterior criteria must strictly pass for go or strictly fail for no-go;
otherwise the result is consider. `monitor` evaluates supplied cumulative
counts and does not retain the history of an already stopped trial.

The two arrays returned by `boundaries()` contain the largest response counts
failing each individual criterion at interim looks. Both must fail to stop;
the combined stopping boundary is their elementwise minimum. A value of -1
means that criterion cannot fail at that look.

Operating characteristics account for absorption at interim stopping. Use
`no_go_probability` for total no-go risk: `final_no_go` excludes earlier stops.
The recursion supports at most 1,000 subjects and bounds scenario-batch work
before allocating state arrays. Larger scenario batches can be split.

The default prior is Beta(0.5, 0.5). Set `prior` explicitly or use
`prior_probability` and `prior_ess` to select another prior. Defaults are
illustrative and do not establish false-decision-rate control.

[Finite-grid parameter optimization](bop2-dc-optimization.md) supports false-go,
false-no-go and optional false-consider constraints. [Paired monitoring](bop2-dc-paired.md)
supports efficacy/toxicity and multiple efficacy endpoints, including exact
operating characteristics from joint outcome probabilities. Paired parameter
calibration remains pending. [Time-to-event monitoring](bop2-dc-survival.md)
is available; survival operating characteristics, calibration and generated app
reports remain pending. See
[source notes](bop2-dc-source.md) and the
[independent numerical references](bop2-dc-reference.md).
