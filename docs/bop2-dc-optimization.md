# BOP2-DC finite-grid optimization

`optimize_bop2_dc` evaluates a declared finite grid of the four BOP2-DC tuning
parameters: `lambda_lrv`, `lambda_cmv`, `gamma_lrv`, and `gamma_cmv`. It uses
the exact binary operating-characteristic recursion from the core design and
evaluates the distinct `theta_futile` and `theta_effective` scenarios.

The constraints are false-go rate at the futile scenario, total false-no-go
rate (including interim no-go stops) at the effective scenario, and an optional
false-consider bound equal to the maximum consider probability across the two
scenarios. The default objective maximizes correct-go rate; `ess_futile`
minimizes expected sample size at the futile scenario. Constraints use exact
`<=` comparisons. The result is explicitly a finite-grid optimum; grids with
no feasible candidate raise `BOP2DCInfeasibleError`.

The default illustrative grid is small (`lambda_lrv` values 0.5, 0.8, 0.9;
`lambda_cmv` values 0.1, 0.3, 0.5; and gamma values 0, 0.5, 1). Production
calibration should provide an explicit grid. Ties are resolved deterministically
by the candidate ordering after the primary objective (higher CGR then lower
futile ESS for the CGR objective; lower futile ESS then higher CGR for the ESS
objective), followed by parameter order. This is a documented Python choice,
not a claim about a unique continuous optimum.
