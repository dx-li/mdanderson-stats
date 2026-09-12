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

```python
from mdanderson_stats import optimize_bop2_dc

result = optimize_bop2_dc(
    40,
    lrv=0.2,
    cmv=0.3,
    theta_futile=0.2,
    theta_effective=0.4,
    prior=(0.1, 0.1),
    looks=[10, 20, 30, 40],
    lambda_lrv_grid=[0.8, 0.9],
    lambda_cmv_grid=[0.3, 0.5],
    gamma_lrv_grid=[0.5, 1.0],
    gamma_cmv_grid=[0.5, 1.0],
    false_go_limit=0.1,
    false_no_go_limit=0.15,
    false_consider_limit=0.3,
)
print(result.design)
```

This 16-candidate example took 0.0155 seconds on the development machine.
It selected lambda values `(0.9, 0.3)` and gamma values `(1.0, 0.5)`, with
false-go rate 0.07426, false-no-go rate 0.09065 and false-consider rate 0.04701.
Correct-go probability was 0.88642. The result retains the grid and requested
error limits; its operating-characteristic objects retain both scenario rates.

The preflight limit is 100,000 candidate combinations and
`2 * candidates * (max_subjects + 1)**2 <= 5,000,000` as a work estimate.
Candidates are evaluated sequentially. Requests exceeding a limit raise an
error before recursion rather than silently truncating the search.
