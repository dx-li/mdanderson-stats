# Independent BOP2-DC categorical reference

The base-R reference in `tools/reference_bop2_dc_categorical.R` follows the cached paper in `research/raw/BOP2-DC/paper.txt`. Section 2.1.4 (printed pp. 6–7) specifies the joint multinomial-Dirichlet model and posterior update `Dirichlet(alpha + X)`, then states that a binary-indicator linear combination has a Beta marginal with shapes equal to the posterior mass in the indicator and its complement. Section 2.2 (printed pp. 7–9) gives the strict dual-cutoff final rule, its “otherwise” consider action, information-scaled interim futility cutoffs, and multiple/co-primary composition. Section 2.4 (printed pp. 12–13) applies independent arm-specific posteriors in randomized trials and defines the effect as experimental minus control.

The reference computes each single-arm tail directly with base-R `pbeta`, choosing the requested lower or upper tail. For randomized arms it integrates the experimental Beta density times the corresponding conditional control tail over `[0,1]`; lower tails are evaluated directly instead of subtracting an upper-tail probability from one. Endpoint posterior probability rows and final/interim decisions are written separately so they can be compared with `monitor` without relying on the Python implementation's internal helpers.

The compact fixtures cover three overlapping indicators over five categories with mixed directions and both `any` and `all` compositions; an independent-arm randomized case with asymmetric Dirichlet priors; a four-category two-binary-endpoint case for reduction to the existing paired design; and prior-only symmetric cases at exact cutoff equality. The randomized tape uses arm assignments `0,0,1,1,0,0,1,1` and category labels `2,5,1,3,2,4,1,5` (one-based R labels; Python replay uses zero-based category IDs). Its first interim action is `stop_no_go`; later CSV rows are forced-tape monitor snapshots for checking the count-to-posterior calculation, not reachable states of a stopped trial. The checker compares replay only with the first reached state.

Run the independent fixture generator from the repository root with:

```sh
Rscript tools/reference_bop2_dc_categorical.R tests/fixtures
```

It uses base R only. The output CSVs are `bop2-dc-categorical-config.csv`, `bop2-dc-categorical-reference.csv`, `bop2-dc-categorical-decisions.csv`, and `bop2-dc-categorical-replay.csv`. The checker `tools/check_bop2_dc_categorical_reference.py` consumes these fixed values through the public design APIs; it is a manual cross-check, not a CI test or a replacement for the project test suite.

This is an independent calculation of the specified posterior marginals, arm contrast probabilities, and decision composition. It does not claim reproduction of native UI behavior, cutoff-calibration optimization, or every native output file. The paper's categorical-indicator posterior property extends beyond two endpoints, but the fixtures exercise only three indicators and five categories.
