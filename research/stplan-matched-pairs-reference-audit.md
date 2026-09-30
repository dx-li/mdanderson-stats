# STPLAN matched-pairs reference audit

This reference exercise checks the matched-pairs power and planning routines in
the cached STPLAN 4.5 source against a separate base-R calculation. It is a
numerical reference for the formulae, not a claim of application or test-level
parity.

The native driver, [`tools/reference_stplan_matched_pairs.f90`](../tools/reference_stplan_matched_pairs.f90),
links the original `qmpd`, `pmpd`, `psimpd`, `fmpd`, and inversion routines from
the ignored `research/raw/STPLAN/source/stplan-4.5/SOURCE` tree. Its only local
shim supplies console unit numbers to `qrange`; it does not replace a numerical
routine. [`tools/reference_stplan_matched_pairs.R`](../tools/reference_stplan_matched_pairs.R)
recomputes the formulae using base R normal quantiles, CDFs, and root solving;
it does not call the Python implementation or the original Fortran.

The cached source files used here have SHA-256 digests: `qmpd.f`
`8c63ecd3bcd9044c75e52d28d9b83ab21784aed24fe53d1dc366000fcf88daad`, `pmpd.f`
`18c75c672d111cfffed4d5398fb9ae3e7f3ea5cdf1fc9a12663b6d443e92139d`,
`psimpd.f` `47b9b6d21528922777f284cd209d6d531f033361751f606168799b6c46faa863`,
`fmpd.f` `6c71205acf02b410c509d0e7144df350d8192901605abfd66264972062bd3a01`,
`qmfinv.f` `9d4470426031602ea8388cc750495b4420b2d71312d7a5f70566d14bd0b6cedb`,
and `qdzero.f` `c690fd16ecff740e7ca117a6429094f9105a2d9e88eb4e8a99ef692398f94062`.

## Source contract

For preliminary matched-pair counts, `psimpd` uses the four cell counts
`z11,z10,z01,z00`, with `N` their total. Let `a=z10+z01`, `b=z10-z01`,
`r=(a+delta*b)/(2*N)`. Its variance term is
`psi = r + sqrt(r^2 - delta*(b-delta*(z11+z00))/N)`. `pmpd` then evaluates

```text
Phi((-z_(1-alpha*)*psi + |delta|*sqrt(n*psi)) /
    sqrt(psi^2 - delta^2*(3+psi)/4))
```

where `alpha*=alpha` for a one-sided test and `alpha/2` for a two-sided test.
The source explicitly says its two-sided result is only the probability of
detecting a difference in the direction of truth, not the sum of both tails.

Without preliminary cell counts, `qmpd` treats `z10` and `z01` as marginal
proportions for groups 1 and 0 and uses
`psi=theta1+theta2-2*theta1*theta2`; they are not joint discordant-cell
probabilities. With these estimates (`iwhich=5`) it returns a conservative
sample size and recommends an initial pilot of `n/4`. With neither estimates
nor pilot data (`iwhich=6`), it assumes group proportions `0.1` and `0.9`,
returns the resulting conservative sample size, and recommends `n/6`. Both
recommendations are unrounded.

For the sample-size inverse, the original code squares
`z_(1-alpha)*psi + z_power*sqrt(psi^2-delta^2*(3+psi)/4)`. When this quantity
is negative, squaring discards its sign and can return a sample size whose
achieved power is not the requested target. The low-target, no-pilot fixture
preserves this native result and separately records its achieved power; the
Python implementation should reject this unsupported inverse rather than
returning the misleading sample size.

The source ranges delta, alpha, and power to `[1e-8, 0.99999999]`, sample size
to `[1, 1e10]`, and preliminary counts to nonnegative values with a total in
`[1, 1e10]`. Estimated group proportions use the same strict probability
range. The reference set avoids invalid square-root geometry because the
original API does not provide a dedicated, reliable geometry status for those
cases.

## Case coverage and comparison

The paired CSV outputs cover one- and two-sided power, including a low-power
two-sided case; pilot counts scaled by ten at fixed proportions; inverse
difference, significance, sample size, and power; estimated-proportion and
no-information sample sizes; and their `n/4` and `n/6` recommendations. A
separate case uses all-concordant pilot counts `(20,0,0,80)` with positive
`delta=0.1`, which is valid and has `psi=0.1`. The final low-target no-pilot
case uses `delta=0.05`, estimated proportions `0.2` and `0.65`, and target
power `0.01` to demonstrate the lost-sign inverse. Native `qmpd` returns
`n=107.20425279175998` with status zero; evaluating that sample size gives
power `0.16502500436240453`, rather than `0.01`.

`tests/fixtures/stplan-matched-pairs-native.csv` records the direct original
routine output, including logical success and native status. The independent
results and achieved-power checks are in
`tests/fixtures/stplan-matched-pairs-reference.csv`. The inverse difference
uses source `QMFINV` tolerances and therefore need not match the base-R root to
full precision: the observed absolute difference is `1.98e-8` for the returned
difference and `8.94e-8` for its achieved power. Across the 12 cases, the
largest difference in returned sample-size recommendations is `1.04e-13`.
Native and base-R generation, including compilation, took `2.102 s`; child
peak RSS was `85,917,696` bytes (`81.94 MiB`) and `ru_nswap` was zero. The
output CSVs are regenerated from the two drivers; no original STPLAN source is
redistributed in this repository.
