# rBOP2 binary source and numerical references

Catalog 150's [official application](https://biostatistics.mdanderson.org/shinyapps/rBOP2/)
was retrieved on September 12, 2026. It identifies itself as PID 1054,
version V1.1.9.0, updated December 2, 2025. The application and help-file hashes
are recorded in [source metadata](rbop2-binary-sources.json).

The binary rules are specified by
[BEhelp2.pdf](https://biostatistics.mdanderson.org/shinyapps/rBOP2/BEhelp2.pdf)
and [BThelp2.pdf](https://biostatistics.mdanderson.org/shinyapps/rBOP2/BThelp2.pdf).
For efficacy, the desirable posterior probability is `P(p_E > p_C + margin)`.
For toxicity, it is `P(p_E < p_C - margin)`. Futility/toxicity stopping uses
`q < lower_cutoff`; superiority uses `q >= upper_cutoff`. Otherwise the trial
continues. These strict versus inclusive comparisons are preserved.

The prior help specifies independent beta distributions, parameterized by
prior mean `p` and effective sample size `n0`: `(a,b)=(p*n0,(1-p)*n0)`.
For each arm, events and non-events update its two shapes separately.
The efficacy prior guide distinguishes calibration under a vague null-centered
prior from operating-characteristic calculation under an informative prior.
This Python core requires explicit priors and cutoff vectors; it does not
perform or claim the native calibration.

The interim help illustrates total looks `[20,40,60,80]` with equal allocation,
ending with 40 patients in each arm. Python accepts explicit per-arm looks,
for example `[[10,10],[20,20],[30,30],[40,40]]`. Its operating characteristics
condition on those declared arm sizes, rather than guessing the application's
allocation-rounding conventions or simulating unconstrained Bernoulli
randomization. The last lower and upper cutoffs must coincide so every final
state has a positive or negative conclusion. This is an explicit supported
Python contract; no undocumented final-gap rule is substituted.

The cited article is Zhao, Yang, Lee, Wang and Yuan (2022),
*Bayesian Optimal Phase II Design for Randomized Clinical Trials*, Statistics
in Biopharmaceutical Research 14:423–432,
[DOI 10.1080/19466315.2022.2050290](https://doi.org/10.1080/19466315.2022.2050290).
Its full calibration formulas and the app's paired-endpoint decision algorithms
were not available from the retrieved sources. They remain open work. The
paired-prior PDF also displays marginal-mean numerators containing only `n10`
or `n01`, although its stated marginal definitions include the joint `n11`
cell. The displayed equations were visually checked; their inconsistency must
be resolved before treating them as a native algorithm specification.

`tools/reference_rbop2_binary.R` enumerates all 16 complete outcome paths for
two patients per arm. Both priors are Beta(1,1), with looks `[[1,1],[2,2]]`,
lower cutoffs `[0.25,0.8]` and upper cutoffs `[0.8,0.8]`. Known rational beta
integrals determine every decision, and an independent base-R integral checks
those fractions. At the final look, the probabilities for counts `(E=1,C=0)`
and `(E=2,C=1)` are exactly `4/5`, so superiority applies at cutoff `0.8`.
No tolerance band is used to reclassify those states.

For equal event rates 0.3, the reference gives early futility and superiority
probabilities 0.21 each, final-only negative and positive probabilities 0.4582
and 0.1218, overall positive probability 0.3318, and expected total enrollment
3.16. These illustrative cutoffs are not calibrated to a nominal type-I error.
Ten scenario/endpoint combinations include deterministic absorbing outcomes.
Six additional base-R integrations check fractional beta priors and margins
-0.1, 0, and 0.1 for both efficacy and toxicity. The resulting fixtures are
`rbop2-binary-reference.csv` and `rbop2-binary-margins.csv`.

Nine focused tests passed on September 12, 2026 in 1.43 seconds, with peak
RSS 131.7 MiB and zero process swaps. A built-wheel example with 20 patients
per arm, three looks, and three rate scenarios took 0.661 seconds and peaked
at 111.8 MiB, also with zero swaps, using one BLAS thread. This benchmark uses
zero margins; shifted beta integration may be more expensive. Affected
Ruff/mypy checks, source/wheel builds, and the isolated wheel workflow passed.
The full repository test suite was not run for this checkpoint.
