# ToxFinder native stochastic prior calibration

`elicit_toxfinder_native_prior` implements the recovered 1.1.1 native calibration
contract using an independent NumPy Gamma random stream. It is separate from
`elicit_toxfinder_prior`, which solves the published probability and Gamma
odds-moment equations. Their beta means differ by construction.

```python
import numpy as np
from mdanderson_stats import elicit_toxfinder_native_prior, simulate_toxfinder_native

elicitation = elicit_toxfinder_native_prior(
    [600, 1200, 1400, 2000],
    [350, 600, 700, 800],
    0.30,
    interaction_mean=[1.0, 1.0],
    interaction_variance=[3.0, 0.9],
    rng=np.random.default_rng(2026),
)
for agent in (elicitation.agent1, elicitation.agent2):
    print(agent.assessed_tail_probability, agent.assessed_probability_mcse)
    print(agent.assessed_target_residual, agent.assessed_cutoff_met)

# Bounded demonstration; increase draws/trials and inspect diagnostics for research.
run = simulate_toxfinder_native(
    [[0.1, 0.1], [0.5, 0.5], [1.0, 1.0]],
    [[0.0, 0.2], [0.3, 0.5]],
    prior=elicitation.prior,
    target=0.30,
    stage1_patients=2,
    stage2_patients=4,
    cohort_size=1,
    trials=2,
    seed=2026,
    draws=8,
    warmup=0,
)
print(run.mean_toxicities, run.mcse_mean_toxicities)
```

Each agent supplies four positive physical doses: negligible, reference,
high-toxicity, and above-target. The first is below the reference; both final
doses are above it. Their mutual ordering is unrestricted. Interaction means and
variances are explicit, so the function never guesses between paper and case-study
specifications. This named native workflow requires positive interaction variances.

## Recovered model and optimizer

For target `t`, high-toxicity probability `qH` and `z3 = dose3/reference`, the
native program fixes

```text
E(alpha) = t/(1-t)
E(beta) = log(qH / ((1-qH)*E(alpha))) / log(z3)
```

This sets odds at the mean beta, not the published Gamma-MGF expectation. For
the example above, beta mean is `8.126872809767802` for both agents. Python's
published-equation elicitor deliberately retains its different formulation.

For each proposed pair of log variances, native calibration uses 30,000 independent
Gamma(alpha), Gamma(beta) pairs. Each pair contributes to both strict tails:
`P[p(negligible)<qL]` and `P[p(above-target)>t]`. The objective is the sum of squared
differences from `0.99`. Native Gamma shape narrowing to float32 is preserved;
Python uses its own Gamma sampler, not the old random generator.

The nine variance starts are alpha `(0.1,0.03,0.3)` crossed with beta `(3,1,10)`
in that order. A three-vertex precheck requires each tail to fall strictly inside
`(1e-6,0.999999)` at least once. Eligible starts get three simplex attempts,
20 outer iterations each, initial log-space steps `(0.1,0.1)`. Every vertex is
reevaluated each iteration; failed attempts restart at the returned point with
steps multiplied by `-0.5`. The first accepted start returns. Exhausting starts
raises an error with no fallback prior.

**Optimizer success does not prove calibration quality.** The original accepts
either its objective cutoff or small relative simplex spread. A constant
`(0.5,0.5)` probability surface is therefore accepted despite objective `0.4802`.
Python preserves that control contract, then uses an additional independent
30,000-pair batch per agent to report probabilities, binomial MCSE, residuals and
whether that assessment meets the native `2e-6` cutoff. `assessed_cutoff_met` is
an estimate, not a guarantee about exact probabilities.

The default total budget is 8,000 probability evaluations including assessment,
with a hard maximum of 10,000. Each evaluation draws 60,000 Gamma values. This
is a work cap, not a substitute optimizer stopping rule. Finite positive moments,
positive beta mean and representable powers/Gamma parameters are required.
Native degenerate/overflow behavior is not emulated. In particular, the separate
MAP optimizer's parameter box is not imposed on elicitation log variances.

## Evidence and limits

The hash-pinned runner `tools/reference_toxfinder_native.py` executes the original
elicitor, precheck, complete optimizer and output transforms with declared
probability callbacks. Fixtures cover cutoff acceptance, flat-objective acceptance,
interior optimization, all-start precheck failure, and all 27 attempts failing.
Every evaluated log-variance point matches the Python control implementation.
Separate original routines execute the 30,000-pair loop and strict comparisons
using explicit Gamma tapes. All eight native fixture sets regenerate as identical
JSON values. Independent Gamma-density integration checks seeded Python results.

The original unmodified stochastic calibration and original seed/RNG equivalence
remain unverified. The reference scopes do not imply otherwise. See the
[elicitation recovery audit](../research/toxfinder-native-elicitation-recovery-2026-10-12.md)
and [complete trial workflow](toxfinder-native-workflow.md).
