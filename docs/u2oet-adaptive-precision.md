# U2OET adaptive posterior precision

The native guide monitors the utility at the four corners of the dose-pair
grid, checking each chain separately. Its precision statistic is batch-means
MCSE divided by that chain's posterior standard deviation. The guide gives a
simulation target range of 0.001–0.05 (default 0.03), a trial-conduct default
of 0.005, and says retained draws per chain are at least the burn-in count.
This Python function requires the target, initial retained draws and maximum
retained draws explicitly; it does not infer simulation or conduct defaults.

```python
import numpy as np

from mdanderson_stats import fit_u2oet_adaptive_precision, u2oet_parameter_names

names = u2oet_parameter_names(2, 2, model="cmi")
prior_mean = np.zeros(len(names) - 1)
prior_sd = np.full(len(prior_mean), 0.3)
for i, name in enumerate(names[:-1]):
    if ".slope." in name:
        prior_mean[i] = 0.8

counts = np.zeros((2, 2, 2, 2))
counts[0, 0, 1, 0] = 3
counts[1, 1, 0, 1] = 2
result = fit_u2oet_adaptive_precision(
    [1, 2],
    [1, 2],
    counts,
    prior_mean=prior_mean,
    prior_sd=prior_sd,
    utility=[[0, 1], [2, 0]],
    target_mcse_ratio=0.01,
    initial_draws=32,
    max_draws_per_chain=64,
    batch_draws=32,
    warmup=16,
    chains=2,
    model="cmi",
    rng=np.random.default_rng(20260929),
)
print(result.target_met, result.termination, result.draws_per_chain)
print(result.mcse_ratio)  # rows are chains; columns follow corner_indices
print(result.corner_split_rhat)  # separate convergence diagnostic, not a stop rule
```

Use `result.fit` with existing posterior summaries, flattening the chain and
draw axes of its `joint` array before calling `u2oet_posterior`. Fixed-budget
fits remain available alongside this adaptive fit option.

`target_met` is true only when all chain/corner ratios are finite and at
most the requested target. `termination="draw_cap"` reports that the explicit
no more valid chunks fit below the requested cap; a final remainder under
eight means the returned draw count can be up to seven below that nominal cap.
A zero-variance corner has an undefined
ratio and cannot make a chain pass; this prevents a stuck or flat chain from
being treated as precise. The result retains the combined `U2OETFit`, plus the
per-chain posterior SD, MCSE and ratio for each corner. Inspect convergence
diagnostics as well: precision alone does not establish mixing across chains.

Python uses nonoverlapping batches of length `max(2, floor(sqrt(draws)))` for
the MCSE calculation and omits a trailing incomplete batch from that
calculation only; the posterior SD and returned fit use every retained draw.
It appends at most `batch_draws` draws at
a time except that a final chunk may absorb up to seven leftover draws so the
sampler's minimum eight-draw batch is respected. If fewer than eight draws
remain under the cap, sampling stops below that upper bound. It performs
warmup once. Later chunks restart each chain from its last
complete parameter vector, including association; the U2OET transition has no
adaptation or latent auxiliary state to carry. A shared Generator advances
sequentially. Since chunk boundaries affect random-number consumption, this
procedure is not promised to reproduce a single fixed-budget fit from the same
seed.

The corner-utility precision target concerns only those four utility
expectations. Meeting it does not certify precision for every model parameter,
other dose-pair utilities, toxicity/efficacy risks, or downstream probabilities.

The sampler currently supports PDS, CMI and PDS+CMI with 2–16 chains. The native
guide lists 1–20 chains; one-chain PSRF is undefined and this implementation
inherits the sampler's narrower range. The separate
[GAO adaptive fitter](u2oet-gao-adaptive-precision.md) applies the same corner
criterion to the 2017 GAO sampler. This wrapper's preflight bounds cumulative worst-case
likelihood work and the retained joint posterior; requests beyond those caps
are rejected before consuming the Generator. The combined live-cell estimate
allows for chunk retention and concatenation/freeze copies, corner traces,
normalized diagnostic traces, batch means, and split-Rhat work arrays; it is an
allocation bound estimate rather than an RSS guarantee. `corner_split_rhat` uses the
project's classical split-Rhat summary implementation and is reported
separately; it is not the guide's precision criterion or a convergence guarantee.

The [adaptive calendar-trial guide](u2oet-adaptive-trial.md) connects this
monitor to interim and final PDS/CMI/hybrid analyses. The fixed-budget trial
path remains the default.
