# Block adaptive randomization (BlockARAND)

Catalog entry **90 is implemented** for the documented two-arm binary-response
simulation workflow: rational block planning, uniform block permutation,
beta-posterior adaptive allocation, patient-wise stopping and repeated-trial
operating characteristics. Arm labels are **0 and 1** in Python.

Sources are the [MD Anderson entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/90),
[2012 guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BlockARAND/UsersGuide.pdf),
and [archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BlockARAND/BlockARAND_V1.0_NoFX4.0.zip),
which includes John D. Cook's January 2011 technical report *Block Adaptive
Randomization*. The library's compiled method bodies were inspected to resolve
choices not specified by the report. No original executable, DLL or document is
bundled. This implementation does not reproduce the Windows conversational/file
interface or the original random-number stream.

## Use

```python
import numpy as np
from mdanderson_stats import (
    BlockArandDesign,
    blockarand_plan,
    blockarand_decision,
    simulate_blockarand,
    simulate_blockarand_oc,
)

plan = blockarand_plan(0.68, min_block=4, max_block=8)
print(plan.arm_zero_count, plan.size)  # 4, 6: the report's example

design = BlockArandDesign(max_patients=200, burnin=50)
decision = blockarand_decision([3, 5], [7, 5], design=design)
trial = simulate_blockarand([0.3, 0.8], design=design, rng=np.random.default_rng(90))
print(trial.decision.selected, trial.successes + trial.failures)
oc = simulate_blockarand_oc(
    [0.3, 0.8], repetitions=500, design=design, rng=np.random.default_rng(91)
)
print(oc.selection_probability, oc.mean_patients)
```

`blockarand_block(plan, rng=...)` samples a uniform permutation of a planned block.
The plan's arm-zero fraction is the closest `k/n` to the target, with
`min_block <= n <= max_block` and `1 <= k < n`. A vectorized candidate search
preserves the original strict-improvement scan: exact floating-point ties choose
the first candidate in increasing denominator, then increasing numerator order.
Blocks are bounded by the guide's sizes 2 through 10. Each complete block contains
both arms; the target itself may lie outside the resulting attainable fractions.

## Posterior and trial conduct

The priors are independent beta distributions, entered as two `(alpha, beta)`
rows. Defaults use Jeffreys Beta(.5,.5) priors from the shipped configuration;
the bare DLL constructor instead contains Beta(1,1) defaults. Posterior parameters
add successes and failures separately. If `p = Pr(theta_0 > theta_1)` and
`q = Pr(theta_1 > theta_0)`, the new-block target is
`p**tuning / (p**tuning + q**tuning)`, with default tuning .5. Python computes both
ordering tails directly and evaluates the tuning transformation in log-odds form.
This avoids unstable powers and subtraction near probability endpoints. Tuning
zero gives equal allocation. The result reports quadrature error; the tolerance
is absolute accuracy, not a guarantee of relative accuracy for arbitrarily tiny
tails or a decision whose cutoff is within numerical error.

Outcomes are immediate independent Bernoulli draws with the supplied true arm
response probabilities. The posterior stopping rule is checked **before every
patient**, including before patient one and throughout burn-in. Strictly exceeding
`early_cutoff` (.95) selects the corresponding arm. At maximum enrollment,
`final_cutoff` (.90 from the DLL constructor) is used if early stopping has not
already selected a winner. If neither exceeds it, the trial has no winner.
A cutoff below .5 is accepted as in the original interface; arm zero takes
precedence when both arms exceed a cutoff. There is no futility boundary.

Burn-in uses balanced blocks of the largest even size not exceeding `max_block`,
even if that is below a user-specified odd `min_block`. At **exactly** `burnin`
patients, unused assignments in its current block are abandoned and a new adaptive
block starts. This follows the library's two separate randomizers. Subsequent
outcomes update monitoring after every patient but change allocation only when
a new block starts. `decision.target_probability` is therefore a target for a
new block, not the next-patient probability in a partly consumed block. Early
stopping and the enrollment cap can truncate any block. Zero burn-in is supported.

`BlockArandTrial` retains `(arm, response, block_index)` records, block plans,
zero-based enrollment positions where blocks start, outcome counts, final
decision and design. Its arrays are immutable. The `selected` field is `None`
when there is no winner. Strong asymmetric priors can stop a trial with zero
patients, matching the source's initial monitoring call.

Repeated simulation releases patient histories after each trial and shares a
bounded 100,000-entry posterior cache across trials within the call. Results
retain per-trial arm enrollment and selection (`-1` for none), selection and
no-selection probabilities, average arm sample sizes, and Monte Carlo standard
errors. Selection MCSEs use the plug-in binomial estimate; zero estimates at
observed proportions zero/one are not confidence bounds. Sample-size MCSEs use
the sample standard deviation. `comparisons` and `cache_hits` expose cache work.
There is no global simulation RNG or persistent posterior cache.

## Validation and limits

The [compiled-method audit](blockarand-reference.json) interprets arithmetic and
branches from the original DLL's `BestRationalApproximation` and
`CheckStoppingRule`. All 5,914 block plans and 180 stopping cases match Python,
including exact fractions, midpoints, strict threshold equality, asymmetric
cutoff ordering and arm-zero precedence. Reproduce after retrieving the archive:

```sh
uv run --with dnfile --with dncil python tools/reference_blockarand.py
```

The interpreter supports only the inspected methods and fails on unsupported
instructions. It does not execute the original numerical library or RNG. The
existing beta-comparison numerical suite is reused; an exact polynomial integral
checks `Pr(Beta(2,1) > Beta(1,2)) = 5/6`. Focused workflow checks cover the report's
4/6 block example, burn-in block truncation, patient-wise monitoring, zero-patient
prior stopping, final selection and operating-characteristic aggregation.

The [two guide-scenario pilots](blockarand-pilot.json) use 500 replicates each,
record seeds and MCSEs, and exercise the full simulator. Independent NumPy streams
and a tighter beta integration tolerance replace the original numerical engine;
rounded guide simulation outputs are not treated as exact reference fixtures.
The guide's displayed averages differ from these shipped-config pilots. The
printed output omits priors, while the shipped configuration and bare library
constructor use different priors. A separate 500-trial check with uniform priors
moved the mean arm enrollments to (5.856, 5.884) and (12.918, 14.556), nearer the
guide's (6.2, 6.3) and (12.1, 13.5). This suggests a prior-version difference but
does not establish which settings produced the historical guide output. Run
`uv run python tools/pilot_blockarand.py` to reproduce the recorded Jeffreys-prior
pilots.
Source control-flow interpretation and independent numerical validation are
separate evidence, not a claim of bitwise whole-program equivalence.
