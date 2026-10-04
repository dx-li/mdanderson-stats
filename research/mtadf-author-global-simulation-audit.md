# MTADF author global-logistic simulation audit

The implementation follows the cached author reference
`research/raw/mtadf-author-reference/targetAgentDF.r`. The operating
characteristic routine `logistic()` is at lines 426–490; the real-trial
`df.logistic()` decision is at lines 516–565. The global fit and point
selection are the same in both functions, while the simulator adds the
one-cohort-lagged safety cap.

For the full dose grid `x=1,...,J`, the source uses
`x1s=(x-mean(x))/(2*sd(x))` and `x2s=x1s^2` (R sample standard deviation).
After each cohort, it expands the accumulated efficacy counts into Bernoulli
rows, fits `bayesglm(yreg ~ x1reg + x2reg, family=binomial(link="logit"))`,
predicts efficacy over all doses, and selects the rightmost maximum (`:465–474`).
The Python core fits the grouped-count equivalent using the recovered
`arm::bayesglm.fit` adaptive working-prior update. It is not claimed to match
R's `arm`/`glm` implementation bit for bit.

The simulator starts at the first dose (`:456–457`). Every cohort draws
toxicity and then efficacy at the assigned dose, updates counts and fits the
global model (`:475–477`). It moves one level toward the rightmost peak, using
the *previous* admissible prefix (`:478–481`), then recomputes that prefix
from the updated toxicity counts (`:482`). The fit occurs even if the fresh
prefix has length one. After the final cohort, selection is the last fitted
peak clipped to the refreshed prefix (`:484–485`). The Python replay exposes
the observed cohort tape, assignments, before/after caps, and final fit;
simulation uses the same conduct engine and retains bounded per-trial dose
counts plus the compact observed-outcome/assignment tapes needed to replay
each trial. It does not retain all-dose potential outcomes or per-fit traces.

Python uses sequential NumPy binomial draws only at the assigned dose. This is
a reproducible explicit convention when given an integer seed, not R RNG
parity. The author prior, logistic fit, and safety calculation are described
in the [global core audit](mtadf-author-global-audit.md). The
local-logistic author rule is a distinct policy; see
[`mtadf-author-local.md`](../docs/mtadf-author-local.md).

The bounded simulation caps the number of cohort fits, generated outcome
cells, and retained summary cells before consuming RNG state. Per-trial
IRLS work is at most 100 iterations per fit. Nonconverged fits abort rather
than being silently counted as completed trials. No posterior draws or
per-fit coefficient traces are retained by the aggregate simulator.
