# Adaptive TITE-Keyboard weights

The adaptive timing method estimates the conditional time-to-DLT distribution
from observed DLT times and pending follow-up, then applies the resulting
posterior-mean follow-up weights to the existing approximate Keyboard
likelihood. It is an explicit Python extension based on the paper's model; it
does not claim to reproduce the app's native sampler or prior defaults.

```python
import numpy as np

from mdanderson_stats import KeyboardDesign
from mdanderson_stats.tite_keyboard_adaptive import tite_keyboard_adaptive_decision

result = tite_keyboard_adaptive_decision(
    KeyboardDesign(target=0.3),
    patients=[3, 3],
    toxicities=[0, 1],
    observed_dlt_times=[[], [35.0]],
    pending_followup=[[], [30.0, 15.0]],
    current_dose=2,
    window=90.0,
    # Explicit independent Gamma(shape, rate) priors for shared timing shapes.
    lambda_prior=(0.5, 0.5),
    gamma_prior=(0.5, 0.5),
    chains=4,
    warmup=3_000,
    draws=12_000,
    rng=np.random.default_rng(2718),
)

assert result.weights.diagnostics_passed
print(result.weights.pending_weights)
print(result.decision.action)
```

Each dose supplies the observed DLT event times, completed non-DLT count (the
wrapper derives it from enrolled counts and pending vectors), and follow-up
times for pending DLT-free patients. Event times must lie strictly inside the
assessment window; pending follow-up may be zero but must remain below the
window. All times use the same units.

The posterior for the shared timing shapes includes both observed DLT-time
densities and pending survival factors. The latter are integrated together with
the dose-specific toxicity probability under the paper's `Beta(1,1)` prior.
This matters when data are observed early: fitting only the observed DLT times
would ignore that pending patients have not yet experienced toxicity.

The Gamma prior arguments are explicit `(shape, rate)` pairs. The paper gives
examples of independent Gamma priors but does not mandate a parameterization or
default. The result exposes posterior means, batch-means MCSEs, classical split
R-hat values, acceptance rates, and work counts. These are MCMC diagnostics, not
convergence guarantees. The decision wrapper raises if the shape-chain R-hat or
pending-weight MCSE thresholds are not met; inspect the standalone weight fit
before adjusting the declared draw budget or thresholds.

The resulting weights are used with the paper's approximate effective-binomial
Keyboard posterior and the existing Python safety/suspension order. They are
not an exact posterior decision over dose toxicity probabilities. The paper
recommends uniform and piecewise-uniform weights for general use because its
adaptive version showed minimal improvement in sparse phase-I timing data; this
extension does not change any default.
