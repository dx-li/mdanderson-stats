# BOP2-DC randomized Normal endpoint

The primary paper, `research/raw/BOP2-DC/paper.txt` §2.4, applies the
single-arm posterior model independently to experimental and control arms and
bases decisions on `P(theta_E - theta_C > theta_LRV/CMV)`. It also gives an
optional interim graduation rule using the O'Brien-Fleming cutoffs; otherwise
interim no-go and final go/consider/no-go follow §2.2. The Python workflow
reuses the shared randomized BOP2-DC decision rules and requires arm-specific
Normal-Inverse-Gamma priors, a fixed allocation tape, total-N looks, and margins
from the caller. It does not infer a randomization ratio, prior, or timing model.

Each arm mean has a Student-t marginal posterior. The difference is evaluated
by bounded deterministic one-dimensional convolution, integrating one arm's
t-quantile over its probability scale against the other arm's t survival
function. The result retains an adaptive quadrature error estimate and the
monitor refuses to classify if that estimate can change a strict decision.
Arm observations and prior means are centered on a shared origin before NIG
updates, preserving a common additive shift while leaving difference margins
unchanged. This is a Python numerical implementation, not a claim of native
BOP2-DC executable parity. Calibration and randomized assignment generation
are separate future scopes.
