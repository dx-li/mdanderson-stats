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


The aggregate simulator uses independent per-arm Normal truths and the fixed
assignment tape. It generates outcomes serially in coordinates centered on the
control truth, returns compact terminal/per-look probabilities and Monte Carlo
standard errors, and records a replayable seed. It retains no simulated patient
or posterior histories. Hard preflight bounds cover patient cells, repeated
look replay work, result cells, and worst-case adaptive-quadrature evaluations;
the 100-trial default is a Python resource choice, not a native recommendation.


## Independent reference and integrated verification

`tools/reference_bop2_dc_randomized_normal.R` updates NIG sufficient statistics
and integrates the treatment Student-t survival function against the control
Student-t density on the real line. This is independent of Python's uniform
quantile integral. Twelve cases and 21 reached looks cover every terminal
action, strict equality, unequal priors/allocation, either empty arm, a common
`1e15` location shift, and `1e-8`/`1e8` changes of units. Six prior-only Cauchy
difference tails also agree with their analytic formula; the difference has
location 1.5 and scale 3.

The checker supplies R with exact generated latent tapes for 48 trials with
78 reached analyses, both with and without interim graduation. It verifies
263 numeric summaries across deterministic cases, analytic tails, simulated
decision/stopping probabilities, enrollment and MCSEs. The graduation-enabled
simulation has 16 early graduations among 24 trials. Common location shifts
preserve simulated decisions and expected enrollment exactly. Maximum posterior
discrepancy is `6.273e-10`, within reported numerical error. The complete check
takes 9.472 seconds after imports; Python and R peaks are 111.77 and 84.89 MiB
(a conservative combined upper bound of 196.66 MiB), with zero Python swaps.

Reference review identified a base-R decimal parsing issue around `1e15`:
parsing two outcome strings whose difference is one produced a difference of
0.75. The reference now stores common shift separately and analytically
cancels it before posterior calculations; Python receives the actual shifted
priors/outcomes. Numeric tolerances were not relaxed to hide the discrepancy.
The simulator also reports maximum quadrature error across every reached look,
not only terminal states. Five focused tests, including nondegenerate shift/
seed replay and pre-RNG work rejection, pass.
