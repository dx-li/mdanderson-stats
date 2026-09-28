# U2OET GAO explicit-prior fitting

The fitted likelihood reuses the 2017 GAO model in `u2oet_gao.py`: raw-dose
threshold-specific intercept/slope pairs, positive endpoint link-shape
parameters, one shared positive interaction, and Gaussian-copula association.
The existing primary-source audit is [u2oet-gao-audit.md](u2oet-gao-audit.md).

`fit_u2oet_gao` defines a transparent Python prior-coordinate contract because
the native GAO prior-vector ordering and transformations remain unverified.
Coordinate order is efficacy thresholds (agent-1 intercept, agent-2 intercept,
agent-1 slope, agent-2 slope), efficacy log-lambda; toxicity thresholds in the
same order, toxicity log-lambda; shared log-kappa; association Fisher-z. Each
coordinate has an independent caller-supplied normal prior; SD zero is an
exact point mass. No transformation Jacobian is applied because these named
coordinates themselves define the prior. This is not a claim of native prior
or fitter parity and does not import the distinct 2010 prior specification.

Complete ordinal counts and optional toxicity-only counts contribute their
respective grouped log probabilities. Independent Gaussian-prior coordinates
are updated by elliptical slice sampling, preserving chain/draw axes. Returned
parameter diagnostics use the package's `summarize_chains`; they do not certify
convergence. A prior-center start is provided for convenience, while dispersed
caller starts are preferable for assessing multimodality. The result retains
fitted probability grids, likelihoods, counts, prior settings and work counts.

Combined retained and summary workspaces are capped before allocation; runtime
likelihood calls are checked against hard-capped evaluation and grid-cell work
budgets. Each Gaussian rectangle also uses the existing bounded deterministic
integrator. Unresolved rectangle errors propagate rather than becoming slice
rejections. Correlations whose finite Fisher-z rounds to exactly plus/minus one
are rejected, since this API represents interior correlations through `tanh`.

A separate one-dimensional normal-prior quadrature regression checks a free
raw efficacy intercept with fixed nonzero margins and a partial-toxicity count.
An independent base-R comparison of two binary-grid posterior cases checked 38
summaries (parameter means, centered second moments, association means and
joint probability cells) within 1.028 Monte Carlo standard errors; maximum
split-Rhat was 1.00347. The reference is validation of this explicit Python
prior convention, not a native-prior comparison.
