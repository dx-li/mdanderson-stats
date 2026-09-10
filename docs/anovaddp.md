# ANOVA DDP: nonlinear observation and conditional-update kernels

MD Anderson entry 67 implements the ANOVA dependent Dirichlet-process model of
De Iorio, Müller, Rosner and MacEachern, “An ANOVA Model for Dependent Random
Measures,” JASA 99(465):205–215 (2004). Its response model describes repeated
measurements using subject-specific nonlinear curves; the distribution of those
curve parameters depends on subject covariates through a DDP mixture.

**Coverage is partial.** The observation curve, Gaussian likelihood and
conditional Gaussian amplitude update, complete subject-level transition and
residual-variance conditional, Gaussian atom posterior and cluster sweep are
available, together with covariance, base-mean and concentration updates and
the complete `fit_anovaddp` MCMC routine and all seven native predictive output
families. Source data readers, output adapters and plotting workflow remain to
be ported.
The low-level conditional kernels are components of the fitting routine; they
do not independently fit a DDP model.

## Supplied nonlinear curve

`anovaddp_curve(parameters, time)` reproduces `regressione.cpp`. Parameters have
final axis six: `(z1, z2, z3, tau1, tau2, b1)`. The source comment incorrectly
lists seven names; its actual implementation reads six and fixes the logistic
intercept at -2. A nonempty time vector is evaluated against every parameter
vector, yielding shape `parameters.shape[:-1] + (len(time),)`.

- Before tau1, the response is z1.
- From tau1 to tau2, it is `r*z1 + (1-r)*(z2+z3*expit(-2))`, with
  `r=(tau2-time)/max(tau2-tau1, 1e-6)`.
- At and after tau2, it is `z2+z3*expit(-2+b1*(time-tau2))`.

Branch order and the 1e-6 denominator floor follow the native code. That floor
can introduce a discontinuity when knots are closer than 1e-6. With reversed
knots, the default retains the original branch behavior rather than silently
sorting them. `repair_order=True` implements `fittare`'s separate prediction
convention: replace tau2 by tau1+1 only when tau2<tau1. A requested replacement
that cannot be represented as greater than tau1 raises an arithmetic error.

```python
from mdanderson_stats import anovaddp_curve, anovaddp_loglikelihood

theta = [2, -1, 4, 1, 3, 0.7]
times = [0, 1, 2, 3, 4, 6]
print(anovaddp_curve(theta, times))
print(anovaddp_loglikelihood(theta, times, [2, 1.9, 0.7, -0.4, 0.3, 2], 0.4))
```

`anovaddp_loglikelihood` sums independent Gaussian observation terms over time.
By default it omits normalizing constants, matching the native `loglik` used
when comparing theta at fixed residual variance. `normalized=True` includes
`-n/2 * log(2*pi*variance)`, which is necessary for comparisons across variances.
A scalar positive variance and finite aligned observations are required.

The sigmoid is evaluated with a stable special function, so very large positive
or negative slopes do not overflow an exponential. Nonfinite final curves,
likelihoods and posterior moments raise clear errors; the function does not
replace them with arbitrary finite values. Returned arrays are immutable.

## Conditional amplitude update

`anovaddp_amplitude_posterior` implements the first Gaussian block of `simtheta`.
It conditions a six-dimensional normal prior on the fixed values of tau1, tau2
and b1, then combines the resulting normal prior for z1,z2,z3 with the Gaussian
observation likelihood. Off-diagonal prior correlations are retained.

```python
import numpy as np
from mdanderson_stats import anovaddp_amplitude_posterior

posterior = anovaddp_amplitude_posterior(
    [2, -1, 4, 1, 3, 0.7],
    [0, 1, 2, 3, 4, 6],
    [2, 1.9, 0.7, -0.4, 0.3, 2],
    prior_mean=[1, 0, 3, 0.5, 2, 0.5],
    prior_covariance=np.eye(6),
    variance=0.4,
)
print(posterior.mean, posterior.covariance)
```

The result contains the three-dimensional posterior mean, covariance and the
observation design matrix. Linear solves and positive-definiteness checks
replace the native explicit matrix inverses. The design is evaluated from the
same curve kernel, ensuring that its matrix product with the amplitudes equals
the likelihood mean. This deliberately resolves a source inconsistency:
`simtheta` omits `regressione`'s 1e-6 denominator floor. Ordinary separated knots
match both native formulas; very close knots follow the observation curve.
These are conditional moments, not samples or unconditional DDP posterior moments.

## Source findings and explicit corrections

The archived `vardati` signature accepts beta0 but never uses it; it draws
residual variance with inverse-gamma shape `(alpha0+N)/2` and scale `SSE/2`.
The R manual instead specifies inverse-gamma prior shape alpha0/2 and scale
beta0/2, which would give posterior scale `(beta0+SSE)/2`. The variance API now makes this discrepancy explicit: its default follows the
documented prior, while `mode="source"` reproduces the omission. Whole-sampler
comparisons will need to state which variance target they use.

Other native routines and the DDP model structure remain under audit. The source
also offers replacement of `regressione.cpp` for user-defined response curves;
that extension point will require a consistent likelihood/conditional-update
contract rather than a callback attached only to the prediction function.

## Validation and provenance

The unmodified `regressione.cpp` was compiled using a minimal adapter providing
only the NEWMAT RowVector indexing it requires. Three parameter vectors were
evaluated at eight times, covering every branch, exact boundaries, reversed
knots and the narrow transition floor. Python matched the native double outputs
exactly on this platform. The conditional update was independently evaluated
with R linear algebra for a correlated six-dimensional prior: maximum mean
and covariance differences were below 5e-15 and 4e-16 respectively.

Two focused tests also cover likelihood normalization, stable sigmoid limits,
explicit knot repair, immutable outputs, positive-definite covariance, and
agreement between the conditional design matrix and the nonlinear curve.
No CI jobs or dependencies were added.

[Source archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/AnovaDDP/anovaddp_1.0.tar.gz).
The case-sensitive download folder is AnovaDDP. Every regular archive member was
checked against its extracted bytes and inventoried in `anovaddp-sources.json`.
That inventory establishes provenance, not completion of every source routine.
The MDACC portion's original Artistic-license declaration is preserved in
`notices/mdanderson-anovaddp-COPYING.txt`. Original source, NEWMAT/RANDLIB
implementations, sample data and the article are not redistributed.


## Subject transition and residual variance

`anovaddp_subject_update` performs one `simtheta` sweep conditional on the current
subject-specific Gaussian prior, residual variance and observations. It first
draws the amplitudes jointly from their exact conditional posterior. Next it
proposes tau1 and tau2 in sequence from their Gaussian conditional priors. The
prior and proposal ratios cancel for these independence proposals, leaving the
likelihood ratio. Finally it proposes b1 by a symmetric normal random walk with
standard deviation 0.45 times its conditional-prior standard deviation, as in
the source. That acceptance ratio includes both likelihood and conditional prior.
Each conditional uses the values already updated earlier in the sweep.

```python
import numpy as np
from mdanderson_stats import anovaddp_subject_update, anovaddp_variance_posterior

step = anovaddp_subject_update(
    [2, -1, 4, 1, 3, 0.7],
    [0, 1, 2, 3, 4, 6],
    [2, 1.9, 0.7, -0.4, 0.3, 2],
    prior_mean=[1, 0, 3, 0.5, 2, 0.5],
    prior_covariance=np.eye(6),
    variance=0.4,
    seed=70,
)
variance = anovaddp_variance_posterior(
    [step.parameters],
    [0, 1, 2, 3, 4, 6],
    [2, 1.9, 0.7, -0.4, 0.3, 2],
    [0, 0, 0, 0, 0, 0],
    alpha0=6,
    beta0=4,
)
print(step.parameters, step.accepted)
print(variance.shape, variance.scale)
```

Acceptance calculations compare logarithms, avoiding exponentiated-ratio
overflow. The result provides updated parameters, three acceptance flags and
clipped log acceptance probabilities in tau1/tau2/b1 order, plus the independent
normal and uniform innovations used in the sweep. No knot-order truncation is
added. The curve/conditional-design consistency repair for nearly coincident
knots described above remains in effect. A seeded local NumPy generator gives
repeatable standalone sweeps; reusing the same seed on every iteration is not
an appropriate way to construct a chain. Use `fit_anovaddp` to manage the chain
and independent transition seeds.

The variance function accepts one six-parameter row per subject and aligned
time, observation and zero-based subject-ID vectors. Subjects need not be
contiguous. Observations are grouped once by sorting, avoiding a full data scan
for each subject. With N observations and squared residual sum SSE, the result
is inverse-gamma shape `(alpha0+N)/2` and scale `(beta0+SSE)/2` by default. In
`mode="source"`, the scale is SSE/2 and beta0 is ignored, as in `vardati`.
All priors must still be positive. A zero source-mode scale raises an error
because it is not a proper inverse-gamma distribution. This API returns
conditional distribution parameters, not a variance draw.

Three complete subject transitions were checked against an independent R
calculation of the archived formulas, using identical supplied normal/uniform
innovations. They exercise accepted and rejected knot/slope moves; maximum
parameter/log-acceptance discrepancy was approximately 1.1e-14. These are
formula-level comparisons, not an unchanged compiled `simtheta` comparison.
Two additional focused tests cover those transitions, exact variance updates
for interleaved subjects, the omitted source prior scale, zero residuals and
response-unit rescaling by 1e100. The fitting routine below assembles these blocks.


## Covariate-dependent atoms and cluster allocation

`anovaddp_atom_posterior` supplies the Gaussian conditional used by `musimul`
and for newly occupied clusters. It accepts a matrix of **conditional random
effects** (subjects by d), a design matrix (subjects by q), their common residual
covariance Scond, and the Gaussian base mean/covariance. These conditional
random effects are not the original six curve parameters. In the supplied model,
d=5 and q=7; the full sampler conditions the last five effects on the first.

The q*d atom coefficients are stored in covariate-major order: q successive
blocks of d elements. Subject i has conditional mean `(x_i kron I_d) @ atom`.
The first design column must be one; the native routines hard-code the intercept
block and ignore values in that column. Python validates it explicitly and
allows compatible dimensions up to 100 coefficients. Posterior precision is
`C^-1 + sum(F_i.T @ Scond^-1 @ F_i)`, with the corresponding prior/data mean term.
The implementation uses linear solves and returns immutable mean/covariance.

```python
import numpy as np
from mdanderson_stats import anovaddp_cluster_sweep

clusters = anovaddp_cluster_sweep(
    conditional_parameters=[[3, 1], [-2, 0], [-1, 1], [4, 2]],
    design=[[1, 0], [1, 1], [1, -1], [1, 2]],
    labels=[0, 1, 1, 2],
    atoms=[[3, 1, 0, 0], [-2, 0, 0.5, 0.2], [4, 2, 0, 0]],
    residual_covariance=[[1, 0.2], [0.2, 0.7]],
    base_mean=np.zeros(4),
    base_covariance=np.eye(4),
    concentration=1.3,
    seed=70,
)
print(clusters.labels, clusters.counts, clusters.atoms)
```

The sweep implements `clusters` followed by `musimul`: remove each subject from
its current group, compute assignment probabilities, assign it, and finally
resample every occupied atom conditional on all its subjects. An existing
cluster receives weight equal to its remaining size times its Gaussian density.
The new-cluster weight integrates out an atom under the Gaussian base measure:
concentration times a normal density with mean F_i*m and covariance
Scond+F_i*C*F_i.T. When that option is selected, its atom is drawn from the
single-subject posterior immediately, making it available to subsequent subjects.
The final atom draws use all assignments after the sweep.

Labels are zero-based and contiguous. Input atoms are rows, unlike the source's
columns, and every input atom must be occupied. The result includes labels,
atoms, counts, and counts of cluster creations/removals. It also records the
assignment uniforms and standard-normal atom innovations, in order of new-atom
creation followed by final occupied-atom resampling, for reproducible auditing.
This remains one conditional block, not a complete fitted DDP model.

**Singleton repair:** native `clusters` deletes the empty cluster count and
renumbers labels but does not delete its MU column. That can associate shifted
labels with the wrong old atom. Python removes the atom row, count and label
together. A one-subject dataset correctly removes its only old cluster before
creating and sampling the single occupied cluster again. No mode reproduces
misaligned labels and atoms.

All assignment calculations use log densities and log-sum-exp normalization,
avoiding native all-zero weights when Gaussian densities underflow. Existing
atom densities are evaluated together in one matrix solve. Cholesky factors
supply log determinants and standardized residuals; no raw determinant or
matrix inverse is used for density evaluation. Nonfinite final log densities
or moments raise errors instead of inventing replacement probabilities.

Independent R calculations verified every posterior mean/covariance element at
the native dimensionality d=5,q=7 and a complete smaller sweep with matching
innovations. The sweep exercises singleton deletion, new-atom generation and
final resampling. Differences were below 5e-15. The comparison uses the corrected
Gaussian/assignment formulas, not an unchanged compiled native sampler. Two
focused tests additionally cover the single-subject case, coherent label/count
invariants, immutable results and density-underflow inputs. The predictive
functions below assemble these blocks; file/plot workflows remain pending.


## Hyperparameter transition

`anovaddp_hyperparameter_update` covers `Basesim`, `SampleCovmu`, `S_Sample`
and `M_sample` in their sampler order. Inputs include raw six-dimensional curve
parameters, design, occupied labels/atoms, the current base covariance, the
fixed base-mean hyperprior mean, the residual-covariance prior, and concentration
prior/current values. The design's first column is one, and coefficient layout
matches the cluster block. The prior mean for the first curve parameter is 2.

```python
import numpy as np
from mdanderson_stats import anovaddp_hyperparameter_update

hyper = anovaddp_hyperparameter_update(
    parameters=[[2, 1, 2, 3, 4, 0.5], [2.1, 2, 1, 2, 3, 0.6]],
    design=[[1], [1]],
    labels=[0, 1],
    atoms=[[1, 2, 3, 4, 0.5], [2, 1, 2, 3, 0.6]],
    base_covariance=np.eye(5),
    base_prior=np.zeros(5),
    covariance_prior=np.eye(6),
    covariance_df=8,
    concentration=1,
    concentration_shape=2,
    concentration_rate=1,
    seed=6712,
)
print(hyper.concentration, hyper.residual_covariance)
```

The base mean has hyperprior N(base_prior,1000 I), matching the archived fixed
precision 0.001 I. Its conditional combines all occupied atoms and the current
base covariance, and a joint normal draw supplies the updated mean. Given that
mean, the intercept coefficient block's covariance has inverse-Wishart degrees
of freedom 10+K and scale `10 I_5 + sum((atom_intercept-mean_intercept) outer
(atom_intercept-mean_intercept))`. K is the number of occupied clusters.
Remaining base-covariance blocks are held fixed, as in `SampleCovmu`.

**Block independence is required:** the intercept block must have zero
cross-covariance with the other coefficient blocks. The native update replaces
only its top-left 5x5 submatrix without conditioning on cross-block dependence;
with nonzero cross-covariances that is not the stated conjugate update and may
not preserve positive definiteness. Python rejects those inputs. Correlations
within the intercept block and within the remaining fixed submatrix are allowed.

The six-dimensional random-effect covariance uses inverse-Wishart degrees of
freedom covariance_df+N and scale
`covariance_df*covariance_prior + sum((theta_i-prior_mean_i) outer
(theta_i-prior_mean_i))`, where prior_mean_i is `[2, F_i*atom_label_i]`.
Covariance_df is an integer at least six. **The native caller discards the return
value of `S_Sample`, so S never changes.** The Python transition returns and
retains the draw; `fit_anovaddp` propagates it to the next iteration. This repairs the omitted
update rather than reproducing a chain with fixed S.

The concentration update follows the source's Escobar–West augmentation:
eta ~ Beta(M+1,N), rate = concentration_rate-log(eta), and a mixture of gamma
shapes concentration_shape+K and concentration_shape+K-1. The higher-shape
mixing probability is `(concentration_shape+K-1) /
(N*rate+concentration_shape+K-1)`. The implementation calculates that probability
in log space and uses the rate parameterization for the gamma draw.

Covariance draws use a Bartlett factorization with triangular solves, avoiding
inversion of a sampled precision matrix. The returned immutable result includes
all draws and the conditional means, covariance scales/degrees of freedom,
beta auxiliary and selected gamma parameters for audit. A fixed seed produces
a repeatable transition. Unrepresentable or nonpositive draws fail explicitly;
there is no covariance clipping or fabricated positive concentration.

Independent R calculations verify the normal conditional, both inverse-Wishart
scales and concentration-mixture parameters. Five thousand two-dimensional
inverse-Wishart draws agree with their analytic expectation within five Monte
Carlo standard errors; the scalar case matches inverse gamma exactly up to
floating-point rounding. Focused checks also verify the retained S update,
positive definiteness, unchanged fixed base-covariance blocks, and rejection of
unsupported cross-block dependence. These do not establish convergence of a
complete chain; file/plot workflows are still pending.


## Complete MCMC fitting routine

`fit_anovaddp` assembles the observation, subject, allocation, atom and
hyperparameter blocks into a complete fitting chain for the supplied nonlinear
model. It accepts flat time/observation arrays and zero-based subject IDs, one
row of design and six initial parameters per subject, the initial random-effect
covariance, the base-mean hyperprior mean and initial base covariance. Each
subject must have observations. Interleaved observations are sorted once for
subject-wise updates without requiring a particular input row order.

```python
import numpy as np
from mdanderson_stats import anovaddp_curve, fit_anovaddp

times = np.tile([0, 1, 2, 3, 4, 6], 3)
subject = np.repeat(np.arange(3), 6)
initial = np.tile([2, 0.5, 1.8, 1, 3, 0.2], (3, 1))
y = np.tile(anovaddp_curve(initial[0], times[:6]), 3)
y += 0.15 * np.sin(np.arange(18))
fit = fit_anovaddp(
    times,
    y,
    subject,
    np.ones((3, 1)),
    initial_parameters=initial,
    initial_covariance=np.diag([1, 1, 1, 0.2, 0.2, 0.1]),
    base_prior=[0.5, 1.8, 1, 3, 0.2],
    base_covariance=np.eye(5),
    covariance_df=8,
    alpha0=6,
    beta0=2,
    concentration_shape=2,
    concentration_rate=1,
    iterations=200,
    burn_in=100,
    seed=67,
)
print(fit.observation_variance.mean(), fit.cluster_count.mean())
```

This short example demonstrates execution, not a sufficient analysis-length
chain. The initial covariance also supplies the fixed prior scale factor for
random-effect covariance updates, matching the source's `covcomfac = S` setup.
The initial cluster is shared by every subject, its atom and base mean equal
base_prior, and concentration is one. Default hyperparameters and 2,000 total /
1,000 warmup iterations come from the supplied demonstration; those iteration
counts do not automatically establish adequate mixing for another dataset.

Every iteration draws residual observation variance, updates each subject's
six parameters, conditions the five nonbaseline effects on its first effect,
updates clusters/atoms, and updates the base mean/covariances/concentration.
The conditional transformation is
`theta[:,1:] - (theta[:,0]-2)*S[1:,0]/S[0,0]`, with covariance
`S[1:,1:] - S[1:,0]*S[0,1:]/S[0,0]`. All blocks use the current state and every
hyperparameter update is propagated. The independent base-intercept covariance
restriction remains in force. `variance_mode="source"` changes only the omitted
beta0 scale behavior; it does not reinstate the singleton or discarded-S bugs.

The result retains complete post-sweep states: subject parameters, observation
variance, random-effect covariance, base mean/covariance, concentration, labels,
occupied atoms and cluster counts. Atom matrices have varying cluster counts
and are stored as an immutable tuple. A normalized observation log likelihood
is recorded for each retained state. It is not the full joint posterior or
model evidence. Acceptance fractions for the two knots and slope are reported
per subject over all iterations, including warmup.

With burn_in=B, retained iteration numbers are B+1, B+1+thin, ... up to the
iteration limit. Thinning controls storage only and consumes no extra randomness;
it does not change the underlying trajectory. A local master generator supplies
the variance draws and separate integer seeds to transition blocks. Repeating a
complete call with the same seed reproduces its states without modifying global
randomness. Current inputs and prior arrays are not mutated. Runtime transition
errors include their iteration number; failed transitions are not silently
skipped or replaced by previous states.

Labels are arbitrary between draws: averaging their integer IDs is meaningless.
Use posterior co-clustering probabilities or permutation-invariant summaries
when analyzing allocations. Acceptance rates and a completed run do not certify
convergence. Run multiple chains, inspect predictive and parameter traces, and
assess effective sample sizes / Monte Carlo errors for the quantities of interest.
The function does not automatically certify convergence or stop when it believes
convergence has occurred.

Memory/work limits are checked before sampling: at most 50 million stored
numeric values under a conservative all-subjects-in-separate-clusters bound,
5 million subject transitions and 100 million observation-iterations. Original
file adapters and plotting remain pending. The prediction functions
below consume the retained posterior states for study and nadir summaries.

Integration checks verify unchanged trajectories under thinning, normalized
log likelihood reconstructed from retained states, positive covariance draws,
occupied label/atom consistency, immutable history and input preservation.
An independent base-R chain implementing the corrected statistical equations
also provides a cross-implementation distributional check; it is not an
unchanged executable comparison with the defective native caller.

Four independent chains per implementation were run for 8,000 iterations with
2,000 discarded, on a three-subject repeated-measurement example. The monitored
summaries were observation variance, occupied-cluster count, concentration and
the mean subject curve at time two. R/Python mean differences were within 1.1
combined Monte Carlo standard errors, using the larger of batch-means and
between-chain errors. The Python runs took about 91 seconds locally for 32,000
iterations in total. Inputs, seeds, settings and both shorter and longer run
summaries are recorded in `tests/fixtures/anovaddp-chain-comparison.json`.

**Validation limitation:** one Python chain continued to mix slowly for the
fitted-response summary. Its across-chain classical split R-hat was about 1.072
in the longer run; cluster count was about 1.044. These are descriptive classical
diagnostics, not modern rank-normalized diagnostics, and the cross-implementation
agreement within estimated error is not a convergence certificate. The example
therefore establishes executable, state-consistent integration and supplies a
qualified distributional check, not a claim that the displayed iteration count
is sufficient for posterior inference. No automatic convergence claim is made.

A separate four-subject integration run also exercised the native seven-covariate,
five-conditional-effect geometry (35 atom coefficients), retaining coherent
35-column atom histories and positive-definite random-effect covariances. This
was a short dimensional/wiring check, not a convergence experiment.


## New-patient and study prediction

`anovaddp_new_atom` ports `Newpatient`. It selects an occupied cluster with
probability n_k/(N+M), or the Gaussian base measure with probability M/(N+M).
An existing cluster's coefficients are **freshly sampled** from their Gaussian
conditional given its subjects, rather than copied from the last saved atom.
The new-cluster option draws from N(base_mean,base_covariance). Returned weights
list occupied clusters first and the base option last; selected cluster -1 means
the base option. This neither adds a patient to the training allocation nor
changes its cluster counts.

`anovaddp_baseline_curves(coefficients,time)` constructs the exact ten nonlinear
transformations in `Baseline`, including its prediction-time knot repair. It
requires 35 coefficients (seven covariate blocks of five). These component
curves reproduce the supplied study layout; they are not differences between
otherwise identical fitted patient curves, and the function does not treat
them as generic ANOVA contrasts for an arbitrary covariate encoding.

`predict_anovaddp(fit,training_design,prediction_design,time=...,seed=...)`
assembles all seven C++ predictive families from each retained fit state. Pass
the exact training design used in fitting; it cannot be reconstructed from the
parameter draws. Both designs must have seven columns and an intercept of one.
At least three prediction rows are required for the source's nadir output. The
default grid is -1,0,...,30, matching the source, but arbitrary nonempty time
vectors are supported in place of the fixed native 32-column buffers.

```python
import numpy as np
from mdanderson_stats import anovaddp_curve, fit_anovaddp, predict_anovaddp

x = np.array(
    [
        [1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1],
    ]
)
times = np.tile([0, 1, 2, 3, 4, 6], 4)
subject = np.repeat(np.arange(4), 6)
initial = np.tile([2, 0.5, 1.8, 1, 3, 0.2], (4, 1))
y = np.tile(anovaddp_curve(initial[0], times[:6]), 4)
y += 0.1 * np.cos(np.arange(24))
fit = fit_anovaddp(
    times,
    y,
    subject,
    x,
    initial_parameters=initial,
    initial_covariance=np.diag([1, 1, 1, 0.2, 0.2, 0.1]),
    base_prior=np.r_[initial[0, 1:], np.zeros(30)],
    base_covariance=np.eye(35),
    iterations=40,
    burn_in=20,
    seed=6707,
)
prediction = predict_anovaddp(fit, x, x[:3], time=[0, 2, 4, 6], seed=67)
print(prediction.prediction_mean)
```

This is a wiring example, not a converged scientific analysis. The fit's mixing
limitations and need for multiple-chain assessment apply equally to predictions.
Python predictions use complete retained post-sweep states. The native program
produces predictions partway through a sweep and consumes the same global random
stream as fitting; exact iteration-by-iteration native prediction parity is not
claimed. Python prediction has its own generator and does not alter the fit.

| Native C++ output | Python result | Meaning |
| --- | --- | --- |
| comeff | common_effect | Per-draw curve for `[2, new_atom[:5]]` |
| predstudy3 | study3 | Per-draw curve for a Gaussian random subject around that mean |
| nadir | nadir | Independent subject draws for the first three prediction rows, transformed as below |
| base | baseline_mean | Posterior Monte Carlo mean of the ten Baseline component curves |
| base2 | baseline_second_moment | Mean squared baseline curve values, not a variance |
| prediction | prediction_mean | Mean latent subject curves for all prediction rows |
| prediction2 | prediction_second_moment | Mean squared latent subject curve values, not a variance |

The R wrapper returns only five of these (`m`, `a0`, `a02`, `f0`, `f02`), dropping
predstudy3 and nadir. Python exposes all seven, plus baseline/prediction draw
arrays, sampled atoms and selected cluster IDs. All result arrays are immutable.

One new atom is shared by every study/design row within each posterior draw,
as in the original simulation. Independent Gaussian subject effects with the
current six-dimensional covariance then produce each latent response curve.
The first study-3 draw is separate from these design-row draws. **No observation
noise is added:** this follows `PREDSTEP`, which draws curve parameters using S
but never uses residual observation variance. Thus these are latent response
curves, not future noisy measurement realizations.

The nadir routine draws three further independent subjects, one per first
prediction row, and returns `z2 + z3*expit(-2)`. That is the curve's nominal
value at tau2 for ordered knots. It need not be the global minimum when slope,
amplitude sign or knot ordering differ, so the name retains a source convention
rather than making a minimization claim. These nadir draws do not reuse the
subject draws in prediction_draws, also matching the source.

Two focused tests compare all ten Baseline rows over the original 32-point grid
against independent R results (maximum difference below 5e-15), verify a
closed-form one-dimensional existing/base atom draw and its weights, and exercise
the full 35-coefficient prediction workflow. The latter checks output shapes,
raw second moments, common-effect construction, independent subject variation,
replay and immutable arrays. Predictions are limited to 30 million saved values.
File export adapters and plot reproduction remain pending.
