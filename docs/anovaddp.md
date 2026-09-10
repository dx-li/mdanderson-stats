# ANOVA DDP: nonlinear observation and conditional-update kernels

MD Anderson entry 67 implements the ANOVA dependent Dirichlet-process model of
De Iorio, Müller, Rosner and MacEachern, “An ANOVA Model for Dependent Random
Measures,” JASA 99(465):205–215 (2004). Its response model describes repeated
measurements using subject-specific nonlinear curves; the distribution of those
curve parameters depends on subject covariates through a DDP mixture.

**Coverage is partial.** The observation curve, Gaussian likelihood and
conditional Gaussian amplitude update, complete subject-level transition and
residual-variance conditional, Gaussian atom posterior and cluster sweep are
available, together with covariance, base-mean and concentration updates. Full MCMC,
new-subject prediction, study contrasts, nadir summaries, source data readers and
plotting workflow remain to be ported. These kernels do not fit a DDP model by
themselves and are not a substitute for the full sampler.

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

## Source findings to resolve in the full sampler

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
an appropriate way to construct a chain. A full-chain API remains pending.

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
response-unit rescaling by 1e100. The full sampler remains unimplemented.


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
invariants, immutable results and density-underflow inputs. Full MCMC and predictive/reporting workflows remain pending.


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
retains the draw; the full sampler must propagate it. This repairs the omitted
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
complete chain: MCMC orchestration and prediction/reporting are still pending.
