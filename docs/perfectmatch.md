# PerfectMatch: normalization, PDNN fitting and expression

Independent Python implementation of the numerical methods in MD Anderson's
[PerfectMatch manual](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/PerfectMatch/PerfectMatchManual.pdf)
(version 2.3, October 29, 2004) and Zhang, Miles and Aldape,
[A model of molecular interactions on short oligonucleotide microarrays](https://doi.org/10.1038/nbt836),
*Nature Biotechnology* 2003, 21:818–821. The site supplies a scanned copy of the
paper; its equations were inspected visually. [Source provenance](perfectmatch-sources.json)
records both documents. Original software, parameter files and array data are not redistributed.

## Available calculations

`quantile_normalize(intensities, reference=None)` accepts a nonnegative matrix
with **probes as rows and samples as columns**. Without a reference, each sorted
rank is mapped to the mean intensity at that rank across samples. A supplied
reference is a vector with one intensity per probe, sorted internally. For input
ties, all tied probes receive the mean target intensity over their occupied ranks.
The manual does not document native tie handling; this deterministic convention
is explicit. Untied samples acquire the same empirical distribution. No missing
values are imputed and no CEL cells are selected automatically.

`pdnn_binding_energy` evaluates paper equations (2) and (3) for 25-base probes:
the energy is the weighted sum of the 24 adjacent-base stacking energies. Inputs
are a 4-by-4 stacking matrix with both axes ordered A,C,G,T, and 24 position
weights in the sequence's 5-prime to 3-prime direction. Lowercase bases are accepted;
ambiguous bases are rejected. Use separate parameters for gene-specific and
nonspecific energies.

`pdnn_signal` evaluates equation (1):
`I = N/(1+exp(E)) + Nstar/(1+exp(Estar)) + B`.
The inputs are specific/nonspecific energies, natural-log expression, nonspecific
amount and background. Calculation in log space avoids overflow from computing
`exp(E)` or `N` separately. Arrays broadcast.

`pdnn_expression` evaluates equation (5) for one sample, **conditional on supplied
energies, Nstar and background**. With specific affinity `a=1/(1+exp(E))` and
residual `r=I-Nstar/(1+exp(Estar))-B`, it uses
`N = sum(r*sqrt(a/I))/sum(a*sqrt(a/I))` separately for each probeset. Negative
residuals are excluded. If both a reference fitted-signal vector and a global
log-intensity mean squared error `fitness` are supplied, probes with
`abs(log(I)-log(reference_fitted)) > 3*sqrt(fitness)` are also excluded, following
the paper. The function does not invent a reference fit or iterate an undocumented
outlier procedure. A probeset without usable positive specific signal raises an
error rather than receiving a fabricated expression value.

The result contains sorted integer probeset IDs, natural-log expression, fitted
signals, an inclusion mask and usable-probe counts. Expression estimates are not
automatically rescaled: the paper's array-average-500 scaling and the manual's
software output conventions require a further workflow audit. Parameters must
be calibrated for the relevant array; these functions do not supply universal
energy parameters.

```python
import numpy as np
from mdanderson_stats import quantile_normalize, pdnn_signal, pdnn_expression

normalized = quantile_normalize([[3, 20], [1, 30], [2, 10]])
np.testing.assert_array_equal(normalized, [[16.5, 11], [5.5, 16.5], [11, 5.5]])

energy = np.array([1.0, 2.0, 3.0, 4.0])
noise_energy = np.array([2.0, 3.0, 2.0, 3.0])
ids = np.array([10, 20, 10, 20])
known = np.array([100.0, 200.0, 100.0, 200.0])
signal = pdnn_signal(
    energy, noise_energy, log_expression=np.log(known), nonspecific_amount=50, background=10
)
fit = pdnn_expression(signal, ids, energy, noise_energy, nonspecific_amount=50, background=10)
np.testing.assert_allclose(np.exp(fit.log_expression), [100, 200])
```

## Learning parameters

`fit_pdnn(sequences, intensities, probeset_ids, initial=PDNNParameters(...))`
minimizes equation (4), the mean squared difference between observed and predicted
natural-log intensities. It fits one array at a time, using all supplied probes.
Supply two 4-by-4 stacking matrices, two 24-position weight vectors, a log-expression
vector in sorted unique probeset-ID order, and scalar `log_nonspecific_amount` and
`log_background`. All amplitudes remain positive through log parameterization.
No universal initial parameters or random initialization are supplied.

The default first fits stacking energies and amplitudes with weights fixed, then
jointly fits all parameters. `fit_weights=False` omits the joint stage;
`fit_energies=False, fit_weights=False` fits amplitudes with fixed energies/weights.
An analytic gradient and L-BFGS-B replace the paper's Monte Carlo optimizer.
`tolerance` sets the gradient threshold, and relative objective reduction can also
terminate a stage at `tolerance*1e-3`; `max_iterations` applies separately to each
stage. Results include initial/final fitness, total iterations, final active-gradient
maximum, optimizer message, fitted signals and learned parameters.

This is local optimization: convergence does not prove a global minimum, and
bilinear energy/weight scaling makes parameter estimates nonunique. Assess fitted
signals and sensitivity to starting parameters; do not interpret individual energy
coefficients as uniquely identified. `PDNNConvergenceError.result` retains a valid
checkpoint that can be passed back as `initial=result.parameters`. Invalid numerical
states raise errors. Outliers are not removed implicitly during fitting. The
log-error objective is distinct from the conditional weighted expression equation
(5); their expression estimates need not coincide with noisy observations.

## Validation and performance

Six focused tests cover both normalization modes and ties; independently summed
nearest-neighbor energies; recovery of known gene-expression levels; direct
weighted equation (5); negative-residual and outlier exclusion; very large
intensities; and log-expression 1,000 without intermediate exponentiation overflow.
The fitting checks compare every analytic gradient component with central finite
differences, recover known amplitudes with fixed energies, verify synthetic joint-fit
error reduction, and exercise a failed optimizer checkpoint. Synthetic intensities
are generated through the separate public signal/energy functions.
Groupwise expression sums use NumPy reductions and stabilized log sums, avoiding
one full-array scan per gene.

On the development machine, 200,000 probes across eight arrays normalized in
about 0.22 seconds. Conditional expression estimation for 200,000 probes in 20,000
probesets took about 0.047 seconds, with maximum synthetic log-expression recovery
error `2.7e-15`. These are local measurements, not cross-machine guarantees.

## Remaining coverage

**Catalog status is partial.** Native parameter-file formats,
Affymetrix text/binary CEL and binCEL workflows, probe sequence/annotation files,
complete quality-control statistics, output formats, native rescaling conventions,
and gene/image/scatter displays remain pending. Synthetic parameter-learning checks
do not validate an end-to-end microarray analysis against real array data. No native
optimization or display parity is claimed.
