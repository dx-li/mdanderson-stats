# Covariate-adjusted principal coordinates analysis

`adjusted_pcoa` implements aPCoA, catalog entry 147, from Shi, Zhang, Do,
Peterson and Jenq (2020), *Bioinformatics* 36:4099–4101,
[doi:10.1093/bioinformatics/btaa276](https://doi.org/10.1093/bioinformatics/btaa276).
The [MD Anderson app](https://biostatistics.mdanderson.org/shinyapps/aPCoA/)
reports version 1.3, PID 1033. The public R package version 1.3 is available from
[CRAN](https://cran.r-project.org/package=aPCoA).
[Provenance](apcoa-sources.json) records the reviewed source artifacts.
This is an independent implementation of the published matrix equations;
original R code and datasets are not redistributed.

## Computation and input conventions

Given a symmetric nonnegative distance matrix D with zero diagonal, define
`A = -D**2/2`, `J = I - 11'/n`, and `G = J A J`. For a numeric covariate matrix X,
aPCoA calculates `Delta = (I-H) G (I-H)`, where H projects onto the column space
of X. For Euclidean distances, this is the inner-product matrix of the residuals
from regressing centered original features on X. Non-Euclidean distances can
produce negative eigenvalues; these are retained in the reported spectrum.

Supply an explicitly encoded n-by-p numeric covariate matrix in **the same sample
order as the distance matrix**. For categorical covariates, supply the intended
dummy columns. Covariates are not automatically centered or augmented with an
intercept. This matches the R function's projection after it removes the formula
intercept. Use `intercept=True` to include the constant direction explicitly.
Changing centering, dummy coding or intercept convention can change the result.
The group of scientific interest belongs in plot labels, not necessarily in X:
including it in X removes its linear contribution.

The implementation scales each covariate column before SVD and projects onto the
numerically identified column space. It accepts redundant and zero columns
without inverting normal equations. The rank threshold is machine epsilon times
the larger matrix dimension times the leading singular value. Distances are
scaled by their maximum before squaring, avoiding squared-distance overflow.
Centering uses row means, and covariate removal uses low-rank matrix products
without materializing the full hat matrix.

Input limits are 2–2000 samples and at most n supplied covariate columns.
Symmetry and zero diagonal are checked at relative tolerance 1e-12; discrepancies
within that tolerance are symmetrized/zeroed. No triangle inequality is required.
Missing and infinite inputs are rejected.

## Results and numerical interpretation

The result contains `original` and `adjusted` ordinations and the identified
`covariate_rank`. Coordinates are eigenvectors times square roots of **positive**
eigenvalues, in the original distance units. By default at most two axes are
returned; `components=None` returns all positive axes. If fewer positive axes
exist, fewer are returned, including an empty n-by-0 matrix for zero inertia.
Signs are fixed for display; tied eigenvalues can still permit arbitrary rotations.

Each ordination retains the full signed `scaled_eigenvalues`, `eigenvectors`,
`scaled_gram_matrix`, and `distance_scale`. Multiply scaled eigenvalues or Gram
entries by the squared distance scale to recover original units. The `eigenvalues`
and `gram_matrix` properties perform that conversion and raise if nonzero values
underflow or overflow; the scaled results remain available.

`eigenvalue_tolerance`, default 1e-12, is relative to the original normalized Gram
matrix's Frobenius norm. Eigenvalues within that threshold are treated as zero
in both decompositions. The retained Gram matrices themselves are not spectrally
truncated. `positive_fraction` divides each displayed eigenvalue by total positive
inertia. `negative_inertia_fraction` is negative inertia magnitude divided by total
absolute inertia. These differ from native plot percentages based on signed trace,
which can be misleading for indefinite matrices. Negative directions are not
silently converted into real coordinates, and no Lingoes/Cailliez correction is
applied. Positive coordinates do not exactly reconstruct an indefinite kernel.

## Usage

```python
import numpy as np
from mdanderson_stats import adjusted_pcoa

rng = np.random.default_rng(147)
batch = np.repeat([0.0, 1.0, 2.0], 8)[:, None]
features = rng.normal(size=(24, 4)) + batch * np.array([3.0, -2.0, 1.0, 0.0])
distance = np.sqrt(((features[:, None] - features[None, :]) ** 2).sum(axis=2))
result = adjusted_pcoa(distance, batch, intercept=True)
assert result.covariate_rank == 2
np.testing.assert_allclose(batch.T @ result.adjusted.coordinates, 0, atol=1e-12)
print(result.adjusted.positive_fraction)
```

With the optional `plot` extra installed:

```python
from mdanderson_stats import plot_adjusted_pcoa

axes = plot_adjusted_pcoa(result, np.repeat(["Site A", "Site B", "Site C"], 8))
# axes[0].figure.savefig("apcoa.png", dpi=150)
```

The plotting function returns original and adjusted axes, colored consistently by
nonempty text group labels. It requires at least two retained positive axes in
both ordinations. It does not show/save a figure or change the global backend.

## Validation and remaining coverage

Three focused tests compare Euclidean adjustment with independent least-squares
residuals, a non-Euclidean fixture with the unmodified R aPCoA 1.3 function, and
extreme units, redundant covariates, complete projection and invalid distances.
For the native comparison, the function was sourced directly alongside ape's
`pcoa` function and `cluster::pam`; native ellipses and medoid connectors were
disabled. Synthetic Bray–Curtis distances were supplied. The adjusted positive
coordinate inner-product matrices agree within 6e-16. No R dependency is required
by Python or its tests.

**Catalog status is partial.** Core ordinations and basic comparison plots are
implemented. Native confidence ellipses, medoid connectors, interactive file/formula
handling and a full app workflow audit remain pending. Matching this native
numerical fixture does not establish parity for every input or display option.
