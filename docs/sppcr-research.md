# SPPCR archive and numerical baseline

Catalog entry **26, SPPCR**, is implemented. The Python package covers the
source's statistical methods, four input modes, repeated analysis, reports and
file workflows. The [completion mapping](sppcr-coverage.md) reconciles all 27
source files and explains the deliberate differences from the historical program.
The numerical baseline below records the original source findings; subsequent
implementation and validation are linked from the completion mapping.

## Archive identity and misplaced material

The pinned [download](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/SPPCR/SPPCR%20%20%20_V1zip.zip)
has SHA-256 `98e0937e26a5209421144c20178a2bd9c77d3baad7a3b28606d0f0f6555e2c98`.
Its 42 regular files include 27 Fortran source files, two build files, two historical
binary distributions, eight foreign documentation/example files, and three
acquisition/build/legal documents. The catalog calls the language Fortran 77;
the actual source uses Fortran modules, allocation and array syntax. Its compiled
banner identifies **SPPCR 1.0, January 2003**, by Barry W. Brown.

The three files under `docs/`, all four files under `EXAMPLE/`, and the root `readme`
refer to **SOGS, Simulation of Genotype Selection**, September 1997. The sample
input concerns chromosome selection, and the example readme explicitly invokes
`sogs`. These files are not an SPPCR manual or validation case. SOGS is independently
tracked as catalog entry 56. Do not port its breeding simulation as part of SPPCR
or use its printed results as allele-frequency oracles. Historical executable
identity has not been established by execution; native references rebuild the source.

`tools/reference_sppcr.py` checks the archive hash, inventories all members, compiles
all 27 sources unchanged in archived order, runs the application startup/exit, and
runs eight isolated fit probes. [The fixture](../tests/fixtures/sppcr.json) retains
member hashes, compiler flags/diagnostics, exact driver and inputs, raw outputs and
exit codes. The archived [Legal.doc](../notices/mdanderson-sppcr-Legal.doc.txt) is
retained exactly. Native code and binaries are not included in the Python wheel.

## Statistical contract recovered from source

For DNA level `d_i`, allele `j` has Poisson mean `d_i*mu_j`. An allele is detected
with probability `p_ij = 1-exp(-d_i*mu_j)`. The input comprises counts of detections
and nondetections at each DNA level, not mutually exclusive multinomial counts:
several alleles can be detected in one well.

Ignoring count-dependent binomial constants, each allele's log likelihood is

```text
L(mu) = sum_i [seen_i * log(1-exp(-d_i*mu)) - unseen_i*d_i*mu]
```

Its derivative is `sum(d_i*(seen_i/expm1(d_i*mu)-unseen_i))`. Interior variance is
the inverse negative second derivative of this log likelihood. Allele frequency
is `mu_j/sum(mu)`, and calibration is `sum(mu)`. Mutant frequency sums alleles
outside the one or two distinct progenitor alleles. The source propagates the
independent mean variances through these ratios using the delta method.

At one DNA level, an interior fit is available independently in closed form:
`mu=-log(1-seen/n)/d`, with observed-information variance
`p/(n*(1-p)*d^2)`. Four native cases, including changed DNA units, heterozygous
progenitors and two levels exactly consistent with one Poisson mean, agree with
these identities. Frequency and mutant variances are checked using a separate
Jacobian/covariance derivation rather than copying the source ratio formula.

The transform is `2*asin(sqrt(p))`, with delta variance `var(p)/(p*(1-p))`;
the source clips p into `[1e-10,1-1e-10]` for that variance. Reports use the inverse
transform and the fixed normal multiplier 1.959964 for confidence limits.

## Observed defects and choices requiring explicit treatment

- `modify_data` says it adjusts an allele with no nondetections at any level.
  Its test actually triggers if **any** level has zero nondetections. It subtracts
  0.5 detections at the minimum DNA level and adds 0.5 nondetections there. A native
  probe changes `[10,20]` to `[9.5,20]` even though the unadjusted likelihood has a
  finite maximum. Another changes `[0,20]` to `[-0.5,20]`, an invalid count, and
  still exits successfully with printed estimates.
- A never-seen allele has boundary MLE zero. `refine_estimate` instead transfers
  0.01 observations from unseen to seen at the last DNA level. The fixture shows a
  positive estimate with zero reported mean variance because curvature is then
  calculated from the unperturbed zero-detection counts.
- The root search is bounded by `1e10` and discards its terminal status. Finite
  output alone does not establish convergence or identify a boundary solution.
- Despite the source's “parametric bootstrap” label, default `set_generate` uses
  each observed detection fraction `seen/(seen+unseen)`, not probabilities from
  fitted means. The truth-specified path instead uses `1-exp(-dna*calibration*frequency)`.
  These are distinct resampling models and must be named explicitly.
- `bootstrap` calls `set_seeds(which=1)`, forcing a clock-based reset even if a
  preceding data-generation step used a phrase. Reproducible Python resampling
  must use explicit RNG state and disclose its seed instead of silently resetting it.
- Simulation probabilities are converted to single precision and well counts to
  integers before the native binomial generator. Replicate count is fixed at 1000.
  `stats_from_accum` computes population variance with divisor B, using the
  cancellation-prone difference of mean squares and squared mean.
- The inverse arcsine confidence-limit formula is periodic outside `[0,pi]`.
  The Python interval implementation clips transformed limits to `[0,pi]` before
  inversion instead of wrapping endpoints through `sin^2`.
- `structures_mod.initialize` attempts allocation of already allocated arrays when
  dimensions change and inconsistently resizes other arrays. Python must use
  independent, correctly sized state for successive data sets.
- The main application can use `write_simulations` before assignment for read or
  interactive data paths. Public optional-output routines also contain unchecked
  optional-argument combinations. Both require explicit Python contracts.

The malformed and perturbed cases in the fixture are defect evidence, not desired
Python results. The fitting API distinguishes original counts, deliberate
boundary adjustments, genuine boundary estimates and numerical failure. Frequency
and uncertainty workflows must retain these distinctions.

## Input conversion findings

`data_in_struct_mod.values_to_structures` doubles input DNA amounts when
converting genome equivalents to allele equivalents, removes never-seen allele
columns and maps progenitor sizes to the retained indices. The numerical Python
APIs accept model DNA amounts directly; the batch data object retains both units
and performs this conversion explicitly. The native removal test sums all 50 rows of `seen_in`, not
only `1:n_dna_in`; an injected stale-row probe now confirms that this changes
retained columns. Removing an unobserved progenitor also leaves its mapped index
at -1 in the native fixture. Python only examines active rows and requires the
retain-all policy when a progenitor is unobserved.

The batch reader requires ordered `nallele`, `nrun`, `nwell`, `allelesizes`,
`progenitor` records followed by the specified number of `run` records. Each run
contains a DNA amount and one integer count per allele. Its lexical/comment rules
and valid conversion results now have native probes
and a validated Python batch parser. The native negative-count probe also confirms
sign loss, which Python rejects. FileMaker numeric rows now have native probes
and validated parsing/formatting. Interactive data entry now has native transcript
probes and a bounded stream API with numeric and identity corrections.

## Completion

The [method and source coverage mapping](sppcr-coverage.md) records the completed
implementation and its validation. Native defects above are retained as evidence,
not reproduced as desired Python behavior.
