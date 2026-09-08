# Third-party notices

## RANGE2 and KWRANGE

The optional legacy grouping procedure in `mdanderson_stats/ranges.py` is a
modified Python adaptation of the RANGE2 and KWRANGE programs distributed by
the University of Texas M. D. Anderson Cancer Center. The original catalog lists
Dennis A. Johnston as contact for both programs. This adaptation is maintained
independently; the original authors are not responsible for its changes.

The original, identical LEGALITIES files from both distributions are preserved in
[`notices/mdanderson-range-LEGALITIES.txt`](notices/mdanderson-range-LEGALITIES.txt).
Those terms apply to the adapted portions; this project does not relicense them.
In particular, the terms permit use and redistribution subject to preservation
of notices and require written permission for incorporation into a commercial
package that will be sold.

Original archives:

- https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/RANGE2/RANGE2_V1.tar.gz
- https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/KWRANGE/KWRANGE_V1.tar.gz

The source archive hashes are recorded in `tests/fixtures/ranges.json`.
No original executable, Fortran source archive, or supporting numerical library
is redistributed by this project.

## MULTI

`_multi_lexer.py` adapts the original QLEX state/action tables and token behavior;
`multi_input.py` provides RDDATA input validation, diagnostics and sorting.
`multi_session.py` coordinates the adapted procedures and exports their results;
`multi_report.py` renders their report tables.
These adaptations are also subject to the original MULTI terms below.

`multiplicity.py`, `schweder.py`, `pvalue_models.py`, `nonparametric.py`, and
`nonparametric_testing.py`, `beta_mixture.py`, `beta_mixture_fit.py`, and
`beta_mixture_ml.py`, `beta_mixture_bootstrap.py`, `beta_mixture_selection.py`, and
`beta_mixture_testing.py` and `schweder_output.py` contain
modified Python implementations of MULTI's adjustment, sharpened-testing, Schweder,
order-statistic, clustered-simulation, S/desktop nonparametric, and beta-mixture
evaluation/initialization/EM/direct-likelihood/selection/simulation/decision algorithms. The original code
is copyright 1996 for The University of Texas M. D. Anderson Cancer Center.
Barry W. Brown is the original contact. This adaptation is maintained independently.

The original copyright and terms from section II of `multi.doc` are preserved in
[`notices/mdanderson-multi-LEGALITIES.txt`](notices/mdanderson-multi-LEGALITIES.txt).
They permit redistribution with the copyright section, permit noncommercial
source use, and require written permission for commercial source use. Those
terms are not replaced by a different project license.

Source archive:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MULTI/MULTI_V1.tar.gz

The hash and reference-build provenance are recorded in `tests/fixtures/multi.json`.
The port uses NumPy/SciPy in place of the original supporting numerical libraries;
the original executable and numerical library sources are not redistributed.

## STUKEL

`stukel.py`, `stukel_objective.py`, `stukel_fit.py`, `stukel_scan.py`, `stukel_output.py`, and `stukel_comparison.py` implement the generalized logistic formulas described by Thérèse A.
Stukel (1988), Generalized Logistic Models, JASA 83(402), 426–431,
DOI 10.1080/01621459.1988.10478613. The MD Anderson STUKEL archive was used to
validate numerical results and identify the S prediction branch discrepancy.
Its README identifies dansera@odin.mda.uth.tmc.edu as the software contact.
No explicit license statement was found in the inspected archive files. No
original STUKEL source or supporting optimizer is redistributed by this package.

`stukel_examples.json` includes the numeric beetles and Warsaw example data from
that archive, with source paths and file hashes. These data reproduce the archived
demonstration and are also checked against the native regression fixtures.

## BP1CI

The independently expressed beta/gamma quantile formulas in `intervals.py` and
the input/output conventions in `bp1ci.py` were checked against BP1CI 2.0
(May 2008), by Barry W. Brown, Floyd M. Spears and Dan M. Serachitopol.
The archived program and manual provide numerical validation and provenance;
no BP1CI Fortran, binary or supporting numerical library is redistributed.
Native reference fixtures record source/archive and executable hashes.

## ONESAMPLE

`onesample.py` independently expresses binomial/Poisson distribution-tail formulas
for the archived ONESAMPLE software. The calculation module and manual were used
for validation and identifying legacy tail cutoffs. `onesample_workflow.py`
independently exposes the four calculations, entry conventions and report content
from the main program and output module as Python calls. No original source or
supporting distribution-library code is bundled. Archive and source hashes and
the reference-only success-status initialization patch are recorded in
`tests/fixtures/onesample.json` and `tools/reference_onesample.py`.

## KSB1CI

`kstage_binomial.py` independently implements the surviving-path probability
recurrence, stage ordering and confidence inversion described by Barry W. Brown's
KSB1CI source and manual, with Python design/report interfaces. The original
[legalities and warranty notice](notices/mdanderson-ksb1ci-LEGALITIES.txt) is preserved;
this project does not relicense the original work. Those terms permit
noncommercial source use and require written permission for commercial source use.
Original numerical routines are extracted only for a local reference build; no
original Fortran or binary is bundled. Reference hashes and compiler provenance
appear in `tests/fixtures/ksb1ci.json`.

## KSBIN1

`ksbin1.py` independently evaluates fixed-design operating characteristics using
shared Python surviving-path calculations. The KSBIN1 main program and auxiliary
numerical routines were consulted for inclusive decision rules and expected-sample-
size definitions. Original source is extracted only into a local reference build;
no original Fortran or binary is distributed. Source/archive hashes and compiler
provenance are recorded in `tests/fixtures/ksbin1.json`.

`binomial_design.py` independently searches discrete binomial critical regions
and reports achieved power and adjacent alternatives, using KSBIN1's XBIN1 output
semantics with explicit empty-region handling. The private reference build applies
only the documented auxiliary success-status repair; its provenance is recorded
in `tests/fixtures/binomial_power.json`.

`binomial_significance` provides KSBIN1's significance solve mode through discrete
count search, with its original alpha search floor removed. Native mode-4 reference
provenance and the shared auxiliary status repair are recorded in
`tests/fixtures/binomial_significance.json`.

`binomial_alternative.py` independently expresses KSBIN1's alternative-probability
solve using inverse beta initialization and checked probability brackets. Mode-2
reference provenance is recorded in `tests/fixtures/binomial_alternative.json`.

`binomial_null.py` independently expresses KSBIN1's null-probability solve through
discrete region selection and checked probability inversion. Mode-1 reference
provenance is recorded in `tests/fixtures/binomial_null.json`.

`binomial_sample_size.py` independently implements the sample-size solve using a
randomized-power bound and ordered integer search, replacing the original
continuous-search/local-lookback heuristic. Mode-3 reference provenance and its
original outputs are recorded in `tests/fixtures/binomial_sample_size.json`.

`ksbin1_table.py` and `ksbin1_study.py` independently implement the source's
boundary-assistance probability definitions, study summaries, revision workflow
and numeric design-export semantics. The table corrects the original lower-cutoff
exclusion. Native table provenance is recorded in `tests/fixtures/ksbin1_table.json`;
original source remains confined to local reference builds.

## KSBIN2

`ksbin2.py` independently implements evidence-statistic formulas and weighted
outcome ordering described by KSBIN2's numerical source and executable menu.
Original Fortran routines are extracted only into a local reference build and
are not shipped. Source and extracted-code hashes and compiler provenance are
recorded in `tests/fixtures/ksbin2.json`. Documentation records the source's
criterion-description inconsistency and the exact-zero likelihood correction.

`ksbin2_probability.py` independently computes ordinary single-stage power and
null-grid significance using shared Python binomial masses. The source probability
routines are consulted and compiled locally for reference; the fixture provenance
is in `tests/fixtures/ksbin2_probability.json`. No original routines are shipped.

The KSBIN2 probability table also independently implements the source BRKARR
mid-p reporting convention and selection of complete tied rejection regions.
The native probability fixture includes unchanged BRKARR outputs. Pointwise
mid-p values are separately named to distinguish adjustment before maximization
from the original adjustment after maximization.

`ksbin2_multistage.py` independently evaluates surviving-path probabilities by
separable binomial-coefficient convolution. The original SSUPD transition/repacking
block is compiled only in a local reference build, with independent scaled-binomial
helpers; its provenance is recorded in `tests/fixtures/ksbin2_transition.json`.

`ksbin2_assistance.py` independently evaluates multistage cumulative rejection
probabilities and reference-completion power loss, following the source SSSIG,
SSPOW and SSPL definitions. It uses Python arrival masses and binomial transition
matrices; exhaustive paired trial paths validate its probabilities.

The multistage boundary table preserves the original BRKARR mid-p display
behavior, including its treatment of prior rejections in the first group.
The pointwise current-group adjustment is separately named. An unchanged native
BRKARR reference and its provenance appear in `tests/fixtures/ksbin2_midp.json`.

`ksbin2_study.py` independently implements study scans, revision and numerical
reporting. It separates actual null-rate operating characteristics from grid-maximum
significance, correcting the original summary's mixing of those quantities.

The multistage region-report methods independently export exact decision sets
as contiguous count ranges. They replace the original REGPRT display assumptions
and formatting while preserving the complete reachable rejection/quitting sets.

## SINGLE

`single.py` independently implements fixed-design Fisher information and delta-method
precision for SINGLE's logistic and log-log models. Original CPROB/CDPDB/MIX/GEXP
are compiled only in a local reference build, with source/compiler provenance in
`tests/fixtures/single.json`. No original source or binary is distributed. SINGLE's
source requires permission for commercial source use; this implementation expresses
the model formulas independently and documents its different numerical treatment.

## SEQBIN

`seqbin.py` independently implements beta-posterior boundary searches and forward
Bernoulli stopping probabilities. SEQBIN 1.5 source routines are compiled only
for local reference tests; source/compiler hashes appear in
`tests/fixtures/seqbin.json`. Original source, binaries and documentation are not
bundled. The archive's LEGALITIES file permits use and redistribution subject
to preserved notices and separate conditions for incorporating original code in
commercial packages; its original beta and root-finding dependencies also carry
ACM notices. These original routines are not incorporated into the Python package.

## CUMINC

`cumulative_incidence.py` and `gray_test.py` independently express competing-risk
incidence, Aalen variance and Gray score/covariance calculations. Original CINC,
CRSTM and CRST are used only in local native reference builds; provenance is
recorded in `tests/fixtures/cuminc.json` and `tests/fixtures/gray.json`. No
original source, binaries or documentation are bundled. The downloaded archive
contains no explicit redistribution license notice. Its cited statistical
references include Aalen (1978), Kalbfleisch and Prentice (1980), and Gray (1988).

## MUHAZ

`muhaz.py` independently expresses the kernel hazard estimator and polynomial
boundary kernels attributed to H. G. Mueller and J. L. Wang (1994). The archive's
Fortran routines carry their copyright notice, with later modifications credited
to Dan M. Serachitopol. No original source or binaries are incorporated; unchanged
HAZDEN/IBNDS/KERNEL are used only in a local reference build. Numerical fixture
provenance is recorded in `tests/fixtures/muhaz-fixed.json`.

`pehaz.py` independently calculates piecewise-exponential event/person-time
ratios. The archived S function is executed unchanged in R only for local
validation; its source/extraction/driver hashes and runtime are recorded in
`tests/fixtures/pehaz.json`. No archived S code is bundled.

`kphaz.py` independently expresses Nelson/product-limit hazard differences and
variance increments. The archived S numerical calculation is run locally in R
with documented syntax adapters; source and adapter provenance are recorded in
`tests/fixtures/kphaz.json`. No archived S code is bundled.

`muhaz_mse.py` independently expresses the archived pilot-convolution diagnostic
criterion. Unchanged MSEMSE and its dependencies are run only in a local Fortran
oracle; source/driver hashes, compiler and floating-point flags are recorded in
`tests/fixtures/muhaz-mse.json`. No original Fortran code is bundled.

`muhaz_global.py` independently combines candidate scoring and fixed-bandwidth
estimation. Local validation invokes unchanged NEW_HAD/GLMIN and dependencies;
source/driver hashes and compiler flags are recorded in
`tests/fixtures/muhaz-global.json`. No original Fortran code is bundled.

`muhaz_local.py` independently implements pointwise bandwidth selection and
kernel regression smoothing. Unchanged NEW_HAD/LOCLMN/BSMOTH are used only in the
local oracle, whose driver explicitly handles undefined source diagnostics.
Source/driver hashes and compiler flags are in `tests/fixtures/muhaz-local.json`.
No original Fortran code is bundled.

`muhaz_neighbors.py` and `muhaz_knn.py` independently implement the archived
nearest-neighbor methods and fitting workflow. Local oracles run unchanged
KNNCEN/OLAFBW/KNNHAD and dependencies, with provenance recorded in
`tests/fixtures/muhaz-neighbors.json` and `tests/fixtures/muhaz-knn.json`.
No original Fortran code is bundled.

`muhaz_summary.py` independently presents the fields described by the archived
S `summary.muhaz`, adding explicit bypass, compatibility and convergence status.
No original S function is bundled.

`muhaz_plot.py` independently implements the archived MUHAZ, PEHAZ and KPHAZ
plot/overlay workflows using the optional Matplotlib dependency. Nonfinite
estimates are consistently displayed as gaps. No original S source is bundled.

## EXPSURV

`expsurv.py` independently implements survival-curve and quantile calculations
from E. Neely Atkinson's EXPSURV XLISP-STAT package. Its source readme permits
redistribution. Original code is used as local reference material and is not
bundled; the source hash and validation limits are recorded in `docs/expsurv.md`.

`expsurv_cutpoint.py` and `expsurv_cutpoint_plot.py` independently implement
EXPSURV's CHOOSE-CUT-PLOT workflow. The density guide has an explicit Gaussian
kernel and bandwidth rule in place of the original runtime's implicit helper.

`expsurv_alignment.py` independently implements the ACCEL-FAIL-PLOT and
PROP-HAZ-PLOT coordinate transformations and slider ranges. No original Lisp
source is bundled.
