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

`expsurv_scatter.py` independently implements SCAT-KM selection and linked
survival views, using Matplotlib rectangle controls in place of XLISP-STAT
mouse modes. Empty selections explicitly clear the survival curve.

`expsurv_event.py` independently implements SCAT-EVENT geometry and status
markers, sharing matrix controls with SCAT-KM. Initial and empty selections are
explicitly rendered; no original code is bundled.

`expsurv_box.py` and `expsurv_box_plot.py` independently implement SCAT-BOX
life-table geometry and linked display. Source interpolation is explicit,
and selections without failures have defined empty geometry.

`expsurv_data.py` replaces EXPSURV file/global-assignment helpers with explicit
named tables. `expsurv_simulation.py` independently implements GEN-EXPO-DATA and
GEN-EXPO-EXAMPLE distributions using NumPy random streams.

`generate_exploratory_data` independently implements GEN-DATA. Its sample-SD
convention was also checked against XLISP-STAT stats.lsp at revision
f1bea6053df658ee48612bf1f63c35de99e2c649; reference runtime code is not bundled.

## CTA

`cta.py` independently implements the CHISQT formulas in the CTA archive,
whose Fortran source is dated February 2, 1998. Original source and executables
are not bundled. [The retained LEGALITIES notice](notices/mdanderson-cta-LEGALITIES.txt)
allows use and redistribution subject to its terms and requires permission for
incorporation of its code into a commercial package sold to others. This Python
implementation is explicitly independent; no broader license is asserted here.
Source provenance and native-reference validation are documented in `docs/cta.md`.

`cta_mcnemar.py` independently implements CTA MCNEMAR pairwise and aggregate
formulas. Zero-discordance cases are defined explicitly and accurate gamma tails
replace the source's forced approximation; the same CTA notice applies.

`cta_kappa.py` independently implements CTA KAPPA, retaining source variance
indices through an explicit legacy option. Default variances use the multinomial
delta method; source provenance and corrections are documented in `docs/cta.md`.

`cta_diagnostic.py` independently implements SENSPEC probabilities. Probability
standard errors are the default; the source count-standard-deviation formula is
available explicitly. The retained CTA notice applies.

`cta_odds.py` independently implements RELRISK odds ratios. Corrected log-Wald
limits are default; the source CDF multiplier is available explicitly. The
retained CTA notice applies.

`cta_fisher.py` independently expresses FISHXT fixed-margin probabilities with
accurate hypergeometric evaluation and explicit source tail/cutoff compatibility.
The retained CTA notice applies.

`cta_binomial.py` independently implements BINCOMP using the existing binomial
tail engine in place of BINOP/BLFEW loops. Source event selection and duplicate-tail
reporting are retained explicitly. The retained CTA notice applies.

`cta_study.py` and `cta_report.py` independently replace the CTA program
selection loop and consolidated numerical output with reusable Python settings
and explicit report files. The retained CTA notice applies.

`cta_detail_report.py` independently supplies CHISQT cell listings and
FISHXT/BINOP term listings using shared numerical calculations. The retained
CTA notice applies.

## RANLIST

`ranlist_random.py` independently implements the phrase mapping and modular
random-stream equations used by RANLIST, whose source credits Barry W. Brown,
James Lovato and Benjamen Rundell and identifies copyright 1988/1992 for the
University of Texas M. D. Anderson Cancer Center. The manual's legal section
identifies copyright 1990 and is retained in
`notices/mdanderson-ranlist-LEGALITIES.txt`; the original readme is retained in
`notices/mdanderson-ranlist-readme.txt`. Their commercial-source-use conditions
are not replaced by a blanket package license. Original Fortran and executables
are local validation material and are not bundled. The generator comments cite
L'Ecuyer and Cote, ACM Transactions on Mathematical Software 17:98–111 (1991).

`ranlist_unrestricted.py` independently expresses IGTUT weighted assignment
and the GENLST/WRKLST cumulative-weight rules. The retained RANLIST notices apply.


## RANDLIB

The RANDLIB generator implementation is an independent Python implementation
of the documented L'Ecuyer–Cote modular recurrence and stream controls. The
archived Fortran 77 and Fortran 95 sources are compiled locally for comparison;
original source and compiled reference programs are not bundled.

The archive distinguishes the authors' public-domain contributions from code
associated with ACM publications. Its original terms and references are retained
in [RANDLIB90 LEGALITIES](notices/mdanderson-randlib90-LEGALITIES.txt) and the
[Fortran readme](notices/mdanderson-randlib-fortran-readme.txt). In particular,
the base generator is attributed to P. L'Ecuyer and S. Cote, “Implementing a
Random Number Package with Splitting Facilities,” ACM Transactions on
Mathematical Software 17(1), 98–111 (1991). The retained ACM policy addresses
use, attribution and redistribution, including direct commercial advantage.
This package does not replace those terms with a blanket license.


RANDLIB's exponential sampler also follows the documented algorithm SA of
J. H. Ahrens and U. Dieter, “Computer Methods for Sampling From the Exponential
and Normal Distributions,” Communications of the ACM 15(10), 873–882 (1972).
The archived attribution and ACM policy are retained in the RANDLIB notices
linked above; the original native implementations are used only as local
validation references.

RANDLIB's normal sampler independently expresses algorithm FL (M=5), using the
archived numerical tables, attributed to J. H. Ahrens and U. Dieter,
“Extensions of Forsythe's Method for Random Sampling from the Normal
Distribution,” Mathematics of Computation 27(124), 927–937 (1973).
The C and Fortran tables differ in several printed constants; this package
retains both numerical variants for explicit source compatibility. The original
source attribution and terms remain available in the RANDLIB notices above.

RANDLIB's gamma sampler independently expresses the archived algorithms GD
and GS, attributed to J. H. Ahrens and U. Dieter: “Generating Gamma Variates
by a Modified Rejection Technique,” Communications of the ACM 25(1), 47–54
(1982), and “Computer Methods for Sampling from Gamma, Beta, Poisson and
Binomial Distributions,” Computing 12, 223–246 (1974). The archived numerical
coefficients and source corrections are reflected in the Python implementation;
the retained RANDLIB notices and their ACM provisions apply as described above.

RANDLIB's central/noncentral chi-square and F samplers independently express
GENCHI, GENNCH, GENF and GENNF's gamma/normal composition and source rounding
rules. They reuse the attributed GS/GD and FL algorithms above. Native reference
executables are compiled locally from unchanged archived source and are not
bundled; the original RANDLIB attribution and terms remain retained.

RANDLIB's beta sampler independently expresses Cheng algorithms BB and BC,
including the archive's numerical coefficients and overflow corrections.
The source attributes them to R. C. H. Cheng, “Generating Beta Variates with
Nonintegral Shape Parameters,” Communications of the ACM 21, 317–322 (1978).
The retained RANDLIB notices include the applicable source attribution and ACM
policy; this package does not replace those terms with a blanket license.

RANDLIB's binomial sampler independently expresses IGNBIN's inversion and
BTPE algorithms, attributed to V. Kachitvichyanukul and B. W. Schmeiser,
“Binomial Random Variate Generation,” Communications of the ACM 31(2),
216–222 (1988). Source coefficients, acceptance tests and inversion restart
rules are retained. The RANDLIB source attribution and ACM terms referenced
above remain available; original native source/executables are not bundled.

RANDLIB’s Poisson sampler independently expresses IGNPOI / RANDOM_POISSON,
attributed to J. H. Ahrens and U. Dieter, “Computer Generation of Poisson
Deviates From Modified Normal Distributions,” ACM Transactions on Mathematical
Software 8(2), 163–179 (1982). Source coefficients, factorials and rejection
rules are retained. The RANDLIB notices and ACM terms referenced above remain
applicable; original native source and executables are not bundled.

RANDLIB’s negative-binomial sampler independently expresses the IGNNBN /
RANDOM_NEGATIVE_BINOMIAL gamma–Poisson mixture. The source cites Luc Devroye,
Non-Uniform Random Variate Generation, Springer-Verlag, New York (1986),
page 480. It reuses the gamma and Poisson algorithms attributed above;
the retained RANDLIB source notices remain available.

RANDLIB’s multinomial sampler independently expresses GENMUL /
RANDOM_MULTINOMIAL’s conditional binomial algorithm. The source cites Luc
Devroye, Non-Uniform Random Variate Generation, Springer-Verlag, New York
(1986), page 559. It reuses the binomial algorithm attributed above and
retains source residual-category and early-exit rules in legacy mode.

RANDLIB multivariate-normal preparation and sampling independently express
SETGMN / GENMN and the Fortran 95 equivalents. Legacy factorization retains
the archived LINPACK SPOFA and BLAS SDOT arithmetic; SPOFA credits Cleve
Moler, University of New Mexico and Argonne National Laboratory (08/14/1978).
Sampling reuses the attributed normal generator above. Original source and
executables are not bundled; retained RANDLIB notices remain available.

RANDLIB phrase seeding reuses the PHRTSD arithmetic already attributed for
RANLIST, with the archived C character table and its defined final-character
behavior available explicitly. Time seeding expresses the Fortran 95
USER_SET_GENERATOR time-of-day formatting and hash composition.

## MULTINOMPOW

The exact multinomial power implementation independently expresses the
Pearson and likelihood-ratio ordering methods in MULTINOMPOW version 1,
Barry W. Brown, MD Anderson Department of Biomathematics (May 2003;
source banner copyright 2002). Original redistribution and warranty terms
are retained in [LEGALITITES](notices/mdanderson-multinompow-LEGALITITES.txt).
The archived ACM `gamln` implementation is not copied or bundled; Python
uses SciPy special functions. Source defects and numerical differences are
identified in [the implementation notes](docs/multinomial-power.md).

## TDTASP

The TDTASP genetic model independently expresses two-locus Mendelian
transmission and the family calculations in version 1.1 (April 2003),
Barry W. Brown and Dan Serachitopol. Copyright 2003, The University of Texas
M.D. Anderson Cancer Center, Department of Biomathematics. Original
[LEGALITIES](notices/mdanderson-tdtasp-LEGALITIES.txt) are retained.
The explicit ASP compatibility option preserves the original duplicate
transmission weighting and cutoff; the default corrects that behavior.
Original source, executables and archived ACM numerical code are not bundled.


## CDFLIB90

The beta, binomial, normal, gamma, chi-square, Poisson, negative-binomial, Student's t, F and
noncentral chi-square/noncentral F/noncentral t CDF/inversion interfaces independently express CDFLIB90 1.2,
by Barry W. Brown, James Lovato and Kathy Russell, with the Fortran 95
conversion by Dan Serachitopol. The manual credits copyright 2002 to the
University of Texas M.D. Anderson Cancer Center, Department of Biomathematics.
Original [LEGALITIES](notices/mdanderson-cdflib90-LEGALITIES.txt) are retained.
The archived Fortran/C sources, binaries and ACM implementations are not
bundled. Python uses SciPy numerical kernels and independently written
array-based searches. Numerical differences and recorded native defects are
explained in [the implementation notes](docs/cdflib90.md).

The legacy `cdff` and `cumf` interfaces also independently implement the
archived DCDFLIB 1.1 F contracts by the same authors. Both C and Fortran reference
implementations are compiled unchanged for validation and remain outside the
package. Python result/error conventions and numerical repairs are documented
in [the legacy F notes](docs/dcdflib-f.md).

The legacy `cdffnc` and `cumfnc` interfaces independently implement the same
archive's noncentral F contracts, including both df inversions. Unchanged C/F77
references, contract differences and numerical repairs are documented in
[the legacy noncentral F notes](docs/dcdflib-nc-f.md); the same retained notices
apply, and the original sources and binaries are not bundled.

The legacy `cdfnor` and `cumnor` interfaces independently implement the archived
DCDFLIB normal contracts using SciPy kernels and array arithmetic. Original C/F77
code is used only for reference validation and is not bundled. The same retained
CDFLIB90 notices apply; [legacy normal notes](docs/dcdflib-normal.md) document
contract differences and numerical repairs.

The legacy `cdft` and `cumt` interfaces independently implement the archived
DCDFLIB Student t contracts using SciPy kernels and vectorized arithmetic.
Unchanged original C/F77 references are used only for validation and are not
bundled. The retained CDFLIB90 notices apply; [legacy Student t notes](docs/dcdflib-t.md)
document wider domains, numerical repairs and validation.

The legacy `cdfgam` and `cumgam` interfaces independently express the archived
C/F77 gamma contracts using SciPy special functions and logarithmic expansions.
No archived numerical kernel is copied or bundled. The retained CDFLIB90 notices
apply; [legacy gamma notes](docs/dcdflib-gamma.md) document reference validation,
source contract differences and numerical repairs.

The legacy `cdfchi` and `cumchi` interfaces independently express the archived
C/F77 chi-square contracts through gamma and exponential-integral identities.
No archived numerical kernel is copied or bundled. The retained CDFLIB90 notices
apply; [legacy chi-square notes](docs/dcdflib-chisq.md) describe independent
validation, bounded inversions and half-value underflow repairs.

The legacy `cdfpoi` and `cumpoi` interfaces independently express the archived
C/F77 Poisson contracts through the incomplete-gamma identity. No archived
numerical kernel is copied or bundled. The retained CDFLIB90 notices apply;
[legacy Poisson notes](docs/dcdflib-poisson.md) describe unchanged references,
independent high-precision validation and numerical repairs.

The legacy `cdfnbn` and `cumnbn` interfaces independently express the archived
C/F77 negative-binomial contracts through beta, power and gamma-limit identities.
No archived numerical kernel is copied or bundled. The retained CDFLIB90 notices
apply; [legacy notes](docs/dcdflib-neg-binomial.md) document numerical methods,
independent validation and the explicit zero-success convention.

The legacy `cdfbin` and `cumbin` interfaces independently express the archived
C/F77 binomial contracts through the shared complementary beta, power and
gamma-limit kernels. The CDFLIB90 authorship and retained legal notice above
apply; [legacy notes](docs/dcdflib-binomial.md) document methods, boundary choices
and unchanged native reference evidence. No archived source is bundled.

The legacy `cdfbet` and `cumbet` interfaces independently express C/F77 beta
contracts through shared numerical kernels and positive beta recurrence
identities. The CDFLIB90 authorship and retained legal notice above apply;
[legacy notes](docs/dcdflib-beta.md) document independent validation, numerical
repairs and boundary conventions. No archived numerical code is bundled.

The legacy `cdfchn` and `cumchn` interfaces independently express the C/F77
noncentral chi-square contracts through shared central and compiled distribution
kernels, shifted-normal identities and Chernoff bounds. The CDFLIB90 authorship
and retained legal notice above apply; [legacy notes](docs/dcdflib-nc-chisq.md)
document numerical methods and independent validation. No archived code is bundled.

## STATTAB

Three discrete probability terms are independently implemented through the existing
beta/gamma factor kernels. Its structured numerical result layer maps all twelve
distributions and 42 computed groups to the existing validated CDFLIB APIs, with
source columns, extra probabilities and neighboring count rows. The
STATTAB request/session layer adds checked positional parsing and isolated saved
values. The console layer supplies distribution/formula help, list editing,
formatted reports and file selection. Python streams replace Fortran unit numbers;
optional confirmation occurs before file creation or truncation. Source and manual
reconciliation is complete; documented Python semantics repair native defects and
manual formula errors.
Source inventory and native-session validation evidence are retained. The archive's
manual and source banner identify version 2.0 despite the download's 1.3 label.
Authors include Barry W. Brown, David Gutierrez, James Lovato, Dan Serachitopol,
Marty Spears and John Venier. Its exact archived
[LEGALITIES](notices/mdanderson-stattab-LEGALITIES.txt), including the referenced
algorithm notices and warranty terms, is retained. Original Fortran source and
platform executables are not bundled in the Python package. These notices do not
replace the original terms with a different license.

## SPPCR source audit

The SPPCR 1.0 (January 2003) source, by Barry W. Brown, was compiled to establish
numerical reference evidence. Independent mathematical checks and source analysis
are provided. The modified Python likelihood fitting core implements mean estimation
and observed information, with explicit repairs to source boundary handling.
Frequency summaries add calibration, mutant frequency and delta-method uncertainty;
direct complementary groups and logarithmic formulas replace unstable subtraction
and powers, and unavailable uncertainty is explicitly marked.
The Python sampling layer adds stable model probabilities and explicit-state
NumPy binomial generation in place of clock seeding and single-precision draws;
historical random-sequence equivalence is not claimed. Bootstrap generation and
replicate fits retain all observations and mark undefined frequencies explicitly.
Centered, scaled population summaries replace cancellation-prone raw squared sums.
Confidence limits use the source multiplier and observed centers, with explicit
support clipping and unbounded reciprocal calibration limits replacing invalid
wrapped or negative limits. The batch reader retains explicit DNA-unit conversion
and valid-file semantics, while checking identities/counts, disclosing omitted
alleles and repairing sign loss, stale-storage dependence and invalid progenitor
indices. FileMaker-style numeric rows add strict CSV field boundaries, repeated
identity validation and checked native output limits. Python tokenization reuses
the package's checked CDFLIB lexer. Interactive entry reuses the tested CDFLIB
console with native field order/ranges, explicit stream ownership and bounded
corrections instead of invalid identity continuation. Labeled analysis and replicate
reports preserve original counts, both DNA units and boundary diagnostics, and
replace fixed-width overflow and ambiguous unavailable values with explicit output.
Native RNG draws and state have been reconciled with existing RANDLIB. An explicit
legacy sampling path retains float32 probabilities and source cell ordering;
RANDLIB repairs the source all-stream reseeding defect and rejects unsafe inputs.
Truth-generation designs preserve native normalization and model DNA units, with
stable weight scaling and explicit validation replacing invalid truth parameters.
Truth dialogue preserves field order and numeric ranges with bounded corrections;
parameter reports disclose units and simulation choices without implicit file writes.
Reusable observed/truth analysis workflows use explicit modern or historical RNG
state and always draw initial simulated data from truth, repairing the source
bootstrap-choice dependence on uninitialized counts.
File workflows use bounded read-only input and complete staged report files with
exclusive publication by default and explicit replacement. A repeated four-mode
menu and seeded CLI use caller-owned streams, preserve prior results on EOF, and
reset per-analysis choices. Interactive output selection retains q/r/o/a choices
and basename conventions, while collecting all choices before staged publication,
protecting inputs and named active streams, and bounding append reads. The archive includes
misplaced SOGS documentation and examples, which are not SPPCR validation evidence.
Its exact [Legal.doc](notices/mdanderson-sppcr-Legal.doc.txt) is retained. Native
source and historical binaries are not bundled in the Python package.

## SURVAN

`survan_km.py` implements its Kaplan–Meier and Simon–Lee confidence calculations.
`survan_cox.py` and `survan_cox_likelihood.py` implement its Breslow-tie Cox
likelihood and inference with new scaled numerical and separation calculations.
`survan_baseline.py` implements the Kalbfleisch–Prentice survivor calculation
with log-domain roots and explicit limiting cases.
`survan_descriptive.py` implements frequency grouping and descriptive summaries,
with stable moments and an explicit legacy standard-error field.
`survan_tests.py` implements SURVAN's log-rank and Gehan–Breslow calculations
with NumPy risk tables and covariance eigendecomposition. This adaptation is
maintained independently of the original authors. Original copyright, use terms
and warranty text are preserved in
[`notices/mdanderson-survan-LEGALITIES.txt`](notices/mdanderson-survan-LEGALITIES.txt).
Those terms are not replaced by a different project license. Source provenance
and validation are described in `docs/survan.md` and `docs/survan-sources.json`.
Original Fortran programs and binaries are not redistributed.

## BLiP

`blip.py` implements BLiP standard distribution-plot geometry and rendering.
Original source: Lee, J. Jack and Tu, Z. Nora (1997), “A Versatile
One-dimensional Distribution Plot: The BLiP Plot,” The American Statistician
51:353–358. The archive permits noncommercial use and distribution with source
citation; its [readme](notices/mdanderson-blip-readme.txt) is preserved. This
independent adaptation does not replace those terms with another license.
Archive hashes and numerical comparisons are documented in `docs/blip.md` and
`docs/blip-sources.json`. Original S code and archives are not redistributed.

## EVENTCHART

`eventchart.py` and `eventchart_goldman.py` independently implement event conversion,
calendar/interval/Goldman timelines and the original current-date boundary geometry
from the MD Anderson EVENTCHART distribution by J. Jack Lee, K. R. Hess and
J. A. Dubin. See Lee JJ, Hess KR, Dubin JA, “Extensions and applications of
event charts,” The American Statistician 54:63–70 (2000). Source provenance and
implementation differences are in `docs/eventchart.md` and
`docs/eventchart-sources.json`. The original S source and example data are not
redistributed. The downloaded readme identifies authors and references but does
not include a separate software license grant.

## SOGS

`sogs.py` implements chromosome recombination, screening and offspring eligibility
from SOGS by Michael M. Weil, Barry W. Brown and Dan M. Seachitopol. Copyright
(1997), The University of Texas, M. D. Anderson Cancer Center, Department of
Biomathematics. The [original legal notice](notices/mdanderson-sogs-Legal.txt)
permits copying/distribution under its stated terms, permits non-commercial
source reuse and requires written permission for commercial source use. These
terms are preserved; this adaptation does not replace them with another license.
Source archives and original implementation files are not redistributed. See
`docs/sogs.md` and `docs/sogs-sources.json` for methods and provenance.
