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
