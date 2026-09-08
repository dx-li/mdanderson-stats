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
