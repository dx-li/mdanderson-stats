# Multc99 source recovery and conversion — October 8, 2026

The institutional [Multc99 2.1 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/Multc99/Multc99_V2.1.zip)
was retrieved with HTTPS verification: 450,705 bytes, SHA-256
`0fb107937ca619681539a5eee9db09cd549f42ba8ccbe7961cc7794691af6f6c`.
Contrary to earlier inaccessible-archive audits, it contains all C modules,
headers, an example transcript, README and build guide. Its source dates mostly
to 1997; the packaged program identifies version 2.1, April 26, 2012.
Original bytes remain ignored under
`research/raw/source-recovery-2026-10-08/multc99`.

The mathematical contract is now independently inspectable. `struct.c` and
`header.h` define elementary Dirichlet outcomes, compound-event aggregation,
conditional intersections and parent-count denominators. `boundary.c` defines
independent historical comparisons, fixed-weight historical mixtures,
upper-inclusive/lower-strict general boundaries and minimum-enrollment rules.
`compute.c` supplies mean/interval Dirichlet elicitation and posterior-width
planning. `simulate.c` supplies cohort and Poisson period-count monitoring;
`random.trial.c` supplies balanced randomized arms, reassignment and selection.
The boundary editor and delta/lambda export are also recovered. The fixed-null
Phase IIa subset remains in the existing independently validated Multc APIs.

## Independent validation

`tools/reference_multc99.py` verifies the archive SHA-256 before extracting
only named source/header files into a temporary directory. It builds unchanged
`compute.c`, `boundary.c`, `cdflib.c`, `zrorj.c` and `ranlib.c` with GCC 14.2.0,
C99, `-fcommon`, function/data sections, garbage-collection of unused sections,
and `-O0`. The authored harness is `reference_multc99_kernel.c`. No native
source is edited, Windows binary run or native file parser trusted as input.

The C harness produces 85 comparison probabilities, eight boundary tables,
three interval-elicitation priors and nine precision-planning sample sizes.
All eight stopping tables agree after normalizing unreachable +/-99 limits;
calibrated prior components differ by at most 9.01e-7, consistent with the
native elicitation tolerance. All nine planned sample sizes agree exactly.

The C comparison routine does not achieve its advertised 1e-6 accuracy for
all recovered cases: deviations reach 1.29036e-4, without an emitted warning
in this fixture. Those outputs are retained as diagnostics, not mathematical
truth. Independently, mpmath 1.3.0 at 45 decimal digits integrates the historical
Beta density against the experimental survival function, splitting at the
negative-margin support boundary. The Python library uses a different
experimental-density/logit-coordinate integration with direct complement
checks. All 85 probabilities agree within 3e-9 and reported quadrature errors
stay within 1e-9. The original kernel and high-precision fixtures regenerate
structurally unchanged with `python tools/reference_multc99.py --check`.
mpmath and GCC are reference-generation dependencies, not package dependencies.

137 focused checks cover these numerical references, conditional exclusion of
outside observations, strict/inclusive cutoff ties, bounded cutoff endpoints,
full 32-tape binary enumeration, a known exact stopping probability and sample
size, time-unit rescaling, all-arm termination, reassignment, conditional final
selection, fair three-way ties, manual-table replay, triangular uniform
difference curve data, immutable arrays and captured-input/report replay.

## Explicit corrections and compatibility boundaries

- Native boundary searches can exceed their allocated cap; Python bounds all
  searches and represents unattainable stops explicitly. Unseparated numerical
  cutoff comparisons raise; a uniform historical rate has an analytical mean
  comparison, preserving its cutoff ties without quadrature.
- Native simulation generates only Nmax-1 outcomes before declaring cap
  completion. Python counts every actually enrolled subject, including the
  final slot, while retaining the no-interim-stop-at-cap policy.
- Randomized initialization writes one element past arm-sized arrays.
  One-survivor reassignment writes the wrong index and can exceed its array;
  zero remaining slots can prematurely stop a trial with active arms. Python
  retains valid future-slot assignments and continues surviving arms.
- The source divides a conditional selection numerator by total arm enrollment
  rather than conditioning observations. Python uses the correct conditional
  posterior mean. Its sequential coin-flip ties bias multiway selection;
  Python uses explicit uniform ties.
- The source overwrites a lower hit when the same event also hits its upper
  boundary. Python retains both sides and all simultaneous event hits, using
  sparse pattern counts instead of an exponential dense breakdown allocation.
- The manual editor leaves conditional lower prefix tables stale after suffix
  changes. Python regenerates the declared lower run-back consistently.
- The plotting menu passes Lambda4 through an incompatible four-argument
  callback even though that function requires seven scalar parameters. Python
  computes the recovered probability directly and exports explicit-grid CSV.
- The precision planner can use impossible observed counts at small sample
  sizes. Python excludes those cases and keeps the documented source rounding
  toward a posterior mean nearer 0.5. Elicitation and planning have explicit
  bounded-search failures instead of unlimited native loops.
- Fixed-null Phase IIa code has inconsistent equality handling between first
  and later boundary rows, plus unbounded searches/uninitialized cap entries.
  The existing Python fixed-reference workflow uses its documented strict
  probability rules, declared minimum looks and completed cap observations.
- `Lambda5` is an unused randomized comparison function; no inspected menu
  calls it and the `Extended` setting is never enabled or used. Its historical
  shape typo is not presented as an additional recovered application workflow.

These are identifiable source defects or explicit numerical/interface
differences, not unknown statistical specifications. All recovered usable
general workflows have Python equivalents. Entry 3 moves to implemented;
entry 12 stays partial because this older C source does not establish the
different Multc Lean generated endpoint-time law or pending-outcome controller.

## Licensing

The README permits noncommercial extraction/adaptation and requires upstream
written permission for commercial source use. Its complete notice accompanies
the adapted Python modules, re-encoded from Windows-1252 to UTF-8. The project
remains mixed-license; this port does not turn restrictive upstream terms into
MIT terms. Original C implementations and the Windows executable are ignored
reference inputs and are not redistributed in the wheel/source distribution.
