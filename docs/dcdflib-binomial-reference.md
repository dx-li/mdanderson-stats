# Legacy binomial reference audit

This audit establishes unchanged C/F77 evidence for the implemented
[`cdfbin`/`cumbin` Python interfaces](dcdflib-binomial.md). The existing
`cdf_binomial` covers the separately documented F95
contract, including an independently reconciled F95 backup source.

## Contract differences

The legacy modes compute (1) paired tails, (2) successes s, (3) trials n, or
(4) chance pr and its complement. Source XN maps to n; PR/OMPR map to pr/cpr.
Counts are real, using I_pr(s+1,n-s) with exchanged probability tails.

Input n is strictly positive with no finite upper cap; input s lies in [0,n].
Computed s has the full [0,n] range, even when n exceeds 1e100. Computed n is
searched over [1e-100,1e100]. For n inversion, the source permits any nonnegative
input s before the search. Both probability pairs accept endpoints and use a
three-machine-epsilon sum check. These rules differ from the F95 count bounds.
The helper returns P=1, Q=0 whenever s>=n, including intermediate search points.

Both languages contain an impossible mode check: `which < 1 AND which > 4`.
Consequently invalid modes 0 and 5 pass validation with otherwise valid inputs,
perform no selected computation, and leave STATUS untouched. Drivers initialize
STATUS to -999, which survives these calls. The fixture's -999 is a driver
sentinel, not a native error code. Python must reject unsupported modes.

The languages differ in their success-count starting value. C uses s=5;
Fortran uses s=n/2. For n<5, the C initial value lies outside its own [0,n]
search interval, and its root-finder terminates the process with exit code 1
and `SMALL, X, BIG not monotone in INVR`. This is an execution failure, not a
returned numerical status. The corresponding Fortran and F95 Python calls work.

At s=n, P=1 does not identify the chance. Native mode 4 nevertheless reports
success with an arbitrary chance near one. The Python F95 implementation rejects
this case. The legacy Python port also rejects this case and documents its boundary choices.

## Reproduction and ordinary validation

`tools/reference_dcdflib_binomial.py` compiles unchanged archived C and F77 code.
The fixture retains source/archive hashes, compiler commands, driver text,
status-initialization policy and empty source-adaptation lists. No native source
or executable is distributed.

Each language has 156 ordinary calls: 48 paired tails and 36 calls per inverse
mode. All Fortran calls return status 0. C returns status 0 on 138 calls and
exits on the 18 success-count inversions with n<5. Eleven invalid-input cases,
nine boundary cases and seven wide-domain cases per language are also recorded.
Wide calls have a three-second limit; the current wide cases all completed.
The recorder distinguishes process errors and timeouts from returned statuses,
retaining exit codes and diagnostics when available.

The 362 tests compare the ordinary calls with the existing F95 Python interface
on their overlapping domain, including Python answers where C exits. Integer
counts receive independent 120-digit binomial PMF-sum checks; s=n uses the exact
unit-CDF endpoint rather than a cancellation residue from the finite-precision
sum. Python inverse parameters are checked against their generating values.
Native inverse outputs are evaluated forward separately: some ordinary chance
inversions have relative small-tail errors near 4e-7, despite status 0.

## Independent false-success evidence

For s=0,n=1, the upper probability Q equals pr exactly. A target Q=1e-100
returns pr about 5e-9 in both native languages with status 0. The existing F95
Python inverse returns 1e-100. At P=0 in the same distribution, native mode 4
returns a positive complement near 5e-9 instead of the exact zero endpoint.

The source searches PR when P<=Q and otherwise searches OMPR, recomputing the
other coordinate by subtraction from one. In the small-Q example it therefore
searches a complement near one; its relative root tolerance does not preserve
the tiny desired pr. Direct complementary beta inversion avoids that loss.

For s=0 and pr=0.5, P=0.5**n. A 150-digit logarithmic calculation gives
n about 1.4426950408889634e-100 when Q=1e-100. Both native languages return the
lower search bound 1e-100 with status 0, a relative parameter error above 30%.

For n=1e200, pr=0.5 and target P=Q=0.5, C returns s about
5.000000024426175e199 with status 0; Fortran returns 5e199. For the C result,
a 150-digit calculation of the variance and distance above the mean gives a
Chebyshev upper-tail bound below 1e-180. Its lower CDF therefore cannot equal
one-half. The source parameter tolerance is insufficient for this narrow CDF
transition even though the relative count error appears small.

Successful wide cases also matter: both languages return P=Q=0.5 at the symmetric
1e308 input and retain the upper tail near 6.931471805599447e-309 at s=0,
n=1e-308, pr=0.5. At n=1e200 and pr=1e-200, the zero-success lower tail is near
exp(-1). The legacy Python interface preserves these examples while repairing the failures.

## Implementation coverage

The separate [legacy API](dcdflib-binomial.md) implements the s/n search bounds,
complement handling, endpoint semantics and inverse forward verification, with
394 implementation tests and recorded batching measurements. This reference
fixture remains unchanged evidence of native behavior.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
