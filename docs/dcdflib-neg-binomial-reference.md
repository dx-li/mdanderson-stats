# Legacy negative-binomial reference audit

This is reference evidence for the forthcoming C/F77 `cdfnbn`/`cumnbn` port.
The wider legacy Python interface is **not yet implemented**. The existing
`cdf_neg_binomial` implements the separately documented F95 domain and semantics.

## Executable contract

The archived `cdfnbn` modes compute (1) p/q, (2) failures, (3) successes, or
(4) success probability and its complement. Source S means failures; source XN
means successes. A Python interface should retain the existing f/s naming and
map source PR/OMPR to pr/cpr explicitly.

Both count inputs allow all finite nonnegative values; count searches are
[0,1e100]. Probabilities and their complements lie in [0,1], and both pair sums
are checked within three machine epsilons. Inverse Q must be strictly positive,
while inverse P may be zero. These differ from some F95 constraints. The source
uses the continuous incomplete-beta identity I_pr(successes, failures+1), so
count inversion is not an integer-valued quantile.

`cumnbn` calls `cumbet` directly with these shapes. At zero successes and pr=0,
both native languages return P=0, Q=1 with status 0. At zero successes and pr>0,
they return P=1, Q=0. The existing Python F95 interface deliberately returns P=1
for zero successes even at pr=0: no trials are needed to achieve zero successes.
The legacy port needs an explicit documented choice at this boundary.

## Reproducible evidence

`tools/reference_dcdflib_neg_binomial.py` compiles the unchanged archived C and
F77 sources. The fixture records archive/source hashes, compiler commands,
drivers and the absence of source adaptations. Source files and binaries are
not distributed. Each language supplies:

- 180 ordinary cases, 45 per computation mode, all reporting status 0;
- nine invalid-input cases, including separate tail-pair and chance-pair errors;
- twelve boundary cases covering zero failures, zero successes and pr endpoints;
- seven wide-domain or small-probability cases.

Wide calls have a three-second subprocess limit. A timeout is recorded with
null status/result and an explicit execution outcome; it is not a native return
code or evidence of a numerical answer. At failures=successes=1e308, pr=0.5,
both languages time out. The initial audit attempt also exhausted a
30-second call limit. The native result at counts=1e200 and pr=0.5 is P=Q=0.5.
A timeout does not prove that the native computation could never finish.

The small-target cases independently demonstrate false success:

- With zero failures and one success, P=pr exactly. Solving P=1e-100 returns
  pr=5e-51 with status 0, over 49 orders of magnitude too large. The existing
  F95 Python inverse returns 1e-100 on this overlapping domain.
- With zero failures and pr=0.5, P=0.5**successes. A target Q=1e-100 therefore
  requires about 1.4426950408889634e-100 successes. Both native languages return
  zero with status 0. A 150-digit logarithmic identity establishes the positive
  answer independently.
- Solving P=0 with one failure and one success returns pr=5e-51 rather than the
  exact endpoint zero. The native root finder uses absolute tolerance 1e-50.

Other wide cases retain successful native evidence: one success, 1e200 failures
and pr=1e-200 gives Q near exp(-1); zero failures, 1e-308 successes and pr=0.5
gives Q near 6.931471805599457e-309. The wider Python implementation must preserve
these useful domains while resolving the failures above.

## Validation scope and remaining work

The 410 tests compare all 360 ordinary C/F77 records against the existing F95
Python interface on its overlapping domain, independently check integer-count
tails with 120-digit negative-binomial PMF sums, and preserve boundary and
false-success evidence. Native inverse outputs are also evaluated forward.
The independent sums use the smaller supplied chance coordinate to retain a
small complement. These tests do not validate unimplemented wide-domain APIs.

Still required: a separate legacy interface, all count search bounds, both
complementary chance coordinates, numerical handling of extreme beta shapes,
explicit degenerate inversions, independent wide-domain tests and batching
measurements. CDFLIB90 remains partial and its ten pending legacy distribution
entry points remain pending after this reference checkpoint.
