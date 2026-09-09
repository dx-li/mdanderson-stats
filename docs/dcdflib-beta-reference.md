# Legacy beta reference audit

This audit establishes unchanged C/F77 evidence for the implemented [`cdfbet` and
`cumbet` interfaces](dcdflib-beta.md). The existing `cdf_beta` implements the
separately bounded F95 contract. The legacy implementation has separate wide-domain tests.

## Source contract

Modes 1–4 compute p/q, x/cx, a and b respectively. Source Y maps to cx.
Input shapes are positive with no finite upper cap. Computed shapes are searched
through [1e-100,1e100], compared with [1e-10,1e10] in the F95 Python API.
Both probability and coordinate pairs accept endpoints, with sum checks within
three machine epsilons. Quantile search spans [0,1]. The source's reference to
a chi-square integral in its P argument description is a documentation typo;
the executable evaluates the incomplete beta ratio.

Both languages correctly reject unsupported modes, unlike the legacy binomial
guard. Forward `cumbet` handles x<=0 and cx<=0 before calling `bratio`.
It does not propagate `bratio`'s error flag, and `cdfbet` sets status 0 after
forward evaluation. A success status alone is therefore insufficient evidence
of numerical validity.

Quantile inversion searches x when p<=q and cx otherwise, recomputing the other
coordinate as one minus the root. Shape inversions start at five. All inverses
use absolute tolerance 1e-50 and relative tolerance 1e-8. These tolerances can
accept materially wrong answers at tiny coordinates or very narrow transitions.

## Reproduction and ordinary validation

`tools/reference_dcdflib_beta.py` compiles the unchanged two C sources and header,
and all 64 F77 source files. The fixture retains source/archive SHA-256 hashes,
compiler versions/commands, driver text and empty adaptation lists. Drivers
initialize status to -999 and bound to zero; bound remains undefined by the
native contract on success. No original source or executable is bundled.

Each language records 256 ordinary cases: 64 tails and 64 calls per inverse
mode. Shapes independently span 0.5,1,2,10 and x spans 0.01,0.2,0.5,0.9.
All 512 calls complete with status 0. The fixture also retains ten invalid,
eight boundary and eleven wide-domain cases per language. The wide calls have
a three-second timeout, with timeout records distinguished from returned status.

The 588 tests compare all ordinary calls with the existing F95 Python API,
checking Python inverse parameters against generating values and native inverse
results through forward evaluation. Seventy-two integer-shape references are
checked independently with 120-digit binomial sums. The identity is
I_x(a,b)=Pr[Binomial(a+b-1,x)>=a], with both tails summed separately.

## Independent failure evidence

For uniform beta, a=b=1, x=p and cx=q exactly. At p=1e-100, both native
quantiles return x=5e-51 with status 0; the reflected small-q case returns
cx=5e-51. Even zero-probability endpoints return the same positive coordinate.
The existing F95 Python quantile preserves 1e-100 and exact zero endpoints.
A legacy implementation must retain those repairs over its broader shape domain.

For b=1, P=x**a. At x=0.5 and Q=1e-100, a 150-digit logarithmic calculation
gives a approximately 1.4426950408889634e-100, inside the legacy search bounds.
Both sources return the lower bound 1e-100 with status 0, an error above 30%.
The reflected b inversion has the same defect. This domain is outside the F95
Python shape bounds and is covered by the legacy implementation.

For a inversion at x=0.5, P=Q=0.5 and fixed b=1e90, symmetry requires a=b.
Both native sources instead return a approximately 1.0000000016027156e90 with
status 0. For that returned beta distribution, its exact mean a/(a+b) exceeds
0.5. A 150-digit calculation using variance ab/((a+b)**2*(a+b+1)) and Chebyshev's
inequality bounds P(X<=0.5) below 1e-70. The requested probability is 0.5;
the apparently small relative parameter error cannot justify success.

At x=0 or 1, matching endpoint probabilities do not identify a or b. Native
inversion returns the arbitrary initial shape five with status 0. At interior
x with a zero target tail, native underflow also produces finite shapes near
1957.5 with status 0 although no finite positive shape attains an exact zero
probability. The existing F95 Python API rejects these shape inversions.

## Successful wide cases and implementation requirements

Both native languages exceed the three-second timeout at a=b=1e308,x=0.5,
where symmetry gives exactly 1/2. The symmetric 1e200 case completes correctly.
The fixture also retains successful subnormal tails at shape 1e-308, with
800-digit power checks for both orientations, and a=1,b=1e200,x=1e-200 where
the upper tail approaches exp(-1). At a=1e-308,b=2e-308 and interior x=0.2,
the limiting pair b/(a+b),a/(a+b) agrees with the returned 2/3,1/3; the interior
first-order correction is far below one float ulp.

The legacy API must cover both small shapes, wide finite inputs, distinct
shape search bounds, direct complementary quantiles, exact endpoints and
inverse forward verification. Existing negative-binomial repairs cover useful
beta subdomains but not the full two-small-shape domain; simply mapping all
beta calls to that count interface would discard valid inputs. The
[legacy implementation](dcdflib-beta.md) now uses positive beta recurrences for this domain and includes 650 implementation tests and batching measurements.

See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
