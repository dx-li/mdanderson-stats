# CDFLIB incomplete-gamma support audit

The unchanged F95 `rcomp`, `gratio` and `grat1` routines now have 162 recorded
native cases and independent mathematical checks. The
[gamma scaling factor](cdflib-gamma-factor.md) implements `rcomp`, and the
[incomplete-gamma port](cdflib-incomplete-gamma.md) implements `gratio` and
`grat1`. CDFLIB90 remains partial.

| Procedure | Source contract |
|---|---|
| `rcomp(a,x)` | exp(−x) x^a / Gamma(a); returns zero for x≤0 |
| `gratio(a,x,p,q,ind)` | Regularized incomplete-gamma tails; nonnegative a,x, not both zero. ind=0 requests full accuracy, ind=1 about six digits, other integers about three digits |
| `grat1(a,x,r,p,q,eps)` | Incomplete-gamma tails for a≤1, with caller-supplied r=exp(−x)x^a/Gamma(a) and convergence tolerance eps |

The driver initializes output variables to −999 before each call so an
unassigned error output remains observable. It computes r using native `rcomp`,
records that value, and optionally scales it before calling `grat1`. These are
driver operations; the archived library and constants are compiled unchanged.

## Findings that affect the port

* `gratio` rejects negative inputs and (0,0) by setting p=2 and leaving q
  unassigned. It also uses this return for equal a=x=1e100 and 1e308, where the
  mathematical tails round to 1/2. Its header's statement about a*x=0 is broader
  than its executable error condition: (0,1) returns (1,0), and (1,0) returns (0,1).
* Both tail routines test the product a*x for zero. Positive inputs whose
  product underflows therefore enter an endpoint branch. For a=1e-100,x=1e-300,
  `gratio` returns (0,1) instead of p rounding to one and q≈6.9019831223e-98.
  `grat1` similarly discards a representable p at a=1/2,x=5e-324.
* In the off-center large-shape branch, `gratio` uses +105 as the leading
  Stirling polynomial coefficient, whereas `rcomp` uses −105. This multiplies
  the computed smaller tail by exp(1/(6a)). Recorded a=20 errors are about
  0.837%; a=100 errors are about 0.167%. Exact integer-shape gamma tails verify
  both the correct values and this specific bias.
* At a=x=20, lower-accuracy `gratio` modes return p≈0.47027 instead of
  0.5297427332. The low-order center branches negate the leading d0 coefficient;
  the full-accuracy center branch does not exhibit this error.
* `grat1` uses r in its continued-fraction branch (x≥1.1 in the recorded a=0.1
  cases), so multiplying r scales q. Its small-x branch ignores r. The port
  must account for this input contract rather than treating r as an arbitrary
  ignored compatibility argument.
* Zero or negative eps at a=0.1,x=1.1 produces NaN outputs after unsuccessful
  continued-fraction iteration. The Python interface needs a valid tolerance
  domain and bounded convergence.
* Recorded `rcomp` values on the nonnegative domain agree with independent
  evaluations, including finite values at a=x=1e308. Its behavior at a=−1/2
  is also recorded, but that out-of-gamma-shape input is not evidence for a
  general negative-shape contract.

## Independent evidence

[The tests](../tests/test_cdflib_incomplete_gamma_reference.py) use 800-digit
Decimal arithmetic, the gamma defining series, finite Poisson sums for integer
shapes, and an upper-tail integration-by-parts expansion with an explicit
first-omitted-term bound. A separate log-gamma recurrence/Stirling calculation
provides scaling factors. Tiny shapes use a local log-gamma expansion to avoid
subtracting an oracle approximation error larger than the requested tail.

Exponential and half-shape error-function identities check the oracle, and an
independent exponential-integral series checks tiny-shape upper tails. The
large equal-shape limits are used only at a≥1e100, where their leading
O(a^−1/2) correction is below float64 resolution. These checks distinguish
documented source failures from ordinary rounding; the audit does not bless
native output merely because it was reproduced.

[The fixture](../tests/fixtures/cdflib_incomplete_gamma.json) retains archive and
source hashes, compiler version, driver text, compiler arguments, and every
outcome. All 162 calls completed. Nonfinite values use JSON strings.

Regenerate with:

```sh
uv run python tools/reference_cdflib_incomplete_gamma.py
uv run pytest tests/test_cdflib_incomplete_gamma_reference.py
```

The archive SHA-256 is
`2f5dd397b93546222a3b31e02073abeee1fc213cea75768c34b17e06c8264a3b`.
Compilation uses gfortran with `-O0 -ffp-contract=off -fcheck=all`. The existing
CDFLIB legal notice remains packaged; no native source or executable is added
to the wheel.
