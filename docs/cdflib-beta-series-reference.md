# CDFLIB beta-series audit

The unchanged F95 `apser`, `fpser` and `bpser` routines are recorded with source
provenance and independent high-precision beta-integral checks: 116 calls,
111 completed and five timed out. The [fpser](cdflib-fpser.md) and
[apser](cdflib-apser.md) ports are implemented; `bpser` remains pending.
CDFLIB90 is partial.

| Helper | Mathematical result | Source's stated series domain |
|---|---|---|
| `apser(a,b,x,eps)` | I_(1-x)(b,a), the upper beta tail | a ≤ min(eps,eps*b), b*x ≤ 1, x ≤ 0.5 |
| `fpser(a,b,x,eps)` | I_x(a,b), the lower beta tail | b < min(eps,eps*a), x ≤ 0.5 |
| `bpser(a,b,x,eps)` | I_x(a,b), the lower beta tail | b ≤ 1 or b*x ≤ 0.7 |

Positive shapes and valid probability coordinates are used. Tests check the
stated domains with Decimal arithmetic for every positive-tolerance fixture
case, including comparisons whose products would underflow in float64. Separate
zero/negative-tolerance probes retain their actual native outcomes.

## Findings relevant to the port

* `apser` returns infinity at x=0, although its mathematical upper tail is one.
  The source computes log(x) without an endpoint branch.
* With a=the smallest positive float64 and b=1e-309, `apser` also returns
  infinity at interior coordinates. Its intermediate psi(b) overflows, although
  the final tail is finite, approximately a/b = 4.9406564584e-15.
* `fpser(1,1e-15,1e-308,5e-15)` returns zero instead of 1e-323. Its power-factor
  cutoff uses the smallest normal value before the final probability is formed.
* `fpser` skips its x**a calculation when a ≤ 1e-3*eps. At x=0 it consequently
  returns b/a instead of zero. Recorded examples include a=1e-20,b=1e-40 and
  a=1e-309,b=the smallest positive float64.
* `bpser(1,0.5,x,5e-15)` exceeds the three-second native limit at x=1 and at
  x=1-1e-12. Both satisfy its stated domain. The exact result is
  1-sqrt(1-x), so the endpoint is one and the nearby value is finite.
* Negative eps causes a timeout in each helper's recorded probe. With a negative
  stopping threshold, even a term that underflows to zero cannot satisfy the
  stopping comparison. Zero tolerance completes in the three recorded cases;
  this does not establish a general convergence guarantee for zero tolerance.
* A loose fpser tolerance of 1e-3 gives about 0.0573% relative error for
  a=1,b=1e-20,x=0.5. This lies within the requested tolerance and is distinguished
  from the endpoint failures.

## Independent evidence

[Tests](../tests/test_cdflib_beta_series_reference.py) use 800-digit Decimal
arithmetic. They integrate the binomial expansion of the beta density term by
term, with independent log-gamma recurrences and Stirling coefficients for the
normalizing beta integral. The power expansion is used only in the recorded
small-coordinate domain. Exact formulas handle unit shapes and endpoints;
uniform, half-shape and symmetric-midpoint identities check the oracle itself.

[The fixture](../tests/fixtures/cdflib_beta_series.json) retains every native
outcome, source and archive hashes, the complete driver, compiler version and
compile arguments. The archived constants and mathematics modules are compiled
unchanged with `-O0 -ffp-contract=off -fcheck=all`. Timeouts are recorded explicitly;
nonfinite native values are JSON strings rather than fabricated finite values.

```sh
uv run python tools/reference_cdflib_beta_series.py
uv run pytest tests/test_cdflib_beta_series_reference.py
```

The archive SHA-256 remains
`2f5dd397b93546222a3b31e02073abeee1fc213cea75768c34b17e06c8264a3b`.
No native source or executable is added to the wheel, and the existing CDFLIB
legal notice remains packaged. This audit adds evidence, not public Python APIs.
