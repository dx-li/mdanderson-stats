# mTPI isotonic posterior interval audit

## Source contract

The primary-paper BioC text is `research/raw/TPI/paper-bioc.xml`. In the
discussion passage at source offset 29582, the authors state that intervals
for the isotonic-transformed dose probabilities can be obtained by (1) drawing
independently from each dose's posterior beta distribution, (2) performing an
isotonic transformation on each posterior sample, and (3) obtaining posterior
intervals numerically from the transformed samples. The passage explicitly
distinguishes inferential MTD estimation from trial-design decisions.

The existing `MTPIDesign.posterior` uses the independent uniform beta prior,
giving `Beta(y+1, n-y+1)`. Existing `select_mtd` at
`src/mdanderson_stats/mtpi.py:169–206` fits isotonic regression once to posterior
means for treated doses and selects the closest admissible dose. It does not
compute posterior intervals. The new computation leaves this selection
behavior unchanged. For a supplied untried dose (`n=0`), it uses the model's
prior `Beta(1,1)` and includes that dose in the isotonic transformation.
The paper does not state whether interval estimation spans untried dose levels;
using the complete supplied grid is an explicit Python scope choice.

## Explicit Python choices and limits

The paper does not specify isotonic weights, Monte Carlo size, or the empirical
quantile convention. The implementation uses equal isotonic weights by
default, permits a caller-supplied positive weight per dose, and uses
equal-tailed `numpy.quantile(method="linear")` intervals at an explicit
confidence level. It uses an explicit NumPy `Generator`, bounded draw and
matrix sizes, and can optionally retain transformed draws. These choices are
not native defaults or spreadsheet parity claims.

## Validation plan

The one-dose case has no isotonic pooling, so its transformed interval must
converge to the ordinary beta posterior interval; the focused test compares
Monte Carlo quantiles against independently evaluated beta quantiles. A
multi-dose retained-draw test checks that every posterior sample is monotone,
including a dose with no observed patients. A budget-rejection test confirms
that an invalid work request is rejected without consuming RNG state.

Validation: focused posterior-interval tests (including one-dose beta quantiles
and the independent two-uniform isotonic projection identity) plus the existing
isotonic-selection test passed (seven tests). Targeted Ruff check/format and
mypy passed. The focused Python process took 2.28 seconds, reached 130.36 MiB
peak RSS on macOS (`ru_maxrss` 136,691,712 bytes), and reported zero swaps. No
broad suite or package build was run.
