# Random survival forest global Brier-gradient split audit

The pinned randomForestSRC 3.2.2 manual advertises `splitrule="bs.gradient"`
as global non-quantile Brier-score gradient splitting and says its horizon is
the observed-event 90th percentile unless `prob` overrides it. The cached
wrapper source resolves the actual scalar contract: `get.grow.splitinfo` maps
`bs.gradient` to split rule 20, and `global.prob.assign` defaults this rule to
`.9`, filters values to `(0,1)`, and takes the first valid probability. The
wrapper source is cran/randomForestSRC commit
`b4d099e262423362a8872c13c468e6dbe2f9e9da`,
`R/utilities.data.R`, Git blob
`bac4cfb07e838cbcae029120d0bf5307c02d9b17`, SHA-256
`16fc1ada2de5b2760489f3c1a19ecde0895e68c53f4fac8eab1400c865109ac3`.

The native `makeSplitRuleObj` dispatches rule 20 to `brierScoreGradient1`
(`src/randomForestSRC.c`, around 20615–21035; enum in `randomForestSRC.h`).
That C function builds the parent event Kaplan–Meier curve and calls
`stackAndGetQETime`. For this rule that helper scans event-grid survival until
it is at most `1-prob`, records the preceding event index, and uses only that
one point for a scalar probability. Equality stops the scan. A first-point
crossing gives index zero, which the split loop skips while leaving its
initialized score at zero; no crossing uses the last event point. This differs
from the manual's descriptive empirical event-time percentile wording.

`stackAndGetLocalGamma` computes reverse-KM censor survival at left limits.
At a selected evaluation point, rows still event-free beyond that time use
`1/G(t-)`; every observed failure by that point uses `1/G(previous event-grid
point-)`, shared across those failures; earlier/equal censorings have weight
zero. The source then computes a parent weighted event-free fraction, forms
`-2*w*(y-fhat)`, and the main candidate loop sums the squared mean gradient in
each daughter weighted by daughter fraction. The Python implementation uses
that scalar source convention rather than a generic all-times integrated
Brier score or subject-specific failure IPCW. Ties are grouped on exact
observed-time equality; reverse-KM factors at censor times tied with the
selected event are excluded by the strict left-limit lookup.

The Python implementation reuses the existing numeric and categorical
candidate generation and scores one streamed `n`-row gradient per node, so it
does not allocate an `n × number_of_event_times` matrix. Work for the KM/score
preparation and each candidate is charged to `max_split_work`. Bootstrap copies
remain repeated rows. If a required inverse weight or resulting gradient is
nonrepresentable, the candidate is unavailable; it is never assigned an
infinite best score. `prob` is recorded on the fit only for this opt-in rule.

`tools/reference_random_survival_brier.R` is an independent arithmetic
reference for the inspected helper contract. It emits point indices, row
gradients, and one candidate score for five cases covering a shared failure
weight, first-point threshold equality, censoring before the first event,
threshold equality after a prior event, and a censor tied to an event.
`tools/reference_random_survival_brier_c.py` extracts the unchanged pinned C
helper bodies, verifies the source blob hash, compiles those helpers with a
small array adapter, and compares the same five ledgers. This validates the
native helper kernels, not the full forest engine or its complete R interface.
The two checked-in fixtures each cover five cases with row-level gradients
and candidate scores. The focused `tests/test_random_survival_forest.py`
run passed (25 tests); Ruff check/format and targeted mypy passed. The
eight-tree public guide example ran successfully. The serial validation
sequence took 3.708 seconds, reached 172,326,912 bytes peak RSS (about 164.3
MiB on macOS), and reported zero swaps.
