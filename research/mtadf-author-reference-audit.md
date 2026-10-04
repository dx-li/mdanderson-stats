# MTADF author-reference and `Iso` contract audit

This audit resolves the source dependency behind the cached 2013 author R
program. The raw author file is dated 2013-08-12 and calls `Iso::ufit` and
`Iso::pava` in its isotonic functions. CRAN archive metadata shows `Iso`
0.0-15 was published 2013-04-18, making it a period-matched dependency to
inspect. The app's actual installed `Iso` version was not recoverable, so this
is an era-matched contract, not proof of the app's runtime version.

The original calls are `ufit(zfit, x=..., type="b")[[2]]`. In the cached
dependency, `type="b"` uniquely abbreviates `"both"`; the second item of that
result is the fitted response vector. The author omits `w`, so `ufit` supplies
unit weights for each dose-level response rate. Counts affect the observed
rate `yeff/n`, but do not weight the regression across dose levels. The
unconstrained mode algorithm visits half-index peak candidates in order,
evaluates squared error, and retains the first candidate at an equal minimum.
Its unimodal fit combines the rising and falling sequences around a peak; it
is not equivalent by definition to fitting independent increasing/decreasing
segments and selecting a split. PAVA defaults to unit weights and increasing
order. Its Fortran implementation pools strict adjacent violations; equal
fitted values need not be physically merged to have the same fitted sequence.

There are two consequential conduct differences between the author simulator
and its per-trial `df.isotonic` function:

1. The simulator initializes an admissible-dose cap, allocates a cohort, and
   uses that pre-cohort cap while selecting the next dose; it updates the cap
   only after selecting that next dose. `df.isotonic` recalculates the cap
   from current counts before deciding. The fixture's all-toxic, favorable
   first cohort therefore moves the simulator from dose 1 to dose 2 using the
   initial cap of 3, even though the freshly computed cap is 1. This is a
   one-cohort lag in simulator code, not a general safety rule.
2. Final simulator selection computes `yeff/(n+0.0001)` over every dose,
   including untried levels. If every efficacy count is zero, all fitted
   values tie at zero; `tail(which(fit==max(fit)),1)` picks the rightmost dose,
   then the code caps it by admissibility. With four admissible doses and
   observations only at dose 1, the reference program can select untried dose
   4. The paper/API policy that restricts selection to observed doses and
   chooses the lowest efficacy tie is a different contract.

`tools/reference_mtadf_author.R` provides an independent base-R numerical
reference. It enumerates contiguous constant-block partitions and minimizes
squared error over feasible unimodal sequences rather than copying/loading
Iso's implementation. Its compact fixture contains two grouped dose-rate
vectors and the two simulator edge cases above. This provides focused source
evidence; it does not claim execution parity with Iso's compiled code, exact
app dependency version, or full operating-characteristic/random-number
parity. CRAN sources and hashes are recorded in
[`docs/mtadf-author-sources.json`](../docs/mtadf-author-sources.json).
