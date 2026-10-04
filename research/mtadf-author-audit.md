# MTADF author-reference isotonic implementation audit

## Primary source

The implementation source is the authors' `targetAgentDF.r` linked from their
institutional TargetAgent code index. A single direct HTTPS retrieval on
2026-10-04 produced a 20,787-byte file with SHA-256
`e27be5581fdcb7c71a7739a4f1b25026dc054896fcbf7134ff9b1a1c8960caac`.
The source file is retained only in the ignored raw cache and was not executed;
the code index listed its modification date as 2013-08-12 (the file itself has
no date header).

The author code calls `Iso::ufit(..., type="b")` without weights. The
historical `Iso` 0.0-15 source (2013-04-18) confirms omitted `w` means unit
weights. Its Fortran enumerates midpoint peak candidates, uses equal-weight
least-squares error, and retains the first candidate on an exact score tie.
Its `unimode` routine fits increasing and reverse-decreasing sides using PAVA
at the midpoint (`Iso/src/ufit.f`, `Iso/src/unimode.f`); the R source then
resolves fitted-value plateaus with the rightmost matching dose. The Python
kernel implements those midpoint candidates directly with unit-weight PAVA.
The author supplies a one-observation prefix after the first cohort, but
Iso 0.0-15's one-point internal mode search has undefined/out-of-range
behavior. The Python kernel explicitly extends the method by returning that
single rate unchanged; first-cohort decisions do not claim native parity.

## Recovered isotonic rules

- `isotonic()` and `df.isotonic()` both calibrate the fixed Beta prior from
  `pbeta(0.3, alpha, 0.5-alpha) = 0.22`, independent of the `phi` and `ct`
  arguments (`targetAgentDF.r:31-48,114-133`). The Python Brent solve gives
  `alpha` about `0.353639544999907`; the R source's default `uniroot`
  tolerance can return a slightly different last few digits, so exact numeric
  equality is not part of the contract. They calculate posterior
  overdose tails, apply increasing `pava`, count adjusted values `<= ct`, and
  set the admissible prefix length to at least one.
- Interim efficacy fits are unweighted. For a non-highest current dose, the
  source fits the contiguous tried prefix and moves one dose toward the
  rightmost maximum (`:72-80,142-150`). At the highest dose it fits all levels;
  if the rightmost peak is below the highest dose it steps down, otherwise it
  stays, before capping at the admissible prefix (`:66-71,136-141`).
  A one-dose observed prefix is passed by the source even in a multi-dose
  design (`:72-80`), but the Iso one-point fit is undefined as above. In this
  Python extension, the first-cohort fitted value is the single observed rate,
  allowing the author-style escalation rule to run.
- Final efficacy rates are `yeff/(n+0.0001)` at every dose, followed by
  all-dose unimodal fitting. The rightmost maximum is capped at the admissible
  prefix (`:83-86`). This differs from the paper-policy implementation's
  observed-only final candidates, lowest-dose ties and possible safety stop.

## Difference between the reference functions

The decision kernel represents the actual-trial function's fresh cap behavior.
The `isotonic()` OC loop instead computes `adm` before a cohort, uses that
previous value for the next-dose cap, and refreshes `adm` only after movement
(`:59-82`). The separate author simulation wrapper must preserve that lagged
order; it must not reuse the fresh-cap transition without passing the previous
cap.

The R file is an author reference, not proof that the current hosted Shiny app
uses the same implementation and hidden settings without modification. Its
header describes isotonic, global logistic and local logistic designs. The
existing Python paper APIs implement their published statistical models, but
the author logistic code has additional source-level choices (including
`arm::bayesglm`, scaled dose coding and lagged safety) and is not covered by
this isotonic kernel. Neither this audit nor this implementation claims full
MTADF method-family or live-app parity.
