# iBOIN final selection and stopping source audit

Inspected September 28, 2026. Catalog entry 145 remains partial.

## Final estimates and historical borrowing

The official app's dedicated
[final-selection prior help](https://biostatistics.mdanderson.org/shinyapps/iBOIN/iBOINprior_for_MTD.pdf),
page 1, resolves the rate estimates used before isotonic regression:

- Without historical borrowing: `y_j / n_j`.
- With borrowing: `(y_j + n0_j*q0_j) / (n_j + n0_j)`, with historical
  toxicity estimate `q0_j` and effective sample size `n0_j`.

The [current official app](https://biostatistics.mdanderson.org/shinyapps/iBOIN/)
exposes prior borrowing for final selection and a separate simulation option.
The help's checkbox wording is inverted relative to the current app, so the
equations establish the two modes without assuming checkbox polarity.

The [main guide](https://biostatistics.mdanderson.org/shinyapps/iBOIN/iBOINGuide.pdf),
Figure 15 on PDF page 10, also shows the optional prior checkbox. The paper's
closest-to-target isotonic selection sentence does not rule out this later
software option. Ordinary BOIN's Beta(.05,.05) posterior means are not the
unborrowed rates stated in this iBOIN help.

These sources do not establish the isotonic weights, tie behavior, treatment
of unobserved doses, or whether robust-prior truncation also changes final
selection's ESS. Native numeric evidence or additional primary documentation
is still required for those details. Sample-size weights may be a reasonable
statistical choice, but they are not verified native behavior from these texts.

## Stopping and final safety constraints

The [author manuscript](https://arxiv.org/abs/2004.12972), page 11, uses ordinary
dose elimination after at least three patients at a dose, with a uniform
Beta(1,1) posterior overdose probability exceeding the cutoff. Elimination
also excludes higher doses; eliminating the lowest dose stops the trial.

The guide, Remarks 2 on PDF page 2, specifies stricter optional safety stopping
only after **more than three** patients at dose 1, at cutoff `P_E - delta`.
Its default offset is .05. This is a stop rule distinct from ordinary dose
elimination; the thresholds must not be conflated.

The live app specifies optional stopping when the patients assigned to a dose
reach a threshold and the next decision is to stay. Its visible text gives no
posterior precision criterion. The guide, Remarks 3 on PDF page 3, adds an
optional final-selection constraint that the isotonic rate not exceed the
de-escalation boundary. A native fallback when no dose satisfies the constraint
has not been established.
