# CondiS workflows beyond the eight learners

This audit records the remaining scope separately from learner implementation.
The primary reference is the CondiS 0.1.2 `vignettes/introduction.Rmd` from the
archive pinned in [the source manifest](../docs/condis-sources.json).
The [MD Anderson app](https://biostatistics.mdanderson.org/shinyapps/CondiS/)
was also inspected as a text-only page on 2026-09-28. Neither source establishes
that the Python package reproduces all interactive app behavior.

## What the vignette demonstrates

The vignette first imputes censored times with `CondiS`, then uses `CondiS_X`
to refine those times. Its Kaplan–Meier illustration compares the original
censored sample with the imputed times treated as all events. It combines
the two curves, displays censor marks, and requests a risk table. The Python
imputation result already retains the original sample and an
`ExploratorySurvival` curve, including risk, event, and censor counts. The
existing `exploratory_survival` calculation can also provide the all-events
curve; a CondiS-specific comparison/summary interface remains to be added.
Do not assume that another module's confidence interval convention matches
the original plotting package or app.

The final vignette example is a separate regression exercise. It makes a
seeded 75/25 train/test split, tunes a radial SVM using repeated cross-validation,
predicts test rows, and reports MAE against both base and refined imputed
times. Its formula excludes `status`, `status2`, `rfstime`, and `pred_time`
from predictors. However, `pred_time_2` has already been added to the data
frame and is **not excluded** by that formula, so the refined imputed time
is itself an input to this example's regression. Both sets of imputations
are also computed before the train/test split. The example therefore does
not establish a deployable predictor or leakage-free evaluation of the full
survival-imputation pipeline. The preprocessed covariate matrix is computed
but not used by the fit.

The current CondiS-X refiners predict the supplied sample and include its
observed event/censor status. A standalone regression fit/predict interface
with an explicit predictor matrix would be a distinct capability. It must
not silently use target-derived features or imply that future censoring
status is known. Matching the vignette's illustrative MAE is a different
claim from measuring future-subject survival prediction accuracy.

## What was observable in the app

The page shell labels Data, Data Summary, Kaplan-Meier Plot, Imputed Survival
Distribution, and Predicted Survival Distribution. Reactive content remained
unobserved; the available in-app browser runtime was unavailable. Upload
formats, column selection controls, exact summary fields, plotting choices,
and download/report outputs have consequently not been verified. Do not
invent these contracts or count them as implemented from the page labels.

## Coverage priority

Finish the random-forest and gradient-boosting numerical workflows before
adding presentation conveniences. Then provide reusable summary/curve data
and an explicitly defined regression prediction workflow where supported
by source evidence. Exact app input/report compatibility remains a separate
unverified item. CondiS stays partial until its remaining scope is resolved;
adding all eight learners alone is not evidence of complete app parity.
