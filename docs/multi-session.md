# MULTI sessions and structured reports

```python
from mdanderson_stats import MultiSession

session = MultiSession(".001 .01 .02 .1 .2 .3 .4 .5 .6 .7 .8 .9", seed=42)
fit = session.run("schweder_fit")
adjustments = session.run("multiple_testing", method="holm", alpha=0.05)
sharpened = session.run("sharpened_testing", null_estimate=fit.null_estimate)

session.change_file("next-pvalues.txt")
next_result = session.run("nonparametric_testing")
session.write_report("multi-report.json")
```

MultiSession supplies the original MAIN program's data-change, procedure-execution,
seed and report lifecycle through explicit Python calls. It does not prompt, hold
the terminal, modify global random state or install an output sink. For terminal
entry syntax including q/Q termination, supply text with terminal=True. The
constructor and change_data accept a source label; change_file records the path.
File/parse failure leaves current data and earlier report history intact. Session
data access returns independent arrays, and changing the returned object cannot
change subsequent analyses.

`session.procedures` lists ten supported numerical APIs:

| Name | Purpose |
| --- | --- |
| schweder_fit | Estimate null count |
| schweder_bootstrap | Bootstrap null estimates |
| multiple_testing | Nine adjustment procedures |
| sharpened_testing | Sharpened Holm/Hochberg decisions |
| fit_beta_mixture_k | Fit a specified mixture component count |
| select_beta_mixture | Automatic mixture selection |
| beta_mixture_testing | Reciprocal-density decisions |
| beta_mixture_bootstrap | Simulated CVM goodness of fit |
| nonparametric_testing | Desktop nonparametric decisions |
| order_statistic_diagnostics | S order-statistic diagnostics |

Each run uses the current accepted observations in entered order and returns its
ordinary package result object. Keyword settings follow the named numerical API;
misspelled names/settings raise. Two session defaults follow the desktop workflow:
mixture selection uses workflow="desktop", and omitted sharpened null_estimate is
freshly estimated with schweder_fit at alpha=.05. This avoids MAIN's QSW cache flag
remaining set across dataset changes. To use a particular prior estimate, supply
it explicitly; it is recorded among the settings. Numerical policies, source
corrections and endpoint conventions remain those of the underlying APIs.

The session does not automatically carry earlier mixture models to another dataset.
For manual mixture extension pass previous=the_previous_fit; for mixture decisions
or bootstrap pass model=the_fit.model. These model inputs are captured in the report.
No successful selection is inferred from merely returning a result: an underlying
component_limit or fit_failed status and message remain part of that result.

## Randomness and reproducibility

The constructor's seed and set_seed reset a NumPy Generator. Calls with an omitted
or None rng use that session generator; an explicit integer or Generator overrides
it for that call. Generator state is recorded before and after each stochastic
call, and set_seed records the resulting initial state. The report also records
the NumPy version. This permits random-stream replay in a compatible environment;
it is not the original two-seed RANF stream or a cross-version bitwise promise.

## Report contract

report() returns a detached dictionary. write_report(path) writes it as UTF-8 JSON,
replacing the path explicitly and propagating file errors. Serialization completes
before writing. Format is mdanderson-stats/MULTI-session, version 1.

The datasets list contains sequential IDs, source labels, terminal flags, entered
and sorted values, original indices, and parser diagnostics. Events reference
those dataset IDs, preserving earlier analyses when current data changes. Analysis
events contain procedure name, all effective settings including API defaults,
returned result fields or explicit numerical failure, and applicable RNG states.
Result arrays retain the ordering documented by their numerical API; their field
names identify the columns/diagnostic vectors. Dataclasses and nested mixture or
bootstrap results are expanded recursively. Reports are snapshots: changing a
returned result, report dictionary or current dataset cannot rewrite prior events.

status="returned" means the function returned, not that every internal fit or
selection succeeded. Numerical ValueError, arithmetic and linear-algebra failures
are recorded and re-raised; no fabricated result is inserted. Failure of automatic
null estimation is recorded with its source. Invalid dispatch arguments raise
before execution. Nonfinite diagnostics use tagged objects such as
{"nonfinite": "inf"}, never invalid bare Infinity/NaN JSON tokens. Unsupported
report value types fail explicitly rather than being stringified without a schema.

Integration tests cover all ten procedure routes across the suite, full setting
capture, file round trips, input-order preservation, immutable historical results,
failed replacement/run handling, current-data null estimation, exact RNG replay,
terminal quit handling, and nonfinite endpoint diagnostics. Underlying numerical
validity is checked separately against the native fixtures. Readable-report validation and source coverage are described below and in
multi-coverage.md.


## Readable reports

```python
text = session.format_report(digits=6)
print(text)
session.write_text_report("multi-report.md", digits=8)
```

The Markdown report follows session event order, keeping analyses attached to
their original datasets. It prints data tables, parser diagnostics, effective
settings, numerical results, failures and random state. digits selects 1–17
significant digits, default 6. File writes replace existing paths explicitly;
formatting completes first and I/O failures propagate.

Decision tables show step, rank, original observation (one-based), p-value,
method-specific numerical columns and an asterisk for rejection, plus a rejection
count. Rank refers to sorted p-values; step follows the algorithm's sequence.
They differ for entered-sequence mixture decisions. Rom uses Critical alpha,
while other adjustments use Adjusted P-value. Mixture/nonparametric reciprocal
densities use Reciprocal-density score, with no posterior-probability claim.

Mixture tables include the uniform component (index 0, beta shapes 1 and 1),
each beta weight/shape pair, and the beta-component count. Nested selected fits,
candidates, bootstrap checks and failure messages remain visible. Other diagnostic
vectors use one-based Index rows in their own API order: in particular,
nonparametric density/bandwidths are fitted-rank vectors, while estimates and
simulated_statistics index simulation replicates. Settings and nonfinite values
are printed explicitly. User-supplied labels/diagnostics are escaped for table
separators, HTML and embedded newlines.

`tools/reference_multi_report.py` extracts original PDISP unchanged, substitutes
only a completed terminal-navigation stub, and records 24 report cases using
existing native adjustment fixtures. Tests compare displayed ranks, observation
numbers, values and markers. Comparison tolerances account for the source's
six-digit display. Additional tests validate Rom thresholds, entered-sequence
step/rank mapping, model/candidate/simulation fields, nonfinite diagnostics,
failures, escaping, precision validation and file output. A sample complete report
was inspected for readable layout. The native report-content audit is maintained
in multi-coverage.md.
