# TOP community report workflow audit

## Source-backed user workflow

The cached primary-source material is under
`../mdanderson-stats/research/raw/TOP/` (captured
application `index.html`, article text/PDF, and published page). The application
capture identifies three endpoint choices: Binary Efficacy, Multiple Efficacy,
and Efficacy & Toxicity. Its simulation UI provides named/add-remove scenario
controls, a number-of-simulations field (default 1,000; displayed range
1,000–100,000), and an operating-characteristics output area. The Protocol tab
shows disabled “Download protocol template” links conditional on endpoint/prior
choices. The downloaded files and report schema are absent from the cache, so
their exact content and control behavior cannot be independently established.

The package already implements the numerical layer: `TOPBinaryDesign` and
`simulate_top_binary` cover single binary efficacy; `TOPMultiEndpointDesign`
and `simulate_top_multiendpoint` cover co-primary efficacy and
efficacy/toxicity. Design methods supply the source-derived decision rules,
complete-data thresholds, pending/suspension rules, and effective-sample-size
crossings. Calendar simulations supply trial-level compact outcomes, decision,
duration, interim suspension, and final wait summaries. Calibration APIs remain
separate, and this report does not rerun or replace calibration.

## Community workflow closure

`mdanderson_stats.top_report` provides typed named scenario inputs and report
factories for binary and two-endpoint designs. It records each design's prior,
cutoffs, looks, analysis timing, windows, and suspension convention along with
each scenario's truth, truth timing, arrival process, accrual/window inputs,
explicit seed, action counts/probabilities/denominators/MCSE, enrollment, event,
pending, and calendar summaries. It calls the existing simulation APIs; it
does not claim native RNG or scheduling parity. Output is an atomic standalone
HTML summary, not a native protocol-template document.

Scenario batches are limited to ten named cases, matching the recovered UI's
visible scenario range, and to two million trial-patient-endpoint cells as a
Python resource guard. Python permits small trial counts for quick examples and
tests; the cached UI's displayed 1,000 minimum is not copied as an engine
requirement. This is an intentional community API difference, not a claim that
the native control accepts smaller counts.

## Evidence boundary and remaining native parity

The cached capture supports an advertised report/template-download workflow but
does not include downloadable artifacts. Consequently this package closes a
usable reproducible report workflow without fabricating a native document
schema, optimizing design settings, or inferring unobserved template content.
Exact file formatting, protocol sections, native report fields, native
scheduling, and native RNG parity remain unverified source questions rather
than claimed implementations.
