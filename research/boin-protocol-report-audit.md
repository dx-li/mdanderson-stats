# BOIN design-report scope and provenance

The report composes existing Python components rather than reimplementing
decision or selection calculations: `boin_protocol` renders methods and the
integer decision table, while `simulate_boin` calculates trial outcomes,
selections, and stop reasons. Report construction validates and snapshots the
design and all scenario inputs, then invokes those components serially from
one recorded NumPy seed. Only bounded aggregate summaries are retained.

The BOIN overdose-risk definitions and their source-specific availability
gate are recorded in [dose-allocation-risk-audit.md](dose-allocation-risk-audit.md).
The report follows that helper's fractions and Bernoulli MCSE convention.
It does not reproduce the desktop animation, native random-number stream, or
observed-trial decisions. HTML is an escaped static presentation of the Python
results; it is not a native report-file-format claim.
