# PerfectMatch per-probeset correlation

The cached PerfectMatch Manual 2.3, §4, describes `corr` as the correlation
coefficient between observed and model-fitted lnPM signals for each probeset.
This Python utility consumes the original probe-level probeset IDs because
`PDNNFit.probeset_ids` stores only the sorted unique labels. It computes Pearson
correlation on the natural logs of the observed and fitted positive intensities.

The manual does not define a correlation variant or degenerate-group behavior.
The implementation makes those choices explicit: singleton groups and groups
constant in either log signal have undefined correlation, represented by NaN.
It summarizes existing fitted signals and does not refit, filter probes, or
claim the rest of native quality-control parity. Inputs are capped at 500,000
probes before conversion to keep the temporary workspace bounded.


The cached manual is pinned by `docs/perfectmatch-sources.json`: SHA-256
`e84bd6578e0bf95ac60b1d32361ce7ec86764a133983502913a38d2b186b95b0`.
Section 4 defines `corr`; it does not identify an outlier subset for that field.
The Python function uses every supplied row. Callers can explicitly subset the
same rows of all three inputs; no native subset-selection parity is claimed.

## Numerical and resource validation

Four focused tests cover grouping, undefined correlations, rescaling, nearby
large signals and pre-conversion rejection. An independent 100-digit Decimal
calculation applied the direct centered Pearson formula to natural logs in
three cases: ordinary signals; adjacent-scale values near `1e200` and `5e200`;
and values spanning the smallest positive binary64 subnormal through its largest
finite value. The maximum absolute correlation discrepancy was `1.11e-16`.
The ordinary vectors were `[1,3,8,2,5]` and `[2,4,7,3,10]`. The near-constant
vectors used binary64 spacing multiples `[0,1,3,6,10]` at `1e200` and
`[0,2,3,8,6]` at `5e200`. The wide-range vector used the smallest positive float,
`1e-150`, `1`, `1e150` and the largest finite float, with its reversal as fitted
signals.

A serial check at the 500,000-probe cap used 250,000 two-row probesets with
observed `[1,2]` and fitted `[3,6]` repeated per group. Every correlation was one
within `2e-15`, and caller arrays remained writable. The combined focused,
high-precision and cap-case checks used 0.249 seconds after imports, 241.25 MiB
peak process RSS and zero process swaps. This is a measured case, not a universal
memory guarantee. No fitting, parallel workers or new dependencies were needed.
