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
