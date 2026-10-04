# PerfectMatch per-array quantile profile

The PerfectMatch 2.3 manual, §2, says the normalization view records the 2nd,
25th, 50th, 75th and 98th percentiles of each array's sorted intensity values
so users can compare chips. `pdnn_array_quantiles` computes that five-value
profile from an already decoded matrix with probes in rows and samples in
columns, the same orientation as [`quantile_normalize`](perfectmatch.md).

```python
import numpy as np
from mdanderson_stats import pdnn_array_quantiles

decoded_intensities = np.array([[120, 180], [150, 175], [210, 250], [400, 330]], dtype=float)
profile = pdnn_array_quantiles(
    decoded_intensities,
    sample_names=("control.CEL", "treated.CEL"),
)
print(profile.probabilities)  # (0.02, 0.25, 0.5, 0.75, 0.98)
print(profile.values)  # one row per sample, one column per percentile
```

Quantiles use linear interpolation at `(n_probes - 1) * p`, matching NumPy's
`method="linear"` convention. The manual does not specify a quantile
interpolation or rank-rounding rule, so exact native-file equality is not
claimed. Input intensities must be finite and nonnegative. The calculation is
bounded to two million matrix cells and 5,000 sample columns; it works on one
sample column at a time and does not mutate the input. Larger collections can
be processed in independent sample-column batches. If names are omitted,
the result uses `sample_1`, `sample_2`, and so on.

This is a descriptive QC profile only. The manual recommends comparing these
values to look for problematic arrays but provides no decision thresholds or
automatic rejection rule. It does not read CEL/binCEL files or recreate the
native Quantile File output format. ScalingFactor and Absent genes should not
be inferred from the manual's example `PDNN.log`: the manual says those fields
were incorrect in that release.

Source provenance is recorded in [`perfectmatch-sources.json`](perfectmatch-sources.json).
The cached manual hash is SHA-256
`e84bd6578e0bf95ac60b1d32361ce7ec86764a133983502913a38d2b186b95b0`.
