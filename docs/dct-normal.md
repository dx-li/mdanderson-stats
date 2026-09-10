# Decentralized clinical trial sample size

The [MD Anderson DCT calculator](https://biostatistics.mdanderson.org/shinyapps/DCTs/)
provides continuous and binary endpoint planning for fully and partially decentralized
trials. This implementation covers **continuous outcomes with equal control and
experimental variances within each onsite/offsite stratum**. Variances may differ
between strata. It implements equations (2.3) and (2.5) in the app's
[embedded paper](https://biostatistics.mdanderson.org/shinyapps/DCTs/pdf_DCT.pdf),
Tian, Lin, Liu and Yuan, *Sample-size determination for decentralized clinical trials*,
[doi:10.1093/ije/dyaf053](https://doi.org/10.1093/ije/dyaf053).

`dct_normal_sample_size` accepts the onsite treatment effect, two stratum SDs,
offsite fraction, relative offsite effect bias, experimental/control randomization
ratio, power, significance level and one/two-sided test choice. The offsite effect
is `(1+relative_bias)*effect`. A fraction of zero or one selects a fully onsite or
fully offsite trial. The bias is treated as known for planning; zero means equal
effects. The effect input is positive in the prespecified treatment direction.

For repeated observations with exchangeable correlation, the variance of a unit
mean is multiplied by `rho + (1-rho)/m`. The allocation counts participants for
longitudinal studies, or clusters when repeats denote cluster sizes. A clustered
study's participant counts require multiplication by the corresponding cluster
sizes. Counts do not include dropout inflation.

The result contains the unrounded total, integer allocation (rows onsite/offsite;
columns control/experimental), and achieved power recomputed from the rounded
allocation. Each of the four arm counts is rounded upward independently; exact
allocation ratios may change slightly. The closed-form requirement uses the
usual dominant-tail normal approximation. Reported two-sided power includes both
tails. No large-sample approximation is presented as an exact small-sample test.

```python
import numpy as np
from mdanderson_stats import dct_normal_sample_size

independent = dct_normal_sample_size(10, 20, 25, relative_bias=-0.2)
np.testing.assert_array_equal(independent.allocation, [[29, 29], [85, 85]])
assert independent.total == 228
repeated = dct_normal_sample_size(
    10,
    20,
    25,
    relative_bias=-0.2,
    onsite_repeats=5,
    offsite_repeats=5,
    onsite_correlation=0.5,
    offsite_correlation=0.5,
)
np.testing.assert_array_equal(repeated.allocation, [[17, 17], [51, 51]])
assert repeated.total == 136
assert repeated.achieved_power >= 0.8
```

Three focused tests check the paper's independent/correlated formulas, these two
app-help allocations, fully decentralized repeated measures (76 participants),
unequal randomization, direct weighted-test power, perfect-correlation limits and
dose-unit scaling by `1e-200` and `1e200`. Calculation uses log information to avoid
squaring extreme SDs or effects. Required totals above one billion units raise.

**Catalog status is partial.** Unequal variances between treatment arms, binary
endpoints, app reporting and exact native rounding remain pending. For example,
the help page lists 128 for a fully decentralized independent continuous trial
with effect 10 and SD 20; direct formula rounding here gives 126. This discrepancy
is documented rather than hidden by adding arbitrary participants. The model
assumes known effect bias, variances and correlations; sensitivity to these inputs
should be assessed when planning a study.
