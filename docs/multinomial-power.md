# MULTINOMPOW exact multinomial power

`multinomial_power` implements exact nonrandomized one-sample multinomial
power with Pearson chi-square and likelihood-ratio ordering. The
[archive audit](multinompow-coverage.md) accounts for all original files and the
complete study/report workflow.

```python
from mdanderson_stats import multinomial_power

result = multinomial_power(
    3, null=[0.5, 0.5], alternatives=[[0.8, 0.2], [0.5, 0.5]], alpha=[0.05, 0.25]
)
# Both statistics: actual sizes [0, 0.25].
# Alternative [0.8, 0.2]: powers [0, 0.52].
```

The implementation enumerates all weak compositions of n into k categories.
It sorts statistics in descending order and adds complete tie groups while
cumulative null probability fits the requested alpha. Adjacent statistics are
tied when their difference is at most `1e-12 * max(abs(a) + abs(b), 1e-20)`,
as in the source. This is a nonrandomized test; its actual size can be much
smaller than its nominal significance level.

The result contains immutable probability inputs, statistic names, critical
values and actual sizes (statistic × alpha), powers (statistic × alternative ×
alpha), and sample-space size. Empty regions have infinite critical values and
zero size/power. Alpha one includes the whole sample space. Critical values
summarize the enumerated regions; use the documented tie convention when
comparing newly computed statistics at a boundary.

Inputs accept n from 1 through 10000 and 2–10 categories. The original interactive
program starts at n=3. Null probabilities must be strictly positive; alternatives
may include zero probabilities. Each probability row must sum to one within
1e-12, after which it is normalized. Scalar or vector alpha values lie in [0, 1];
duplicate and unsorted levels are preserved. These strict probability checks
replace the source interface's 0.99–1.01 sum acceptance window.

The point count is `comb(n + k - 1, k - 1)`, checked before allocation against
`max_points` (default 1,000,000). Large problems still have combinatorial cost.
Statistics and log probabilities use NumPy and SciPy compiled operations.
Alternatives are processed separately, avoiding a three-dimensional probability
array. Joint evaluation measured 11.7–12.8 times faster than 15 repeated calls
for the tested workloads; see the [benchmark scope](multinompow-coverage.md).

Probabilities use double-precision log factorials and zero-safe log products.
Each enumerated distribution must sum to one within 1e-8 before normalization;
otherwise calculation fails. Significance comparisons allow relative rounding
error of 32 machine epsilons; alpha zero explicitly excludes all points.
Extremely small masses can underflow double precision. Statistic overflow raises
an error. “Exact” describes enumeration rather than arbitrary-precision arithmetic.

## Source findings and validation

Source: [MULTINOMPOW version 1](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/19),
Barry W. Brown, MD Anderson Department of Biomathematics, May 2003.
Archive: [MULTINOMPOW _V1.tar.gz](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MULTINOMPOW/MULTINOMPOW%20_V1.tar.gz).
Archive SHA-256: `c420cd6e346f41c7990c88db884c00af697ab95ca745b9a250596f8a4b7ff453`.
Original notices are retained in
[LEGALITITES](../notices/mdanderson-multinompow-LEGALITITES.txt).
The original ACM gamma routine is not copied; this implementation uses SciPy.

Source inspection identified these differences in `mp_setup_mod.f90`:

- `traverse_points` does not initialize `point_prob(1)`. Python evaluates every
  null probability, including the first point.
- `chi_sq` and `log_multinom_con` return default REAL, with repeated
  single-precision rounding in the multinomial coefficient. Python uses doubles.
- `set_crit` leaves a -1 sentinel when no complete tail group fits. The power
  routine subsequently includes all nonnegative statistics. Python returns an
  empty region, with zero size and power.
- The final tie group is not considered by the native boundary loop. Python
  explicitly supports alpha one and includes all outcomes.

`tests/test_multinomial_power.py` compares both statistics with independent
Cartesian enumeration and rational factorial probabilities for small examples.
Additional checks cover binary ties, null power equaling actual size, degenerate
alternatives, category permutations, repeated/unsorted levels, immutable results,
input validation, preallocation budgets, and numerical overflow.

`tools/reference_multinompow.py` compiles two explicitly repaired native profiles
and records 52 studies in `tests/fixtures/multinompow_native.json`. Both initialize
the omitted first null probability and translate empty-region sentinels at the
driver boundary. The `corrected_double` profile additionally promotes the two
single-precision return types. All other numerical routines are unchanged.
The fixture records the archive/source hashes, compiler, build commands, driver,
and exact repairs. It is not an unmodified-native reference.

The 26 double-precision studies cover n=3, 7, 20, 60, 999, 1000, and 1001;
2–4 categories; three alternatives; and five significance levels. The larger
binary studies cross the original log-factorial table/gamma-function boundary.
Maximum observed size/power difference from Python is 5.8e-13. The profile retaining
single precision differs by up to 0.000244 and has null-total error up to 0.000237;
these diagnostic results are recorded but are not Python compatibility targets.
`tests/test_multinomial_power_native.py` checks critical values, sizes, powers,
and native null totals against the corrected-double profile.

The native comparisons exclude alpha one because the archived boundary loop
omits the final tie group. Each study runs in a fresh process to avoid the
original repeated-deallocation defect. Independent tests cover alpha one and
zero-probability alternatives. Python does not reproduce undefined memory or
single-precision behavior.

## Reports

```python
from pathlib import Path
from mdanderson_stats import format_multinomial_power

text = format_multinomial_power(result, digits=8)
print(text)
Path("multinomial-power.txt").write_text(text)
```

The report includes every alternative and requested level, all critical values,
actual sizes and powers, and explicitly labels empty regions. It uses readable
text rather than reproducing historical terminal spacing. Printing and file
writing remain under the caller's control.
