# MULTINOMPOW archive coverage

The version 1 archive contains 37 regular files: 17 Fortran source files,
two build files, two Windows binaries/installers, and 16 documentation/notice
files. `tools/audit_multinompow.py` verifies each extracted file against its
archive bytes, records hashes and routine names, and rejects unknown members.
The resulting inventory is [multinompow-archive.json](multinompow-archive.json).
Original code and executables remain outside the package.

| Source responsibility | Python implementation and evidence |
|---|---|
| `mp_main`, `mp_struct_mod`: study inputs, both/one statistic, alternative and level batches, repeated studies | `multinomial_power`, immutable `MultinomialPower`; validation, batch, permutation and repeated-study tests |
| `mp_setup_mod`: expectations, chi-square/LR statistics, null masses, tie groups and critical regions | NumPy statistics and sorted whole-group probabilities; independent rational and repaired native tests |
| `power_mod`, `calc_point_prob_mod`: alternative masses and power accumulation | `_mass` and ordered cumulative probabilities; native three-alternative studies and null-equals-size checks |
| `log_factorial_mod`, `gamma_mod`: factorial coefficients | SciPy `gammaln`; native n=999–1001 comparisons straddle the archived table boundary; no ACM implementation copied |
| `n_partitions_mod`, `partial_sums_mod`, `update_partition_mod`: point counts and enumeration | Exact integer `math.comb` and stars-and-bars compositions; independent Cartesian enumeration, sample-space limits and native comparisons |
| `perm_sort_array_mod`: comparator sorting and index permutations | Stable NumPy `argsort`, original adjacent tie tolerance; ties and category-permutation tests |
| `summation_mod`: separate sums for small/large masses | Checked normalization and NumPy long-double accumulation; probability invariants and native comparisons |
| `write_answers_mod`, `print_array_mod`: problem, setup and power reports | `format_multinomial_power`; all input vectors and output tables, both statistics, empty-region labels and report persistence tests |
| `get_numbers_mod`, `print_it`, `open_file_mod`: console input, prompts, holds, file dialogs | Python arguments and exceptions; returned report text supports standard printing and caller-controlled file writing |

Generic integer/real sort overloads, numeric prompt overloads, format-editing
helpers and screen-clearing routines are supporting machinery for the console
application. Their relevant behavior is replaced by the Python interfaces above;
they are not exposed as unrelated utility APIs. Build scripts and Windows
installers are replaced by the existing package build/install workflow and CI.
The Unix/Windows manuals and duplicated notices are accounted for in the
inventory; the original legal terms are retained in the wheel.

The report preserves all substantive fields rather than the old terminal's
fixed columns, pagination and interactive file-overwrite dialog. Callers choose
whether to print, save, append, or retain the returned text. Repeated Python
calls own separate state and do not encounter the source's double deallocation.
The Python API allows more alternatives/levels than the old console's 50/10
limits; the exponential sample-space cost remains explicitly bounded.

## Correctness and compatibility

The [method notes](multinomial-power.md) detail source repairs, numerical limits,
52 native reference studies across two explicit profiles, and independent
rational-probability tests. No uninitialized source output is treated as ground
truth. The manual's n=100, three-category example is checked for its 5,151 points,
printed critical values, null power and probability bounds. The manual prints
power 1.00000471 for its extreme alternative; Python removes coefficient
roundoff through checked normalization and returns a valid probability.

The manual also prints inconsistent preliminary sample-space counts and writes
the expected count using the number of categories where n belongs. Python uses
`n * p0`, as the actual source does. Likelihood-ratio critical values use twice
the log likelihood ratio, matching the executable's formula. Asymptotic claims
in the manual are unnecessary for the exact enumeration.

## Measured batching benefit

[Benchmark results](multinompow-benchmark.json) compare one joint call for three
alternatives and five significance levels with 15 separate calls to the same
Python API, checking numerical agreement first. Both compute both statistics.
On the recorded environment, joint evaluation was 11.7–12.8 times faster across
1,001, 1,891 and 39,711-point problems. The largest joint call took about 25 ms.
These are median-of-five local measurements, not comparisons with native Fortran
or promises for other hardware. The combinatorial point count still limits
large n/category combinations.
