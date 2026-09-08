# RANDLIB archive coverage audit

This audit covers catalog entry 27's `RANDLIB_V90.tar.gz`, including its
Fortran 95, Fortran 77 and C libraries. It does not complete other catalog
entries. The archive SHA-256 is
`c90251cc129ed21616d80df7bc87acefcce57d1a9e7fe7a2dd23e6d119bed466`.

`tools/audit_randlib.py` inventories all 88 regular archive members, compares
each extracted file byte-for-byte to the archive, records hashes, and rejects
unclassified members or missing implementation/test paths. The generated
[`randlib-archive.json`](randlib-archive.json) gives the per-file mapping.
There are 61 library files (36 Fortran 77 files, 22 Fortran 95 modules and
three C implementation files), six demonstration programs, two C declaration
headers, three native build/instruction files and 16 documentation/terms files.
The inventory is a scope check; the behavioral evidence is the native fixtures
and statistical tests described below, not file presence alone.

## Library routine mapping

The methods below belong to `RandlibGenerator` unless another name is given.
Fortran 95 distribution modules use the `random_..._mod` naming convention;
the inventory links each one individually. C implementations are in `com.c`,
`randlib.c` and `linpack.c`.

| Archived routines / modules | Python behavior and evidence |
| --- | --- |
| ADVNST / advance_state | `advance_state`; block/stream and exact native-state tests in `test_randlib.py` |
| GETCGN, SETCGN / get_current_generator, set_current_generator / GSCGN | `stream`, `select`; 32 independent streams, with validated stream selection |
| GETSD / get_seeds | `get_seeds`; all native distribution fixtures check consumed component states |
| SETSD / set_current_seed | `set_seeds`; selected stream initial/block/current reset |
| SETALL / set_all_seeds | `set_all_seeds`; all-stream reset, including selection-independent repair of the source reset bug |
| INITGN / reinitialize_current_generator | `reinitialize(-1/0/1)`; original start, current block and next block tests |
| SETANT / set_antithetic | `set_antithetic`; native antithetic fixtures across samplers |
| IGNLGI / random_large_integer | `integers`; exact modular recurrence and native values/states |
| RANF / random_standard_uniform | `uniform`; separate C/Fortran rounding in legacy mode |
| GENUNF / random_uniform | `uniform(low=..., high=...)`; bounded and degenerate cases |
| IGNUIN / random_uniform_integer | `integer_uniform`; unbiased default and original rejection rule in legacy mode |
| GENPRM / random_permutation | `permutation`; forward Fisher–Yates, native values/state and duplicate-input tests |
| GENEXP, SEXPO / random_exponential, random_standard_exponential | `exponential`, including standard mean 1; native exponential fixture |
| GENNOR, SNORM / random_normal, random_standard_normal | `normal`, including standard mean 0/sd 1; separate C/Fortran FL tables |
| GENGAM, SGAMMA / random_gamma, random_standard_gamma | `gamma`, including rate 1; both GS/GD regimes and native fixture |
| GENCHI, GENNCH / random_chisq, random_nc_chisq | `chi_square`, `noncentral_chi_square`; native composition/order and near-one behavior |
| GENF, GENNF / random_f, random_nc_f | `f`, `noncentral_f`; quantile checks, source composition and warned legacy truncation |
| GENBET / random_beta | `beta`; original BB/BC and inverse-CDF default |
| IGNBIN / random_binomial | `binomial`; original inversion/BTPE, counts and states |
| IGNPOI / random_poisson | `poisson`; inverse table, normal/exponential rejection and repaired table lifetime |
| IGNNBN / random_negative_binomial | `negative_binomial`; gamma–Poisson mixture with a shared transaction/budget |
| GENMUL / random_multinomial | `multinomial`; conditional binomials, residual counts, original early exit |
| SETGMN, GENMN / set_random_multivariate_normal, random_multivariate_normal | `RandlibMultivariateNormal` and `multivariate_normal`; immutable prepared parameters, native factors/vectors/states |
| SPOFA, SDOT / nested Fortran 95 equivalents | Private source factorization and dot accumulation in `randlib_multivariate.py`; all three native factor conventions validated |
| PHRTSD, LENNOB / phrase_to_seed | `ranlist_seeds` and source-aware `set_phrase`; full phrase hash and trailing-space handling |
| phrase_set_seeds, inter_phrase_set_seeds, user_set_all | `set_phrase`, constructor and `set_all_seeds`/`select`; explicit inputs replace terminal prompting, and the missing interactive reset is repaired |
| time_set_seeds, set_seeds(which) in user_set_generator | `set_time` or `set_phrase`; explicit choice, local-clock or supplied-time seeding with returned replay seeds |
| MLTMOD / multiply_modulo | Exact integer modular arithmetic and modular powers in shared RNG code; avoids native overflow-decomposition machinery |
| INRGCM, QRGNIN, QRGNSN, RGNQSD, GSRGS, GSSST, set_default | Constructor-owned initialized state/constants replace process-global initialization flags. No public bank can exist in a partially initialized state. |
| FSIGN, FTNSTOP | Explicit sign logic and exceptions at corresponding source boundaries; a bad call does not terminate the Python process |

## Demonstration and support files

The C and Fortran `tstbot` programs demonstrate generator blocks, advancing,
and replay. Those behaviors are exercised by generator-state tests and native
stream fixtures. `tstmid` demonstrates distribution sampling, permutations,
and numerical summaries; the Python methods return arrays and distribution
tests check means, variances, CDFs and categorical counts. Its local STAT,
STATISTICS, TRSTAT/TRSTATISTICS and printing helpers are test/reporting
utilities, not additional library sampling algorithms. `tstgmn` demonstrates
covariance setup and sampled covariance, covered by native packed-factor/vector
fixtures and covariance tests. Their local covariance/statistics routines map
to ordinary array means and covariance calculations in the tests.

These six terminal demonstration menus and their typography are not reproduced
as a new interactive application. Their library operations are available as
Python calls. C headers declare the native interfaces; native make/compile
scripts are replaced by the package's existing build tooling. Documentation
formats describe the same library, and original attribution/terms remain in
`THIRD_PARTY_NOTICES.md` and `notices/`. Original code, native executables and
the raw archive are kept outside the distributed wheel.

## Evidence and known differences

`tests/fixtures/randlib_*.json` cover every distribution family, stream controls,
bounded sampling, factor preparation and seeding. Corresponding
`tools/reference_randlib_*.py` files record compiler flags/source hashes and
produce the fixture data from archived native implementations. Validation
checks random-number consumption in addition to output values; this catches
incorrect rejection paths that a moments-only check could miss. Distribution
tests additionally check mathematical moments, tails/marginals, covariance,
parameter boundaries, scalar/batch behavior, resource limits and rollback.
The isolated built-wheel checks exercise the same fixture comparisons on the
installed artifact. CI runs the complete package suite, lint/format checks,
type checking and builds on Python 3.12, 3.13 and 3.14.

The [RANDLIB behavior document](randlib.md) specifies deliberate differences:
explicit per-bank state; inverse-CDF defaults; unbiased integer rejection;
bounded work and atomic state commits; common safe integer/float ranges;
source-specific rounding; repaired all-stream reset, Poisson table storage
and Fortran 95 multivariate parameter allocation; rejection of unsafe C phrase
lookups; and explicit seeding instead of terminal prompts. Native equivalence
claims refer to recorded builds and defined inputs, not undefined reads,
integer overflow, uninitialized state, or every compiler optimization setting.

The performance report in [`randlib-benchmark.json`](randlib-benchmark.json)
compares batched and repeated scalar calls to this Python API, including raw
integers, uniforms, bounded integers and all distribution families. Prepared
multivariate parameters are reused in both modes. Permutations retain their
inherently sequential swap dependency while sharing bounded sampling and
avoiding user-input mutation. Legacy rejection algorithms preserve source
consumption and precision rather than promising the same throughput as the
vectorized defaults. These benchmarks do not claim speedups over native C or
Fortran.
