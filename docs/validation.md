# Validation evidence and remaining scope

The live catalog retrieved on 2026-09-08 UTC contains 80 desktop and 58 online
entries. The inventory is not a claim of implementation. Each entry has an explicit
status and feature/test references in `src/mdanderson_stats/catalog.json`.

## BP1CI

Source: [BP1CI 2.0 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BP1CI/BP1CI_V2.0x.zip),
by Barry W. Brown, Floyd M. Spears, and Dan M. Serachitopol.

The unmodified Fortran program was compiled with gfortran and executed to produce
24 reference intervals, including binomial zero/all-success cases, fractional
Poisson input, large Poisson counts, and three confidence levels. The fixture
records archive SHA-256, compiler version, build command, and printed precision.
`tools/bp1ci_reference.py` reproduces the fixture from a locally built executable.

The binomial formula agrees with equal-tailed Clopper–Pearson inversion. Tests
independently evaluate binomial tail sums and frequentist coverage on a probability
grid for sample sizes 1, 5, 20, and 50. The published 12/30 example is also checked.

The original Poisson routine passes `s=k` into both tail inversions. Its lower
bound is therefore a gamma quantile with shape `k+1`, whereas the exact Garwood
lower bound uses shape `k`. `bp1ci_poisson_interval` preserves the original formula
and accepts positive fractional input as the original does. It rejects zero,
which the original interactive interface also rejects. `poisson_interval` provides
the standard exact interval and accepts zero. Direct probability sums validate
its tail equations independently of inverse special functions.

The guide's final bread-example scaling sentence has a decimal-place error;
the reported bounds on 10,000 events divided by 100 are about 97.45 and 102.60.
The test oracle uses the executable's unscaled output, not that sentence.

Remaining BP1CI compatibility work: generalized fractional binomial inputs,
interactive confidence-range restrictions, and a wider extreme-input comparison.
The Python API is not an emulation of the original menus or executable.

## Implementation strategy

For every remaining catalog entry: acquire its source/manual and examples, identify
its numerical and simulation features, implement those features in the same Python
package, validate against original output or independent published results, then
benchmark representative workloads. A generic statistical primitive does not count
as implementation of every program that uses it. Shared methods may share code,
but each catalog entry needs its own coverage evidence.

Original archives are research inputs and are not bundled. Their licenses must be
reviewed individually before incorporating any original code. Current interval code
uses independently expressed formulas with NumPy/SciPy kernels.
