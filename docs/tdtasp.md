# TDTASP genetic model

TDTASP version 1.1 (April 2003), by Barry W. Brown and Dan Serachitopol,
plans transmission-disequilibrium (TDT) and affected-sibling-pair (ASP) studies.
This catalog entry is **partial**. The genetic, ascertainment, power and sample-size layers are implemented;
study orchestration and reports are also available. Template files and the complete
archive audit remain outstanding.

Source: [TDTASP catalog entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/20)
and its [version 1 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TDTASP/TDTASP%20%20_V1.tar.gz).
Despite the archive filename, the enclosed source and manual are version 1.1.
Original [legal notices](../notices/mdanderson-tdtasp-LEGALITIES.txt) are retained.
Original source, executables and the archived ACM numerical routines are not
bundled with the Python package.

```python
from mdanderson_stats import tdtasp_genetics, tdtasp_haplotype_frequencies

frequencies = tdtasp_haplotype_frequencies(
    marker_frequency=0.3, disease_frequency=0.2, relative_disequilibrium=0.7
)
model = tdtasp_genetics(frequencies, penetrance=[0.8, 0.3, 0.01], recombination=0.1)
```

Haplotype probabilities use **AD, Ad, BD, Bd** order; penetrances use **DD, Dd,
dd** order. A/B is the marker locus and D/d the disease locus. The frequency
vector must sum to one within 1e-12 and is then normalized. Penetrance values
lie in [0, 1] and recombination lies in [0, 0.5]. These strict probability
checks replace the console's approximate frequency-sum acceptance window.

`tdtasp_haplotype_frequencies` accepts population frequencies m of A and p of D,
and relative disequilibrium D prime. The absolute disequilibrium is bounded by
`-min(m*p, (1-m)*(1-p))` and `min(m*(1-p), (1-m)*p)`. D prime scales the bound
of the corresponding sign. Negative D prime and degenerate population allele
frequencies are supported for genetic calculations; the original one-sided
planning interface assumes positive association between A and D.

`TDTASPGenetics` contains immutable arrays for all 256 ordered parental
combinations. `parents` columns are father's two haplotypes and mother's two
haplotypes; codes are **0=Bd, 1=BD, 2=Ad, 3=AD**, matching the Fortran enumeration.
The rows are in lexicographic order. Outputs include:

- Each family's population probability under independent haplotype sampling.
- Father/mother marker-heterozygosity flags.
- The probability a child is affected, conditional on that parental family.
- TDT probability of transmitting A to an affected child, averaged over the
  marker-heterozygous parents in that family.
- ASP probability of transmitting the same marker allele to two independently
  generated affected siblings, averaged over those informative parents.
- Population prevalence, weighting affected-child probabilities by family mass.

A family with no informative parent has transmission/sharing zero. If a family
cannot have an affected child, its conditional transmission/sharing outputs are
also zero, following the source convention; `affected_probability` identifies
those rows. The zero is a sentinel for an uninformative/impossible conditioning
event, not a defined conditional probability. Families with zero population
mass remain present so downstream ascertainment can use the original indexing.

## Corrected and original ASP calculations

The original `offspring_mod.f90` first multiplies penetrance by offspring
transmission probability. Its `calc_exp_p_same` then multiplies both offspring
probabilities by their transmission probabilities again. That extra weighting
changes the conditional distribution when recombination is between zero and
one half. The default Python calculation uses each transmission probability
once. Given parental marker-A transmission probabilities q conditional on an
affected child, sibling sharing is `q*q + (1-q)*(1-q)`, averaged over informative
parents. This follows directly from independent sibling transmissions.

`legacy_asp=True` reproduces the archived duplicate path weighting and its
`1e-100` cutoff on pair contributions to the numerator. The denominator still
includes those tiny pairs, as in the source. Keeping transmission paths distinct
is necessary for that compatibility calculation, even when they encode the
same gamete. If positive affected probability produces a zero legacy pair
denominator through underflow, Python raises `ArithmeticError`; it does not
return a native NaN. The corrected calculation conditions before forming sibling
probabilities and avoids that unnecessary squaring of small masses. Both modes
use floating-point arithmetic, not arbitrary precision.

## Validation

`tools/reference_tdtasp_genetics.py` compiles the **unmodified** original
`offspring_mod.f90` with a separate driver. The fixture records archive/source
hashes, compiler, build command, driver and output column definitions. Twenty-one
models combine seven penetrance patterns and recombination 0, 0.1 and 0.5,
covering **5,376 parental-family rows**. Tests compare heterozygosity counts,
affected-child and transmission probabilities, and legacy ASP sharing.

Independent tests enumerate Mendelian gametes and all affected-sibling pairs
for every parental family at three recombination settings. These validate the
corrected ASP result separately from the archived defect. Other checks cover
analytic dominant/recessive examples, population prevalence under random mating,
allele marginals and disequilibrium bounds, impossible conditioning events,
legacy tiny-probability behavior, input validation and immutable results.

The numerical layer evaluates all families and offspring paths with NumPy.
The default sharing calculation uses conditional marginals instead of explicitly
enumerating sibling pairs. Compatibility mode uses a bounded 256-by-16-by-16
array to reproduce the original cutoff. Performance has not yet been benchmarked.

## Ascertainment

```python
from mdanderson_stats import tdtasp_ascertainment

selection = tdtasp_ascertainment(
    model,
    mean_offspring=2,
    test="tdt",
    sampling="individual",
    eligibility="one",
    minimum_affected=2,
    all_affected=True,
)
```

Family offspring counts are Poisson with common mean in [0.1, 10]. Given a
parental family with affected-child probability a, its affected count N is
Poisson with mean mu = mean_offspring × a. The selectable list contains families
with at least k affected children, or affected individuals from those families.
TDT accepts k=1–10 and one or all affected children. ASP uses k=2 and one sibling
pair. Unlike the old keyboard dialogue, Python also allows k>1 when only one
TDT child is used. No offspring are randomly drawn by this calculation.

For family ascertainment, a family's list weight is its population frequency
multiplied by P(N≥k). For individual ascertainment, the weight is population
frequency multiplied by E[N 1(N≥k)]. Parent eligibility is `father` (the source's
random parent), `one` (at least one), or `both` marker-heterozygous parents.
As in the source, all informative parents contribute after eligibility is
established, including under the father criterion.

The immutable result provides the selected family distribution, per-family
conditional affected counts and contributions, eligibility probability within
the list, expected affected counts, expected informative parents and expected
contributions. `answer_probability` averages TDT transmission or ASP sharing
using contribution weights. The genetic model controls the ASP compatibility
choice separately. Impossible selection raises a clear error.

Individual sampling size-biases the affected count within a family. Its mean
is E[N² 1(N≥k)] / E[N 1(N≥k)], whereas family sampling uses E[N | N≥k].
`legacy_moments=True` uses the latter mean for both sampling schemes, reproducing
the original within-family moment convention. The corrected mean matters for
TDT when all affected children contribute. `expected_affected` averages the
chosen conditional mean over **selected** families.

`population_average_truncated_mean` separately preserves the source's diagnostic
population-weighted average of within-family truncated means, assigning zero
when affected probability is zero. It is not the expected count in the selected
sample; the original program labels and uses this different quantity as `exp_n`.
Keeping these quantities distinct avoids silently substituting one for the other
when the study-planning layer is added.

`log_list_mass` and `log_eligible_mass` preserve rare list weights. For individual
sampling these masses represent expected affected counts per population family,
not probabilities. `log_screening_for_minimum` is the log of the expected number
of samples from the corresponding k=1 list needed to reach a k-qualified family,
before parental eligibility screening. It uses a ratio of total list masses.
This corrects the source's population-weighted average of conditional tail ratios.
The log value is returned directly because the count can exceed floating-point
range.

Poisson tails are evaluated through a convergent positive series after factoring
out their leading probability. Log normalization avoids cancellation and tiny-tail
underflow. Stable series ratios provide conditional means even when direct tail
ratios would be 0/0. This replaces the archived subtraction of a lower-tail sum
from one and its `mu <= 1e-10` zero cutoff. Positive means that underflow during
input multiplication raise an error.

`tools/reference_tdtasp_ascertainment.py` records 18 native studies spanning TDT
one/all-child modes, ASP, both list types and all three parental criteria. The
driver only exposes private variables/routines using PUBLIC declarations;
numerical routine bodies are unchanged. Native comparisons use `legacy_moments`
and the genetic `legacy_asp` option. Separate tests explicitly enumerate the joint
family/Poisson-count distribution to validate corrected selection, moments,
contributions, test probabilities and screening counts, including rare-event,
zero-probability-family and input-boundary cases.

## Power

```python
from mdanderson_stats import tdtasp_fixed_power, tdtasp_power

fixed = tdtasp_fixed_power([10, 20, 50], answer_probability=0.7, sides=2)
study_power = tdtasp_power(selection, families=100, alpha=0.05, sides=2)
```

The fixed-observation calculation tests a binomial probability of one half.
Inputs are nonnegative integer observation counts; scalar or array inputs are
accepted. One-sided direction follows the specified alternative, using the
lower tail at equality. Two-sided tests include both equal-alpha tails. The
result provides inclusive lower and upper critical values, actual size and
power. A cutoff of -1 or n+1 means the corresponding tail is absent. Zero
observations and unattainable significance levels give empty regions and zero
power. Degenerate alternatives zero and one are supported.

For the study calculation, K, the number of eligible families, is binomial with
`families` trials and the ascertainment eligibility probability. The observation
count is the mean contribution per eligible family multiplied by K, rounded to
the nearest integer with halves rounded upward. This matches the archived
`power_bin1` routine's ANINT behavior; its comment claiming truncation is
inconsistent with the code. Every possible K is evaluated and weighted by its
binomial probability. `max_terms` (default 100,001) is checked before allocation;
deterministic counts need only one term. The result exposes weights and all
conditional fixed-observation results as immutable arrays, plus average size
and power.

This is the original **mean-contribution planning model**, not a full joint
model of random contributions and dependent tests within families. The mixture
is exact for that approximation, up to numerical arithmetic. The program's
short-tail summation and large-variance normal/Hermite approximation are replaced
by complete discrete averaging. Large studies therefore have explicit resource
limits rather than an automatic normal approximation. Probabilities that
underflow to zero at the ascertainment boundary raise an error, and observation
counts must remain below 2**53.

By default the scale uses `expected_contributions` from ascertainment. For
all-child TDT, `legacy_scale=True` instead multiplies expected informative
parents by `population_average_truncated_mean`, as the original statistics
module does. The source's nominally two-sided calculation halves alpha but
computes only one tail's power; `legacy_two_sided=True` reproduces that convention
when sides=2. Genetic/ascertainment compatibility options remain separate so
numerical differences can be isolated.

`tools/reference_tdtasp_power.py` compiles six unmodified original numerical
source files. Its fixture contains 52 successful fixed-observation cases and
11 recorded source failures. In those failures the native continuous cutoff
inversion produces an out-of-range count and aborts, although the correct result
is an empty rejection region. Tests verify zero Python power for these cases;
they are not relabeled as successful native references. Successful critical
values and powers are compared directly, including n=1000.

Independent rational binomial enumeration validates one-sided, two-sided and
source-tail behavior, empty regions, and boundary probabilities. Explicit sums
over binomial eligible-family probabilities validate the complete mixture.
Tests also cover half-up rounding, null power equaling size, immutable output,
input/resource limits and the original scale option. A regression example
shows why power must not be assumed monotone in a future discrete sample-size
search: adding an observation can move the rejection cutoff and reduce power.

## Sample-size search

```python
from mdanderson_stats import tdtasp_fixed_sample_size, tdtasp_sample_size

fixed_design = tdtasp_fixed_sample_size(0.7, target_power=0.8, sides=2)
family_design = tdtasp_sample_size(selection, target_power=0.8, max_families=10000)
print(family_design.design.families, family_design.design.power)
```

Both searches return the **first qualifying integer design within inclusive
user bounds**. The target lies strictly between zero and one. If no design
qualifies, the function raises `ValueError`; it does not return a clipped bound
as a solution. The default maximum is 100,000 observations or screened families.
Fixed-observation search uses ordered vectorized batches (default 512). This
preserves nonmonotone discrete power: for example, at alternative 0.7 and alpha
0.05, n=5 meets target 0.16 while n=6 does not.

Family search caches the conditional power f(k) at each eligible-family count.
Its prefix maximum g(k)=max(f(0), …, f(k)) bounds f and is nondecreasing.
Since K~Binomial(n,q) increases stochastically with n, E[g(K)] is a monotone upper
bound on actual power E[f(K)]. Bisection finds where this bound first permits a
solution, then an ordered scan finds the first actual qualifying design. Counts
skipped by the bound cannot qualify. A conservative 1e-12 upward cushion on the
upper bound protects comparisons near numerical ties. The returned design is
recomputed using the public forward function before being accepted.

`TDTASPSampleSize` records that forward design, target, search bounds, first
scanned count and number of mixture evaluations. This is a first-crossing
criterion; it does not assert that every larger sample size also meets the
target. The source's continuous root-finding and local adjustment algorithms
are replaced by this explicitly discrete search.

The conditional curve grows geometrically and reuses cached values. `max_terms`
(default 100,001) bounds the cache before allocation, including when the forward
mixture would have deterministic eligibility. Prefetch growth stops at the
budget rather than rejecting an otherwise permitted count. A requested search
that needs more terms raises a resource-limit error. The power model, rounding,
and genetic/ascertainment/scale/tail compatibility options remain those described
above; sample-size search does not remove the mean-contribution approximation.

Validation compares both searches with exhaustive forward evaluation over the
relevant bounded ranges, including all parental criteria, both sampling methods,
TDT/ASP, one/two-sided tests, legacy choices, degenerate alternatives, equality
bounds, resource limits and known downward power jumps. In one tested model the
first design is 527 screened families with power about 0.800438; the upper bound
starts direct scanning at 514, requiring 34 mixture evaluations overall.

[Search benchmarks](tdtasp-search-benchmark.json) compare the bounded search with
an ordered scan using the same Python forward API, verifying agreement on the
first qualifying count. They report median-of-three local timings and include
all model parameters. They are not comparisons with native Fortran.


## Complete study calls and reports

```python
from pathlib import Path
from mdanderson_stats import tdtasp_study, format_tdtasp_study

study = tdtasp_study(
    [0.3, 0.2, 0.1, 0.4],
    [0.8, 0.5, 0.2],
    mean_offspring=2,
    recombination=0.1,
    target_power=0.8,
    max_families=1000,
)
Path("tdtasp-report.txt").write_text(format_tdtasp_study(study))
```

Supply exactly one of `families` (forward power, including zero) and
`target_power` (sample-size design). The function computes genetics and
ascertainment once, then calls the validated statistical components. Search mode
returns both the first qualifying family design and the fixed-observation
comparison printed by the original program. Both must succeed; an insufficient
bound or resource limit raises an error rather than returning a partial study.
Separate inclusive `min_families`/`max_families` and
`min_observations`/`max_observations` constrain the searches. Search bounds and
`batch_size` are inactive for a forward calculation.

`TDTASPStudy.design` provides the full family-power result, including its
ascertainment and genetic inputs. In search mode, `search` contains the family
search diagnostics, `fixed_design` contains the fixed-observation result, and
`fixed_search_bounds` retains its bounds. These fields are `None` in forward
mode. All four compatibility options are explicit and independent; no legacy
behavior is enabled implicitly. Like the numerical API, study calls allow
two-sided ASP, extending the original console's one-sided restriction.

`format_tdtasp_study` returns text for callers to print or save. It includes
population inputs, selection rules, moments, list masses, the screening factor,
all compatibility flags, contribution scaling, screened and expected eligible
family counts, actual significance and power. Search reports add the target,
bounds, diagnostics and fixed-observation comparison, including its critical
values. Rare-event masses and screening factors remain on the natural-log scale.
The report identifies the mean-contribution approximation and the first-crossing
search criterion. `digits=1` through `17` changes displayed significant figures
only; it does not change the underlying calculation.

Integration tests compare complete forward calls with real component calls
across TDT/ASP, both sampling schemes, all parental criteria, one/two-sided tests
and both compatibility settings. They also validate searched designs, separate
inclusive bounds, zero-family studies, propagated failures, input ownership and
report saving. The numerical component validations above remain the statistical
correctness evidence.
