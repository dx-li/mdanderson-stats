# SOGS: genotype-selection breeding simulation

SOGS simulates backcross breeding to eliminate unwanted donor genome while
retaining specified donor regions. The Python implementation provides chromosome recombination, marker screening,
all four offspring-selection rules, repeated backcross simulation and reports.

```python
from mdanderson_stats import sogs_recombine, sogs_screen, sogs_eligible

cross = sogs_recombine([[0, 20]], length=20, breakpoints=[3, 8, 14])
# retained: [[0,3], [8,14]]; transferred: [[3,8], [14,20]]
scan = sogs_screen([[5.01, 9.99]], length=20)
assert scan.missed  # donor material lies between markers at 5 and 10 cM

eligible = sogs_eligible(
    [[True, True, False], [False, False, True]],
    chromosome_lengths=[2, 3, 20],
    rule=3,
)
# eligible == [1]: greatest whole-chromosome length classified as IPT
```

## Representation and recombination

A chromosome is represented by increasing, separated `[start,end]` donor-segment
intervals in centimorgans. The other homologue is entirely inbred-partner type
(IPT). An empty segment array represents a pure IPT chromosome. Inputs must be
finite, intervals must have positive lengths and lie in `[0,length]`, and touching
intervals should be merged before calling. Chromosome length is limited to
10000 cM and input interval/breakpoint arrays to 10000 entries each.

`sogs_recombine` starts on the donor-containing homologue and switches at every
supplied breakpoint. Breakpoints must be strictly increasing and within the
chromosome. It returns both complementary donor-segment sets as immutable arrays.
An odd number of breakpoints transfers the final interval through the chromosome
end. Zero breakpoints retain the original chromosome in `retained` and leave
`transferred` empty; random choice between homologues is a separate simulation
step. Returned segments stay ordered, and donor length is conserved. This uses
interval partitioning rather than the source's fixed ten-segment storage.

## Screening

`sogs_screen` places `floor(length/5)+1` markers, including both chromosome ends,
with equal separation `length/floor(length/5)`. Thus the spacing is approximately
5 cM, not necessarily exactly 5. For lengths below 5 cM, Python uses the two
endpoints instead of the source's division by zero; all 19 source mouse chromosome
lengths exceed 5 cM. Segment endpoints count as marker hits.

The result exposes marker positions, total donor length, whether a marker
detected donor material, whether the chromosome is pure IPT, and whether donor
material was missed. A pure IPT chromosome has `missed=False`; the source's
internal `is_missed` returns true for an empty chromosome, but its simulation
checks purity first and never counts such a chromosome as missed.

`SOGS_MOUSE_LENGTHS` preserves the source's 19 chromosome lengths and their
ordering as an immutable cM vector. Chromosomes containing desired donor regions are excluded by `sogs_simulate`.

## Selection

`sogs_eligible` takes an offspring-by-chromosome **boolean** matrix indicating
which chromosomes appear to be pure IPT, plus the whole chromosome lengths. It
returns every eligible offspring's zero-based index, without breaking ties:

| Rule | Eligibility |
| --- | --- |
| 0 | All offspring; random selection happens afterward |
| 1 | Maximum number of apparent IPT chromosomes |
| 2 | Maximum number, then maximum total length of those chromosomes |
| 3 | Maximum total length of apparent IPT chromosomes, ignoring their number |

A missed donor/mixed chromosome counts as apparent IPT when screening error is
being considered. The length criterion sums *whole chromosome lengths* classified
as IPT, not the amount of IPT material within mixed chromosomes. Up to 100000
candidates and two million matrix entries are accepted. Fixed-order NumPy sums
avoid BLAS-dependent tie ordering; calculations use float64 rather than source
single precision.

## Kernel validation

The original `sim_input.f90` and `sim_info.f90` were compiled unchanged. A harness
calls `one_event`, `is_missed` and `b_stretch` for six synthetic cases covering
splits, multiple donor intervals, complete transfer, endpoint detection and
missed fragments. Recorded segment coordinates and lengths match within 1e-5 cM,
consistent with source single precision. Intervals are compared in coordinate
order; the source sometimes stores them in append order. Marker classifications
match exactly.

The original `one_simulation.f90` was also compiled unchanged against a controlled
candidate generator. Its real eligibility/selection code chose the same first
eligible candidate for all four rules, including tie handling. The candidate
harness does **not** validate the stochastic offspring generator. Additional
checks verify donor-length conservation, marker endpoints and invalid overlapping
segments. See `tests/test_sogs.py` and `tests/fixtures/sogs-native-kernels.json`.

## Breeding simulation

```python
from mdanderson_stats import sogs_simulate, sogs_summary, format_sogs

result = sogs_simulate(
    offspring=(10,) * 10,
    exclude=(16,),
    rule=2,
    screening_error=True,
    avoidance=5,
    replicates=1000,
    seed=1997,
)
summary = sogs_summary(result)
print(format_sogs(result))
```

`offspring` supplies 1..10 backcross generations, with 1..20 eligible male
offspring per generation. These offspring are assumed already to carry the
required donor region: its ascertainment is not separately simulated, matching
SOGS. `exclude` contains unique chromosome numbers in 1..19 that carry the desired
donor regions; at least one chromosome must remain. An empty exclusion tuple
simulates all 19 chromosomes. The sex chromosome is outside the source model.

The initial F1 parent has a fully donor-derived chromosome paired with a pure
IPT chromosome at each simulated locus. Each candidate inherits one of the two
recombined homologues with probability one-half. All candidates undergo the
requested screening and selection rule; ties are broken uniformly at random.
The selected parent is backcrossed again to a pure IPT partner. Pure IPT
chromosomes remain pure, and descendant donor segments are subsets of their
parent's donor segments.

Crossover counts follow Poisson(length / 100), with uniform chromosome positions.
The source first decides whether there are any crossovers, then redraws a positive
Poisson count. Its marginal count distribution is the same ordinary Poisson law
used here. NumPy draws counts and homologue choices in batches. Interval traversal
uses the same partition rule as `sogs_recombine`, with already-established internal
invariants; the source's ten-segment storage limitation is removed. Simulation
stops evolving a replicate once all chromosomes are truly pure IPT.

`screening_error=False` classifies chromosomes by their actual donor content.
With screening enabled, the source's approximately 5-cM marker grid determines
apparent purity. `avoidance` is 0, 5 or 10 cM and must be zero when screening is
disabled. It is a **soft** avoidance distance: up to nine candidate positions are
tried for each additional breakpoint, then the final candidate is accepted even
if too close. `avoidance_failures` counts these fallback placements across all
simulated candidates. It is not a count of selected offspring or a strict
spacing guarantee.

`replicates` is 1..100000 and `seed` is a nonnegative integer. Work is bounded by
20 million replicate/offspring/chromosome combinations. Random state is local;
repeating the same inputs reproduces the result, without claiming S/RANDLIB seed
parity. A local workload of 1000 replicates with 18 chromosomes, ten backcrosses
and ten offspring per backcross completed in 1.44 seconds (screening enabled,
5-cM avoidance). This is a measured example, not a performance guarantee.

## Generation-aligned results and reports

Result arrays have shape `(replicates, number_of_backcrosses + 1)`. Column zero
is the initial F1 state; subsequent columns are BC1, BC2, etc. `pure_ipt` contains
the true pure-chromosome counts, `missed` counts donor-containing chromosomes
missed by screening, and `apparent_dmt` equals chromosome count minus both.
`donor_length` and `missed_length` contain total and missed donor-segment lengths
in cM. Arrays are read-only. Simulation configuration is retained in the result.

`sogs_summary` returns stage labels, means and Monte Carlo standard errors for
apparent DMT counts, missed DMT counts, total donor cM, missed donor cM, and their
two percentages. Percentages divide by the total length of the **simulated single
chromosome copies**, as in SOGS: the F1 baseline is 100%, not its 50% diploid donor
DNA proportion. Excluded chromosomes do not enter that denominator.
`apparent_dmt_cdf` and `missed_dmt_cdf` have one row per stage and one column per
count k=0..number_of_chromosomes, giving P(count <= k). For a single replicate,
Monte Carlo SE is NaN rather than a fabricated precision estimate.

`format_sogs` returns the configuration, all six mean/SE tables and both count
CDF tables as text. The caller chooses whether to print or save it. All quantities
use the same F1/BC stage labels. This corrects `chrom_sim`'s report mismatch: its
count summaries include the initial state and omit the last simulated backcross,
while its donor-length summaries begin after the first backcross.

## Stochastic validation and source compiler issue

The original recombination, breeding and RANDLIB source routines were compiled
with bounds checking for the native comparison. Eight scenarios cover every
selection rule with and without screening, using two chromosomes and the variable
offspring schedule (3,2,4,3). Each native scenario used 20000 replicates. Python
means over 1000 independent replicates agree within five combined Monte Carlo
standard errors for apparent/missed counts and total/missed donor length across
all four backcrosses. Native means/SEs are recorded in
`tests/fixtures/sogs-native-simulation.json`.

A source compiler dependency initially caused disagreement: `ignpoi.f90` saves
its Poisson cache state but omits the cached `pp` probability table from `SAVE`.
The reference therefore uses `-fno-automatic` to retain local state, along with
explicit zero initialization of avoidance when screening is disabled. Adding
**only `SAVE pp`** to a private reference copy, without `-fno-automatic`, reproduced
the same native output exactly in all eight scenarios. Original source files are
unchanged and are not shipped. Python's NumPy generator does not use this cache.

Independent checks confirm the random-breeding expectation that donor length
halves at every backcross, and the first-backcross probability of a pure IPT
chromosome, exp(-length/100)/2. Additional checks cover permanent purity,
nonincreasing donor length, missed-versus-total bounds, inner-kernel agreement,
replay, generation alignment and report distributions. See
`tests/test_sogs_simulation.py`. No CI workflow was expanded.

**Coverage:** chromosome-level recombination, marker error, all four selection
rules, exclusions, fixed/variable offspring schedules, replicate breeding and all
advertised distribution/mean reports are implemented. The Python API and text
report replace the original interactive prompts and legacy report layout; no
exact legacy RNG sequence or incorrectly shifted generation table is claimed.

Source: [MD Anderson SOGS](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/56),
[archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/SOGS/SOGS_V1.tar.gz).
Weil MM, Brown BW, Seachitopol DM, “Genotype Selection to Rapidly Breed Congenic
Strains,” Genetics 146:1061–1069 (1997). Source hashes are in `sogs-sources.json`;
original legal terms are preserved in the package notices.
