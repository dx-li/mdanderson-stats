# SOGS: genotype-selection kernels

SOGS simulates backcross breeding to eliminate unwanted donor genome while
retaining specified donor regions. This port currently provides its deterministic
chromosome recombination, marker-screening and offspring-selection calculations.
The stochastic multi-generation simulation and report are still pending.

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
ordering as an immutable cM vector. Chromosomes containing desired donor regions
will be excluded when the complete breeding simulator is added.

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

## Validation and remaining work

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

**Status: partial.** Still required: random crossover generation and its avoidance
rule, offspring generation, repeated breeding under all four rules, chromosome
exclusions, variable offspring schedules, replicate distributions and Monte Carlo
summaries, and the report workflow. These kernels do not substitute for those
simulations.

Source inspection identified two details for the next phase:

- Avoidance is a soft retry rule: at most nine candidate positions are tried for
  a breakpoint, then the final candidate is accepted even if too close. The
  source draws a preliminary Poisson count to choose zero/nonzero crossover;
  the nonzero branch draws a fresh positive count. When screening is disabled,
  the source does not initialize its avoidance-distance variable.
- `chrom_sim` shifts count summaries by one generation to include the initial
  state, while donor-length summaries use the post-backcross generations directly.
  The Python report must distinguish these stages rather than silently pairing
  inconsistent generation labels.

Source: [MD Anderson SOGS](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/56),
[archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/SOGS/SOGS_V1.tar.gz).
Weil MM, Brown BW, Seachitopol DM, “Genotype Selection to Rapidly Breed Congenic
Strains,” Genetics 146:1061–1069 (1997). Source hashes are in `sogs-sources.json`;
original legal terms are preserved in the package notices.
