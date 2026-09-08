# KSB1CI: confidence intervals for multistage binomial trials

Catalog entry 47 is implemented through `KStageBinomial`: design entry, stage-ordered
probabilities, confidence intervals, changing confidence or observation points,
and design/result reports. The source is Barry W. Brown's KSB1CI Version 1.1
(February 1994), archived as KSB1CI_V1.tar.gz.

```python
from mdanderson_stats import KStageBinomial

trial = KStageBinomial([14, 28, 42], low=[0, 1], high=[3, 4])
lower, upper = trial.interval(stage=3, events=3, confidence=0.95)
# approximately (0.0179281, 0.241485)
print(trial.report(3, [3, 5]))
trial.write_report("intervals.tsv", 2, 6, confidence=0.95)
less, greater = trial.tails(3, 3, probability=[0.06, 0.2])
```

Sample sizes are cumulative. A low boundary stops when the accumulated number of
events is at most that value; a high boundary stops at or above that value. Use
-1 to disable a boundary. Supply boundaries for interim stages only. Stages are
numbered from one. Events, probabilities and confidence levels broadcast over
independent cases for a fixed observation stage.

## Statistical definition

The less tail sums all earlier low stopping outcomes and all paths reaching the
observed stage with at most the observed count. The greater tail sums earlier
high stopping outcomes and paths reaching the observed stage with at least that
count. Both include the observed outcome; their sum is generally greater than one.
These are stage-ordered probabilities, not ordinary binomial tails conditional on
reaching a stage. Earlier high stops are excluded from the less tail, and earlier
low stops from the greater tail.

The equal-tailed interval inverts each inclusive tail at (1-confidence)/2. With
no early stopping this reduces to the Clopper–Pearson interval. The observation
may end the trial within its planned continuation region, as the original manual
explicitly permits; earlier-stage stopping rules still apply. An observation
unreachable under those earlier rules raises rather than fabricating an interval.

The design caches coefficients counting surviving Bernoulli paths, using NumPy
convolution to combine each stage's new subjects. Probability evaluation uses
complete log-weighted terms to avoid premature underflow. Both tails are summed
directly, avoiding subtraction from one. Inversion bisects [0,1] for 55 iterations
and returns exact zero/one bounds when the relevant tail never crosses its target.
This controls absolute probability-parameter error; it is not a relative-accuracy
guarantee for bounds extremely close to zero.

## Source coverage and deliberate changes

| Original functionality | Python implementation |
| --- | --- |
| SETKS and coefficient access entries | Validated immutable design and cached path coefficients |
| PRSTAT, PRSTLO, PRSTBY | `tails` combines stage arrivals and earlier low/high stops |
| PLOCON, PHICON, FPRSTB and QRZERO | `interval` inverts inclusive tails with binary64 bisection |
| Enter cumulative design and stopping boundaries | Constructor arguments |
| Change confidence / stopping point / design | Repeated calls / construct another design |
| Print design and interval | `report` and `write_report` |
| Exit, screen clearing, pause and input retry | Ordinary caller control flow and exceptions |

The original shared mutable Fortran state, preprocessor control flow, machine I/O
and fixed-width whitespace are replaced by Python calls and TSV output. File
writes explicitly replace the named UTF-8 file and propagate errors. Reports echo
stage increments, cumulative sizes, stopping boundaries and every broadcast case;
precision is configurable from 1 to 17 significant digits.

The Python domain is 1–10 stages with increasing positive cumulative sizes at most
200. The original coefficient helper additionally limits a single increment to
100; this allocation restriction is removed within the total-200 limit. Boundaries
may be disabled or lie between zero and the stage total. Overlapping stops and
unreachable planned stages raise. Redundant, nonmonotone boundaries are allowed
when continuation remains possible; the path recurrence applies them literally.
Original input menus impose additional boundary-order restrictions. Confidence
must be strictly between zero and one, instead of passing endpoint confidence
levels into a root finder that cannot generally bracket the target. The source's
p<1e-20 shortcut is replaced by mathematical probabilities. Binary32 arithmetic
and its root tolerance of 1e-4 are replaced by binary64 calculations.

## Validation and performance

`tools/reference_ksb1ci.py` extracts 11 complete numerical routines unchanged from
the original source and compiles them with gfortran -O2 -std=legacy. The isolated
driver records 36 cases covering single/multistage designs, disabled boundaries,
interim observations and three confidence levels. Source/archive/extracted-code
hashes and compiler provenance are in `tests/fixtures/ksb1ci.json`. Native tail
checks allow binary32 accumulation error; interval checks allow 1.1e-4 absolute
error for the source solver's tolerance.

Independent tests exhaustively enumerate Bernoulli paths for small designs,
verify both tail orientations, check interval inversion and frequentist coverage,
and compare no-stopping designs with Clopper–Pearson intervals. Boundary,
broadcasting, unreachable observations, file output and validation are tested.
A 100-digit Decimal mass sum independently verifies the representable tail
P(Binomial(200,0.02)>=190) = approximately 2.881774153016122e-307. In the local
SciPy 1.18.1 build, the incomplete-beta call used by the separate `binomial_test`
API returns about 2.218453989e-307 for this extreme case. That pre-existing
limitation is not used as a reference for this test and is not fixed by this port.

A local Python 3.13 / NumPy 2.5.3 check evaluated 5,000 probabilities from 0.001 to
0.999 for the example design at stage 3, count 3: one vector call took 0.0075 s
versus 0.352 s for separate scalar calls (about 47x). The results agreed. This is
a workload-specific timing, not a universal performance guarantee.
