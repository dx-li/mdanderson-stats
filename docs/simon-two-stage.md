# Simon's two-stage design

Catalog entry **113** implements the statistical methods in MD Anderson's
[Simon2S application](https://biostatistics.mdanderson.org/shinyapps/Simon2S/).
The application links [Simon (1989)](https://biostatistics.mdanderson.org/shinyapps/Simon2S/Simon2Stage.pdf);
[source provenance](simon-two-stage-source.json) records the inspected paper.
The live input defaults are null response rate .05, alternative .15, one-sided
type I error .05, power .80 and maximum search size 100.

```python
from mdanderson_stats import simon_two_stage

result = simon_two_stage()
print(result.optimal)  # SimonDesign(n1=23, n=56, r1=1, r=5)
print(result.minimax)  # SimonDesign(n1=30, n=52, r1=1, r=5)
print(result.protocol("optimal"))

oc = result.optimal.operating_characteristics([0.05, 0.15, 0.25])
print(oc.reject_null)  # achieved type I error, target-rate power, higher-rate power
print(oc.stop_early)
print(oc.expected_sample_size)
```

Enroll `n1` participants initially. Stop for futility if at most `r1` respond.
Otherwise accrue to `n` participants and reject the null hypothesis only if the
total response count exceeds `r`. There is no early efficacy stop, even when
stage-one responses already exceed `r`.

Optimal designs minimize expected enrollment under the null. Minimax designs
minimize total enrollment, breaking ties by expected enrollment under the null.
The search considers `1 <= n1 < n <= max_n`, `0 <= r1 < n1` and `r1 <= r < n`.
It selects the largest final cutoff satisfying target power for each stage-one
configuration, then checks type I error. Optimal ties prefer smaller total size;
remaining ties prefer smaller first-stage size and smaller first-stage cutoff.

`max_n` accepts 2–500. **Optimality is within that bound.** An infeasible search
raises `ValueError`; it does not prove that no larger design exists. Null and
alternative rates must satisfy `0 < p0 < p1 < 1`, with `0 < alpha < power < 1`.
Error constraints use floating-point binomial probabilities without relaxing
the requested bounds. The mathematical search is exact rather than simulated.

`SimonDesign(n1, n, r1, r)` also evaluates an already specified design (up to
100,000 total participants). Its operating characteristics broadcast over rates
in [0, 1], including both endpoints. Rejection probabilities use positive sums
of binomial masses times direct survival probabilities, preserving small upper
tails. Continuation and early stopping are evaluated separately. Enrollment has
two-point support at `n1` and `n`; its mean and standard deviation are returned.
Output arrays are owned and read-only.

```python
design = result.optimal
assert design.decision(1) == "stop_futility"
assert design.decision(2) == "continue"
assert design.decision(2, 6) == "reject_null"
```

Decisions require completed stages and compatible response counts. Supplying
second-stage results after a first-stage futility stop raises an error.
`result.protocol("optimal" | "minimax")` produces a statistical design paragraph
with the stopping rules and achieved error probabilities. The original Shiny
interface and its download formatting are not reproduced.

## Numerical validation

Eleven focused tests cover six optimal/minimax pairs from Simon's Table 1,
unpruned rational enumeration of every design through total size 10 for two
parameter sets, exhaustive binary-path operating characteristics, stage decisions,
the source defaults, infeasible search bounds, and a rejection probability near
`5e-120`. Cached binomial tables and NumPy cumulative sums evaluate multiple
cutoffs together; expected-enrollment bounds prune configurations that cannot
improve the incumbent design. The default search through size 100 took about
0.04 seconds locally; this is a workload-specific timing, not a general guarantee.
