# MERIT interim sample-size search

`merit_interim_sample_size` combines the existing integer final-boundary search
with an explicit `MERITInterims` schedule. The interim posterior targets must
match the acceptable alternative rates supplied to the search:

```python
from mdanderson_stats import MERITInterims, merit_interim_sample_size

result = merit_interim_sample_size(
    interims=MERITInterims(
        toxicity_target=0.2,
        efficacy_target=0.4,
        toxicity_looks=[8, 17],
        efficacy_looks=[13],
    ),
    toxicity_null=0.4,
    toxicity_alternative=0.2,
    efficacy_null=0.2,
    efficacy_alternative=0.4,
    alpha=0.1,
    power=0.6,
    max_patients_per_arm=40,
    trials=5000,
    rng=160,
)
print(result.design)
```

The search starts above the last scheduled look and tests each maximum per-arm
sample size in order. At each n, it evaluates every integer final toxicity
maximum and efficacy minimum from zero through n. A candidate must meet the
estimated global type-I error constraint and the selected estimated Power I or
Power II target across the paper's corner configurations. Ties prefer greater
chosen power, smaller global error, smaller toxicity maximum, and then greater
efficacy minimum.

Interim boundaries use `MERITInterims.boundaries`: raw arm-specific event
counts, its explicit Beta prior/targets, and strict posterior probability
cutoffs. Arms stop permanently on either interim rule. Stopped arms receive no
later observations, are excluded from final pooling, and do not reallocate
patients to other arms. Surviving arms can reach n. These are the package's
documented interim conduct conventions, not a claim of native app parity.

The result reports selected-boundary error and power estimates with binomial
MCSEs for each corner, plus mean enrollment by dose and corner with trial-level
MCSEs. These are Monte Carlo estimates and do not guarantee constraints in the
underlying probability model. The function bounds scenario state, boundary-grid
scratch, and a conservative work estimate before using the random generator.
See [`research/merit-interim-search-audit.md`](../research/merit-interim-search-audit.md)
for equations, corner ordering, and implementation policies.
