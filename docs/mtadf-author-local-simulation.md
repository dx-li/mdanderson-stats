# MTADF author-local complete-cohort simulation

`simulate_mtadf_author_local` replays the rendered author program's local
logistic conduct. It is separate from the paper-policy local API described in
[MTADF logistic](mtadf-logistic.md): it uses the adjacent-dose slope gates,
the source's cap-one branch, and the author final isotonic selection.

```python
from mdanderson_stats.mtadf_author_local_simulation import simulate_mtadf_author_local

simulation = simulate_mtadf_author_local(
    true_toxicity=[0.05, 0.15, 0.30],
    true_efficacy=[0.15, 0.55, 0.40],
    cohorts=3,
    cohort_size=3,
    trials=2,
    draws=256,
    warmup=128,
    chains=2,
    rng=871,
)
print(simulation.selection_probability, simulation.trial_seeds)
```

The first complete cohort goes to dose index zero. After it, the source
provisionally sets the next assignment to dose index one; before each later
cohort, an admissibility cap of one overrides that assignment to dose zero
and skips MCMC. Otherwise the source observes the current cohort, chooses the
next dose using the cap from before that cohort, then refreshes the cap. This
preserves its lagged-cap and floor behavior. Trials use at least two doses
and two cohorts, complete outcomes, and have no safety stop.

The source thresholds are named `ce1` and `ce2`. At the lowest dose, move up
when the forward positive-slope probability exceeds `ce1`. At the highest,
move down when the backward nonpositive-slope probability exceeds `1-ce1`.
At an interior dose with an untried next level, move down above `1-ce1`, move
up at or below `1-ce2`, and otherwise stay. When the next level has data, move
up only when the backward probability is at or below `1-ce2` and the forward
probability exceeds `ce1`; move down when the backward probability exceeds
`1-ce1`; otherwise stay. Defaults are `ce1=0.3` and `ce2=0.4`.

`replay_mtadf_author_local_trial` accepts potential cohort outcome tables with
shape `(cohorts, doses)` and an explicit unsigned 64-bit `sampler_seed`. Only
the assigned dose's entries are observed. Its immutable ledger includes
assignments, caps before and after each cohort, slope probabilities where
fitted, fit counts, compact MCMC diagnostics, and the final selected dose.
The simulator draws independent toxicity and efficacy binomial counts for
each potential cohort-dose cell, then uses the same replay. Its
`trial_seeds[:, 0]` and `trial_seeds[:, 1]` hold separate outcome and sampler
seeds for reproducing the outcome table and decision path. When a trial takes
the cap-one branch throughout, slope-fit diagnostics are NaN because no fit
was performed; they are not zero-valued estimates. For trials with fits,
review the returned acceptance, R-hat and slope-probability MCSE diagnostics
before interpreting operating characteristics. NumPy randomness
and MCMC are explicit Python conventions; no R random-stream parity is
claimed.

Final selection uses the fresh toxicity cap and the author all-dose fit of
`responses / (subjects + 0.0001)`, selecting its rightmost fitted maximum and
capping that dose by admissibility. The author prior is fixed by
`pbeta(0.3, alpha, 0.5-alpha) = 0.22`; toxicity and safety defaults are `.3`
and `.8`. The local model uses the globally standardized ordinal dose
coordinates and independent Cauchy coefficient scales 10 and 2.5. Python
defaults to 2,000 retained draws, 1,000 warmup draws, and four chains; this
does not reproduce the source's exact `metrop` sample.

The example is intentionally small and demonstrates mechanics rather than
precise operating characteristics or convergence. Work, two-fit peak memory,
potential-outcome cells, and aggregate storage are checked before random
draws, and posterior arrays are released after each cohort. See the
[source crosswalk and audit](../research/mtadf-author-local-simulation-audit.md).
