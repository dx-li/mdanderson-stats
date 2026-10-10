# Recovered fixed-ray interaction-index study

`simulate_interaction_index_fixed_ray_study` implements Scenario 2 from the
original CI-IIV2 source archive. The recovered ratio is `d2/d1=2`. Single-agent
slopes are -1 with median doses 1 and 2; the total-mixture-dose slope is -2 with
median dose 1.5. Five dose points per curve receive independent normal errors
on the logit-effect scale. Defaults are SDs .2/.4, seven datasets per SD,
500 coefficient draws and 43 effects from .10 through .94.

```python
from mdanderson_stats import simulate_interaction_index_fixed_ray_study

study = simulate_interaction_index_fixed_ray_study(
    error_sd=(0.2,),
    replicates=1,
    samples=100,
    rng=65,
)
study.write_json("fixed-ray-study.json")
figure = study.plot(0)  # requires the plot extra
figure.savefig("fixed-ray-study.png")
```

Each dataset retains fitted curves, observed responses, the corrected log-delta
interval, the normal-coefficient Monte Carlo comparator, the observed-combination
pooled-error interval and its length relative to the MC interval. Native reporting
floors MC lower limits at .0001. The study records those limits separately from
the unmodified MC limits; the generic MC API continues to return raw limits.
Plots show pointwise intervals on a log axis, rather than simultaneous bands.

The recovered `CI.delta` function omits squared inverse mixture dose from its
mixture-variance term. Python retains the mathematically correct gradient
already used by `interaction_index_ray`. Original-function references retain
both the erroneous native interval and an independent corrected base-R interval.
The original observed-combination and MC routines agree within `1e-12` for two
synthetic designs with unequal curve sample sizes and externally replayed
coefficient draws. This also resolves the native residual-df pooling denominator.
See the [workflow audit](../research/interaction-index-workflow-audit.md).

PCG64 and child dataset seeds provide Python replay; original S-Plus/R random
streams are not reproduced. All numerical failures identify their dataset and
raise, without retrying or discarding it. Aggregate coefficient, work and storage
limits are checked before random generation. Defaults bound storage to 64 MB;
controls and hard ceilings are in the function signature. Saved JSON captures
controls, responses and results. It is a numerical record, not an executable
native session.

For recovered Scenario 1 presets, pass `INTERACTION_INDEX_SOURCE_SCENARIOS` to
`simulate_interaction_index_three_drug_study`, with `error_sd=(.4,.1)`,
`replicates=1000` and an explicit Python seed. The original seventh constant is
`1/.6`, rather than the printed `1.67`; the existing paper-based default remains
unchanged. `retain_samples=True` stores at most 50,000 index estimates and
supports `.plot_qq(cell)` for index/log-index panels using original `ppoints`
normal quantiles. `.write_json(path)` captures inputs, summaries and optional
samples. Exact source page layout and RNG identity are compatibility differences.
