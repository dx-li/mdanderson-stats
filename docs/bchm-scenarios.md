# BCHM community scenario reports

`BCHMScenario` runs one named subgroup dataset through the existing `bchm_fit`
numerical implementation. `fit_bchm_scenarios` runs a bounded set in order.
These helpers are also available from the package's top-level namespace.

```python
from mdanderson_stats import BCHMScenario, fit_bchm_scenarios

scenario = BCHMScenario(
    "illustrative basket",
    successes=[1, 2, 7],
    trials=[15, 18, 20],
    subgroup_labels=("A", "B", "C"),
    seed=158,
    burn_in=50,
    iterations=100,
    warmup=32,
    draws=64,
    chains=2,
)
batch = fit_bchm_scenarios((scenario,))
print(batch.report())
scenario.write_input_csv("bchm-input.csv")
batch.write_report("bchm-report.txt")
batch.write_samples_csv("bchm-samples.csv")
```

The input CSV is a Python-defined format, not a claimed copy of the app's hidden
upload schema. It has one row per subgroup and these exact columns, in order:
`scenario, subgroup, successes, trials, seed, mu, sigma02, sigmaD2, alpha, d0,
alpha1, beta1, tau2, phi1, deltaT, thetaT, burn_in, iterations, draws, warmup,
chains`. Scenario-level settings repeat on every row and must agree when loaded
with `BCHMScenario.from_input_csv`. Counts are integers; subgroup labels are
unique, nonempty strings. Inputs are limited to 20 subgroups and 256 KiB.

The sample CSV contains one row per retained target probability, with columns
`scenario, subgroup, target_index, chain, draw, probability`. Indices are
zero-based. Rows stream to a temporary file before an atomic replacement, so
the entire sample table is not copied into memory. Input, report and sample
files replace an existing destination only after a complete successful write;
their parent directory must already exist.

The report records subgroup counts, all clustering and borrowing settings,
root and spawned seeds, the three similarity stages, representative partition,
posterior means, efficacy estimates and native-rounded decisions. It includes
batch-means Monte Carlo standard errors for posterior means, clustering
similarities and efficacy-event indicators, plus split R-hat and empirical
80% intervals for posterior probabilities. The event diagnostics use the
sampler's original logit-scale indicators, preserving endpoint behavior. These
are finite-run diagnostics, not convergence guarantees. Clustering MCSE uses
non-overlapping sequential batch means from its single allocation chain;
efficacy MCSE and R-hat use the existing chain-aware summary method.

Native R/JAGS starts four chains but extracts the first and uses R/JAGS random
streams. This Python workflow uses the requested Python chains and NumPy seed
streams; results and diagnostics therefore need not match the native app.
The cached application page shows input/save, PDF and sample-download controls,
but no server handlers or file schemas. These Python formats are intentionally
explicit rather than guessed native contracts. See the [source crosswalk](../research/bchm-scenario-report-audit.md).

At most 20 scenarios are accepted. Every scenario is validated before any fit
starts, and the aggregate bound is one million units of estimated clustering
and borrowing work. Core per-fit limits still apply. Use small settings for an
illustration; choose larger MCMC runs based on convergence assessment rather
than treating the examples as recommended inferential settings.
