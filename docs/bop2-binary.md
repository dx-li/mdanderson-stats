# BOP2 binary efficacy and toxicity monitoring

Catalog entry **112**, [BOP2](https://biostatistics.mdanderson.org/shinyapps/BOP2/),
is partially implemented for binary efficacy and toxicity: specified-parameter monitoring,
exact operating characteristics, and power-maximizing finite-grid calibration.
[Ordinal and multiple efficacy](bop2-paired.md) and
[joint efficacy/toxicity](bop2-efftox.md) are also available. Time-to-event endpoints
and integrated reports remain pending. Categorical [sample-size optimization](bop2-sample-size.md)
is available.

The app snapshot is version 1.4.27.0, updated September 4, 2026. Its binary-prior
and error-control guides are pinned in [provenance](bop2-sources.json). The original
method is Zhou, Lee and Yuan (2017), DOI 10.1002/sim.7338. The strict futility
rule and power-function cutoff are also given in the clinical-methods study
[Bayesian sequential monitoring strategies for trials of digestive cancer therapeutics](https://link.springer.com/article/10.1186/s12874-024-02278-3).
The related [BOP2-TE paper](https://arxiv.org/abs/2408.05816) explains the restriction
of the power exponent to `[0,1]`; its joint efficacy/toxicity design is implemented [separately](bop2-efftox.md).

## A specified efficacy design

```python
from mdanderson_stats import bop2_binary_design

design = bop2_binary_design(
    40,
    null_rate=0.2,
    cutoff_scale=0.86,
    gamma=0.95,
    looks=[10, 20, 30, 40],
)
print(design.futility_max)
print(design.monitor(events=1, sample_size=10).decision)
print(design.operating_characteristics([0.2, 0.4]).positive_conclusion)
```

With responses `y` among `n` fully observed subjects, use a beta-binomial posterior
and stop for futility if

$$
\Pr(p\le p_0\mid y,n)>1-\lambda(n/N)^\gamma.
$$

The implementation compares the directly evaluated upper tail against
`lambda*(n/N)**gamma`, avoiding subtraction from a near-unit lower-tail cutoff.
Equality continues. At the final analysis, the same rule determines whether the
treatment is promising. There is no early success stopping. Parameters require
`0<p0<1`, `0<lambda<1`, and `0<=gamma<=1`. Cutoffs that underflow to zero
raise an explicit numerical error.

The default prior is `Beta(p0,1-p0)`, with effective sample size one. A specified
positive two-shape `prior` may be used when constructing a design directly. This
factory does **not** calibrate its supplied parameters or guarantee a particular
frequentist error rate. Inspect its operating characteristics before using them.

The returned `BayesianMonitoringDesign` reuses the package's binary monitoring
engine. `futility_max` lists the largest response count leading to futility at
each scheduled look (`-1` means none). The final successful-response minimum is
`final_positive_min`. `monitor` accepts broadcast counts; off-schedule evaluations
continue. `monitor_outcomes` processes complete binary paths and retains their
first stopping decision. The final look must equal `max_subjects`; default looks
start at 10, occur every 5 subjects, and include the final analysis. A custom
schedule is needed when the maximum is below the default initial look.

Exact operating characteristics use forward binomial probability propagation.
They include the final positive/negative probabilities, early futility probability
at each look, sample-size distribution and expectation, and response-count
summaries. Operating characteristics broadcast over true response rates and use
no simulated outcomes.

## Finite-grid calibration

```python
from mdanderson_stats import optimize_bop2_binary

fit = optimize_bop2_binary(40, 0.2, 0.4, looks=[10, 20, 30, 40], type1_error=0.1)
print(fit.cutoff_scale, fit.gamma)
print(fit.calibration_oc.positive_conclusion)  # null error, alternative power
print(fit.calibration_oc.expected_sample_size)
```

By default, `optimize_bop2_binary` maximizes exact power over the supplied `cutoff_scales` and
`gammas` grids. Defaults are scales `0.50,0.51,...,0.99` and exponents
`0,0.05,...,1`. Repeated integer stopping boundaries are evaluated once. Power
ties favor lower expected sample size under the null, followed by input grid order.
This is an optimum over that **finite grid**, not over all real-valued parameters
or all possible stopping boundaries. The grids and tie-breaking policy are explicit
Python choices; native app optimization parity has not been established.

`error_control="strict"` is the default and enforces exact numerical type I error
at or below the supplied nominal level. `error_control="closest"` instead first
minimizes the absolute error-rate distance to nominal and then maximizes power;
it can select a design above nominal. This distinction is exposed because the
app's current guide permits closest-to-nominal error control by default. A grid
with no feasible strict candidate raises a clear error rather than relaxing the
constraint. Calibration supports up to 200 subjects and 100,000 parameter pairs;
specified designs support up to 1,000 subjects.

Calibration always uses the null-centered prior. Optional `analysis_prior` then
recomputes the decision boundaries and operating characteristics with the chosen
parameters under that prior. Both `calibration_design`/`calibration_oc` and
`analysis_design`/`analysis_oc` are retained. An informative analysis prior can
increase type I error; the calibration guarantee does not transfer to that
changed analysis. This separation follows the current app's prior guide.

Validation checks boundaries against an exact rational beta-binomial identity,
explicit equality handling, and operating characteristics against enumeration of
all 64 binary paths in a small trial. A separate small-grid search verifies the
chosen optimum, and an informative-prior case confirms that calibration parameters
stay fixed while the resulting error rate is separately reported.

## Binary toxicity

Set `endpoint="toxicity"` in either factory. Count observed toxicities as events;
the null toxicity rate is unacceptably high, and the alternative must be lower.
The app's [null guide](https://biostatistics.mdanderson.org/shinyapps/BOP2/Null2Help.pdf)
defines type I error as concluding that treatment is safe at this unacceptable rate.
Its [alternative guide](https://biostatistics.mdanderson.org/shinyapps/BOP2/Alter2Help.pdf)
defines power as concluding safety at the lower rate. The
[toxicity-prior guide](https://biostatistics.mdanderson.org/shinyapps/BOP2/BThelp.pdf)
specifies the same null-centered calibration prior and separate analysis prior.
The guide's opening reference to an “efficacy rate” is inconsistent with its
“Prob(Toxicity)” input; here the beta prior is explicitly on the toxicity rate.

The binary BOP2 acceptability rule reverses direction:

$$
\Pr(p<p_0\mid y,n)<\lambda(n/N)^\gamma
$$

stops for excessive toxicity. This is the efficacy rule after the mathematical
substitution `q=1-p`, reversing event counts and swapping beta prior shapes.
The implementation evaluates the beta **lower tail directly at `p0`**, without
forming `1-p0` as a new efficacy threshold or subtracting a near-unit upper tail.
This preserves representable small toxicity thresholds. Equality continues;
there is no early declaration of safety. Native app optimizer parity is not
claimed; the same explicit finite-grid calibration policy applies to both endpoints.

```python
from mdanderson_stats import optimize_bop2_binary

fit = optimize_bop2_binary(
    40,
    null_rate=0.4,
    alternative_rate=0.2,
    endpoint="toxicity",
    looks=[10, 20, 30, 40],
    type1_error=0.1,
)
print(fit.calibration_design.positive_min)  # smallest unsafe toxicity count at each look
print(fit.calibration_success_probability)  # null error, power at the lower toxicity rate
print(fit.analysis_success_probability)
print(fit.calibration_oc.expected_sample_size)
```

The shared monitoring result retains its existing high-event convention:

| Output | Toxicity interpretation |
| --- | --- |
| `positive_min`, `final_positive_min` | Minimum count requiring an unsafe conclusion |
| `stop_toxicity` | Early unsafe conclusion |
| `final_positive` | Unsafe at the final analysis |
| `final_negative` | Safe at the final analysis |
| `low_probability` | Posterior probability of a toxicity rate below the null |
| `high_probability`, `final_probability` | Posterior probability above the null |
| `oc.positive_conclusion` | Probability of concluding unsafe, early or at completion |
| `oc.complete_negative` | Probability of concluding safe |

`futility_max` is always `-1` for toxicity. Both optimization-result
`*_success_probability` properties use the appropriate endpoint direction, in
`[null, alternative]` order. Use these properties for type I error and power.
The existing efficacy defaults and result interpretation are unchanged.

Toxicity validation independently checks rational beta lower tails, equality at
a stopping cutoff, a very small null rate, all 64 paths in a six-subject trial
(including expected toxicities and the sample-size distribution), and agreement
with the outcome-reversed efficacy calibration. A separate informative-prior
case verifies that analysis success probabilities can exceed calibrated error.
