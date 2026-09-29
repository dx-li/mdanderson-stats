# BOP2-DC continuous Normal endpoint

Catalog #156's primary paper, Zhao, Li, Liu and Yuan,
[arXiv:2112.10880](https://arxiv.org/abs/2112.10880), §2.1.2 specifies a Normal
outcome with unknown mean and variance and a Normal–inverse-gamma prior.
The implementation reuses the package's conjugate updater, then applies the
paper's strict dual-criterion rules. All four proper prior parameters are
explicit. Higher outcome means indicate benefit. Analysis schedules and
cutoff/exponent parameters remain caller choices.

The public surface includes posterior monitoring, absorbing complete-outcome
replay, and serial independent-Normal trial simulation with decision and
enrollment Monte Carlo errors. Full potential outcomes are generated but
decisions only use the observed prefix. The simulator returns a replayable
seed and compact per-trial summaries; it does not retain posterior histories
for every simulated trial. Work and storage are bounded before RNG use.
There is no accrual or delayed-data model. Calibration and randomized
comparisons remain separate method coverage.

## Numerical correction

Review found that constructing an absolute posterior mean before subtracting
a threshold lost meaningful precision with large measurement offsets. For
`[-.5, 0, .25, .75]`, prior `(0,.5,1.5,.2)`, LRV 0 and CMV .5, adding
`1e15` to outcomes, prior mean and thresholds changed the LRV posterior tail
from .7050523792 to .7272865016, despite exactly representable input increments.

The corrected monitor shifts observations, prior mean and thresholds to the
same origin before updating and evaluating Student-t tails. It returns the
centered posterior location and origin separately; the absolute-location
property is only their rounded sum. The shifted and original probabilities
now agree. An affine regression accompanies the correction.

The simulator also works in truth-centered coordinates: it generates residuals
and shifts the prior and thresholds, avoiding rounded absolute observations.
This matters even after the monitor fix, because variation erased during
generation cannot be recovered by subsequent centering. In a 32-trial,
20-patient comparison at seed 0, the previous implementation changed decisions
or enrollment for trials 21 and 24 solely from a `1e15` offset. The corrected
implementation agrees exactly, and this discriminating case is the regression.
The independent 64-path R comparison also still passes after the correction.

## Independent evidence

`tools/reference_bop2_dc_normal.R` computes the NIG sufficient statistics and
Student-t tails directly in base R. It does not call package routines. Its
inputs and outputs are under `tests/fixtures/bop2-dc-normal-*`.

Eight posterior cases cover negative outcomes, final go/no-go/consider,
strict equality, large offsets, and positive unit changes by `1e-100` and
`1e100`, with prior variance scale changed by the square of the unit factor.
Sixteen posterior analyses agree. The large-offset case is constructed by
arithmetic from its offset and centered data: this R runtime's decimal parser
otherwise loses fractional bits while reading some values around `1e15`.
That fixture correction concerns input parsing, not the statistical formula.

The fixed simulation tape contains 64 six-patient paths from NumPy's seed
1560931 under Normal mean .6 and SD 1.4. Base R independently replays them at
looks 2, 4 and 6, producing 154 reached analyses. Python's posterior summaries,
all reached decisions, terminal decisions, enrollment and Monte Carlo errors
agree. Decision counts are 22 early no-go, 20 final go, 19 consider and 3 final
no-go. This comparison checks identical observations, without requiring R and
NumPy random generators to agree.

`tools/check_bop2_dc_normal.py` verifies 898 numeric summaries in 0.1083 seconds
after imports, using 119.19 MiB peak resident memory and no reported swaps.
The five focused tests, including both affine corrections, passed in the worker;
the parent independently verified old-code failure and corrected-code success
for the simulation regression.
No native executable, app-prior, RNG or optimizer parity is claimed, and no
new CI workflow is added.
