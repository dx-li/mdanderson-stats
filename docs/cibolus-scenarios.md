# CiBolus scenario truth

`cibolus_interpolated_truth` constructs the categorical response/toxicity truth
used by the CiBolus calibration and simulation workflows from elicited
probabilities at standardized times zero and one. It is a transparent scenario
builder: it does not fit parameters or infer missing probabilities.

The response inputs are cumulative response probabilities including the
instantaneous bolus response at time zero. `response_zero` is therefore the
bolus atom, while `response_one` is total response probability by the infusion
endpoint. The remaining response mass is assigned to failure. Response inputs
must satisfy `response_one >= response_zero` regimen by regimen. The toxicity
inputs are conditional probabilities for the bolus response, response at time
one, and failure, respectively; failure toxicity must be supplied explicitly.
Each input can be a scalar or an array broadcastable to the concentration by
bolus grid.

Both response and conditional-toxicity probabilities use one of the profiles
described in Section 5 of the CiBolus paper. For endpoint probabilities `p0`
and `p1`, a profile is `p(s) = p0 + (p1-p0) g(s)`, where `g(s)` is `s` for
`linear`, `s²` for `below_linear`, and `sqrt(s)` for `above_linear`. The
`s_shaped` profile uses `g(s)=2s²` up to one half and
`g(s)=1/2 + sqrt(2s-1)/2` thereafter. These profiles are applied separately
to response and toxicity using `response_curve` and `toxicity_curve`.

The returned shape is `(n_concentrations, n_bolus_fractions, n_categories, 2)`.
The category order is the bolus atom, one interval category for each supplied
endpoint, then failure. Endpoints partition `(0,1]`, and the last endpoint must
be one. For an interval ending at `s`, Eq. 8 of the paper assigns its response
mass `F(s)-F(s_previous)` and conditions toxicity at that interval's right
endpoint. The final failure category has mass `1-response_one` and uses
`toxicity_failure`. The final axis is no toxicity then toxicity. The returned
array is owned and read-only; it can be passed to
`calibrate_cibolus_prior` as `joint_probabilities`.

```python
import numpy as np

from mdanderson_stats import cibolus_interpolated_truth

joint = cibolus_interpolated_truth(
    concentrations=[0.3, 0.5],
    bolus_fractions=[0.1],
    endpoints=[0.5, 1.0],
    response_zero=0.08,
    response_one=0.55,
    toxicity_zero=0.03,
    toxicity_one=0.12,
    toxicity_failure=0.20,
    response_curve="below_linear",
    toxicity_curve="linear",
)
utility = np.array([[0.2, -0.5], [0.7, -0.2], [1.0, -0.3], [0.0, -0.4]])
expected_utility = np.einsum("cqkt,kt->cq", joint, utility)
assert np.allclose(joint.sum(axis=(-2, -1)), 1.0)
```

The paper does not define these interpolation profiles as fitted response or
toxicity models. Choosing among profiles and supplying endpoint values remain
scenario assumptions made by the caller. The implementation uses stable
profile increments when constructing interval masses and validates that each
regimen's joint cells form a probability distribution.

Source: Thall et al., “Optimizing the Concentration and Bolus of a Drug
Delivered by Continuous Infusion,” *Biometrics* 67 (2011), Section 5 and Eqs.
7–8; cached source: `research/raw/CiBolus/paper.pdf` and `paper.txt`.
