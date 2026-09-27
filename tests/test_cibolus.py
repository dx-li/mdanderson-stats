import numpy as np
import pytest

from mdanderson_stats.cibolus import (
    CiBolusObservation,
    cibolus_loglikelihood,
    cibolus_predict,
    cibolus_response,
    cibolus_toxicity,
)

LOG_PARAMETERS = np.log([0.5, 0.7, 0.8, 0.08, 1.4, 1.6, 0.9, 0.12, 0.25, 0.2, 0.1])


def test_response_predictions_preserve_categories_and_failure_toxicity() -> None:
    response = cibolus_response([0, 0.25, 0.5, 1], 0.4, 0.2, LOG_PARAMETERS)
    assert response.bolus_probability > 0
    assert np.all(np.diff(response.cdf) > 0)
    toxicity = cibolus_toxicity([1], 0.4, 0.2, LOG_PARAMETERS)
    failure_toxicity = cibolus_toxicity([1], 0.4, 0.2, LOG_PARAMETERS, failure=True)
    assert failure_toxicity[0] > toxicity[0]

    prediction = cibolus_predict(
        LOG_PARAMETERS,
        [0.2, 0.4],
        [0.1, 0.2],
        [0.25, 0.5, 1.0],
        utility=np.zeros((5, 2)),
    )
    assert prediction.joint.shape == (2, 2, 5, 2)
    np.testing.assert_allclose(prediction.joint.sum(axis=(-2, -1)), 1, atol=1e-12)
    np.testing.assert_allclose(
        prediction.response_at_one, prediction.response_probability[..., :-1].sum(axis=-1)
    )


def test_observation_kinds_and_zero_bolus_response_support() -> None:
    observations = [
        CiBolusObservation(0.4, 0.2, "bolus", False),
        CiBolusObservation(0.4, 0.2, "exact", True, time=0.3),
        CiBolusObservation(0.4, 0.2, "interval", False, lower=0, upper=0.5),
        CiBolusObservation(0.4, 0.2, "failure", True),
    ]
    assert np.isfinite(cibolus_loglikelihood(observations, LOG_PARAMETERS))
    impossible_bolus = CiBolusObservation(0.4, 0.0, "bolus", False)
    assert cibolus_loglikelihood([impossible_bolus], LOG_PARAMETERS) == -np.inf
    with pytest.raises(ValueError, match="endpoint"):
        cibolus_toxicity([0.5], 0.4, 0.2, LOG_PARAMETERS, failure=True)
