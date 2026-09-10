import numpy as np
import pytest

from mdanderson_stats import confint_survival_solve


def test_confint_survival_inverse_independent_r_roots():
    references = {
        "hazard": [
            0.58823649265708855,
            9.52621161601316224,
            16.4673455370656967,
            0.62988237208446096,
        ],
        "mean": [
            0.60455419609370864,
            9.68985628651595476,
            16.85328656888810883,
            1.91927425730301726,
        ],
    }
    for target, expected in references.items():
        for parameter, bounds, assurance, answer in zip(
            ["max_length", "accrual_rate", "accrual_time", "followup_time"],
            [(0.1, 2), (1, 20), (10, 30), (0, 10)],
            [0.5, 0.9, 0.8, 0.8],
            expected,
            strict=True,
        ):
            values = dict(
                hazard=1,
                accrual_rate=5,
                accrual_time=16 if parameter == "followup_time" else 10,
                followup_time=0,
                max_length=0.5,
            )
            values[parameter] = None
            result = confint_survival_solve(
                **values, bounds=bounds, assurance=assurance, target=target
            )
            assert result.parameter == parameter
            np.testing.assert_allclose(result.value, answer, rtol=2e-9)
            np.testing.assert_allclose(result.achieved.probability, assurance, atol=1e-10)
            assert result.achieved.omitted_probability <= 1e-14


def test_confint_survival_hazard_branches_and_unattainable_quantile():
    kwargs = dict(accrual_rate=5, accrual_time=10, followup_time=0, max_length=0.5, assurance=0.5)
    low = confint_survival_solve(**kwargs, bounds=(0.00001, 0.01))
    high = confint_survival_solve(**kwargs, bounds=(0.5, 1.5))
    np.testing.assert_allclose(
        [low.value, high.value], [0.0028583338690436641, 0.8409063518812448512], rtol=2e-9
    )
    with pytest.raises(ValueError, match="bracket"):
        confint_survival_solve(**kwargs, bounds=(0.00001, 10))
    # Too few expected events: the chance of any defined interval is below .9.
    with pytest.raises(ValueError, match="bracket"):
        confint_survival_solve(
            hazard=1, accrual_rate=0.01, accrual_time=1, followup_time=0, assurance=0.9
        )


def test_confint_survival_inverse_time_unit_scaling():
    for target, reference in [("hazard", 0.58823649265708855), ("mean", 0.60455419609370864)]:
        for scale in [1e-200, 1e200]:
            length_scale = 1 / scale if target == "hazard" else scale
            result = confint_survival_solve(
                hazard=1 / scale,
                accrual_rate=5 / scale,
                accrual_time=10 * scale,
                followup_time=0,
                target=target,
                assurance=0.5,
                bounds=(0.1 * length_scale, 2 * length_scale),
            )
            np.testing.assert_allclose(result.value / length_scale, reference, rtol=2e-9)
