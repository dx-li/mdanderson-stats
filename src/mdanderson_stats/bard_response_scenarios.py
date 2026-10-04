"""Published BARD response-generation scenarios.

These immutable records preserve scenario inputs from the paper and its
supplement. They do not decide which dose is the OBD or implement a response
model; callers can pass the recorded intercepts and factor profiles to the
shared BARD response-probability calculation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_RESPONSE_COEFFICIENTS = (1.7, -1.5, 0.4)
_FACTOR_LEVELS = (1, 2)
_FACTOR_LEVEL_PROBABILITIES = (0.5, 0.5)
_FIVE_DOSE_SOURCE = "Paper Table 3 and Supplementary Tables S1; Section 3"
_THREE_DOSE_SOURCE = "Supplementary Tables S4-S5; Section 3"


@dataclass(frozen=True, slots=True)
class BARDResponseScenario:
    """One published scenario's marginal truths and response-model inputs.

    Dose labels are ordinal levels, not physical dose amounts. Response
    probabilities in ``published_response_probabilities`` are the paper's
    printed marginal truths; ``response_intercepts`` are the separate
    covariate-conditional logistic-model inputs.
    """

    scenario_id: str
    dose_levels: tuple[int, ...]
    toxicity_probabilities: tuple[float, ...]
    published_response_probabilities: tuple[float, ...]
    response_intercepts: tuple[float, ...]
    response_coefficients: tuple[float, float, float] = _RESPONSE_COEFFICIENTS
    factor_levels: tuple[int, int] = _FACTOR_LEVELS
    factor_level_probabilities: tuple[float, float] = _FACTOR_LEVEL_PROBABILITIES
    factor_joint_distribution: str = "not specified by source"
    source: str = _FIVE_DOSE_SOURCE

    def __post_init__(self) -> None:
        if any(
            not isinstance(values, tuple)
            for values in (
                self.dose_levels,
                self.toxicity_probabilities,
                self.published_response_probabilities,
                self.response_intercepts,
                self.response_coefficients,
                self.factor_levels,
                self.factor_level_probabilities,
            )
        ):
            raise TypeError("scenario vectors must be immutable tuples")
        n_doses = len(self.dose_levels)
        if not self.scenario_id or n_doses not in (3, 5):
            raise ValueError("scenario_id is required and dose_levels must contain 3 or 5 levels")
        if any(
            len(values) != n_doses
            for values in (
                self.toxicity_probabilities,
                self.published_response_probabilities,
                self.response_intercepts,
            )
        ):
            raise ValueError("all dose-specific scenario values must match dose_levels")
        if self.dose_levels != tuple(range(1, n_doses + 1)):
            raise ValueError("dose_levels must be consecutive ordinal levels starting at 1")
        if self.factor_levels != (1, 2) or self.factor_level_probabilities != (0.5, 0.5):
            raise ValueError("source factor coding is levels 1/2 with equal marginal probabilities")
        if len(self.response_coefficients) != 3 or len(self.source) == 0:
            raise ValueError("three response coefficients and a source label are required")
        numeric = (
            *self.toxicity_probabilities,
            *self.published_response_probabilities,
            *self.response_intercepts,
            *self.response_coefficients,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not float("-inf") < value < float("inf")
            for value in numeric
        ):
            raise ValueError("scenario values must be finite real numbers")
        if any(
            not 0.0 <= value <= 1.0
            for value in (*self.toxicity_probabilities, *self.published_response_probabilities)
        ):
            raise ValueError("published toxicity and response probabilities must lie in [0, 1]")


_SCENARIOS: Mapping[str, BARDResponseScenario] = MappingProxyType(
    {
        "five-dose-1": BARDResponseScenario(
            "five-dose-1",
            (1, 2, 3, 4, 5),
            (0.12, 0.25, 0.42, 0.49, 0.55),
            (0.181, 0.349, 0.439, 0.519, 0.596),
            (-2.197, -1.099, -0.619, -0.201, 0.201),
            source=_FIVE_DOSE_SOURCE,
        ),
        "five-dose-2": BARDResponseScenario(
            "five-dose-2",
            (1, 2, 3, 4, 5),
            (0.04, 0.12, 0.25, 0.43, 0.63),
            (0.152, 0.181, 0.349, 0.439, 0.519),
            (-2.442, -2.197, -1.099, -0.619, -0.201),
            source=_FIVE_DOSE_SOURCE,
        ),
        "five-dose-3": BARDResponseScenario(
            "five-dose-3",
            (1, 2, 3, 4, 5),
            (0.02, 0.06, 0.10, 0.25, 0.40),
            (0.103, 0.152, 0.181, 0.349, 0.439),
            (-2.944, -2.442, -2.197, -1.099, -0.619),
            source=_FIVE_DOSE_SOURCE,
        ),
        "five-dose-4": BARDResponseScenario(
            "five-dose-4",
            (1, 2, 3, 4, 5),
            (0.02, 0.05, 0.08, 0.11, 0.25),
            (0.046, 0.103, 0.152, 0.181, 0.349),
            (-3.892, -2.944, -2.442, -2.197, -1.099),
            source=_FIVE_DOSE_SOURCE,
        ),
        "five-dose-5": BARDResponseScenario(
            "five-dose-5",
            (1, 2, 3, 4, 5),
            (0.12, 0.25, 0.42, 0.49, 0.55),
            (0.349, 0.349, 0.359, 0.359, 0.359),
            (-1.099, -1.099, -1.046, -1.046, -1.046),
            source=_FIVE_DOSE_SOURCE,
        ),
        "five-dose-6": BARDResponseScenario(
            "five-dose-6",
            (1, 2, 3, 4, 5),
            (0.04, 0.12, 0.25, 0.43, 0.63),
            (0.181, 0.349, 0.349, 0.359, 0.359),
            (-2.197, -1.099, -1.099, -1.046, -1.046),
            source=_FIVE_DOSE_SOURCE,
        ),
        "five-dose-7": BARDResponseScenario(
            "five-dose-7",
            (1, 2, 3, 4, 5),
            (0.02, 0.06, 0.10, 0.25, 0.40),
            (0.152, 0.181, 0.349, 0.349, 0.359),
            (-2.442, -2.197, -1.099, -1.099, -1.046),
            source=_FIVE_DOSE_SOURCE,
        ),
        "five-dose-8": BARDResponseScenario(
            "five-dose-8",
            (1, 2, 3, 4, 5),
            (0.02, 0.05, 0.08, 0.11, 0.25),
            (0.103, 0.152, 0.181, 0.349, 0.349),
            (-2.944, -2.442, -2.197, -1.099, -1.099),
            source=_FIVE_DOSE_SOURCE,
        ),
        "three-dose-1": BARDResponseScenario(
            "three-dose-1",
            (1, 2, 3),
            (0.12, 0.25, 0.40),
            (0.181, 0.349, 0.439),
            (-2.197, -1.099, -0.619),
            source=_THREE_DOSE_SOURCE,
        ),
        "three-dose-2": BARDResponseScenario(
            "three-dose-2",
            (1, 2, 3),
            (0.04, 0.12, 0.25),
            (0.152, 0.181, 0.349),
            (-2.442, -2.197, -1.099),
            source=_THREE_DOSE_SOURCE,
        ),
        "three-dose-3": BARDResponseScenario(
            "three-dose-3",
            (1, 2, 3),
            (0.12, 0.25, 0.42),
            (0.349, 0.349, 0.359),
            (-1.099, -1.099, -1.046),
            source=_THREE_DOSE_SOURCE,
        ),
        "three-dose-4": BARDResponseScenario(
            "three-dose-4",
            (1, 2, 3),
            (0.04, 0.12, 0.25),
            (0.181, 0.349, 0.349),
            (-2.197, -1.099, -1.099),
            source=_THREE_DOSE_SOURCE,
        ),
    }
)


def bard_response_scenario(scenario_id: str) -> BARDResponseScenario:
    """Return one of the twelve response scenarios printed in the BARD paper."""
    if not isinstance(scenario_id, str):
        raise TypeError("scenario_id must be a string")
    try:
        return _SCENARIOS[scenario_id]
    except KeyError as exc:
        raise ValueError(f"unknown BARD response scenario {scenario_id!r}") from exc
