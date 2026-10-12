"""Recovered desktop combination decision conventions over independent BOIN math."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from .boin_combination import (
    BOINCombDecision,
    BOINCombDesign,
    _validate_counts,
    _validate_dose,
)


@dataclass(frozen=True)
class BOINDesktopCombDesign(BOINCombDesign):
    """Desktop interactive one-of-three rule and convergence-aware stopping.

    The run-in flag is offered by the original GUI only at target .25 and
    cohort size three. The study adapter applies both conditions. This class
    requires target .25; aggregate counts alone do not encode cohort size.
    """

    run_in: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.run_in, (bool, np.bool_)):
            raise ValueError("run_in must be boolean")
        if self.run_in:
            if self.target != 0.25:
                raise ValueError("desktop run_in requires target .25")
            object.__setattr__(self, "_base", replace(self._base, stay_at_one_of_three=True))

    def desktop_next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: tuple[int, int],
        *,
        rng: int | np.random.Generator | None = None,
    ) -> BOINCombDecision:
        """Analyze completed counts using the active desktop single-MTD convention.

        Unlike the older public-R interactive routine, the desktop stops for
        precision only when its current dose has converged: neither escalation
        nor de-escalation is indicated/possible. It omits the old empirical
        higher-dose blockers. Mathematical safety and explicit random ties use
        the existing Python core; native random-stream identity is not claimed.
        """
        n, y = _validate_counts(patients, toxicities)
        dose = _validate_dose(current_dose, n.shape)
        i, j = dose[0] - 1, dose[1] - 1
        if n[i, j] < 1:
            raise ValueError("current dose must contain completed outcomes")
        state, posterior = self._state(n, y, None)
        if state[0, 0]:
            return BOINCombDecision("stop_safety", None, state, posterior)
        table = self.boundary_table(int(n[i, j]))
        index = int(n[i, j]) - 1
        neighbors = [
            (a, b) for a, b in ((i + 1, j), (i, j + 1)) if a < n.shape[0] and b < n.shape[1]
        ]
        cannot_escalate = not any(not state[a, b] for a, b in neighbors)
        wants_no_escalation = y[i, j] > table.escalate_max[index] or cannot_escalate
        wants_no_deescalation = y[i, j] < table.deescalate_min[index] or dose == (1, 1)
        if (
            self.early_stop_patients is not None
            and n[i, j] >= self.early_stop_patients
            and wants_no_escalation
            and wants_no_deescalation
        ):
            return BOINCombDecision("stop_precision", None, state, posterior)
        return super().next_dose(n, y, dose, source_simulation=True, rng=rng)
