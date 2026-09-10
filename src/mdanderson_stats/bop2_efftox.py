"""Joint efficacy/toxicity BOP2 monitoring, including separate assessment schedules."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .bayesian_monitoring import _owned
from .bop2_binary import _boundaries, _candidate, bop2_binary_design
from .bop2_paired import BOP2PairedDesign, _cells


def bop2_efftox_design(
    max_subjects: int,
    null_rates: ArrayLike,
    *,
    cutoff_scales: ArrayLike,
    gamma: float,
    null_joint_rate: float | None = None,
    efficacy_looks: ArrayLike | None = None,
    toxicity_looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    toxicity_exponent_factor: float = 1 / 3,
    equality_continues: bool = False,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2PairedDesign:
    """Stop on either futility or toxicity; both criteria must pass at completion.

    Category order is (efficacy,toxicity): 11,10,01,00. Default prior has ESS one,
    centered on the null (independent unless null_joint_rate is supplied).
    Defaults use BOP2-TE's strict go inequality and toxicity exponent gamma/3.
    """
    rates = finite(null_rates, "null_rates")
    if rates.shape != (2,) or np.any((rates <= 0) | (rates >= 1)):
        raise ValueError("null_rates requires efficacy and toxicity thresholds in (0,1)")
    cells = _cells(
        rates, float(np.prod(rates)) if null_joint_rate is None else null_joint_rate, "multiple"
    )
    shapes = cells if prior is None else finite(prior, "prior")
    if shapes.shape != (4,) or np.any(shapes <= 0) or not np.isfinite(shapes.sum()):
        raise ValueError("prior requires four positive finite shapes with a finite sum")
    scales = finite(cutoff_scales, "cutoff_scales")
    if scales.shape != (2,) or np.any((scales <= 0) | (scales >= 1)):
        raise ValueError("cutoff_scales requires efficacy and toxicity scales in (0,1)")
    factor = scalar(toxicity_exponent_factor, "toxicity_exponent_factor")
    exponent = scalar(gamma, "gamma")
    if not 0 <= factor <= 1 or not 0 <= exponent <= 1:
        raise ValueError("gamma and toxicity_exponent_factor must lie in [0,1]")
    if not isinstance(equality_continues, (bool, np.bool_)):
        raise ValueError("equality_continues must be boolean")
    if (
        isinstance(max_subjects, (bool, np.bool_))
        or not isinstance(max_subjects, (int, np.integer))
        or not 1 <= max_subjects <= 200
    ):
        raise ValueError("max_subjects must be an integer in [1,200]")
    masks = np.array([[True, True], [True, False], [False, True], [False, False]])
    margins = []
    for j, (endpoint, looks, power) in enumerate(
        [("efficacy", efficacy_looks, exponent), ("toxicity", toxicity_looks, exponent * factor)]
    ):
        baseline = bop2_binary_design(
            max_subjects,
            float(rates[j]),
            cutoff_scale=float(scales[j]),
            gamma=power,
            endpoint=endpoint,
            looks=looks,
            prior=[shapes[masks[:, j]].sum(), shapes[~masks[:, j]].sum()],
            min_subjects=min_subjects,
            cohort_size=cohort_size,
        )
        margins.append(
            _candidate(
                baseline,
                _boundaries(
                    baseline, float(scales[j]), power, equality_continues=equality_continues
                ),
            )
        )
    return BOP2PairedDesign("efficacy_toxicity", _owned(shapes), (margins[0], margins[1]))
