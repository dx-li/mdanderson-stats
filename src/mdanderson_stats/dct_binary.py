"""Binary decentralized-trial planning via the supplement's normal approximation."""

import numpy as np

from ._validation import scalar
from .dct_normal import DCTNormalSampleSize, dct_normal_sample_size


def dct_binary_sample_size(
    onsite_control: float,
    onsite_experimental: float,
    offsite_control: float,
    offsite_experimental: float,
    *,
    offsite_fraction: float = 0.75,
    randomization_ratio: float = 1,
    onsite_repeats: int = 1,
    offsite_repeats: int = 1,
    onsite_correlation: float = 0,
    offsite_correlation: float = 0,
    power: float = 0.8,
    alpha: float = 0.05,
    sides: int = 2,
) -> DCTNormalSampleSize:
    """Plan a weighted difference-in-proportions z-test, supplement E.1/E.2.

    All response probabilities must be interior, with experimental > control in
    each stratum. Variances are p*(1-p), with the exchangeable design effect for
    repeated/clustered responses. This is an asymptotic normal approximation,
    not an exact binomial calculation or a pooled-null-variance test.
    """
    p = np.array(
        [
            [
                scalar(onsite_control, "onsite_control"),
                scalar(onsite_experimental, "onsite_experimental"),
            ],
            [
                scalar(offsite_control, "offsite_control"),
                scalar(offsite_experimental, "offsite_experimental"),
            ],
        ]
    )
    differences = p[:, 1] - p[:, 0]
    if np.any((p <= 0) | (p >= 1)) or np.any(differences <= 0):
        raise ValueError("require 0 < control < experimental < 1 in both strata")
    sd = np.sqrt(p) * np.sqrt(1 - p)
    return dct_normal_sample_size(
        float(differences[0]),
        float(sd[0, 0]),
        float(sd[1, 0]),
        onsite_experimental_sd=float(sd[0, 1]),
        offsite_experimental_sd=float(sd[1, 1]),
        relative_bias=float(differences[1] / differences[0] - 1),
        offsite_fraction=offsite_fraction,
        randomization_ratio=randomization_ratio,
        onsite_repeats=onsite_repeats,
        offsite_repeats=offsite_repeats,
        onsite_correlation=onsite_correlation,
        offsite_correlation=offsite_correlation,
        power=power,
        alpha=alpha,
        sides=sides,
    )
