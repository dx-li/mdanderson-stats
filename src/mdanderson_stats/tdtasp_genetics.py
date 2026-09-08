"""TDTASP two-locus genetic and offspring probabilities."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite, scalar

FloatArray = NDArray[np.float64]


def _freeze(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _probability(value: float, name: str) -> float:
    value = scalar(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def tdtasp_haplotype_frequencies(
    marker_frequency: float, disease_frequency: float, relative_disequilibrium: float
) -> FloatArray:
    """Return AD, Ad, BD, Bd probabilities from marker A, disease D and D prime.

    D prime is in [-1, 1]. Positive values associate A with D, as required by
    the original planning interface; negative values are allowed here for
    genetic-model exploration. Boundary allele frequencies imply zero linkage
    disequilibrium regardless of D prime.
    """
    m = _probability(marker_frequency, "marker_frequency")
    p = _probability(disease_frequency, "disease_frequency")
    relative = scalar(relative_disequilibrium, "relative_disequilibrium")
    if not -1 <= relative <= 1:
        raise ValueError("relative_disequilibrium must lie in [-1, 1]")
    lower = -min(m * p, (1 - m) * (1 - p))
    upper = min(m * (1 - p), (1 - m) * p)
    delta = relative * (upper if relative >= 0 else -lower)
    return _freeze(
        [m * p + delta, m * (1 - p) - delta, (1 - m) * p - delta, (1 - m) * (1 - p) + delta]
    )


@dataclass(frozen=True)
class TDTASPGenetics:
    """All 256 ordered parental families before ascertainment.

    ``parents`` columns are father's two and mother's two haplotypes, in
    lexicographic order, with codes 0=Bd, 1=BD, 2=Ad, 3=AD. Probability arrays
    have one row per family. Transmission and sharing probabilities average
    over marker-heterozygous parents. They are zero when no parent is
    informative or the family cannot have an affected child.
    """

    haplotype_frequencies: FloatArray
    penetrance: FloatArray
    recombination: float
    legacy_asp: bool
    parents: NDArray[np.int64]
    parent_probability: FloatArray
    father_heterozygous: NDArray[np.bool_]
    mother_heterozygous: NDArray[np.bool_]
    affected_probability: FloatArray
    transmission_probability: FloatArray
    sharing_probability: FloatArray
    population_affected_probability: float


def tdtasp_genetics(
    haplotype_frequencies: ArrayLike,
    penetrance: ArrayLike,
    recombination: float = 0,
    *,
    legacy_asp: bool = False,
) -> TDTASPGenetics:
    """Compute parental frequencies and affected-child TDT/ASP probabilities.

    Haplotype frequencies use AD, Ad, BD, Bd order and sum to one within 1e-12
    (then normalized). Penetrances use DD, Dd, dd order, each in [0, 1].
    Recombination is in [0, 0.5]. Parent haplotypes are sampled independently
    from the population, matching the source's random-mating model.

    The default ASP result conditions two independent siblings on both being
    affected. ``legacy_asp=True`` reproduces the source's extra multiplication
    by offspring transmission probabilities and its 1e-100 pair cutoff.
    Underflow that makes that legacy calculation undefined raises an error.
    This is the genetic layer; ascertainment and study power are separate.
    """
    frequencies = finite(haplotype_frequencies, "haplotype_frequencies")
    if frequencies.shape != (4,) or np.any((frequencies < 0) | (frequencies > 1)):
        raise ValueError(
            "haplotype_frequencies must contain four probabilities in AD, Ad, BD, Bd order"
        )
    if abs(float(frequencies.sum()) - 1) > 1e-12:
        raise ValueError("haplotype_frequencies must sum to one within 1e-12")
    frequencies = frequencies / frequencies.sum()
    penetrances = finite(penetrance, "penetrance")
    if penetrances.shape != (3,) or np.any((penetrances < 0) | (penetrances > 1)):
        raise ValueError("penetrance must contain three probabilities in DD, Dd, dd order")
    theta = scalar(recombination, "recombination")
    if not 0 <= theta <= 0.5:
        raise ValueError("recombination must lie in [0, 0.5]")
    if not isinstance(legacy_asp, (bool, np.bool_)):
        raise ValueError("legacy_asp must be boolean")
    parents = np.array(list(product(range(4), repeat=4)), dtype=np.int64)
    parent_probability = np.prod(frequencies[::-1][parents], axis=1)
    heterozygous = parents[:, [0, 2]] // 2 != parents[:, [1, 3]] // 2
    informative = heterozygous.sum(axis=1)
    # Keep the four transmission paths distinct, including duplicate gametes,
    # because the historical ASP weighting operates on paths rather than types.
    first = parents[:, [0, 2]]
    second = parents[:, [1, 3]]
    gametes = np.stack(
        (first, second, (first & 2) | (second & 1), (second & 2) | (first & 1)), axis=-1
    )
    weights = np.array([1 - theta, 1 - theta, theta, theta]) / 2
    offspring_weights = np.outer(weights, weights).reshape(16)
    father = np.repeat(gametes[:, 0], 4, axis=1)
    mother = np.tile(gametes[:, 1], (1, 4))
    affected = penetrances[::-1][(father & 1) + (mother & 1)] * offspring_weights
    affected_probability = affected.sum(axis=1)
    conditional = np.divide(
        affected,
        affected_probability[:, None],
        out=np.zeros_like(affected),
        where=affected_probability[:, None] > 0,
    )
    paternal_a = father > 1
    maternal_a = mother > 1
    qa = np.stack(
        ((conditional * paternal_a).sum(axis=1), (conditional * maternal_a).sum(axis=1)), axis=1
    )
    transmission = np.divide(
        (qa * heterozygous).sum(axis=1), informative, out=np.zeros(256), where=informative > 0
    )
    if legacy_asp:
        # Original calc_exp_p_same weights each pair twice and omits tiny terms
        # from the numerator only. This option preserves that defined behavior.
        weighted = affected * offspring_weights
        pair = weighted[:, :, None] * weighted[:, None, :]
        denominator = pair.sum(axis=(1, 2))
        if np.any((affected_probability > 0) & (denominator == 0)):
            raise ArithmeticError(
                "legacy ASP pair probabilities underflow; use corrected calculation"
            )
        same = (
            (paternal_a[:, :, None] == paternal_a[:, None, :]) * heterozygous[:, 0, None, None]
        ).astype(float)
        same += (maternal_a[:, :, None] == maternal_a[:, None, :]) * heterozygous[:, 1, None, None]
        numerator = np.where(pair >= 1e-100, pair * same, 0).sum(axis=(1, 2))
        sharing = np.divide(
            numerator,
            denominator * informative,
            out=np.zeros(256),
            where=(denominator > 0) & (informative > 0),
        )
    else:
        sharing = np.divide(
            ((qa**2 + (1 - qa) ** 2) * heterozygous).sum(axis=1),
            informative,
            out=np.zeros(256),
            where=informative > 0,
        )
        sharing[affected_probability == 0] = 0
    parents = np.frombuffer(parents.tobytes(), dtype=np.int64).reshape(256, 4)
    flags = np.frombuffer(heterozygous.tobytes(), dtype=np.bool_).reshape(256, 2)
    return TDTASPGenetics(
        _freeze(frequencies),
        _freeze(penetrances),
        theta,
        bool(legacy_asp),
        parents,
        _freeze(parent_probability),
        flags[:, 0],
        flags[:, 1],
        _freeze(affected_probability),
        _freeze(transmission),
        _freeze(sharing),
        float(parent_probability @ affected_probability),
    )
