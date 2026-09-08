"""Family/individual ascertainment for TDTASP genetic models."""

from dataclasses import dataclass
from math import lgamma
from operator import index

import numpy as np
from scipy.special import logsumexp

from ._validation import scalar
from .tdtasp_genetics import FloatArray, TDTASPGenetics, _freeze


def _poisson_series(mu: FloatArray, k: int) -> FloatArray:
    """Tail divided by its leading Poisson probability; mu is at most ten."""
    term = np.ones_like(mu)
    total = term.copy()
    for j in range(1, 129):
        term *= mu / (k + j)
        total += term
        if np.all(term <= np.finfo(float).eps * total):
            return total
    raise ArithmeticError("Poisson tail series did not converge")


def _log_tail(mu: FloatArray, k: int) -> FloatArray:
    if k == 0:
        return np.zeros_like(mu)
    with np.errstate(divide="ignore"):
        return -mu + k * np.log(mu) - lgamma(k + 1) + np.log(_poisson_series(mu, k))


@dataclass(frozen=True)
class TDTASPAscertainment:
    """Selected family distribution and contribution-weighted test probability.

    Array rows follow ``genetics.parents``. ``selection_probability`` is the
    parental eligibility rate within the chosen affected-family/individual list.
    Log masses retain extremely rare selection events without underflow.
    """

    genetics: TDTASPGenetics
    mean_offspring: float
    test: str
    sampling: str
    eligibility: str
    minimum_affected: int
    all_affected: bool
    legacy_moments: bool
    family_probability: FloatArray
    mean_affected_by_family: FloatArray
    contributions_by_family: FloatArray
    selection_probability: float
    expected_affected: float
    population_average_truncated_mean: float
    expected_heterozygous_parents: float
    expected_contributions: float
    answer_probability: float
    log_list_mass: float
    log_eligible_mass: float
    log_screening_for_minimum: float


def tdtasp_ascertainment(
    genetics: TDTASPGenetics,
    mean_offspring: float,
    *,
    test: str = "tdt",
    sampling: str = "family",
    eligibility: str = "one",
    minimum_affected: int | None = None,
    all_affected: bool = False,
    legacy_moments: bool = False,
) -> TDTASPAscertainment:
    """Select eligible families assuming Poisson numbers of offspring.

    ``sampling`` is 'family' or 'individual'; the latter samples affected
    offspring and size-biases family selection. ``eligibility`` is 'father',
    'one' (at least one parent), or 'both' marker-heterozygous parents. All
    informative parents contribute, including under father-based eligibility.

    Mean offspring is in [0.1, 10]. TDT permits minimum affected from 1 through
    10 (default one), either one or all affected children. ASP requires two
    affected siblings and ``all_affected=False``. ``legacy_moments`` retains
    the source's non-size-biased within-family mean under individual sampling.
    The genetic model independently chooses corrected or legacy ASP sharing.
    """
    if not isinstance(genetics, TDTASPGenetics):
        raise TypeError("genetics must be a TDTASPGenetics")
    mean = scalar(mean_offspring, "mean_offspring")
    if not 0.1 <= mean <= 10:
        raise ValueError("mean_offspring must lie in [0.1, 10]")
    if test not in ("tdt", "asp") or sampling not in ("family", "individual"):
        raise ValueError("test must be 'tdt'/'asp' and sampling must be 'family'/'individual'")
    if eligibility not in ("father", "one", "both"):
        raise ValueError("eligibility must be 'father', 'one', or 'both'")
    if not isinstance(all_affected, (bool, np.bool_)) or not isinstance(
        legacy_moments, (bool, np.bool_)
    ):
        raise ValueError("all_affected and legacy_moments must be boolean")
    if minimum_affected is None:
        k = 2 if test == "asp" else 1
    else:
        try:
            k = index(minimum_affected)
        except TypeError as exc:
            raise ValueError("minimum_affected must be an integer in [1, 10]") from exc
        if isinstance(minimum_affected, (bool, np.bool_)) or not 1 <= k <= 10:
            raise ValueError("minimum_affected must be an integer in [1, 10]")
    if test == "asp" and (k != 2 or all_affected):
        raise ValueError("ASP requires minimum_affected=2 and all_affected=False")
    mu = mean * genetics.affected_probability
    if np.any((genetics.affected_probability > 0) & (mu == 0)):
        raise ArithmeticError("affected-offspring means underflow")
    with np.errstate(divide="ignore"):
        log_population = np.log(genetics.haplotype_frequencies[::-1][genetics.parents]).sum(axis=1)
        log_mu = np.log(mu)
    truncated_mean = k * _poisson_series(mu, k - 1) / _poisson_series(mu, k)
    if sampling == "family":
        log_weights = log_population + _log_tail(mu, k)
        log_base = log_population + _log_tail(mu, 1)
        conditional_mean = truncated_mean
    else:
        log_weights = log_population + log_mu + _log_tail(mu, k - 1)
        log_base = log_population + log_mu
        if legacy_moments:
            conditional_mean = truncated_mean
        elif k == 1:
            conditional_mean = 1 + mu
        else:
            conditional_mean = 1 + (k - 1) * _poisson_series(mu, k - 2) / _poisson_series(mu, k - 1)
    conditional_mean = np.where(mu > 0, conditional_mean, 0)
    father, mother = genetics.father_heterozygous, genetics.mother_heterozygous
    eligible = {"father": father, "one": father | mother, "both": father & mother}[eligibility]
    log_list = float(logsumexp(log_weights))
    log_eligible = float(logsumexp(log_weights[eligible]))
    if not np.isfinite(log_list) or not np.isfinite(log_eligible):
        raise ValueError("no eligible families with affected offspring exist under this model")
    selected = np.exp(np.where(eligible, log_weights, -np.inf) - log_eligible)
    selected /= selected.sum()
    heterozygous = father.astype(float) + mother
    contributions = (
        heterozygous * conditional_mean if test == "tdt" and all_affected else heterozygous
    )
    expected_contributions = float(selected @ contributions)
    answer = genetics.transmission_probability if test == "tdt" else genetics.sharing_probability
    return TDTASPAscertainment(
        genetics,
        mean,
        test,
        sampling,
        eligibility,
        k,
        bool(all_affected),
        bool(legacy_moments),
        _freeze(selected),
        _freeze(conditional_mean),
        _freeze(contributions),
        float(np.exp(log_eligible - log_list)),
        float(selected @ conditional_mean),
        float(genetics.parent_probability @ np.where(mu > 0, truncated_mean, 0)),
        float(selected @ heterozygous),
        expected_contributions,
        float((selected * contributions) @ answer / expected_contributions),
        log_list,
        log_eligible,
        float(logsumexp(log_base) - log_list),
    )
