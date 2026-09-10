"""Expected-enrollment and minimax searches for complex categorical BOP2 endpoints."""

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ._bop2_sample_size import SampleSizeResult, interim_schedule, looks_at, search_sizes
from .bop2_efftox_optimization import BOP2EffToxOptimization, optimize_bop2_efftox
from .bop2_paired_optimization import BOP2PairedOptimization, optimize_bop2_paired


@dataclass(frozen=True)
class BOP2PairedSampleSizeOptimization(SampleSizeResult[BOP2PairedOptimization]):
    """Ordinal/multiple efficacy sample-size search results."""


@dataclass(frozen=True)
class BOP2EffToxSampleSizeOptimization(SampleSizeResult[BOP2EffToxOptimization]):
    """Joint efficacy/toxicity sample-size search results."""


def optimize_bop2_paired_sample_size(
    sample_sizes: ArrayLike,
    null_rates: ArrayLike,
    alternative_rates: ArrayLike,
    *,
    minimum_power: float,
    type1_error: float = 0.1,
    endpoint: str = "ordinal",
    objective: str = "expected_sample_size",
    null_joint_rate: float | None = None,
    alternative_joint_rate: float | None = None,
    interim_looks: ArrayLike | None = None,
    cutoff_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2PairedSampleSizeOptimization:
    """Search ordinal/multiple efficacy sizes with strict point-null error and power limits.

    Minimize null expected enrollment, or maximum sample size ('minimax'). Within
    N, minimize null expected enrollment then maximize power. Correlation is retained
    using the supplied null/alternative joint probabilities for multiple efficacy.
    """
    schedule = interim_schedule(interim_looks)

    def optimize(n: int, floor: float) -> BOP2PairedOptimization:
        return optimize_bop2_paired(
            n,
            null_rates,
            alternative_rates,
            endpoint=endpoint,
            null_joint_rate=null_joint_rate,
            alternative_joint_rate=alternative_joint_rate,
            type1_error=type1_error,
            looks=looks_at(schedule, n),
            cutoff_scales=cutoff_scales,
            gammas=gammas,
            analysis_prior=analysis_prior,
            objective="expected_sample_size",
            minimum_power=floor,
            min_subjects=min_subjects,
            cohort_size=cohort_size,
        )

    return BOP2PairedSampleSizeOptimization(
        objective, *search_sizes(sample_sizes, minimum_power, objective, optimize)
    )


def optimize_bop2_efftox_sample_size(
    sample_sizes: ArrayLike,
    null_rates: ArrayLike,
    alternative_rates: ArrayLike,
    *,
    minimum_power: float,
    type1_error: ArrayLike = (0.05, 0.1, 0.1),
    objective: str = "expected_sample_size",
    joint_rates: ArrayLike | None = None,
    efficacy_interim_looks: ArrayLike | None = None,
    toxicity_interim_looks: ArrayLike | None = None,
    efficacy_scales: ArrayLike | None = None,
    toxicity_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    toxicity_exponent_factor: float = 1 / 3,
    equality_continues: bool = False,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2EffToxSampleSizeOptimization:
    """Search joint endpoint sizes with all three null limits and a power floor.

    Expected enrollment is evaluated under H00 (futile and toxic). Independent or
    explicitly correlated scenarios and distinct assessment schedules are supported.
    """
    efficacy = interim_schedule(efficacy_interim_looks)
    toxicity = interim_schedule(toxicity_interim_looks)

    def optimize(n: int, floor: float) -> BOP2EffToxOptimization:
        return optimize_bop2_efftox(
            n,
            null_rates,
            alternative_rates,
            type1_error=type1_error,
            joint_rates=joint_rates,
            efficacy_looks=looks_at(efficacy, n),
            toxicity_looks=looks_at(toxicity, n),
            efficacy_scales=efficacy_scales,
            toxicity_scales=toxicity_scales,
            gammas=gammas,
            analysis_prior=analysis_prior,
            objective="expected_sample_size",
            minimum_power=floor,
            toxicity_exponent_factor=toxicity_exponent_factor,
            equality_continues=equality_continues,
            min_subjects=min_subjects,
            cohort_size=cohort_size,
        )

    return BOP2EffToxSampleSizeOptimization(
        objective, *search_sizes(sample_sizes, minimum_power, objective, optimize)
    )
