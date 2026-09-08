"""Reproducible TDTASP study orchestration and plain-text reports."""

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike

from .tdtasp_ascertainment import tdtasp_ascertainment
from .tdtasp_genetics import tdtasp_genetics
from .tdtasp_power import TDTASPFixedPower, TDTASPPower, tdtasp_power
from .tdtasp_sample_size import (
    TDTASPSampleSize,
    tdtasp_fixed_sample_size,
    tdtasp_sample_size,
)


@dataclass(frozen=True)
class TDTASPStudy:
    """A forward study or a searched family design with fixed-size comparison."""

    design: TDTASPPower
    search: TDTASPSampleSize | None
    fixed_design: TDTASPFixedPower | None
    fixed_search_bounds: tuple[int, int] | None = None


def tdtasp_study(
    haplotype_frequencies: ArrayLike,
    penetrance: ArrayLike,
    mean_offspring: float,
    recombination: float = 0,
    *,
    families: int | None = None,
    target_power: float | None = None,
    test: str = "tdt",
    sampling: str = "family",
    eligibility: str = "one",
    minimum_affected: int | None = None,
    all_affected: bool = False,
    alpha: float = 0.05,
    sides: int = 1,
    legacy_asp: bool = False,
    legacy_moments: bool = False,
    legacy_scale: bool = False,
    legacy_two_sided: bool = False,
    min_families: int = 1,
    max_families: int = 100_000,
    min_observations: int = 1,
    max_observations: int = 100_000,
    batch_size: int = 512,
    max_terms: int = 100_001,
) -> TDTASPStudy:
    """Run genetics, ascertainment and exactly one statistical calculation.

    Supply families for forward power or target_power for both family and
    fixed-observation searches. Search bounds/batch_size apply only to searches.
    Each search must succeed; a failed comparison does not yield a partial study.
    Two-sided ASP is permitted, extending the original console's restriction.
    All numerical conventions and limits are those of the component APIs.
    """
    if (families is None) == (target_power is None):
        raise ValueError("supply exactly one of families and target_power")
    genetics = tdtasp_genetics(
        haplotype_frequencies, penetrance, recombination, legacy_asp=legacy_asp
    )
    ascertainment = tdtasp_ascertainment(
        genetics,
        mean_offspring,
        test=test,
        sampling=sampling,
        eligibility=eligibility,
        minimum_affected=minimum_affected,
        all_affected=all_affected,
        legacy_moments=legacy_moments,
    )
    if families is not None:
        design = tdtasp_power(
            ascertainment,
            families,
            alpha,
            sides=sides,
            legacy_scale=legacy_scale,
            legacy_two_sided=legacy_two_sided,
            max_terms=max_terms,
        )
        return TDTASPStudy(design, None, None)
    assert target_power is not None
    search = tdtasp_sample_size(
        ascertainment,
        target_power,
        alpha,
        sides=sides,
        legacy_scale=legacy_scale,
        legacy_two_sided=legacy_two_sided,
        min_families=min_families,
        max_families=max_families,
        max_terms=max_terms,
    )
    fixed = tdtasp_fixed_sample_size(
        ascertainment.answer_probability,
        target_power,
        alpha,
        sides=sides,
        legacy_two_sided=legacy_two_sided,
        min_observations=min_observations,
        max_observations=max_observations,
        batch_size=batch_size,
    )
    return TDTASPStudy(
        search.design, search, fixed, (index(min_observations), index(max_observations))
    )


def format_tdtasp_study(study: TDTASPStudy, *, digits: int = 8) -> str:
    """Return a complete plain-text result; callers choose printing or file I/O.

    Screening factors stay on the natural-log scale to preserve rare events.
    digits controls significant figures (1 through 17), not calculation precision.
    """
    if not isinstance(study, TDTASPStudy):
        raise TypeError("study must be a TDTASPStudy")
    if isinstance(digits, (bool, np.bool_)):
        raise ValueError("digits must be an integer from 1 to 17")
    try:
        precision = index(digits)
    except TypeError as exc:
        raise ValueError("digits must be an integer from 1 to 17") from exc
    if not 1 <= precision <= 17:
        raise ValueError("digits must be an integer from 1 to 17")

    def number(value: float) -> str:
        return format(value, f".{precision}g")

    d = study.design
    a = d.ascertainment
    g = a.genetics
    c = d.conditional
    ad, a_d, bd, b_d = map(float, g.haplotype_frequencies)
    father = float(a.family_probability @ g.father_heterozygous)
    mother = float(a.family_probability @ g.mother_heterozygous)
    lines = [
        f"TDTASP {a.test.upper()} study",
        "Calculation: " + ("sample size" if study.search is not None else "power"),
        "Haplotype frequencies (AD, Ad, BD, Bd): "
        + ", ".join(number(float(x)) for x in g.haplotype_frequencies),
        "Penetrances (DD, Dd, dd): " + ", ".join(number(float(x)) for x in g.penetrance),
        f"Recombination: {number(g.recombination)}",
        f"Marker allele A frequency: {number(ad + a_d)}",
        f"Disease allele D frequency: {number(ad + bd)}",
        f"Linkage disequilibrium D: {number(ad * b_d - a_d * bd)}",
        f"Population affected probability: {number(g.population_affected_probability)}",
        f"Mean offspring: {number(a.mean_offspring)}",
        f"Sampling list: {a.sampling}; parental eligibility: {a.eligibility}",
        f"Minimum affected offspring: {a.minimum_affected}; all affected: {a.all_affected}",
        f"Eligibility probability within list: {number(a.selection_probability)}",
        f"Father heterozygosity probability among eligible families: {number(father)}",
        f"Mother heterozygosity probability among eligible families: {number(mother)}",
        f"Expected affected offspring per eligible family: {number(a.expected_affected)}",
        "Expected heterozygous parents per eligible family: "
        f"{number(a.expected_heterozygous_parents)}",
        f"Expected contributions per eligible family: {number(a.expected_contributions)}",
        "Population-average truncated affected mean: "
        f"{number(a.population_average_truncated_mean)}",
        f"Natural log of list mass: {number(a.log_list_mass)}",
        f"Natural log of eligible mass: {number(a.log_eligible_mass)}",
        f"Natural log of screening factor from k=1 list: {number(a.log_screening_for_minimum)}",
        f"Alternative answer probability: {number(a.answer_probability)}; null: 0.5",
        f"Sides: {c.sides}; requested significance: {number(c.alpha)}",
        f"Compatibility: legacy_asp={g.legacy_asp}, legacy_moments={a.legacy_moments}, "
        f"legacy_scale={d.legacy_scale}, legacy_two_sided={c.legacy_two_sided}",
        "Model: binomial eligible-family count; mean contributions with half-up rounding.",
        "This planning approximation does not model the full within-family dependence.",
        f"Contributions used per eligible family: {number(d.contribution_per_family)}",
        f"Families screened from qualifying list: {d.families}",
        f"Expected eligible families: {number(d.families * a.selection_probability)}",
        f"Actual significance: {number(d.actual_size)}",
        f"Power: {number(d.power)}",
    ]
    if a.sampling == "individual":
        lines.append(
            "Individual-list sampling assumes repeated selection of a family is negligible."
        )
    if study.search is not None:
        s = study.search
        lines.extend(
            [
                f"Target power: {number(s.target_power)}",
                f"Family search bounds (inclusive): {s.min_families}, {s.max_families}",
                f"First directly scanned count: {s.search_start}; "
                f"mixture evaluations: {s.evaluations}",
                "First qualifying integer within bounds; larger counts need not all qualify.",
            ]
        )
    if study.fixed_search_bounds is not None:
        lo, hi = study.fixed_search_bounds
        lines.append(f"Fixed-observation search bounds (inclusive): {lo}, {hi}")
    if study.fixed_design is not None:
        f = study.fixed_design
        lines.extend(
            [
                f"Fixed-observation comparison: {int(f.observations)} observations",
                f"Fixed actual significance: {number(float(f.actual_size))}",
                f"Fixed power: {number(float(f.power))}",
                f"Fixed inclusive critical values: {int(f.lower_critical)}, "
                f"{int(f.upper_critical)}",
                "A critical value of -1 or n+1 denotes an absent tail.",
            ]
        )
    return "\n".join(lines) + "\n"
