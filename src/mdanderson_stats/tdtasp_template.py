"""Validated TDTASP template specifications, parsing and blank/filled forms."""

import re
from dataclasses import dataclass
from decimal import Decimal
from operator import index

import numpy as np

from ._validation import scalar
from .tdtasp_ascertainment import tdtasp_ascertainment
from .tdtasp_genetics import tdtasp_genetics, tdtasp_haplotype_frequencies
from .tdtasp_power import tdtasp_fixed_power
from .tdtasp_study import TDTASPStudy, tdtasp_study


@dataclass(frozen=True)
class TDTASPTemplate:
    """Immutable, validated inputs representable by the archived template.

    Frequencies and penetrances use AD/Ad/BD/Bd and DD/Dd/dd order. Exactly one
    calculation input is required. ASP uses two siblings and one side; one-child
    TDT uses minimum_affected=1. These are template, not general API restrictions.
    """

    haplotype_frequencies: tuple[float, ...]
    penetrance: tuple[float, ...]
    mean_offspring: float
    recombination: float = 0
    families: int | None = None
    target_power: float | None = None
    test: str = "tdt"
    sampling: str = "family"
    eligibility: str = "one"
    minimum_affected: int | None = None
    all_affected: bool = False
    alpha: float = 0.05
    sides: int = 1

    def __post_init__(self) -> None:
        if (self.families is None) == (self.target_power is None):
            raise ValueError("supply exactly one of families and target_power")
        g = tdtasp_genetics(self.haplotype_frequencies, self.penetrance, self.recombination)
        a = tdtasp_ascertainment(
            g,
            self.mean_offspring,
            test=self.test,
            sampling=self.sampling,
            eligibility=self.eligibility,
            minimum_affected=self.minimum_affected,
            all_affected=self.all_affected,
        )
        checked = tdtasp_fixed_power(0, a.answer_probability, self.alpha, sides=self.sides)
        if a.test == "asp" and checked.sides != 1:
            raise ValueError("ASP templates require sides=1")
        if a.test == "tdt" and not a.all_affected and a.minimum_affected != 1:
            raise ValueError("one-child TDT templates require minimum_affected=1")
        if self.families is not None:
            if isinstance(self.families, (bool, np.bool_)):
                raise ValueError("families must be a nonnegative integer smaller than 2**53")
            try:
                n = index(self.families)
            except TypeError as exc:
                raise ValueError(
                    "families must be a nonnegative integer smaller than 2**53"
                ) from exc
            if not 0 <= n < 2**53:
                raise ValueError("families must be a nonnegative integer smaller than 2**53")
            object.__setattr__(self, "families", n)
        else:
            assert self.target_power is not None
            target = scalar(self.target_power, "target_power")
            if not 0 < target < 1:
                raise ValueError("target_power must lie in (0, 1)")
            object.__setattr__(self, "target_power", target)
        object.__setattr__(
            self, "haplotype_frequencies", tuple(map(float, g.haplotype_frequencies))
        )
        object.__setattr__(self, "penetrance", tuple(map(float, g.penetrance)))
        object.__setattr__(self, "recombination", g.recombination)
        object.__setattr__(self, "mean_offspring", a.mean_offspring)
        object.__setattr__(self, "minimum_affected", a.minimum_affected)
        object.__setattr__(self, "all_affected", a.all_affected)
        object.__setattr__(self, "alpha", checked.alpha)
        object.__setattr__(self, "sides", checked.sides)

    def run(
        self,
        *,
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
        """Run with explicit numerical compatibility options and resource bounds.

        These controls are not stored in the legacy form; they are recorded in
        the resulting study/report where applicable. Defaults use corrected math.
        """
        return tdtasp_study(
            self.haplotype_frequencies,
            self.penetrance,
            self.mean_offspring,
            self.recombination,
            families=self.families,
            target_power=self.target_power,
            test=self.test,
            sampling=self.sampling,
            eligibility=self.eligibility,
            minimum_affected=self.minimum_affected,
            all_affected=self.all_affected,
            alpha=self.alpha,
            sides=self.sides,
            legacy_asp=legacy_asp,
            legacy_moments=legacy_moments,
            legacy_scale=legacy_scale,
            legacy_two_sided=legacy_two_sided,
            min_families=min_families,
            max_families=max_families,
            min_observations=min_observations,
            max_observations=max_observations,
            batch_size=batch_size,
            max_terms=max_terms,
        )


_FIELDS = (
    "tdt_or_asp",
    "n_offspring",
    "penetrances",
    "theta",
    "how_enter_popn_freq",
    "popn_haplotype_frequencies",
    "prevalence",
    "relative_disequilibrium",
    "family_or_individual",
    "parent_hetero_requirement",
    "include_all_affected",
    "mininum_n_affected",
    "sided",
    "significance",
    "which_calc",
    "power",
    "sample_size",
)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?", re.ASCII)


def _assignments(text: str) -> dict[str, str]:
    if not isinstance(text, str):
        raise TypeError("template must be text")
    result: dict[str, str] = {}
    pending = ""
    for line in text.splitlines():
        if "#" in line or not line.strip():
            continue
        pending += " " + line.strip()
        if pending.count("(") > 1 or pending.count(")") > 1:
            raise ValueError("nested or multiple lists are not allowed")
        if pending.count("(") > pending.count(")"):
            continue
        if pending.count("(") != pending.count(")"):
            raise ValueError("unmatched template parenthesis")
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*?)\s*", pending)
        if match is None:
            raise ValueError(f"invalid template assignment: {pending.strip()}")
        name, value = match.groups()
        name = name.lower()
        if name == "minimum_n_affected":
            name = "mininum_n_affected"
        if name not in _FIELDS:
            raise ValueError(f"unknown template field: {name}")
        if name in result:
            raise ValueError(f"duplicate template field: {name}")
        result[name] = value.replace("*", " ").strip()
        pending = ""
    if pending:
        raise ValueError("unterminated template list")
    return result


def _numeric(value: str, name: str) -> float:
    if _NUMBER.fullmatch(value) is None:
        raise ValueError(f"{name} requires a numeric value")
    number = float(value.replace("D", "e").replace("d", "e"))
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: str, name: str) -> int:
    _numeric(value, name)
    number = Decimal(value.replace("D", "e").replace("d", "e"))
    if number != number.to_integral_value() or abs(number) >= 2**53:
        raise ValueError(f"{name} requires an integer smaller than 2**53 in magnitude")
    return int(number)


def _choice(value: str, name: str, choices: tuple[str, ...]) -> str:
    matches = [choice for choice in choices if value and choice.startswith(value.lower())]
    if len(matches) != 1:
        raise ValueError(f"{name} requires one of {', '.join(choices)}")
    return matches[0]


def _list(value: str, name: str, labels: tuple[str, ...]) -> tuple[float, ...]:
    if not value.startswith("(") or not value.endswith(")"):
        raise ValueError(f"{name} requires a parenthesized list")
    parts = [part.strip() for part in value[1:-1].split(",")]
    if len(parts) != len(labels):
        raise ValueError(f"{name} requires {len(labels)} values")
    if not any("=" in part for part in parts):
        return tuple(_numeric(part, name) for part in parts)
    found: dict[str, float] = {}
    for part in parts:
        pair = part.split("=")
        if len(pair) != 2:
            raise ValueError(f"{name} cannot mix named and positional values")
        label, number = (item.strip() for item in pair)
        if name == "penetrances" and label == "dD":
            label = "Dd"
        if label not in labels or label in found:
            raise ValueError(f"{name} has an unknown or duplicate label: {label}")
        found[label] = _numeric(number, name)
    return tuple(found[label] for label in labels)


def parse_tdtasp_template(text: str) -> TDTASPTemplate:
    """Read a legacy form, validating active values and ignoring inactive values.

    Lines containing # are ignored in full. Field order is free; field names
    and choice values are case-insensitive. Named genotype labels retain case.
    Active placeholders, duplicate/unknown fields and fractional counts fail.
    Both spellings of mininum/minimum_n_affected are accepted, but not together.
    """
    fields = _assignments(text)

    def value(name: str) -> str:
        if name not in fields or not fields[name]:
            raise ValueError(f"missing or unfilled active template field: {name}")
        return fields[name]

    test = _choice(value("tdt_or_asp"), "tdt_or_asp", ("tdt", "asp"))
    method = _choice(value("how_enter_popn_freq"), "how_enter_popn_freq", ("direct", "diseq"))
    if method == "direct":
        frequencies = _list(
            value("popn_haplotype_frequencies"),
            "popn_haplotype_frequencies",
            ("AD", "Ad", "BD", "Bd"),
        )
    else:
        p, m = _list(value("prevalence"), "prevalence", ("p", "m"))
        frequencies = tuple(
            map(
                float,
                tdtasp_haplotype_frequencies(
                    m, p, _numeric(value("relative_disequilibrium"), "relative_disequilibrium")
                ),
            )
        )
    all_affected = (
        test == "tdt"
        and _choice(value("include_all_affected"), "include_all_affected", ("yes", "no")) == "yes"
    )
    minimum = (
        2
        if test == "asp"
        else (_integer(value("mininum_n_affected"), "mininum_n_affected") if all_affected else 1)
    )
    calculation = _choice(value("which_calc"), "which_calc", ("sample-size", "power"))
    eligibility = _choice(
        value("parent_hetero_requirement"), "parent_hetero_requirement", ("random", "one", "both")
    )
    return TDTASPTemplate(
        frequencies,
        _list(value("penetrances"), "penetrances", ("DD", "Dd", "dd")),
        _numeric(value("n_offspring"), "n_offspring"),
        _numeric(value("theta"), "theta"),
        families=_integer(value("sample_size"), "sample_size") if calculation == "power" else None,
        target_power=_numeric(value("power"), "power") if calculation == "sample-size" else None,
        test=test,
        sampling=_choice(
            value("family_or_individual"), "family_or_individual", ("family", "individual")
        ),
        eligibility="father" if eligibility == "random" else eligibility,
        minimum_affected=minimum,
        all_affected=all_affected,
        alpha=_numeric(value("significance"), "significance"),
        sides=_integer(value("sided"), "sided") if test == "tdt" else 1,
    )


def format_tdtasp_template(specification: TDTASPTemplate | None = None) -> str:
    """Return a blank form or a filled direct-frequency form with 17-digit numbers.

    All 17 source fields are emitted in original order, including inactive fields.
    The original misspelling mininum_n_affected is retained for compatibility.
    Caller owns file I/O. Mathematical input bounds follow the Python components.
    """
    values = dict.fromkeys(_FIELDS, "********")
    values.update(
        penetrances="( DD=********, dD=********, dd=******** )",
        popn_haplotype_frequencies="( AD=********, Ad=********, BD=********, Bd=******** )",
        prevalence="( p=********, m=******** )",
    )
    if specification is not None:
        if not isinstance(specification, TDTASPTemplate):
            raise TypeError("specification must be a TDTASPTemplate")
        s = specification
        values.update(
            tdt_or_asp=s.test,
            n_offspring=format(s.mean_offspring, ".17g"),
            penetrances="( "
            + ", ".join(
                f"{label}={v:.17g}"
                for label, v in zip(("DD", "dD", "dd"), s.penetrance, strict=True)
            )
            + " )",
            theta=format(s.recombination, ".17g"),
            how_enter_popn_freq="direct",
            popn_haplotype_frequencies="( "
            + ", ".join(
                f"{label}={v:.17g}"
                for label, v in zip(("AD", "Ad", "BD", "Bd"), s.haplotype_frequencies, strict=True)
            )
            + " )",
            family_or_individual=s.sampling,
            parent_hetero_requirement="random" if s.eligibility == "father" else s.eligibility,
            include_all_affected="yes" if s.all_affected else "no",
            mininum_n_affected=str(s.minimum_affected),
            sided=str(s.sides),
            significance=format(s.alpha, ".17g"),
            which_calc="power" if s.families is not None else "sample-size",
            power="********" if s.target_power is None else format(s.target_power, ".17g"),
            sample_size="********" if s.families is None else str(s.families),
        )
    instructions = (
        "# TDTASP input form: replace active asterisks with values.\n"
        "# Any line containing # is ignored in full; do not add inline comments.\n"
        "# tdt_or_asp: tdt/asp; how_enter_popn_freq: direct/diseq.\n"
        "# direct uses haplotypes; diseq uses prevalence p (disease), m (marker), and D'.\n"
        "# Haplotype probabilities sum to 1; penetrances lie in [0,1]; theta in [0,0.5].\n"
        "# Mean offspring lies in [0.1,10]; relative disequilibrium lies in [-1,1].\n"
        "# Sampling: family/individual; parents: random/one/both.\n"
        "# TDT: include_all_affected yes/no; if yes, minimum affected is an integer 1..10.\n"
        "# One-child TDT fixes minimum affected to 1. ASP fixes it to 2 and ignores all/sided.\n"
        "# sided: 1/2; significance lies in (0,1); which_calc: sample-size/power.\n"
        "# sample-size requires power in (0,1); power requires integer sample_size >=0.\n"
        "# Inactive fields may retain asterisks; run options control numerical compatibility.\n"
    )
    return instructions + "\n" + "\n".join(f"{name} = {values[name]}" for name in _FIELDS) + "\n"
