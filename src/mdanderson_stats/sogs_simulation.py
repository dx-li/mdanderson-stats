"""SOGS stochastic backcross breeding with chromosome-level recombination."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .sogs import SOGS_MOUSE_LENGTHS, sogs_eligible

Segments = list[tuple[float, float]]


def _breakpoints(
    count: int, length: float, avoidance: float, rng: np.random.Generator
) -> tuple[list[float], int]:
    if avoidance == 0:
        return sorted(rng.uniform(0, length, count).tolist()), 0
    points: list[float] = []
    failures = 0
    for _ in range(count):
        for attempt in range(9):
            point = float(rng.uniform(0, length))
            if all(abs(point - old) > avoidance for old in points):
                break
            if attempt == 8:
                failures += 1
        points.append(point)
    return sorted(points), failures


def _gamete(parent: Segments, breaks: list[float], homologue: int) -> Segments:
    """Inner interval partition, skipping validation already enforced by construction."""
    if not breaks:
        return parent if homologue == 0 else []
    child: Segments = []
    for start, end in parent:
        lo = bisect_right(breaks, start)
        hi = bisect_left(breaks, end)
        cuts = [start, *breaks[lo:hi], end]
        child.extend(
            (cuts[j], cuts[j + 1])
            for j in range(len(cuts) - 1)
            if (lo + j) % 2 == homologue and cuts[j] < cuts[j + 1]
        )
    return child


def _missed(segments: Segments, probes: list[float]) -> bool:
    return bool(segments) and not any(probes[bisect_left(probes, a)] <= b for a, b in segments)


@dataclass(frozen=True)
class SOGSSimulation:
    chromosomes: tuple[int, ...]
    offspring: tuple[int, ...]
    rule: int
    screening_error: bool
    avoidance: float
    seed: int
    pure_ipt: NDArray[np.int64]
    missed: NDArray[np.int64]
    donor_length: FloatArray
    missed_length: FloatArray
    avoidance_failures: int

    @property
    def apparent_dmt(self) -> NDArray[np.int64]:
        result = len(self.chromosomes) - self.pure_ipt - self.missed
        result.flags.writeable = False
        return result


def sogs_simulate(
    *,
    offspring: tuple[int, ...] = (10,) * 10,
    exclude: tuple[int, ...] = (16,),
    rule: int = 2,
    screening_error: bool = True,
    avoidance: float = 0.0,
    replicates: int = 1000,
    seed: int = 1997,
) -> SOGSSimulation:
    """Simulate selected parents from F1 through each requested backcross.

    Columns are initial F1, BC1, BC2, ...; rows are independent replicates.
    Offspring counts refer to eligible males already carrying the desired donor
    region; excluded chromosomes are not simulated. NumPy RNG replay is local.
    """
    if not 1 <= len(offspring) <= 10 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 1 <= v <= 20
        for v in offspring
    ):
        raise ValueError("offspring must contain 1..10 counts, each an integer in 1..20")
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 1 <= v <= 19
        for v in exclude
    ) or len(set(exclude)) != len(exclude):
        raise ValueError("exclude must contain unique chromosome numbers in 1..19")
    chromosomes = tuple(i for i in range(1, 20) if i not in exclude)
    if not chromosomes:
        raise ValueError("at least one chromosome must remain")
    if isinstance(rule, bool) or not isinstance(rule, int) or rule not in (0, 1, 2, 3):
        raise ValueError("rule must be 0, 1, 2 or 3")
    if not isinstance(screening_error, bool):
        raise ValueError("screening_error must be boolean")
    if np.ndim(avoidance) != 0 or avoidance not in (0, 5, 10):
        raise ValueError("avoidance must be 0, 5 or 10 cM")
    if not screening_error and avoidance != 0:
        raise ValueError("nonzero avoidance requires screening_error=True, as in SOGS")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or not 1 <= replicates <= 100000
    ):
        raise ValueError("replicates must be an integer in 1..100000")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n = len(chromosomes)
    if replicates * sum(offspring) * n > 20_000_000:
        raise ValueError("simulation exceeds 20000000 offspring/chromosome combinations")
    lengths = SOGS_MOUSE_LENGTHS[np.array(chromosomes) - 1]
    probes = []
    for length in lengths:
        intervals = int(length / 5)
        grid = [float(i * (length / intervals)) for i in range(intervals)]
        probes.append([*grid, float(length)])
    rng = np.random.default_rng(seed)
    shape = (replicates, len(offspring) + 1)
    pure = np.zeros(shape, dtype=np.int64)
    missed = np.zeros(shape, dtype=np.int64)
    donor = np.zeros(shape)
    missed_length = np.zeros(shape)
    donor[:, 0] = lengths.sum()
    failures = 0
    for rep in range(replicates):
        parent: list[Segments] = [[(0.0, float(length))] for length in lengths]
        for generation, count in enumerate(offspring, 1):
            if not any(parent):
                pure[rep, generation:] = n
                break
            # Independent Poisson counts and homologue choices are drawn in batches.
            crossover_counts = rng.poisson(lengths / 100, (count, n))
            homologues = rng.integers(0, 2, (count, n))
            candidates = []
            appears = np.zeros((count, n), dtype=bool)
            missed_flags = np.zeros((count, n), dtype=bool)
            for candidate in range(count):
                child = []
                for chrom, segments in enumerate(parent):
                    if segments:
                        breaks, fallback = _breakpoints(
                            int(crossover_counts[candidate, chrom]),
                            float(lengths[chrom]),
                            avoidance,
                            rng,
                        )
                        failures += fallback
                        inherited = _gamete(segments, breaks, int(homologues[candidate, chrom]))
                    else:
                        inherited = []
                    child.append(inherited)
                    missed_flags[candidate, chrom] = screening_error and _missed(
                        inherited, probes[chrom]
                    )
                    appears[candidate, chrom] = not inherited or missed_flags[candidate, chrom]
                candidates.append(child)
            eligible = sogs_eligible(appears, lengths, rule=rule)
            chosen = int(eligible[rng.integers(eligible.size)])
            parent = candidates[chosen]
            pure[rep, generation] = sum(not c for c in parent)
            missed[rep, generation] = missed_flags[chosen].sum()
            amounts = np.array([sum(b - a for a, b in c) for c in parent])
            donor[rep, generation] = amounts.sum()
            missed_length[rep, generation] = amounts[missed_flags[chosen]].sum()
    for array in (pure, missed):
        array.flags.writeable = False
    return SOGSSimulation(
        chromosomes,
        tuple(map(int, offspring)),
        rule,
        screening_error,
        float(avoidance),
        seed,
        pure,
        missed,
        _freeze(donor),
        _freeze(missed_length),
        failures,
    )
