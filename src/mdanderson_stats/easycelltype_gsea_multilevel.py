"""Ranked EasyCellType GSEA with the pinned fgsea multilevel workflow.

The implementation follows the EasyCellType 1.5.4 wrapper and the selected
Bioconductor 3.18 profile (clusterProfiler 4.10.1, DOSE 3.28.2, fgsea 1.28.0).
The fgsea splitting calculation is independently implemented from its MIT-
licensed source; Python random streams do not reproduce R/C++ streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincinv, digamma, polygamma

from mdanderson_stats.easycelltype import _labels, _scores
from mdanderson_stats.easycelltype_gsea import (
    EasyCellTypeGSEASet,
    _fgsea_stat,
    easycelltype_gsea_es,
)

_MAX_ROWS = 200_000
_MAX_SETS_PER_CLUSTER = 20_000
_MAX_PILOT_PERMUTATIONS = 100_000
_MAX_SPLITTING_SAMPLES = 501
_MAX_RETAINED_SPLIT_POSITIONS = 2_000_000
_MAX_WORK = 500_000_000
_MAX_RETAINED_IDS = 1_000_000


@dataclass(frozen=True)
class EasyCellTypeGSEAInferenceSet:
    """One cell-type enrichment with the observed score and inference."""

    observed: EasyCellTypeGSEASet
    normalized_enrichment_score: float | None
    p_value: float | None
    adjusted_p_value: float | None
    log2_error: float | None
    inference_method: str
    status: str | None
    reported: bool
    pilot_mode_count: int
    pilot_extreme_count: int | None


@dataclass(frozen=True)
class EasyCellTypeGSEAInferenceCluster:
    """All tested cell types for one cluster in reference order."""

    cluster: str
    sets: tuple[EasyCellTypeGSEAInferenceSet, ...]


@dataclass(frozen=True)
class EasyCellTypeGSEAInference:
    """Observed GSEA, normalized scores and adjusted tail probabilities."""

    clusters: tuple[EasyCellTypeGSEAInferenceCluster, ...]
    score_type: str
    exponent: float
    min_size: int
    max_size: int
    p_cut: float
    sample_size: int
    n_perm_simple: int
    eps: float


@dataclass
class _Pilot:
    mode_count: int
    extreme_count: int
    p_value: float | None
    nes: float | None
    simple_error: float | None
    method: str
    log2_error: float | None
    status: str | None = None


@dataclass
class _Work:
    remaining: int

    def spend(self, units: int) -> None:
        if units < 0 or units > self.remaining:
            raise ValueError("GSEA adaptive-splitting work exceeds max_work")
        self.remaining -= units


def easycelltype_gsea(
    query_genes: ArrayLike,
    clusters: ArrayLike,
    scores: ArrayLike,
    reference_genes: ArrayLike,
    reference_cell_types: ArrayLike,
    *,
    score_type: str = "std",
    exponent: float = 1.0,
    min_size: int = 1,
    max_size: int = 500,
    p_cut: float = 0.5,
    sample_size: int = 101,
    n_perm_simple: int = 1000,
    eps: float = 1e-10,
    rng: np.random.Generator | int | None = None,
    max_work: int = 100_000_000,
) -> EasyCellTypeGSEAInference:
    """Run weighted ranked GSEA with fgsea's pilot and adaptive splitting.

    Each cluster is analyzed independently. Pilot gene sets are sampled as a
    uniform subset at the largest tested set size, and smaller sizes use
    prefixes of that same subset, as in the pinned fgsea cumulative kernel.
    The source-specific epsilon replacement for zero random hit weights is
    retained. Splitter samples are bounded to two million retained positions
    across the old and replacement sample arrays. RNG controls NumPy
    randomness only; it does not reproduce R's RNG or the C++ mt19937 stream.

    p_value and adjusted_p_value are fgsea tail probabilities and BH values.
    reported applies DOSE's two-sided p-cut rule: both values must be at most
    p_cut. A mode count below ten leaves probability and NES unavailable.
    qvalue is not part of the EasyCellType label workflow.
    """
    _check_int(sample_size, "sample_size", 1, _MAX_SPLITTING_SAMPLES)
    sample_size = max(3, int(sample_size))
    if sample_size % 2 == 0:
        sample_size += 1
    if sample_size > _MAX_SPLITTING_SAMPLES:
        raise ValueError(f"sample_size must be at most {_MAX_SPLITTING_SAMPLES - 1}")
    _check_int(n_perm_simple, "n_perm_simple", 1, _MAX_PILOT_PERMUTATIONS)
    _check_int(max_work, "max_work", 1, _MAX_WORK)
    if isinstance(p_cut, (bool, np.bool_)) or not isinstance(p_cut, Real):
        raise ValueError("p_cut must be a finite number in [0, 1]")
    p_cut = float(p_cut)
    if not isfinite(p_cut) or not 0 <= p_cut <= 1:
        raise ValueError("p_cut must be a finite number in [0, 1]")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real):
        raise ValueError("eps must be a finite number in [0, 1]")
    eps = float(eps)
    if not isfinite(eps) or not 0 <= eps <= 1:
        raise ValueError("eps must be a finite number in [0, 1]")

    observed = easycelltype_gsea_es(
        query_genes,
        clusters,
        scores,
        reference_genes,
        reference_cell_types,
        score_type=score_type,
        exponent=exponent,
        min_size=min_size,
        max_size=max_size,
    )
    cluster_labels = _labels(clusters, "clusters")
    score_values = _scores(scores)
    rng_obj = np.random.default_rng(rng)

    rows_by_cluster: dict[str, list[int]] = {item.cluster: [] for item in observed.clusters}
    for row, label in enumerate(cluster_labels):
        rows_by_cluster[label].append(row)
    work = _Work(int(max_work))
    total_ids = 0
    final_clusters: list[EasyCellTypeGSEAInferenceCluster] = []
    for cluster_result in observed.clusters:
        rows = rows_by_cluster[cluster_result.cluster]
        if not rows or not cluster_result.sets:
            final_clusters.append(EasyCellTypeGSEAInferenceCluster(cluster_result.cluster, ()))
            continue
        if len(cluster_result.sets) > _MAX_SETS_PER_CLUSTER:
            raise ValueError("each cluster is limited to 20,000 tested cell types")
        order = np.asarray(sorted(rows, key=lambda i: -score_values[i]), dtype=np.intp)
        ranked_scores = np.asarray([score_values[int(i)] for i in order], dtype=np.float64)
        transformed, log_scale = _powered_weights(ranked_scores, observed.exponent)
        if log_scale == float("-inf"):
            raise ValueError(
                "full GSEA inference is undefined for an all-zero prepared ranking; "
                "use easycelltype_gsea_es for observed scores"
            )
        pilot_work = int(n_perm_simple) * (
            int(ranked_scores.size) + sum(int(item.set_size) for item in cluster_result.sets)
        )
        work.spend(pilot_work)

        sizes = [int(item.set_size) for item in cluster_result.sets]
        pilots = _simple_pilot(
            transformed,
            sizes,
            [item.enrichment_score for item in cluster_result.sets],
            score_type=observed.score_type,
            log_weight_scale=log_scale,
            single_set=len(cluster_result.sets) == 1,
            n_perm=int(n_perm_simple),
            rng=rng_obj,
        )
        multilevel_sets: dict[int, list[int]] = {}
        for index, item in enumerate(cluster_result.sets):
            pilot = pilots[index]
            if pilot.mode_count < 10:
                pilot.p_value = None
                pilot.nes = None
                pilot.simple_error = None
                pilot.method = "unavailable"
                pilot.status = "insufficient_directional_pilot"
                continue
            if pilot.nes is None or pilot.p_value is None or pilot.simple_error is None:
                pilot.method = "unavailable"
                pilot.status = "zero_directional_null_mean"
                continue
            mult_error = _multilevel_error(
                (pilot.extreme_count + 1.0) / (int(n_perm_simple) + 1.0), sample_size
            )
            if mult_error < pilot.simple_error:
                pilot.method = "multilevel"
                multilevel_sets.setdefault(item.set_size, []).append(index)
            else:
                pilot.method = "simple"
                pilot.log2_error = _simple_log2_error(pilot.extreme_count, int(n_perm_simple))

        for size, indices in multilevel_sets.items():
            pos_ranks = transformed.copy()
            target_es = [cluster_result.sets[i].enrichment_score for i in indices]
            eps_group = eps * min(
                (pilots[i].mode_count + 1.0) / (int(n_perm_simple) + 1.0) for i in indices
            )
            positive_ruler = None
            negative_ruler = None
            if any(value >= 0 for value in target_es):
                positive_ruler = _EsRuler(pos_ranks, size, sample_size, rng_obj, work)
                positive_ruler.extend(max(max(0.0, float(value)) for value in target_es), eps_group)
            if any(value < 0 for value in target_es):
                negative_ruler = _EsRuler(pos_ranks[::-1].copy(), size, sample_size, rng_obj, work)
                negative_ruler.extend(
                    max(max(0.0, -float(value)) for value in target_es), eps_group
                )
            for index in indices:
                es = cluster_result.sets[index].enrichment_score
                ruler = positive_ruler if es >= 0 else negative_ruler
                if ruler is None:
                    raise RuntimeError("missing directional GSEA splitting ruler")
                conditional_p, cp_ge_half = ruler.p_value(abs(es), score_type != "std")
                denominator = (pilots[index].mode_count + 1.0) / (int(n_perm_simple) + 1.0)
                p_value = min(1.0, conditional_p / denominator)
                log_error = _multilevel_error(p_value, sample_size) if cp_ge_half else None
                if not cp_ge_half:
                    pilots[index].status = "conditional_probability_below_half"
                if p_value < eps:
                    p_value = eps
                    log_error = None
                    pilots[index].status = _append_status(pilots[index].status, "below_eps")
                pilots[index].p_value = p_value
                pilots[index].log2_error = log_error

        if multilevel_sets:
            for pilot in pilots:
                if pilot.p_value is not None and pilot.p_value < eps:
                    pilot.p_value = eps
                    pilot.log2_error = None
                    pilot.status = _append_status(pilot.status, "below_eps")

        adjusted = _bh([pilot.p_value for pilot in pilots])
        cluster_sets: list[EasyCellTypeGSEAInferenceSet] = []
        for item, pilot, adj in zip(cluster_result.sets, pilots, adjusted, strict=True):
            reported = (
                pilot.p_value is not None
                and adj is not None
                and pilot.p_value <= p_cut
                and adj <= p_cut
            )
            total_ids += len(item.fgsea_leading_edge or ()) + len(item.core_enrichment or ())
            if total_ids > _MAX_RETAINED_IDS:
                raise ValueError("retained GSEA gene IDs exceed the 1,000,000-ID limit")
            status = pilot.status
            cluster_sets.append(
                EasyCellTypeGSEAInferenceSet(
                    observed=item,
                    normalized_enrichment_score=pilot.nes,
                    p_value=pilot.p_value,
                    adjusted_p_value=adj,
                    log2_error=pilot.log2_error,
                    inference_method=pilot.method,
                    status=status,
                    reported=reported,
                    pilot_mode_count=pilot.mode_count,
                    pilot_extreme_count=pilot.extreme_count if pilot.p_value is not None else None,
                )
            )
        final_clusters.append(
            EasyCellTypeGSEAInferenceCluster(cluster_result.cluster, tuple(cluster_sets))
        )
    return EasyCellTypeGSEAInference(
        tuple(final_clusters),
        observed.score_type,
        observed.exponent,
        observed.min_size,
        observed.max_size,
        p_cut,
        sample_size,
        int(n_perm_simple),
        eps,
    )


def _check_int(value: object, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _powered_weights(
    scores: NDArray[np.float64], exponent: float
) -> tuple[NDArray[np.float64], float]:
    if exponent == 0:
        return np.ones(scores.size, dtype=np.float64), 0.0
    scale = float(np.max(np.abs(scores), initial=0.0))
    if scale == 0:
        return np.zeros(scores.size, dtype=np.float64), float("-inf")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = (np.abs(scores) / scale) ** exponent
    if np.any((scores != 0) & (result == 0)):
        raise ValueError("score dynamic range is too large for stable full GSEA weights")
    return result, exponent * log(scale)


def _simple_pilot(
    weights: NDArray[np.float64],
    sizes: list[int],
    observed_es: list[float],
    *,
    score_type: str,
    log_weight_scale: float,
    single_set: bool,
    n_perm: int,
    rng: np.random.Generator,
) -> list[_Pilot]:
    """Source cumulative pilot: one max-size random set supplies every prefix."""
    count = len(sizes)
    largest = max(sizes)
    n = int(weights.size)
    less_es = np.zeros(count, dtype=np.int64)
    greater_es = np.zeros(count, dtype=np.int64)
    less_zero = np.zeros(count, dtype=np.int64)
    greater_zero = np.zeros(count, dtype=np.int64)
    less_zero_sum = np.zeros(count, dtype=np.float64)
    greater_zero_sum = np.zeros(count, dtype=np.float64)
    for _ in range(n_perm):
        selected = _sample_positions(n, largest, rng)
        if single_set:
            pilot_es = [
                _fgsea_stat(weights, np.sort(selected), exponent=1.0, score_type=score_type)[0]
            ]
        else:
            positive = weights[selected]
            positive_min = float(np.min(positive[positive > 0], initial=np.inf))
            if log_weight_scale == float("-inf"):
                relative_floor = 1e-5
            else:
                log_relative_floor = log(1e-5) - log_weight_scale
                if log_relative_floor > log(np.finfo(float).max):
                    relative_floor = float("inf")
                elif log_relative_floor < log(np.nextafter(0.0, 1.0)):
                    relative_floor = 0.0
                else:
                    relative_floor = float(np.exp(log_relative_floor))
            stat_eps = (
                min(relative_floor, positive_min) / 1024.0
                if isfinite(positive_min)
                else relative_floor / 1024.0
            )
            if stat_eps <= 0 and np.any(positive == 0):
                raise ValueError("pilot epsilon is not representable at this score scale")
            pilot_es = _prefix_es(weights, selected, sizes, stat_eps, score_type)
        for i, value in enumerate(pilot_es):
            es = observed_es[i]
            if value <= es:
                less_es[i] += 1
            if value >= es:
                greater_es[i] += 1
            if value <= 0:
                less_zero[i] += 1
                less_zero_sum[i] += min(value, 0.0)
            if value >= 0:
                greater_zero[i] += 1
                greater_zero_sum[i] += max(value, 0.0)
    result: list[_Pilot] = []
    for i, es in enumerate(observed_es):
        ge_mean = greater_zero_sum[i] / greater_zero[i] if greater_zero[i] else 0.0
        le_mean = less_zero_sum[i] / less_zero[i] if less_zero[i] else 0.0
        if score_type == "std":
            denominator = ge_mean if es > 0 else abs(le_mean)
            nes = es / denominator if denominator else None
        elif score_type == "pos":
            nes = es / ge_mean if es >= 0 and ge_mean else None
        else:
            nes = es / abs(le_mean) if es <= 0 and le_mean else None
        if score_type == "std":
            mode_count = int(greater_zero[i] if es >= 0 else less_zero[i])
            extreme = int(greater_es[i] if es > 0 else less_es[i])
        elif score_type == "pos":
            mode_count, extreme = int(greater_zero[i]), int(greater_es[i])
        else:
            mode_count, extreme = int(less_zero[i]), int(less_es[i])
        p_value = None
        simple_error = None
        if nes is not None:
            p_value = min(
                (1.0 + less_es[i]) / (1.0 + less_zero[i]),
                (1.0 + greater_es[i]) / (1.0 + greater_zero[i]),
            )
            simple_error = _simple_error(extreme, n_perm)
        result.append(_Pilot(mode_count, extreme, p_value, nes, simple_error, "simple", None))
    return result


def _prefix_es(
    weights: NDArray[np.float64],
    selected: NDArray[np.intp],
    sizes: list[int],
    stat_eps: float,
    score_type: str,
) -> list[float]:
    """Compute source forward/reverse cumulative ES for requested prefixes."""
    n = int(weights.size)
    answer: list[float] = []
    for k in sizes:
        hit_positions = np.sort(selected[:k])
        hit_weights = np.maximum(weights[hit_positions], stat_eps)
        total_weight = float(np.sum(hit_weights))
        if total_weight <= 0.0 or not isfinite(total_weight):
            raise ValueError("random pilot hit weights are not representable at this score scale")
        cumulative = np.cumsum(hit_weights) / total_weight
        misses = hit_positions - np.arange(k, dtype=np.intp)
        top = cumulative - misses / (n - k)
        forward = float(np.max(top))
        reverse_positions = n - 1 - hit_positions[::-1]
        reverse_cumulative = np.cumsum(hit_weights[::-1]) / total_weight
        reverse_misses = reverse_positions - np.arange(k, dtype=np.intp)
        reverse_top = reverse_cumulative - reverse_misses / (n - k)
        reverse = float(np.max(reverse_top))
        if score_type == "pos":
            answer.append(forward)
        elif score_type == "neg":
            answer.append(-reverse)
        elif forward == reverse:
            answer.append(0.0)
        elif forward > reverse:
            answer.append(forward)
        else:
            answer.append(-reverse)
    return answer


def _simple_error(extreme: int, n_perm: int) -> float:
    crude = log((extreme + 1.0) / (n_perm + 1.0), 2)
    left = (
        float("-inf")
        if extreme == 0
        else float(np.log2(betaincinv(extreme, n_perm - extreme + 1, 0.025)))
    )
    right = (
        0.0
        if extreme == n_perm
        else float(np.log2(betaincinv(extreme + 1, n_perm - extreme, 0.975)))
    )
    return 0.5 * max(crude - left, right - crude)


def _multilevel_error(p_value: float, sample_size: int) -> float:
    if p_value <= 0:
        return float("inf")
    return sqrt(
        max(0.0, np.floor(-np.log2(p_value) + 1.0))
        * (polygamma(1, (sample_size + 1.0) / 2.0) - polygamma(1, sample_size + 1.0))
    ) / log(2.0)


def _simple_log2_error(extreme: int, n_perm: int) -> float:
    return sqrt(float(polygamma(1, extreme + 1) - polygamma(1, n_perm + 1))) / log(2.0)


def _bh(values: list[float | None]) -> list[float | None]:
    finite = [(i, value) for i, value in enumerate(values) if value is not None and isfinite(value)]
    m = len(finite)
    adjusted: list[float | None] = [None] * len(values)
    order = sorted(finite, key=lambda pair: pair[1])
    running = 1.0
    for rank in range(m, 0, -1):
        index, value = order[rank - 1]
        running = min(running, float(value) * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


class _EsRuler:
    """Conditional fixed-cardinality splitter for one direction and set size."""

    def __init__(
        self,
        ranks: NDArray[np.float64],
        pathway_size: int,
        sample_size: int,
        rng: np.random.Generator,
        work: _Work,
    ) -> None:
        self.ranks = ranks
        self.n = int(ranks.size)
        self.k = int(pathway_size)
        self.sample_size = int(sample_size)
        self.rng = rng
        self.work = work
        self.half = (self.sample_size + 1) // 2
        self.levels: list[float] = []
        self.corrector: list[int] = []

    def extend(self, threshold: float, eps: float) -> None:
        if 2 * self.sample_size * self.k > _MAX_RETAINED_SPLIT_POSITIONS:
            raise ValueError("adaptive GSEA splitter samples exceed the retained-position limit")
        self.work.spend(self.sample_size * self.k * 2)
        samples = [
            np.sort(_sample_positions(self.n, self.k, self.rng)) for _ in range(self.sample_size)
        ]
        self._duplicate(samples)
        while self.levels[-1] <= threshold - 1e-10:
            accepted = 0
            target = self.sample_size * self.k
            while accepted < target:
                for i in range(self.sample_size):
                    accepted += self._perturb(samples, i, self.levels[-1])
            self.work.spend(self.sample_size * self.k)
            self._duplicate(samples)
            level_count = len(self.levels) // self.half
            if eps != 0 and level_count > -np.log2(0.5 * eps):
                break

    def p_value(self, es: float, signed: bool) -> tuple[float, bool]:
        if es >= self.levels[-1]:
            index = len(self.levels) - 1
        else:
            index = int(np.searchsorted(self.levels, es, side="left"))
        levels = index // self.half
        remainder = self.sample_size - (index % self.half)
        log_p = levels * _beta_mean_log(self.half, self.sample_size)
        log_p += _beta_mean_log(remainder + 1, self.sample_size)
        if signed:
            return float(np.clip(np.exp(log_p), 0.0, 1.0)), True
        correction, adequate = self._correction(index, remainder)
        return float(np.clip(np.exp(log_p + correction), 0.0, 1.0)), adequate

    def _correction(self, index: int, remainder: int) -> tuple[float, bool]:
        log_mean = _beta_mean_log(self.corrector[index] + 1, remainder)
        return log_mean, bool(np.exp(log_mean) >= 0.5)

    def _duplicate(self, samples: list[NDArray[np.intp]]) -> None:
        statistics = [(_positive_es(self.ranks, sample), i) for i, sample in enumerate(samples)]
        statistics.sort()
        positive_count = sum(
            _signed_es(self.ranks, samples[sample_index]) > 0.0 for _, sample_index in statistics
        )
        for value, sample_index in statistics[: self.half]:
            self.levels.append(float(value))
            if _signed_es(self.ranks, samples[sample_index]) > 0.0:
                positive_count -= 1
            self.corrector.append(positive_count)
        new_samples: list[NDArray[np.intp]] = []
        for i in range((self.sample_size - 1) // 2):
            retained = samples[statistics[self.sample_size - 1 - i][1]]
            new_samples.extend((retained.copy(), retained.copy()))
        new_samples.append(samples[statistics[self.sample_size >> 1][1]].copy())
        samples[:] = new_samples

    def _perturb(
        self,
        samples: list[NDArray[np.intp]],
        sample_index: int,
        bound: float,
    ) -> int:
        iterations = max(1, int(self.k * 0.1))
        moved = 0
        for _ in range(iterations):
            sample = samples[sample_index]
            self.work.spend(self.k + 1)
            old_i = int(self.rng.integers(self.k))
            old_value = int(sample[old_i])
            new_value = int(self.rng.integers(self.n))
            if new_value == old_value:
                moved += 1
                continue
            insertion = int(np.searchsorted(sample, new_value))
            if insertion < self.k and int(sample[insertion]) == new_value:
                continue
            candidate = np.delete(sample, old_i)
            candidate = np.insert(candidate, int(np.searchsorted(candidate, new_value)), new_value)
            if _positive_es(self.ranks, candidate) > bound:
                samples[sample_index] = candidate.astype(np.intp, copy=False)
                moved += 1
        return moved


def _positive_es(ranks: NDArray[np.float64], positions: NDArray[np.intp]) -> float:
    n = int(ranks.size)
    k = int(positions.size)
    selected_weights = ranks[positions]
    total = float(np.sum(selected_weights))
    if total <= 0 or not isfinite(total):
        return 0.0
    cumulative = np.cumsum(selected_weights / total)
    misses = positions - np.arange(k, dtype=np.intp)
    return float(np.max(cumulative - misses / (n - k), initial=0.0))


def _signed_es(ranks: NDArray[np.float64], positions: NDArray[np.intp]) -> float:
    n = int(ranks.size)
    k = int(positions.size)
    selected_weights = ranks[positions]
    total = float(np.sum(selected_weights))
    down = 1.0 / (n - k)
    if total <= 0 or not isfinite(total):
        hit_increments = np.full(k, np.nan, dtype=np.float64)
    else:
        hit_increments = selected_weights / total
    current = 0.0
    result = 0.0
    last = -1
    for position, increment in zip(positions, hit_increments, strict=True):
        current -= down * (int(position) - last - 1)
        if abs(current) > abs(result):
            result = current
        current += float(increment)
        if abs(current) > abs(result):
            result = current
        last = int(position)
    return result


def _sample_positions(n: int, k: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Uniform fixed-size sample in O(k) memory (Floyd's algorithm)."""
    chosen: set[int] = set()
    ordered: list[int] = []
    for upper in range(n - k, n):
        candidate = int(rng.integers(upper + 1))
        value = upper if candidate in chosen else candidate
        chosen.add(value)
        ordered.append(value)
    rng.shuffle(ordered)
    return np.asarray(ordered, dtype=np.intp)


def _beta_mean_log(a: int, b: int) -> float:
    return float(digamma(float(a)) - digamma(float(b + 1)))


def _append_status(status: str | None, addition: str) -> str:
    if status is None:
        return addition
    if addition in status.split(";"):
        return status
    return f"{status};{addition}"
