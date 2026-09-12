"""Paired-endpoint BOP2-DC monitoring with a joint Dirichlet prior."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned


@dataclass(frozen=True)
class BOP2DCPairedState:
    sample_size: NDArray[np.int64]
    counts: NDArray[np.int64]
    posterior_shapes: FloatArray
    marginal_posterior: FloatArray
    endpoint_decision: NDArray[np.str_]
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BOP2DCPairedOperatingCharacteristics:
    category_probability: FloatArray
    looks: NDArray[np.int64]
    stop_no_go: FloatArray
    final_go: FloatArray
    final_consider: FloatArray
    final_no_go: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray

    @property
    def no_go_probability(self) -> FloatArray:
        return self.stop_no_go.sum(axis=-1) + self.final_no_go


@dataclass(frozen=True)
class BOP2DCPairedDesign:
    endpoint: str
    max_subjects: int
    lrv: FloatArray
    cmv: FloatArray
    success_lrv: FloatArray
    success_cmv: FloatArray
    lambda_lrv: FloatArray
    lambda_cmv: FloatArray
    gamma_lrv: FloatArray
    gamma_cmv: FloatArray
    prior: FloatArray
    looks: NDArray[np.int64]

    @staticmethod
    def _strict_above(value: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
        return value > threshold

    @staticmethod
    def _strict_below(value: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
        return value < threshold

    def _posterior_marginal(self, marginal: np.ndarray, total: int | np.ndarray) -> np.ndarray:
        alpha = np.array([self.prior[[0, 1]].sum(), self.prior[[0, 2]].sum()])
        beta = np.array([self.prior[[2, 3]].sum(), self.prior[[1, 3]].sum()])
        if self.endpoint == "efficacy_toxicity":
            alpha[1] = self.prior[[1, 3]].sum()
            beta[1] = self.prior[[0, 2]].sum()
        total_array = np.asarray(total)
        a, b = alpha + marginal, beta + (total_array[..., None] - marginal)
        posterior = np.stack(
            (betaincc(a, b, self.success_lrv), betaincc(a, b, self.success_cmv)), axis=-1
        )
        # Symmetry gives an exact half probability; preserve strict cutoff semantics.
        for j, threshold in enumerate((self.success_lrv, self.success_cmv)):
            posterior[..., j] = np.where((a == b) & (threshold == 0.5), 0.5, posterior[..., j])
        if self.endpoint == "efficacy_toxicity":
            toxicity = total_array - marginal[..., 1]
            toxicity_a = self.prior[[0, 2]].sum() + toxicity
            toxicity_b = self.prior[[1, 3]].sum() + marginal[..., 1]
            posterior[..., 1, 0] = betainc(toxicity_a, toxicity_b, self.lrv[1])
            posterior[..., 1, 1] = betainc(toxicity_a, toxicity_b, self.cmv[1])
            for j, threshold in enumerate((self.lrv[1], self.cmv[1])):
                posterior[..., 1, j] = np.where(
                    (toxicity_a == toxicity_b) & (threshold == 0.5), 0.5, posterior[..., 1, j]
                )
        return posterior

    def _marginal_counts(self, counts: np.ndarray) -> np.ndarray:
        if self.endpoint == "multiple_efficacy":
            return np.stack(
                (counts[..., 0] + counts[..., 1], counts[..., 0] + counts[..., 2]), axis=-1
            )
        # Categories are (efficacious,toxic), (efficacious,not toxic),
        # (not efficacious,toxic), (not efficacious,not toxic).
        return np.stack((counts[..., 0] + counts[..., 1], counts[..., 1] + counts[..., 3]), axis=-1)

    def _posterior(self, counts: np.ndarray) -> np.ndarray:
        return self._posterior_marginal(self._marginal_counts(counts), counts.sum(axis=-1))

    def operating_characteristics(
        self, category_probability: ArrayLike
    ) -> BOP2DCPairedOperatingCharacteristics:
        p = finite(category_probability, "category_probability")
        if p.ndim == 0 or p.shape[-1] != 4 or np.any((p < 0) | (p > 1)):
            raise ValueError("category_probability requires a final category axis of length four")
        total_probability = p.sum(axis=-1)
        if np.any(np.abs(total_probability - 1) > 1e-14):
            raise ValueError("category probabilities must sum to one")
        scenarios = p.size // 4
        if scenarios * (self.max_subjects + 1) ** 3 > 5_000_000:
            raise ValueError("scenario workload too large; split into smaller batches")
        p = p / total_probability[..., None]
        probabilities = p.reshape(scenarios, 4)
        stop = np.zeros((scenarios, self.looks.size))
        final_go = np.zeros(scenarios)
        final_consider = np.zeros(scenarios)
        final_no_go = np.zeros(scenarios)
        increments = ((1, 1), (1, 0), (0, 1), (0, 0))
        if self.endpoint == "efficacy_toxicity":
            increments = ((1, 0), (1, 1), (0, 0), (0, 1))
        interim_looks = {int(look): j for j, look in enumerate(self.looks[:-1])}
        for s, probabilities_s in enumerate(probabilities):
            surviving = np.ones((1, 1))
            for n in range(1, self.max_subjects + 1):
                arriving = np.zeros((n + 1, n + 1))
                for probability, (first, second) in zip(probabilities_s, increments):
                    arriving[first : first + n, second : second + n] += surviving * probability
                if n in interim_looks:
                    m1: np.ndarray
                    m2: np.ndarray
                    m1, m2 = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
                    posterior = self._posterior_marginal(np.stack((m1, m2), axis=-1), n)
                    lrv_cut = self.lambda_lrv * (n / self.max_subjects) ** self.gamma_lrv
                    cmv_cut = self.lambda_cmv * (n / self.max_subjects) ** self.gamma_cmv
                    no_go = np.all(
                        self._strict_below(posterior[..., :, 0], lrv_cut)
                        & self._strict_below(posterior[..., :, 1], cmv_cut),
                        axis=-1,
                    )
                    if self.endpoint == "efficacy_toxicity":
                        no_go = np.any(
                            self._strict_below(posterior[..., :, 0], lrv_cut)
                            & self._strict_below(posterior[..., :, 1], cmv_cut),
                            axis=-1,
                        )
                    j = interim_looks[n]
                    stop[s, j] = arriving[no_go].sum()
                    arriving[no_go] = 0
                surviving = arriving
            m1, m2 = np.meshgrid(
                np.arange(self.max_subjects + 1),
                np.arange(self.max_subjects + 1),
                indexing="ij",
            )
            posterior = self._posterior_marginal(np.stack((m1, m2), axis=-1), self.max_subjects)
            go = self._strict_above(posterior[..., :, 0], self.lambda_lrv) & self._strict_above(
                posterior[..., :, 1], self.lambda_cmv
            )
            no_go = self._strict_below(posterior[..., :, 0], self.lambda_lrv) & self._strict_below(
                posterior[..., :, 1], self.lambda_cmv
            )
            if self.endpoint == "multiple_efficacy":
                final_go[s] = surviving[np.any(go, axis=-1)].sum()
                final_no_go[s] = surviving[np.all(no_go, axis=-1)].sum()
                consider = ~(np.any(go, axis=-1) | np.all(no_go, axis=-1))
            else:
                final_go[s] = surviving[np.all(go, axis=-1)].sum()
                final_no_go[s] = surviving[np.any(no_go, axis=-1)].sum()
                consider = ~(np.all(go, axis=-1) | np.any(no_go, axis=-1))
            final_consider[s] = surviving[consider].sum()
        sample_size_probability = stop.copy()
        sample_size_probability[:, -1] = final_go + final_consider + final_no_go
        expected = sample_size_probability @ self.looks
        shape = p.shape[:-1]
        return BOP2DCPairedOperatingCharacteristics(
            _owned(p),
            self.looks,
            _owned(stop.reshape(*shape, self.looks.size)),
            _owned(final_go.reshape(shape)),
            _owned(final_consider.reshape(shape)),
            _owned(final_no_go.reshape(shape)),
            _owned(sample_size_probability.reshape(*shape, self.looks.size)),
            _owned(expected.reshape(shape)),
        )

    def monitor(self, counts: ArrayLike) -> BOP2DCPairedState:
        x = count(counts, "counts")
        if x.ndim == 0 or x.shape[-1] != 4:
            raise ValueError("counts must have final category axis of length four")
        if x[..., 0].size > 100_000:
            raise ValueError("count batch exceeds 100000 scenarios")
        if np.any(x.sum(axis=-1) > self.max_subjects):
            raise ValueError("total counts exceed max_subjects")
        x = x.astype(np.int64)
        n = x.sum(axis=-1).astype(np.int64)
        posterior = self._posterior(x)
        endpoint_decision = np.full((*n.shape, 2), "continue", dtype="U16")
        final = n == self.max_subjects
        for j in range(2):
            for look in self.looks[:-1]:
                at = n == look
                bad = (
                    self._strict_below(
                        posterior[..., j, 0],
                        self.lambda_lrv[j] * (look / self.max_subjects) ** self.gamma_lrv[j],
                    )
                ) & (
                    self._strict_below(
                        posterior[..., j, 1],
                        self.lambda_cmv[j] * (look / self.max_subjects) ** self.gamma_cmv[j],
                    )
                )
                endpoint_decision[..., j][at & bad] = "no_go"
            go = (
                final
                & self._strict_above(posterior[..., j, 0], self.lambda_lrv[j])
                & self._strict_above(posterior[..., j, 1], self.lambda_cmv[j])
            )
            no = (
                final
                & self._strict_below(posterior[..., j, 0], self.lambda_lrv[j])
                & self._strict_below(posterior[..., j, 1], self.lambda_cmv[j])
            )
            endpoint_decision[..., j][go] = "go"
            endpoint_decision[..., j][no] = "no_go"
            endpoint_decision[..., j][final & ~(go | no)] = "consider"
        decision = np.full(n.shape, "continue", dtype="U24")
        if self.endpoint == "multiple_efficacy":
            interim_bad = endpoint_decision[..., 0] == "no_go"
            interim_bad &= endpoint_decision[..., 1] == "no_go"
            final_go = final & (
                (endpoint_decision[..., 0] == "go") | (endpoint_decision[..., 1] == "go")
            )
            final_no = final & np.all(endpoint_decision == "no_go", axis=-1)
        else:
            interim_bad = np.any(endpoint_decision == "no_go", axis=-1)
            final_go = final & np.all(endpoint_decision == "go", axis=-1)
            final_no = final & np.any(endpoint_decision == "no_go", axis=-1)
        decision[(n < self.max_subjects) & interim_bad] = "stop_no_go"
        decision[final_go] = "final_go"
        decision[final_no] = "final_no_go"
        decision[final & ~(final_go | final_no)] = "final_consider"
        endpoint_decision.flags.writeable = False
        decision.flags.writeable = False
        return BOP2DCPairedState(
            _owned(n),
            _owned(x),
            _owned(x + self.prior),
            _owned(posterior),
            endpoint_decision,
            decision,
        )


def bop2_dc_paired_design(
    max_subjects: int,
    endpoint: str,
    lrv: ArrayLike,
    cmv: ArrayLike,
    *,
    lambda_lrv: ArrayLike = (0.9, 0.9),
    lambda_cmv: ArrayLike = (0.5, 0.5),
    gamma_lrv: ArrayLike = (0.5, 0.5),
    gamma_cmv: ArrayLike = (0.5, 0.5),
    prior: ArrayLike | None = None,
    looks: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2DCPairedDesign:
    if endpoint not in ("multiple_efficacy", "efficacy_toxicity"):
        raise ValueError("endpoint must be multiple_efficacy or efficacy_toxicity")
    n_value = scalar(max_subjects, "max_subjects")
    n = int(n_value)
    if n_value != n or not 1 <= n <= 1000:
        raise ValueError("max_subjects must be an integer in [1,1000]")
    lower, clinical = finite(lrv, "lrv"), finite(cmv, "cmv")
    for values, name in ((lower, "lrv"), (clinical, "cmv")):
        if values.shape != (2,) or np.any((values <= 0) | (values >= 1)):
            raise ValueError(f"{name} must contain two values in (0,1)")
    if endpoint == "multiple_efficacy" and np.any(lower >= clinical):
        raise ValueError("each efficacy lrv must be below its cmv")
    if endpoint == "efficacy_toxicity" and (lower[0] >= clinical[0] or not lower[1] > clinical[1]):
        raise ValueError("efficacy lrv must be below cmv and toxicity lrv must exceed cmv")
    arrays = [
        finite(v, name)
        for v, name in (
            (lambda_lrv, "lambda_lrv"),
            (lambda_cmv, "lambda_cmv"),
            (gamma_lrv, "gamma_lrv"),
            (gamma_cmv, "gamma_cmv"),
        )
    ]
    if (
        any(v.shape != (2,) for v in arrays)
        or np.any((arrays[0] <= 0) | (arrays[0] >= 1))
        or np.any((arrays[1] <= 0) | (arrays[1] >= 1))
        or np.any((arrays[2] < 0) | (arrays[2] > 1))
        or np.any((arrays[3] < 0) | (arrays[3] > 1))
    ):
        raise ValueError("cutoffs must be two values in (0,1), and gammas in [0,1]")
    if prior is None:
        p, q = lower
        shapes = np.array([p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)])
    else:
        shapes = finite(prior, "prior")
    if shapes.shape != (4,) or np.any(shapes <= 0) or not np.isfinite(shapes.sum()):
        raise ValueError("prior must contain four positive finite shapes")
    if looks is None:
        first_value, step_value = (
            scalar(min_subjects, "min_subjects"),
            scalar(cohort_size, "cohort_size"),
        )
        first, step = int(first_value), int(step_value)
        if first_value != first or step_value != step or first < 1 or step < 1 or first > n:
            raise ValueError("min_subjects and cohort_size must be valid integers")
        schedule = np.unique(np.r_[np.arange(first, n, step), n]).astype(np.int64)
    else:
        raw_schedule = count(looks, "looks")
        if (
            raw_schedule.ndim != 1
            or not raw_schedule.size
            or np.any(raw_schedule < 1)
            or np.any(raw_schedule > n)
        ):
            raise ValueError("looks must be nonempty and lie in [1,max_subjects]")
        schedule = raw_schedule.astype(np.int64)
        if schedule[-1] != n or np.any(np.diff(schedule) <= 0):
            raise ValueError("looks must increase and end at max_subjects")
    success_lrv, success_cmv = lower.copy(), clinical.copy()
    if endpoint == "efficacy_toxicity":
        success_lrv[1], success_cmv[1] = 1 - lower[1], 1 - clinical[1]
    return BOP2DCPairedDesign(
        endpoint,
        n,
        _owned(lower),
        _owned(clinical),
        _owned(success_lrv),
        _owned(success_cmv),
        _owned(arrays[0]),
        _owned(arrays[1]),
        _owned(arrays[2]),
        _owned(arrays[3]),
        _owned(shapes),
        _owned(schedule),
    )
