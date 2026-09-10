"""TOP delayed binary-response posterior decisions and effective-size boundaries."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import betaincc

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import BayesianMonitoringDesign, _inputs
from .boin import _owned
from .bop2_binary import bop2_binary_design
from .tite_keyboard import tite_effective_sample_size


@dataclass(frozen=True)
class TOPBinaryDecision:
    effective_sample_size: FloatArray
    posterior_alpha: FloatArray
    posterior_beta: FloatArray
    acceptable_probability: FloatArray
    acceptable_cutoff: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class TOPBinaryBoundaries:
    patients: NDArray[np.int64]
    complete_go_min: NDArray[np.int64]
    suspend_pending_min: NDArray[np.int64]
    futility_effective_size: FloatArray


@dataclass(frozen=True)
class TOPBinaryDesign:
    max_subjects: int
    null_rate: float
    cutoff_scale: float
    gamma: float
    prior: ArrayLike | None = None
    looks: ArrayLike | None = None
    suspension: str = "table"

    def __post_init__(self) -> None:
        p, c, g = (
            scalar(v, name)
            for v, name in [
                (self.null_rate, "null_rate"),
                (self.cutoff_scale, "cutoff_scale"),
                (self.gamma, "gamma"),
            ]
        )
        if not 0 < p < 1 or not 0 < c < 1 or not 0 <= g <= 1:
            raise ValueError("require null_rate/cutoff_scale in (0,1) and gamma in [0,1]")
        n, prior, looks = _inputs(
            self.max_subjects,
            (p, 1 - p) if self.prior is None else self.prior,
            self.looks,
            min(10, self.max_subjects),
            5,
        )
        if n > 200 or self.suspension not in ("table", "strict"):
            raise ValueError("require at most 200 subjects and suspension='table' or 'strict'")
        object.__setattr__(self, "max_subjects", n)
        object.__setattr__(self, "null_rate", p)
        object.__setattr__(self, "cutoff_scale", c)
        object.__setattr__(self, "gamma", g)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "looks", looks)

    def _cutoff(self, n: ArrayLike) -> FloatArray:
        return self.cutoff_scale * (np.asarray(n) / self.max_subjects) ** self.gamma

    def _pending_min(self, n: ArrayLike) -> NDArray[np.int64]:
        n = np.asarray(n, dtype=np.int64)
        # Table 4 uses >= n^2/N, while the prose says strictly > n^2/N.
        minimum = (
            (n * n + self.max_subjects - 1) // self.max_subjects
            if self.suspension == "table"
            else n * n // self.max_subjects + 1
        )
        return np.where(n == self.max_subjects, 1, minimum)

    def evaluate(
        self,
        patients: ArrayLike,
        responses: ArrayLike,
        pending: ArrayLike,
        pending_weight: ArrayLike,
    ) -> TOPBinaryDecision:
        """Batched decisions at scheduled looks, with summed pending time weights.

        Each pending weight is conditional time-to-response CDF in [0,1]. Uniform
        timing uses follow-up/window. The beta posterior is TOP's approximation.
        Final decisions wait for all outcomes. Suspension precedes futility when
        observed responses do not already satisfy the complete-data go boundary.
        """
        n, r, m, w = np.broadcast_arrays(
            count(patients, "patients"),
            count(responses, "responses"),
            count(pending, "pending"),
            finite(pending_weight, "pending_weight"),
        )
        if (
            n.size > 2000000
            or np.any(~np.isin(n, np.asarray(self.looks, dtype=np.int64)))
            or np.any((r + m > n) | (w < 0) | (w > m))
        ):
            raise ValueError(
                "require scheduled counts, responses+pending<=patients and 0<=weight<=pending"
            )
        a, b = np.asarray(self.prior)
        effective = n - m + w
        alpha, beta = a + r, b + effective - r
        probability = betaincc(alpha, beta, self.null_rate)
        complete_probability = betaincc(a + r, b + n - r, self.null_rate)
        cutoff = self._cutoff(n)
        if np.any(~np.isfinite(probability)) or np.any(~np.isfinite(complete_probability)):
            raise ArithmeticError("posterior beta probability evaluation failed")
        final = n == self.max_subjects
        suspend = (final & (m > 0)) | (
            (~final) & (complete_probability < cutoff) & (m >= self._pending_min(n))
        )
        action = np.where(
            suspend,
            "suspend",
            np.where(probability < cutoff, "stop_futility", np.where(final, "success", "continue")),
        )
        return TOPBinaryDecision(
            *map(_owned, (effective, alpha, beta, probability, cutoff, action))
        )

    def evaluate_followup(
        self,
        nonpending: ArrayLike,
        responses: ArrayLike,
        pending_followup: ArrayLike,
        window: float,
    ) -> TOPBinaryDecision:
        """Uniform conditional response timing; pending patients occupy the last axis."""
        ess = tite_effective_sample_size(nonpending, pending_followup, window)
        observed = count(nonpending, "nonpending")
        pending = ess.pending_weights.shape[-1]
        return self.evaluate(observed + pending, responses, pending, ess.pending_weights.sum(-1))

    def boundaries(self) -> TOPBinaryBoundaries:
        """Exact ESS crossing by response count; stop strictly above the crossing.

        Rows are looks; columns are observed response counts 0..max_subjects.
        NaN marks impossible counts; infinity means no crossing up to enrolled n.
        Negative infinity means futility throughout the feasible ESS domain.
        Suspension is applied separately.
        """
        looks = np.asarray(self.looks, dtype=np.int64)
        roots = np.full((looks.size, self.max_subjects + 1), np.nan)
        go = np.empty(looks.size, dtype=np.int64)
        a, b = np.asarray(self.prior)
        for i, n in enumerate(looks):
            cutoff = float(self._cutoff(n))
            responses = np.arange(n + 1)
            complete = betaincc(a + responses, b + n - responses, self.null_rate)
            accepted = np.flatnonzero(complete >= cutoff)
            go[i] = accepted[0] if accepted.size else n + 1
            for r in responses:
                if complete[r] >= cutoff:
                    roots[i, r] = np.inf
                elif betaincc(a + r, b, self.null_rate) < cutoff:
                    roots[i, r] = -np.inf
                else:
                    roots[i, r] = brentq(
                        lambda size: betaincc(a + r, b + size - r, self.null_rate) - cutoff,
                        float(r),
                        float(n),
                        xtol=1e-11,
                    )
        return TOPBinaryBoundaries(
            _owned(looks), _owned(go), _owned(self._pending_min(looks)), _owned(roots)
        )

    def complete_data_design(self) -> BayesianMonitoringDesign:
        """Equivalent BOP2 design for complete outcomes, with its exact OC methods."""
        return bop2_binary_design(
            self.max_subjects,
            self.null_rate,
            cutoff_scale=self.cutoff_scale,
            gamma=self.gamma,
            looks=self.looks,
            prior=self.prior,
        )
