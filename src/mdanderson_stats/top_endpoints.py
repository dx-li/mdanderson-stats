"""TOP monitoring for co-primary efficacy and efficacy/toxicity endpoints.

The joint four-cell Dirichlet prior is retained, but TOP monitors each endpoint
with its own marginal Beta approximation. In particular, different pending
patterns do not define a joint fractional Dirichlet posterior.
"""

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _inputs
from .boin import _owned


@dataclass(frozen=True)
class TOPMultiEndpointDecision:
    """Endpoint posterior summaries and their source-defined combined action."""

    effective_sample_size: FloatArray
    posterior_alpha: FloatArray
    posterior_beta: FloatArray
    acceptable_probability: FloatArray
    cutoff: FloatArray
    endpoint_status: NDArray[np.str_]
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class TOPMultiEndpointBoundaries:
    """Complete-data count limits and fractional-ESS crossings by endpoint."""

    patients: NDArray[np.int64]
    complete_event_threshold: NDArray[np.int64]
    suspend_pending_min: NDArray[np.int64]
    effective_size_crossing: FloatArray


def _pair(value: ArrayLike, name: str, *, positive: bool) -> FloatArray:
    if np.shape(value) != (2,):
        raise ValueError(f"{name} must contain two values")
    result = finite(value, name).copy()
    if result.shape != (2,) or np.any(result <= 0 if positive else result < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must contain two {bound} values")
    return result


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _readonly_int(value: ArrayLike) -> NDArray[np.int64]:
    result = np.array(value, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class TOPMultiEndpointDesign:
    """Two-endpoint TOP design using endpoint-specific marginal monitoring.

    `null_joint_probabilities` and `prior_concentration` define Dirichlet
    parameters in the cell order ``(1,1), (1,0), (0,1), (0,0)``. The two binary
    coordinates are the monitored endpoints. Their marginal prior shapes are
    aggregated from that joint prior; after pending-data weighting, each margin
    receives TOP's separate fractional Beta update.

    ``mode='coprimary'`` combines endpoint acceptability with an OR rule.
    ``mode='efficacy_toxicity'`` stops for either endpoint's adverse conclusion.
    ``suspension='table'`` uses ``ceil(n²/N)`` pending patients; ``'strict'``
    uses ``floor(n²/N)+1``. At the final look, any pending endpoint is suspended
    before the source combination rule is applied. This can permit a co-primary
    success while the other endpoint remains pending.
    """

    max_subjects: int
    null_joint_probabilities: ArrayLike
    cutoff_scale: float
    gamma: float
    mode: str = "coprimary"
    prior_concentration: float = 1.0
    windows: ArrayLike = (1.0, 1.0)
    timing_probabilities: ArrayLike | None = None
    looks: ArrayLike | None = None
    suspension: str = "table"

    def __post_init__(self) -> None:
        scale = scalar(self.cutoff_scale, "cutoff_scale")
        gamma = scalar(self.gamma, "gamma")
        concentration = scalar(self.prior_concentration, "prior_concentration")
        if not 0 < scale < 1 or not 0 <= gamma <= 1 or concentration <= 0:
            raise ValueError(
                "require cutoff_scale in (0,1), gamma in [0,1], and positive prior concentration"
            )
        if self.mode not in ("coprimary", "efficacy_toxicity"):
            raise ValueError("mode must be 'coprimary' or 'efficacy_toxicity'")
        if self.suspension not in ("table", "strict"):
            raise ValueError("suspension must be 'table' or 'strict'")

        if np.shape(self.null_joint_probabilities) != (4,):
            raise ValueError("null_joint_probabilities must contain four cell probabilities")
        joint = finite(self.null_joint_probabilities, "null_joint_probabilities")
        if joint.shape != (4,) or np.any(joint <= 0):
            raise ValueError(
                "null_joint_probabilities must contain four positive cell probabilities"
            )
        total = float(joint.sum())
        if not np.isfinite(total) or abs(total - 1.0) > 32 * np.finfo(float).eps:
            raise ValueError("null_joint_probabilities must sum to one")
        joint = joint / total
        windows = _pair(self.windows, "windows", positive=True)
        if self.timing_probabilities is None:
            timing = np.full((2, 3), 1 / 3, dtype=float)
        else:
            if np.shape(self.timing_probabilities) not in ((3,), (2, 3)):
                raise ValueError("timing_probabilities must be a triple or a 2-by-3 array")
            timing = finite(self.timing_probabilities, "timing_probabilities").copy()
            if timing.shape == (3,):
                timing = np.broadcast_to(timing, (2, 3)).copy()
            if (
                timing.shape != (2, 3)
                or np.any(timing < 0)
                or np.any(np.abs(timing.sum(axis=1) - 1) > 32 * np.finfo(float).eps)
            ):
                raise ValueError(
                    "timing_probabilities must be two nonnegative triples summing to one"
                )
            timing /= timing.sum(axis=1, keepdims=True)
        n, _, looks = _inputs(
            self.max_subjects,
            (1.0, 1.0),
            self.looks,
            min(10, self.max_subjects),
            5,
        )
        if n > 200:
            raise ValueError("max_subjects must not exceed 200")
        for value in (joint, windows, timing, looks):
            value.flags.writeable = False
        object.__setattr__(self, "max_subjects", n)
        object.__setattr__(self, "cutoff_scale", scale)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "prior_concentration", concentration)
        object.__setattr__(self, "null_joint_probabilities", joint)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "timing_probabilities", timing)
        object.__setattr__(self, "looks", looks)

    @property
    def marginal_null(self) -> FloatArray:
        """Null event rates for the two binary coordinates."""
        p = np.asarray(self.null_joint_probabilities)
        return _readonly_float((p[0] + p[1], p[0] + p[2]))

    @property
    def marginal_prior(self) -> FloatArray:
        """Endpoint Beta prior shapes ``(alpha, beta)`` in endpoint order."""
        p = np.asarray(self.null_joint_probabilities)
        total = self.prior_concentration
        prior = np.asarray(
            (
                (total * (p[0] + p[1]), total * (p[2] + p[3])),
                (total * (p[0] + p[2]), total * (p[1] + p[3])),
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(prior)) or np.any(prior <= 0):
            raise ValueError(
                "prior concentration produces nonfinite or underflowed marginal shapes"
            )
        return _readonly_float(prior)

    def _cutoff(self, patients: NDArray[np.int64]) -> FloatArray:
        return self.cutoff_scale * (patients / self.max_subjects) ** self.gamma

    def _pending_min(self, patients: NDArray[np.int64]) -> NDArray[np.int64]:
        n = patients.astype(np.int64, copy=False)
        if self.suspension == "table":
            minimum = (n * n + self.max_subjects - 1) // self.max_subjects
        else:
            minimum = n * n // self.max_subjects + 1
        return np.where(n == self.max_subjects, 1, minimum)

    def timing_weight(self, followup: ArrayLike) -> FloatArray:
        """Conditional event-time CDF for follow-up fractions over three thirds."""
        if len(np.shape(followup)) == 0 or np.shape(followup)[-1] != 2:
            raise ValueError("followup must have endpoint as its final axis")
        if prod(np.shape(followup)) > 2_000_000:
            raise ValueError("followup exceeds the 2,000,000-cell limit")
        t = finite(followup, "followup")
        if np.any(t < 0):
            raise ValueError("followup must have endpoint as its final axis and be nonnegative")
        windows = np.asarray(self.windows)
        if np.any(t > windows):
            raise ValueError("followup must not exceed its endpoint assessment window")
        u = t / windows
        pi = np.asarray(self.timing_probabilities)
        out = np.zeros_like(u)
        for third in range(3):
            out += pi[:, third] * np.clip(3 * u - third, 0, 1)
        out = np.clip(out, 0, 1)
        return _readonly_float(out)

    def evaluate(
        self,
        patients: ArrayLike,
        events: ArrayLike,
        pending: ArrayLike,
        pending_weight: ArrayLike,
    ) -> TOPMultiEndpointDecision:
        """Evaluate summary counts and summed time weights at scheduled looks.

        ``events``, ``pending`` and ``pending_weight`` end in the two endpoints;
        ``patients`` broadcasts over all preceding dimensions. Each pending
        patient's weight is its endpoint-specific conditional event-time CDF.
        """
        patient_shape = np.shape(patients)
        event_shape, pending_shape, weight_shape = map(np.shape, (events, pending, pending_weight))
        if any(
            len(shape) == 0 or shape[-1] != 2
            for shape in (event_shape, pending_shape, weight_shape)
        ):
            raise ValueError("events, pending and pending_weight require a final two-endpoint axis")
        try:
            batch_shape = np.broadcast_shapes(
                patient_shape, event_shape[:-1], pending_shape[:-1], weight_shape[:-1]
            )
        except ValueError as exc:
            raise ValueError("summary inputs must be broadcast-compatible") from exc
        shape = (*batch_shape, 2)
        if prod(shape) > 2_000_000:
            raise ValueError("summary inputs exceed the 2,000,000-cell limit")
        n_base = np.broadcast_to(count(patients, "patients"), batch_shape)
        n = np.broadcast_to(n_base[..., None], shape)
        event = np.broadcast_to(count(events, "events"), shape)
        miss = np.broadcast_to(count(pending, "pending"), shape)
        weight = np.broadcast_to(finite(pending_weight, "pending_weight"), shape)
        # n is provided per endpoint for vectorized batched APIs, but both margins
        # share a cohort size by design.
        if (
            np.any(n[..., 0] != n[..., 1])
            or np.any(n[..., 0] < 1)
            or np.any(n[..., 0] > self.max_subjects)
            or np.any(~np.isin(n[..., 0], np.asarray(self.looks)))
            or np.any(event + miss > n)
            or np.any((weight < 0) | (weight > miss))
        ):
            raise ValueError(
                "require scheduled patient counts, events+pending<=patients, and 0<=weight<=pending"
            )
        n1 = n[..., 0].astype(np.int64)
        cutoff = self._cutoff(n1)[..., None]
        prior = np.asarray(self.marginal_prior)
        null = np.asarray(self.marginal_null)
        ess = n - miss + weight
        alpha = prior[:, 0] + event
        failures = (n - miss - event) + weight
        beta = prior[:, 1] + failures
        complete_beta = prior[:, 1] + (n - event)
        acceptable = np.empty(shape, dtype=float)
        complete = np.empty(shape, dtype=float)
        efficacy_index = (0, 1) if self.mode == "coprimary" else (0,)
        toxicity_index = (1,) if self.mode == "efficacy_toxicity" else ()
        for j in efficacy_index:
            acceptable[..., j] = betaincc(alpha[..., j], beta[..., j], null[j])
            complete[..., j] = betaincc(alpha[..., j], complete_beta[..., j], null[j])
        for j in toxicity_index:
            acceptable[..., j] = betainc(alpha[..., j], beta[..., j], null[j])
            complete[..., j] = betainc(alpha[..., j], complete_beta[..., j], null[j])
        if np.any(~np.isfinite(acceptable) | ~np.isfinite(complete)):
            raise ArithmeticError("posterior beta probability evaluation failed")

        status = np.where(acceptable < cutoff, "unacceptable", "acceptable").astype("<U12")
        final = n1 == self.max_subjects
        min_pending = self._pending_min(n1)[..., None]
        final_suspension = final[..., None] & (miss > 0)
        suspension_complete = complete < cutoff
        if self.mode == "efficacy_toxicity":
            suspension_complete = suspension_complete.copy()
            suspension_complete[..., 1] = complete[..., 1] >= cutoff[..., 0]
        interim_suspension = ~final[..., None] & (miss >= min_pending) & suspension_complete
        status[final_suspension | interim_suspension] = "suspend"

        if self.mode == "coprimary":
            has_acceptable = np.any(status == "acceptable", axis=-1)
            all_unacceptable = np.all(status == "unacceptable", axis=-1)
            decision = np.where(
                has_acceptable,
                np.where(final, "success", "continue"),
                np.where(all_unacceptable, "stop_futility", "suspend"),
            )
        else:
            efficacy_unacceptable = status[..., 0] == "unacceptable"
            toxicity_unacceptable = status[..., 1] == "unacceptable"
            both_acceptable = np.all(status == "acceptable", axis=-1)
            decision = np.where(
                efficacy_unacceptable & toxicity_unacceptable,
                "stop_futility_toxicity",
                np.where(
                    toxicity_unacceptable,
                    "stop_toxicity",
                    np.where(efficacy_unacceptable, "stop_futility", ""),
                ),
            )
            decision = np.where(
                decision != "",
                decision,
                np.where(both_acceptable, np.where(final, "success", "continue"), "suspend"),
            )
        return TOPMultiEndpointDecision(
            _readonly_float(ess),
            _readonly_float(alpha),
            _readonly_float(beta),
            _readonly_float(acceptable),
            _readonly_float(np.broadcast_to(cutoff, shape)),
            _owned(status),
            _owned(decision),
        )

    def evaluate_followup(
        self, outcomes: ArrayLike, followup: ArrayLike
    ) -> TOPMultiEndpointDecision:
        """Evaluate patient outcomes with endpoint-specific pending follow-up.

        ``outcomes`` and ``followup`` have shape ``(..., patients, 2)``. Outcomes
        are 0/1 or NaN for pending. Follow-up is measured from enrollment and may
        be supplied for every endpoint record; only pending values contribute.
        """
        yshape = np.shape(outcomes)
        fshape = np.shape(followup)
        if yshape != fshape or len(yshape) < 2 or yshape[-1] != 2:
            raise ValueError("outcomes and followup must have matching (..., patients, 2) shapes")
        if prod(yshape) > 2_000_000 or yshape[-2] > self.max_subjects:
            raise ValueError("patient-level input exceeds the 2,000,000-cell or patient limit")
        y = np.asarray(outcomes, dtype=np.float64)
        if np.any(np.isinf(y)) or np.any(~(np.isnan(y) | (y == 0) | (y == 1))):
            raise ValueError("outcomes must contain only 0, 1, or NaN for pending")
        pending_mask = np.isnan(y)
        f = np.asarray(followup, dtype=float)
        pending_followup = f[pending_mask]
        if np.any(~np.isfinite(pending_followup)) or np.any(pending_followup < 0):
            raise ValueError("pending followup must be finite and nonnegative")
        pending_windows = np.broadcast_to(np.asarray(self.windows), f.shape)[pending_mask]
        if np.any(pending_followup > pending_windows):
            raise ValueError("pending followup must not exceed its endpoint assessment window")
        safe_followup = np.where(pending_mask, f, 0.0)
        weights = self.timing_weight(safe_followup)
        events = np.nansum(y, axis=-2)
        pending = pending_mask.sum(axis=-2)
        weight = np.where(pending_mask, weights, 0).sum(axis=-2)
        return self.evaluate(yshape[-2], events, pending, weight)

    def boundaries(self) -> TOPMultiEndpointBoundaries:
        """Complete-data event thresholds and fractional effective-size crossings.

        For efficacy, `complete_event_threshold` is the minimum response count
        meeting endpoint acceptability. For toxicity it is the minimum DLT count
        making the endpoint unacceptable. Crossing entries are NaN when there is
        no interior crossing in effective-size range [events, patients].
        """
        looks = np.asarray(self.looks, dtype=np.int64)
        thresholds = np.full((looks.size, 2), -1, dtype=np.int64)
        crossings = np.full((looks.size, 2, self.max_subjects + 1), np.nan)
        prior = np.asarray(self.marginal_prior)
        null = np.asarray(self.marginal_null)
        for i, n in enumerate(looks):
            cutoff = float(self.cutoff_scale * (n / self.max_subjects) ** self.gamma)
            for j in range(2):
                events = np.arange(n + 1, dtype=float)
                aa = prior[j, 0] + events
                bb = prior[j, 1] + (n - events)
                if self.mode == "efficacy_toxicity" and j == 1:
                    complete = betainc(aa, bb, null[j])
                    indices = np.flatnonzero(complete < cutoff)
                    tail = betainc
                else:
                    complete = betaincc(aa, bb, null[j])
                    indices = np.flatnonzero(complete >= cutoff)
                    tail = betaincc
                thresholds[i, j] = int(indices[0]) if indices.size else int(n + 1)
                for r in range(n + 1):

                    def f(eff: float) -> float:
                        return float(
                            tail(prior[j, 0] + r, prior[j, 1] + (eff - r), null[j]) - cutoff
                        )

                    left, right = float(r), float(n)
                    f_left, f_right = f(left), f(right)
                    if f_left == 0:
                        crossings[i, j, r] = left
                    elif f_right == 0:
                        crossings[i, j, r] = right
                    elif f_left * f_right < 0:
                        crossings[i, j, r] = brentq(f, left, right, xtol=1e-11)
        return TOPMultiEndpointBoundaries(
            _readonly_int(looks),
            _readonly_int(thresholds),
            _readonly_int(self._pending_min(looks)),
            _readonly_float(crossings),
        )
