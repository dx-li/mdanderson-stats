"""Marginal Bayesian monitoring for the MD Anderson Multc Lean design."""

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior, _owned
from .beta_comparison import compare_beta_difference

type IntArray = NDArray[np.int64]


def _int_owned(value: ArrayLike) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


def _str_owned(value: ArrayLike) -> NDArray[np.str_]:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MultcState:
    sample_size: FloatArray
    responses: FloatArray
    toxicities: FloatArray
    response_probability: FloatArray
    toxicity_probability: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class MultcBoundaries:
    """Raw marginal boundaries at scheduled looks; values -1/N+1 mean never/all."""

    looks: IntArray
    response_stop_max: IntArray
    toxicity_stop_min: IntArray


@dataclass(frozen=True)
class MultcPotentialBoundaries:
    """Compact count intervals reachable from surviving paths at each look."""

    looks: IntArray
    reachable_response: tuple[tuple[int, int], ...]
    reachable_toxicity: tuple[tuple[int, int], ...]
    response_stop: tuple[tuple[int, int], ...]
    toxicity_stop: tuple[tuple[int, int], ...]
    stop_both: tuple[tuple[tuple[int, int], tuple[int, int]], ...]


@dataclass(frozen=True)
class MultcOperatingCharacteristics:
    scenario_probability: FloatArray
    looks: IntArray
    stop_response_only: FloatArray
    stop_toxicity_only: FloatArray
    stop_both: FloatArray
    cap_completion: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    sample_size_sd: FloatArray
    expected_responses: FloatArray
    expected_toxicities: FloatArray


@dataclass(frozen=True)
class _Historical:
    constant: float | None
    prior: BetaBinomialPosterior | None


@dataclass(frozen=True)
class MultcLeanDesign:
    max_subjects: int
    response_prior: BetaBinomialPosterior
    toxicity_prior: BetaBinomialPosterior
    historical_response: _Historical
    historical_toxicity: _Historical
    response_margin: float
    toxicity_margin: float
    response_cutoff: float
    toxicity_cutoff: float
    min_subjects: int
    cohort_size: int
    looks: IntArray
    pretrial_check: bool
    _bounds: MultcBoundaries

    def _prob(self, endpoint: str, n: int, events: int) -> float:
        response = endpoint == "response"
        prior = self.response_prior if response else self.toxicity_prior
        historical = self.historical_response if response else self.historical_toxicity
        margin = self.response_margin if response else self.toxicity_margin
        posterior = prior.update(events, n - events)
        if historical.constant is not None:
            threshold = historical.constant + margin
            if response:
                if threshold <= 0:
                    return 0.0
                if threshold >= 1:
                    return 1.0
                return float(betainc(float(posterior.alpha), float(posterior.beta), threshold))
            if threshold <= 0:
                return 1.0
            if threshold >= 1:
                return 0.0
            return float(betaincc(float(posterior.alpha), float(posterior.beta), threshold))
        assert historical.prior is not None
        cutoff = self.response_cutoff if response else self.toxicity_cutoff
        tolerance = 1e-8
        if cutoff in (0.0, 1.0):
            result = compare_beta_difference(
                historical.prior, posterior, margin, absolute_tolerance=1e-8
            )
            return float(result.below_margin if response else result.above_margin)
        for _ in range(3):
            result = compare_beta_difference(
                historical.prior, posterior, margin, absolute_tolerance=tolerance
            )
            probability = float(result.below_margin if response else result.above_margin)
            error = float(result.absolute_error)
            if error == 0 or abs(probability - cutoff) > error:
                return probability
            tolerance /= 100
        raise ArithmeticError(
            "Multc stopping decision remains unresolved after quadrature refinement"
        )

    def _stop(self, endpoint: str, probability: float) -> bool:
        response = endpoint == "response"
        cutoff = self.response_cutoff if response else self.toxicity_cutoff
        if cutoff == 1:
            return False
        if cutoff == 0:
            historical = self.historical_response if response else self.historical_toxicity
            threshold = historical.constant
            if threshold is not None:
                threshold += self.response_margin if response else self.toxicity_margin
                return threshold > 0 if response else threshold < 1
            return True
        return probability > cutoff

    def monitor(
        self, responses: ArrayLike, toxicities: ArrayLike, sample_size: ArrayLike
    ) -> MultcState:
        """Monitor cumulative response/toxicity counts at explicit scheduled sample sizes."""
        return self.monitor_counts(responses, toxicities, sample_size)

    def monitor_counts(
        self, responses: ArrayLike, toxicities: ArrayLike, sample_size: ArrayLike
    ) -> MultcState:
        out_shape = np.broadcast_shapes(
            np.shape(responses), np.shape(toxicities), np.shape(sample_size)
        )
        if prod(out_shape) > 10_000:
            raise ValueError("Multc monitor broadcast exceeds the 10,000-snapshot limit")
        r, t, n = np.broadcast_arrays(
            count(responses, "responses"),
            count(toxicities, "toxicities"),
            count(sample_size, "sample_size"),
        )
        if np.any(n > self.max_subjects) or np.any(r > n) or np.any(t > n):
            raise ValueError("event counts must not exceed sample size or max_subjects")
        pr, pt = np.empty(r.shape), np.empty(r.shape)
        decisions = np.full(r.shape, "continue", dtype="U24")
        for ix in np.ndindex(r.shape):
            ni, ri, ti = int(n[ix]), int(r[ix]), int(t[ix])
            if ni == 0:
                pr[ix] = self._prob("response", 0, 0)
                pt[ix] = self._prob("toxicity", 0, 0)
            elif ni not in self.looks:
                raise ValueError("sample_size must be a scheduled look")
            else:
                pr[ix], pt[ix] = self._prob("response", ni, ri), self._prob("toxicity", ni, ti)
            stop_r = self._stop("response", float(pr[ix]))
            stop_t = self._stop("toxicity", float(pt[ix]))
            if ni == 0:
                if not self.pretrial_check:
                    stop_r = stop_t = False
            if ni == 0 and self.pretrial_check:
                if stop_r and stop_t:
                    decisions[ix] = "stop_both"
                elif stop_r:
                    decisions[ix] = "stop_response"
                elif stop_t:
                    decisions[ix] = "stop_toxicity"
            elif ni < self.max_subjects:
                if stop_r and stop_t:
                    decisions[ix] = "stop_both"
                elif stop_r:
                    decisions[ix] = "stop_response"
                elif stop_t:
                    decisions[ix] = "stop_toxicity"
            elif ni == self.max_subjects:
                decisions[ix] = "cap_complete"
        return MultcState(
            _owned(n), _owned(r), _owned(t), _owned(pr), _owned(pt), _str_owned(decisions)
        )

    def monitor_outcomes(self, outcomes: ArrayLike) -> MultcState:
        outcome_shape = np.shape(outcomes)
        if len(outcome_shape) != 2 or outcome_shape[1] != 2 or outcome_shape[0] > self.max_subjects:
            raise ValueError("outcomes must have shape (n,2), with n <= max_subjects")
        x = count(outcomes, "outcomes")
        if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] > self.max_subjects:
            raise ValueError("outcomes must have shape (n,2), with n <= max_subjects")
        if np.any(x > 1):
            raise ValueError("outcomes must contain binary response/toxicity indicators")
        n = x.shape[0]
        cs = np.concatenate((np.zeros((1, 2)), np.cumsum(x, axis=0)), axis=0)
        sizes = np.arange(n + 1)
        mask = np.isin(sizes, self.looks) | ((sizes == 0) & self.pretrial_check)
        mask[0] = True
        sizes, cs = sizes[mask], cs[mask]
        result = self.monitor_counts(cs[:, 0], cs[:, 1], sizes)
        stopping = np.flatnonzero(
            np.isin(result.decision, ["stop_response", "stop_toxicity", "stop_both"])
        )
        if stopping.size:
            stop_at = int(stopping[0])
            stop_at += 1
            return MultcState(
                result.sample_size[:stop_at],
                result.responses[:stop_at],
                result.toxicities[:stop_at],
                result.response_probability[:stop_at],
                result.toxicity_probability[:stop_at],
                _str_owned(result.decision[:stop_at]),
            )
        return result

    def stopping_bounds(self) -> MultcBoundaries:
        return self._bounds

    def potential_boundaries(self) -> MultcPotentialBoundaries:
        if self.pretrial_check and (
            self._stop("response", self._prob("response", 0, 0))
            or self._stop("toxicity", self._prob("toxicity", 0, 0))
        ):
            empty = tuple((1, 0) for _ in self.looks)
            return MultcPotentialBoundaries(
                self.looks, empty, empty, empty, empty, tuple((e, e) for e in empty)
            )
        previous_n = 0
        live_r = (0, 0)
        live_t = (0, 0)
        can_live = True
        reachable_r, reachable_t, response, toxicity, both = [], [], [], [], []
        for n in self.looks:
            j = int(np.searchsorted(self.looks, n))
            n1 = int(n)
            gap = n1 - previous_n
            reach_r = (live_r[0], live_r[1] + gap) if can_live else (1, 0)
            reach_t = (live_t[0], live_t[1] + gap) if can_live else (1, 0)
            reachable_r.append(reach_r)
            reachable_t.append(reach_t)
            rmax = int(self._bounds.response_stop_max[j])
            tmin = int(self._bounds.toxicity_stop_min[j])
            response.append((reach_r[0], min(reach_r[1], rmax)))
            toxicity.append((max(reach_t[0], tmin), reach_t[1]))
            both.append(((reach_r[0], min(reach_r[1], rmax)), (max(reach_t[0], tmin), reach_t[1])))
            live_r = (max(reach_r[0], rmax + 1), reach_r[1])
            live_t = (reach_t[0], min(reach_t[1], tmin - 1))
            can_live = live_r[0] <= live_r[1] and live_t[0] <= live_t[1]
            previous_n = n1
        return MultcPotentialBoundaries(
            self.looks,
            tuple(reachable_r),
            tuple(reachable_t),
            tuple(response),
            tuple(toxicity),
            tuple(both),
        )

    def operating_characteristics(
        self, scenario_probabilities: ArrayLike
    ) -> MultcOperatingCharacteristics:
        if np.shape(scenario_probabilities) != (4,):
            raise ValueError("scenario_probabilities must have four entries")
        p = finite(scenario_probabilities, "scenario_probabilities")
        if p.shape != (4,) or np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0):
            raise ValueError("scenario_probabilities must be [p11,p10,p01,p00] summing to one")
        if sum((n + 1) ** 2 for n in range(1, self.max_subjects + 1)) > 10_000_000:
            raise ValueError("Multc exact OC exceeds the 10-million-state calculation limit")
        if 2 * (self.max_subjects + 1) ** 2 * 8 > 4_000_000:
            raise ValueError("Multc exact OC working arrays exceed the 4 MB allocation limit")
        dist = np.zeros(self.max_subjects + 1)
        rsp = tsp = both = cap = 0.0
        if self.pretrial_check:
            pre_r = self._stop("response", self._prob("response", 0, 0))
            pre_t = self._stop("toxicity", self._prob("toxicity", 0, 0))
            if pre_r or pre_t:
                rsp = float(pre_r and not pre_t)
                tsp = float(pre_t and not pre_r)
                both = float(pre_r and pre_t)
                dist[0] = 1.0
                mean = 0.0
                expected_r = 0.0
                expected_t = 0.0
                return MultcOperatingCharacteristics(
                    _owned(p),
                    self.looks,
                    _owned(rsp),
                    _owned(tsp),
                    _owned(both),
                    _owned(cap),
                    _owned(dist),
                    _owned(mean),
                    _owned(0.0),
                    _owned(expected_r),
                    _owned(expected_t),
                )
        prev = 0
        live: FloatArray = np.ones((1, 1), dtype=float)
        prev = 0
        for n0 in self.looks:
            n = int(n0)
            for step in range(prev + 1, n + 1):
                old = live
                live = np.zeros((step + 1, step + 1))
                live[1:, 1:] += old * p[0]
                live[1:, :-1] += old * p[1]
                live[:-1, 1:] += old * p[2]
                live[:-1, :-1] += old * p[3]
            j = int(np.searchsorted(self.looks, n))
            rmax = int(self._bounds.response_stop_max[j])
            tmin = int(self._bounds.toxicity_stop_min[j])
            w_r = float(live[: rmax + 1, :tmin].sum())
            w_t = float(live[rmax + 1 :, tmin:].sum())
            w_b = float(live[: rmax + 1, tmin:].sum())
            if n < self.max_subjects:
                rsp += w_r
                tsp += w_t
                both += w_b
                dist[n] += w_r + w_t + w_b
                live[: rmax + 1, :] = 0
                live[:, tmin:] = 0
            else:
                cap = float(live.sum())
                dist[n] += cap
            prev = n
        mean = float(dist @ np.arange(self.max_subjects + 1))
        variance = float(dist @ np.arange(self.max_subjects + 1) ** 2 - mean**2)
        # Bounded stopping time and iid paired outcomes give Wald's identity.
        expected_r = mean * float(p[0] + p[1])
        expected_t = mean * float(p[0] + p[2])
        return MultcOperatingCharacteristics(
            _owned(p),
            self.looks,
            _owned(rsp),
            _owned(tsp),
            _owned(both),
            _owned(cap),
            _owned(dist),
            _owned(mean),
            _owned(np.sqrt(np.maximum(variance, 0))),
            _owned(expected_r),
            _owned(expected_t),
        )

    def operating_characteristics_independent(
        self, response_rate: float, toxicity_rate: float
    ) -> MultcOperatingCharacteristics:
        pr, pt = scalar(response_rate, "response_rate"), scalar(toxicity_rate, "toxicity_rate")
        if not 0 <= pr <= 1 or not 0 <= pt <= 1:
            raise ValueError("true rates must lie in [0,1]")
        return self.operating_characteristics(
            [pr * pt, pr * (1 - pt), (1 - pr) * pt, (1 - pr) * (1 - pt)]
        )


def _historical(value: float | tuple[float, float], name: str) -> _Historical:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} beta prior must have two shapes")
        a, b = (scalar(v, name) for v in value)
        if not (0 < a <= 1000 and 0 < b <= 1000):
            raise ValueError(f"{name} beta shapes must lie in (0,1000]")
        return _Historical(None, BetaBinomialPosterior(a, b))
    p = scalar(value, name)
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return _Historical(p, None)


def multc_lean_design(
    max_subjects: int,
    response_prior: tuple[float, float],
    toxicity_prior: tuple[float, float],
    *,
    historical_response: float | tuple[float, float],
    historical_toxicity: float | tuple[float, float],
    response_margin: float = 0,
    toxicity_margin: float = 0,
    response_cutoff: float = 0.95,
    toxicity_cutoff: float = 0.95,
    min_subjects: int = 1,
    cohort_size: int = 1,
    pretrial_check: bool = True,
) -> MultcLeanDesign:
    """Build a frozen Multc Lean design using independent response/toxicity marginals."""
    ints = (max_subjects, min_subjects, cohort_size)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in ints):
        raise ValueError("sample-size controls must be integers")
    nmax, minimum, cohort = map(int, ints)
    if not 3 <= nmax <= 1000 or not 1 <= minimum <= nmax or cohort < 1 or nmax % cohort:
        raise ValueError(
            "require Nmax 3..1000, min_subjects 1..Nmax, and cohort_size dividing Nmax"
        )
    if minimum >= cohort and minimum % cohort:
        raise ValueError("min_subjects must be a cohort multiple or less than one cohort")
    priors = []
    for pair, name in ((response_prior, "response_prior"), (toxicity_prior, "toxicity_prior")):
        if len(pair) != 2:
            raise ValueError(f"{name} requires two shapes")
        a, b = (scalar(v, name) for v in pair)
        if not (0 < a <= 100 and 0 < b <= 100):
            raise ValueError(f"{name} shapes must lie in (0,100]")
        priors.append(BetaBinomialPosterior(a, b))
    mr, mt = scalar(response_margin, "response_margin"), scalar(toxicity_margin, "toxicity_margin")
    cr, ct = scalar(response_cutoff, "response_cutoff"), scalar(toxicity_cutoff, "toxicity_cutoff")
    if not -1 < mr < 1 or not -1 < mt < 1 or (mr < 0 and mt > 0):
        raise ValueError(
            "margins must lie in (-1,1), and cannot have response<0 and toxicity>0 together"
        )
    if not 0 <= cr <= 1 or not 0 <= ct <= 1:
        raise ValueError("cutoffs must lie in [0,1]")
    if not isinstance(pretrial_check, (bool, np.bool_)):
        raise ValueError("pretrial_check must be boolean")
    schedule: IntArray = np.arange(cohort, nmax + 1, cohort, dtype=np.int64)
    schedule = schedule[schedule >= minimum]
    if schedule.size == 0 or schedule[-1] != nmax:
        raise ValueError("scheduled looks must include max_subjects")
    hist_r = _historical(historical_response, "historical_response")
    hist_t = _historical(historical_toxicity, "historical_toxicity")
    # Temporary design supports the same verified comparison method used by monitoring.
    blank = MultcBoundaries(
        _int_owned(schedule),
        _int_owned(np.zeros(schedule.size)),
        _int_owned(np.zeros(schedule.size)),
    )
    design = MultcLeanDesign(
        nmax,
        priors[0],
        priors[1],
        hist_r,
        hist_t,
        mr,
        mt,
        cr,
        ct,
        minimum,
        cohort,
        _int_owned(schedule),
        bool(pretrial_check),
        blank,
    )
    comparisons = 0
    rb, tb = [], []
    for n0 in schedule:
        n = int(n0)
        lo, hi = -1, n + 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            comparisons += 1
            if comparisons > 20_000:
                raise ValueError("boundary search exceeds the 20,000-comparison limit")
            if design._stop("response", design._prob("response", n, mid)):
                lo = mid
            else:
                hi = mid
        rb.append(lo)
        lo, hi = -1, n + 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            comparisons += 1
            if comparisons > 20_000:
                raise ValueError("boundary search exceeds the 20,000-comparison limit")
            if design._stop("toxicity", design._prob("toxicity", n, mid)):
                hi = mid
            else:
                lo = mid
        tb.append(hi)
    bounds = MultcBoundaries(_int_owned(schedule), _int_owned(rb), _int_owned(tb))
    return MultcLeanDesign(
        nmax,
        priors[0],
        priors[1],
        hist_r,
        hist_t,
        mr,
        mt,
        cr,
        ct,
        minimum,
        cohort,
        _int_owned(schedule),
        bool(pretrial_check),
        bounds,
    )
