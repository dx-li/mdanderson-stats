"""Named BCHM analyses with reproducible, saveable community reports."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .bchm import BCHMFit, bchm_fit
from .hierarchical_binomial import summarize_chains

_MAX_SCENARIOS = 20
_MAX_TOTAL_WORK = 1_000_000
_INPUT_FIELDS = (
    "scenario",
    "subgroup",
    "successes",
    "trials",
    "seed",
    "mu",
    "sigma02",
    "sigmaD2",
    "alpha",
    "d0",
    "alpha1",
    "beta1",
    "tau2",
    "phi1",
    "deltaT",
    "thetaT",
    "burn_in",
    "iterations",
    "draws",
    "warmup",
    "chains",
)


def _atomic_csv(
    destination: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, object]]
) -> None:
    if not destination.parent.is_dir():
        raise ValueError("CSV parent directory must already exist")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent, delete=False
        ) as stream:
            temporary = stream.name
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise


@dataclass(frozen=True)
class BCHMScenario:
    """One named subgroup dataset and its complete BCHM design and sampler request."""

    label: str
    successes: ArrayLike
    trials: ArrayLike
    seed: int
    mu: float = 0.2
    sigma02: float = 20.0
    sigmaD2: float = 0.01
    alpha: float = 0.001
    d0: float = 0.0
    alpha1: float = 30.0
    beta1: float = 6.0
    tau2: float = 0.1
    phi1: float = 0.2
    deltaT: float = 0.15
    thetaT: float = 0.5
    burn_in: int = 1000
    iterations: int = 2000
    draws: int = 1000
    warmup: int = 500
    chains: int = 2
    subgroup_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.label, str)
            or not self.label.strip()
            or len(self.label) > 120
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in self.label)
        ):
            raise ValueError("label must be a nonempty string of at most 120 characters")
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or not 0 <= self.seed <= 2**63 - 1
        ):
            raise ValueError("seed must be an integer in [0, 2**63-1]")
        y = np.asarray(self.successes)
        n = np.asarray(self.trials)
        if y.ndim != 1 or n.shape != y.shape or y.size < 1:
            raise ValueError("successes and trials must be matching vectors")
        object.__setattr__(self, "successes", tuple(y.tolist()))
        object.__setattr__(self, "trials", tuple(n.tolist()))
        labels = self.subgroup_labels
        if labels is None:
            labels = tuple(f"group_{i + 1}" for i in range(y.size))
        if (
            not isinstance(labels, (tuple, list))
            or len(labels) != y.size
            or any(
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 120
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
                for value in labels
            )
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("subgroup_labels must be unique nonempty strings matching the data")
        object.__setattr__(self, "subgroup_labels", tuple(labels))
        from ._validation import scalar
        from .bchm import _borrow_parameters, _sampling, _validate

        _validate(self.successes, self.trials)
        _sampling(self.draws, self.warmup, self.chains, y.size, all_targets=True)
        _borrow_parameters(self.alpha1, self.beta1, self.tau2, self.phi1, self.deltaT, self.thetaT)
        mu, var0, vard, alpha = (
            scalar(v, name)
            for v, name in (
                (self.mu, "mu"),
                (self.sigma02, "sigma02"),
                (self.sigmaD2, "sigmaD2"),
                (self.alpha, "alpha"),
            )
        )
        if (
            abs(mu) > 1e4
            or not 1e-8 <= var0 <= 1e8
            or not 1e-8 <= vard <= 1e8
            or not 1e-300 <= alpha <= 1e100
        ):
            raise ValueError("clustering hyperparameters outside stable range")
        if not 0 <= scalar(self.d0, "d0") <= 1:
            raise ValueError("d0 must be in [0,1]")
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in (self.burn_in, self.iterations)
        ):
            raise ValueError("MCMC counts must be integers")
        if (
            self.burn_in < 0
            or self.iterations < 1
            or (self.burn_in + self.iterations) * y.size**2 > 2_000_000
            or self.iterations * y.size > 200_000
        ):
            raise ValueError("allocation MCMC budget too large")

    def write_input_csv(self, path: str | os.PathLike[str]) -> Path:
        """Save the documented Python scenario-input format, repeating settings per subgroup."""
        destination = Path(path)
        rows = []
        for label, y, n in zip(self.subgroup_labels, self.successes, self.trials, strict=True):
            rows.append(
                {
                    "scenario": self.label,
                    "subgroup": label,
                    "successes": y,
                    "trials": n,
                    **{name: getattr(self, name) for name in _INPUT_FIELDS[4:]},
                }
            )
        _atomic_csv(destination, _INPUT_FIELDS, rows)
        return destination

    @classmethod
    def from_input_csv(cls, path: str | os.PathLike[str]) -> BCHMScenario:
        """Read the Python-defined input CSV; all design values must be present and consistent."""
        source = Path(path)
        if source.stat().st_size > 262_144:
            raise ValueError("input CSV exceeds the 256 KiB limit")
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _INPUT_FIELDS:
                raise ValueError(
                    "input CSV headers do not match the documented BCHM scenario format"
                )
            rows = []
            for row in reader:
                if len(rows) == 20:
                    raise ValueError("input CSV must contain at most 20 subgroup rows")
                rows.append(row)
        if not rows or any(
            row.get(None) is not None
            or any(value is None for key, value in row.items() if key is not None)
            for row in rows
        ):
            raise ValueError("input CSV must contain 1..20 complete subgroup rows")
        settings: dict[str, object] = {}
        integer_names = {"seed", "burn_in", "iterations", "draws", "warmup", "chains"}
        try:
            for name in _INPUT_FIELDS[4:]:
                parsed = int(rows[0][name]) if name in integer_names else float(rows[0][name])
                if any(
                    (int(row[name]) if name in integer_names else float(row[name])) != parsed
                    for row in rows[1:]
                ):
                    raise ValueError(f"input CSV has inconsistent {name} values")
                settings[name] = parsed
            if any(not row["scenario"] == rows[0]["scenario"] for row in rows):
                raise ValueError("input CSV has inconsistent scenario labels")
            return cls(
                label=rows[0]["scenario"],
                subgroup_labels=tuple(row["subgroup"] for row in rows),
                successes=tuple(int(row["successes"]) for row in rows),
                trials=tuple(int(row["trials"]) for row in rows),
                **settings,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid BCHM scenario input CSV: {exc}") from exc


@dataclass(frozen=True)
class BCHMScenarioSummary:
    scenario: BCHMScenario
    fit: BCHMFit
    clustering_seed: int
    borrowing_seeds: tuple[int, ...]
    similarity_mcse: np.ndarray
    efficacy_mcse: np.ndarray
    efficacy_split_rhat: np.ndarray


def _array_text(value: ArrayLike) -> str:
    return np.array2string(
        np.asarray(value),
        precision=17,
        floatmode="unique",
        formatter={"float_kind": lambda item: f"{item:.17g}"},
    )


def _co_cluster_mcse(allocations: np.ndarray) -> np.ndarray:
    batch_size = max(2, int(np.sqrt(allocations.shape[0])))
    batch_count = allocations.shape[0] // batch_size
    if batch_count < 2:
        return np.full((allocations.shape[1], allocations.shape[1]), np.nan)
    retained = allocations[: batch_count * batch_size]
    result = np.zeros((allocations.shape[1], allocations.shape[1]))
    for left in range(allocations.shape[1]):
        for right in range(left, allocations.shape[1]):
            indicator = (retained[:, left] == retained[:, right]).reshape(batch_count, batch_size)
            mcse = np.std(indicator.mean(axis=1), ddof=1) / np.sqrt(batch_count)
            result[left, right] = result[right, left] = mcse
    return result


@dataclass(frozen=True)
class BCHMScenarioBatch:
    """Ordered scenario results that render or atomically save a plain-text report."""

    scenarios: tuple[BCHMScenarioSummary, ...]

    def report(self) -> str:
        lines = [
            "BCHM named-scenario report",
            "Python BCHM implementation; see the accompanying source crosswalk for limits.",
            "Native R/JAGS uses different RNG streams and chain handling.",
            "This report is not a native PDF or MCMC export.",
            "",
        ]
        for item in self.scenarios:
            s, fit = item.scenario, item.fit
            lines.extend(
                (
                    f"Scenario: {s.label}",
                    f"  seed={s.seed}; clustering_seed={item.clustering_seed};",
                    f"  borrowing_seeds={item.borrowing_seeds!r}",
                    f"  successes={s.successes!r}; trials={s.trials!r}",
                    f"  subgroup_labels={s.subgroup_labels!r}",
                    "  clustering_design="
                    + repr(
                        {
                            "mu": s.mu,
                            "sigma02": s.sigma02,
                            "sigmaD2": s.sigmaD2,
                            "alpha": s.alpha,
                            "d0": s.d0,
                            "burn_in": s.burn_in,
                            "iterations": s.iterations,
                        }
                    ),
                    "  borrowing_design="
                    + repr(
                        {
                            "alpha1": s.alpha1,
                            "beta1": s.beta1,
                            "tau2": s.tau2,
                            "phi1": s.phi1,
                            "deltaT": s.deltaT,
                            "thetaT": s.thetaT,
                            "prior_mean_logit": float(
                                np.log(np.mean(np.asarray(s.successes) / np.asarray(s.trials)))
                                - np.log1p(-np.mean(np.asarray(s.successes) / np.asarray(s.trials)))
                            ),
                            "tau1_prior": f"Gamma(shape={s.alpha1}, rate={s.beta1})",
                            "draws": s.draws,
                            "warmup": s.warmup,
                            "chains": s.chains,
                            "sampler": "Gaussian-prior elliptical slice; shared-mean and "
                            "precision Gibbs",
                        }
                    ),
                    f"  representative_partition={fit.cluster.result.representative!r};",
                    f"  representative_silhouette={fit.cluster.result.representative_score:.17g}",
                    f"  raw_similarity={_array_text(fit.raw_similarity)}",
                    f"  reported_similarity={_array_text(fit.similarity)}",
                    f"  borrowing_similarity={_array_text(fit.borrowing_similarity)}",
                    f"  similarity_batch_means_mcse={_array_text(item.similarity_mcse)}",
                    f"  posterior_mean={_array_text(fit.posterior_mean)}",
                    f"  efficacy_probability_raw={_array_text(fit.raw_probability)}",
                    f"  efficacy_probability_native_rounded={_array_text(fit.native_probability)}",
                    f"  efficacy_decision={_array_text(fit.decision)}",
                    f"  efficacy_indicator_batch_means_mcse={_array_text(item.efficacy_mcse)}",
                    f"  efficacy_indicator_split_rhat={_array_text(item.efficacy_split_rhat)}",
                )
            )
            for target, summary in enumerate(fit.summaries):
                lines.append(
                    f"  target_{target}_posterior_mean_mcse={summary.batch_mean_mcse[0]:.17g}; "
                    f"split_rhat={summary.split_rhat[0]:.17g}; "
                    f"80pct_interval={_array_text(summary.interval[0])}"
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write_report(self, path: str | os.PathLike[str]) -> Path:
        """Atomically write the report; parent directory must already exist."""
        destination = Path(path)
        if not destination.parent.is_dir():
            raise ValueError("report parent directory must already exist")
        text = self.report()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
            ) as stream:
                temporary = stream.name
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
            raise
        return destination

    def write_samples_csv(self, path: str | os.PathLike[str]) -> Path:
        """Stream retained target probabilities as scenario/subgroup/chain/draw/value rows."""
        destination = Path(path)
        fields = ("scenario", "subgroup", "target_index", "chain", "draw", "probability")

        def rows():
            for item in self.scenarios:
                for target, (subgroup, borrow) in enumerate(
                    zip(item.scenario.subgroup_labels, item.fit.borrowing, strict=True)
                ):
                    for chain, samples in enumerate(borrow.samples):
                        for draw, probability in enumerate(samples):
                            yield {
                                "scenario": item.scenario.label,
                                "subgroup": subgroup,
                                "target_index": target,
                                "chain": chain,
                                "draw": draw,
                                "probability": format(float(probability), ".17g"),
                            }

        _atomic_csv(destination, fields, rows())
        return destination


def fit_bchm_scenarios(scenarios: Sequence[BCHMScenario]) -> BCHMScenarioBatch:
    """Fit a bounded named batch of observed subgroup datasets."""
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ValueError("scenarios must be a sequence of BCHMScenario values")
    if not 1 <= len(scenarios) <= _MAX_SCENARIOS or any(
        not isinstance(s, BCHMScenario) for s in scenarios
    ):
        raise ValueError(f"scenarios must contain 1..{_MAX_SCENARIOS} BCHMScenario values")
    if len({s.label for s in scenarios}) != len(scenarios):
        raise ValueError("scenario labels must be unique")

    # Validate all requests and aggregate the core's dominant borrowing work before sampling.
    work = 0
    for s in scenarios:
        from .bchm import _borrow_parameters, _sampling, _validate

        y, n = _validate(s.successes, s.trials)
        _sampling(s.draws, s.warmup, s.chains, len(y), all_targets=True)
        _borrow_parameters(s.alpha1, s.beta1, s.tau2, s.phi1, s.deltaT, s.thetaT)
        for value, name in ((s.sigma02, "sigma02"), (s.sigmaD2, "sigmaD2"), (s.alpha, "alpha")):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        if not 0 <= s.d0 <= 1:
            raise ValueError("d0 must be in [0,1]")
        if not np.isfinite(s.mu) or abs(s.mu) > 1e4:
            raise ValueError("mu is outside the stable range")
        if (
            not isinstance(s.burn_in, (int, np.integer))
            or isinstance(s.burn_in, (bool, np.bool_))
            or s.burn_in < 0
        ):
            raise ValueError("burn_in must be a nonnegative integer")
        if (
            not isinstance(s.iterations, (int, np.integer))
            or isinstance(s.iterations, (bool, np.bool_))
            or s.iterations < 8
        ):
            raise ValueError("iterations must be an integer >=8")
        work += s.chains * (s.draws + s.warmup) * len(y) ** 2
        work += (s.burn_in + s.iterations) * len(y) ** 2
    if work > _MAX_TOTAL_WORK:
        raise ValueError("aggregate scenario MCMC work exceeds 1000000")

    summaries = []
    for s in scenarios:
        fit_args = {
            name: getattr(s, name)
            for name in (
                "mu",
                "sigma02",
                "sigmaD2",
                "alpha",
                "d0",
                "alpha1",
                "beta1",
                "tau2",
                "phi1",
                "deltaT",
                "thetaT",
                "burn_in",
                "iterations",
                "draws",
                "warmup",
                "chains",
            )
        }
        fit = bchm_fit(s.successes, s.trials, seed=int(s.seed), **fit_args)
        master = np.random.default_rng(s.seed)
        cluster_seed = int(master.integers(2**31))
        borrowing_seeds = tuple(int(master.integers(2**31)) for _ in s.successes)
        indicators_by_target = []
        for borrow in fit.borrowing:
            if borrow.efficacy_indicators is None:
                raise ArithmeticError("BCHM fit omitted efficacy indicators")
            indicators_by_target.append(borrow.efficacy_indicators)
        indicators = np.stack(indicators_by_target, axis=-1)
        efficacy_diagnostics = summarize_chains(indicators.astype(float))
        similarity_mcse = _co_cluster_mcse(fit.allocations)
        efficacy_mcse = efficacy_diagnostics.batch_mean_mcse
        efficacy_rhat = efficacy_diagnostics.split_rhat
        for value in (similarity_mcse, efficacy_mcse, efficacy_rhat):
            value.setflags(write=False)
        summaries.append(
            BCHMScenarioSummary(
                s,
                fit,
                cluster_seed,
                borrowing_seeds,
                similarity_mcse,
                efficacy_mcse,
                efficacy_rhat,
            )
        )
    return BCHMScenarioBatch(tuple(summaries))
