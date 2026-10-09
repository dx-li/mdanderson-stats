"""Multc Lean desktop model inputs and complete captured legacy study workflow."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .multc_core import MultcLeanDesign, MultcOperatingCharacteristics, multc_lean_design
from .multc_legacy_boundaries import MultcLegacyBoundaries, multc_legacy_boundaries
from .multc_legacy_simulation import MultcLegacySimulation, _integer, simulate_multc_legacy_duration
from .multc_study import _atomic_write, _validate_scenario_names

_MAX_BYTES = 1_048_576
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def _real(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be representable as a finite float64 scalar")
    return result


@dataclass(frozen=True)
class MultcLeanEndpointInput:
    """Native endpoint row, preserving inactive historical fields too."""

    standard_a: float
    standard_b: float
    experimental_a: float
    experimental_b: float
    use_standard_constant: bool
    standard_constant: float

    def __post_init__(self) -> None:
        if not isinstance(self.use_standard_constant, (bool, np.bool_)):
            raise ValueError("use_standard_constant must be boolean")
        object.__setattr__(self, "use_standard_constant", bool(self.use_standard_constant))
        for field in (
            "standard_a",
            "standard_b",
            "experimental_a",
            "experimental_b",
            "standard_constant",
        ):
            object.__setattr__(self, field, _real(getattr(self, field), field))
        if not 0 < self.experimental_a <= 100 or not 0 < self.experimental_b <= 100:
            raise ValueError("experimental beta shapes must lie in (0,100]")
        if self.use_standard_constant:
            if not 0 <= self.standard_constant <= 1:
                raise ValueError("active standard_constant must lie in [0,1]")
        elif not 0 < self.standard_a <= 1000 or not 0 < self.standard_b <= 1000:
            raise ValueError("active historical beta shapes must lie in (0,1000]")

    @property
    def historical(self) -> float | tuple[float, float]:
        return (
            self.standard_constant
            if self.use_standard_constant
            else (self.standard_a, self.standard_b)
        )


@dataclass(frozen=True)
class MultcLeanScenario:
    """Joint truth plus native optional timing fields in consistent time units.

    Both positive time fields enable simulation; zero in either disables it,
    as in the desktop runner. The source UI accepts sum error up to 1e-10;
    Python retains the core's stricter 1e-12 tolerance and never renormalizes.
    """

    joint_probabilities: tuple[float, float, float, float]
    mean_interarrival: float = 0
    response_window: float = 0

    def __post_init__(self) -> None:
        values = self.joint_probabilities
        if not isinstance(values, (tuple, list, np.ndarray)) or np.shape(values) != (4,):
            raise ValueError("joint_probabilities must have four entries")
        probabilities = tuple(_real(v, "joint_probabilities") for v in values)
        if any(not 0 <= v <= 1 for v in probabilities) or not np.isclose(
            sum(probabilities), 1, atol=1e-12, rtol=0
        ):
            raise ValueError("four joint probabilities must be in [0,1] and sum to one")
        object.__setattr__(self, "joint_probabilities", probabilities)
        for field in ("mean_interarrival", "response_window"):
            value = _real(getattr(self, field), field)
            if not 0 <= value <= 10_000:
                raise ValueError(f"{field} must lie in [0,10000] in the desktop profile")
            object.__setattr__(self, field, value)

    @property
    def simulation_enabled(self) -> bool:
        return self.mean_interarrival > 0 and self.response_window > 0

    @classmethod
    def independent(
        cls,
        response: float,
        toxicity: float,
        *,
        mean_interarrival: float = 0,
        response_window: float = 0,
    ) -> MultcLeanScenario:
        """Convert marginal independent rates to the desktop's four categories."""
        r, t = _real(response, "response"), _real(toxicity, "toxicity")
        if not 0 <= r <= 1 or not 0 <= t <= 1:
            raise ValueError("independent response/toxicity rates must lie in [0,1]")
        return cls(
            (r * t, r * (1 - t), (1 - r) * t, (1 - r) * (1 - t)), mean_interarrival, response_window
        )


@dataclass(frozen=True)
class MultcLeanModel:
    """Native .model fields, with bounded Python study execution and exports."""

    response: MultcLeanEndpointInput
    toxicity: MultcLeanEndpointInput
    max_subjects: int = 30
    min_subjects: int = 1
    cohort_size: int = 1
    response_cutoff: float = 0.95
    response_margin: float = 0
    toxicity_cutoff: float = 0.95
    toxicity_margin: float = 0
    scenarios: tuple[MultcLeanScenario, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.response, MultcLeanEndpointInput) or not isinstance(
            self.toxicity, MultcLeanEndpointInput
        ):
            raise ValueError("response/toxicity must be MultcLeanEndpointInput objects")
        cap = _integer(self.max_subjects, "max_subjects", 3, 1000)
        minimum = _integer(self.min_subjects, "min_subjects", 1, cap)
        cohort = _integer(self.cohort_size, "cohort_size", 1, cap)
        if cap % cohort or (minimum >= cohort and minimum % cohort):
            raise ValueError(
                "cohort must divide cap; minimum must be below a cohort or a cohort multiple"
            )
        for field, integer in (
            ("max_subjects", cap),
            ("min_subjects", minimum),
            ("cohort_size", cohort),
        ):
            object.__setattr__(self, field, integer)
        for field in ("response_cutoff", "toxicity_cutoff", "response_margin", "toxicity_margin"):
            value = _real(getattr(self, field), field)
            if not (0 <= value <= 1 if field.endswith("cutoff") else -1 < value < 1):
                raise ValueError(f"invalid {field}")
            object.__setattr__(self, field, value)
        if self.response_margin < 0 and self.toxicity_margin > 0:
            raise ValueError("cannot combine negative response and positive toxicity margins")
        if not isinstance(self.scenarios, (tuple, list)) or len(self.scenarios) > 100:
            raise ValueError("scenarios must be a sequence of at most 100 scenarios")
        if any(not isinstance(s, MultcLeanScenario) for s in self.scenarios):
            raise ValueError("scenarios must contain MultcLeanScenario objects")
        object.__setattr__(self, "scenarios", tuple(self.scenarios))

    @classmethod
    def example(cls, *, scenarios: Sequence[MultcLeanScenario] = ()) -> MultcLeanModel:
        """Original 2.1 installed configuration's reset/startup example."""
        if not isinstance(scenarios, Sequence) or len(scenarios) > 100:
            raise ValueError("scenarios must be a sequence of at most 100 scenarios")
        return cls(
            MultcLeanEndpointInput(30, 70, 0.6, 1.4, False, 0.3),
            MultcLeanEndpointInput(25, 75, 0.5, 1.5, False, 0.25),
            scenarios=tuple(scenarios),
        )

    def to_design(self) -> MultcLeanDesign:
        """Bind the native fields to verified marginal posterior monitoring."""
        r, t = self.response, self.toxicity
        return multc_lean_design(
            self.max_subjects,
            (r.experimental_a, r.experimental_b),
            (t.experimental_a, t.experimental_b),
            historical_response=r.historical,
            historical_toxicity=t.historical,
            response_cutoff=self.response_cutoff,
            toxicity_cutoff=self.toxicity_cutoff,
            response_margin=self.response_margin,
            toxicity_margin=self.toxicity_margin,
            min_subjects=self.min_subjects,
            cohort_size=self.cohort_size,
            pretrial_check=True,
        )

    def to_model_text(self) -> str:
        """Write the desktop schema with 17-digit, locale-independent floats.

        Original .NET general formatting can round to fewer digits. Python's
        round-trip representation is readable by the native reader and keeps
        input precision; byte-for-byte native timestamp/formatting is not claimed.
        """

        def number(value: float | int) -> str:
            return format(value, ".17g")

        lines = ["% Multc Lean model: mdanderson-stats compatibility export"]
        for endpoint in (self.response, self.toxicity):
            values = [
                number(v)
                for v in (
                    endpoint.standard_a,
                    endpoint.standard_b,
                    endpoint.experimental_a,
                    endpoint.experimental_b,
                )
            ]
            values += [str(endpoint.use_standard_constant), number(endpoint.standard_constant)]
            lines.append(" ".join(values))
        lines.append(
            " ".join(
                number(v)
                for v in (
                    self.max_subjects,
                    self.min_subjects,
                    self.cohort_size,
                    self.response_cutoff,
                    self.response_margin,
                    self.toxicity_cutoff,
                    self.toxicity_margin,
                )
            )
        )
        lines.append(str(len(self.scenarios)))
        lines.extend(
            " ".join(
                number(v) for v in (*s.joint_probabilities, s.mean_interarrival, s.response_window)
            )
            for s in self.scenarios
        )
        return "\n".join(lines) + "\n"

    def write_model(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_model_text())

    def run(
        self,
        *,
        seed: int,
        trials: int = 10_000,
        max_unit_exponentials_per_trial: int = 10_000,
        max_total_draws: int = 500_000_000,
        max_total_exact_work: int = 50_000_000,
        max_storage_bytes: int = 64_000_000,
        scenario_names: Sequence[str] | None = None,
        show_potential_boundaries: bool = False,
    ) -> MultcLeanStudy:
        """Run exact scenarios plus recovered legacy duration where times enable it.

        The native runner computes exact count OCs, then substitutes only the
        simulated duration/balk means. Python keeps the two estimates labeled
        separately. The source repetition default is 10,000; a captured explicit
        seed replaces its clock-based default. All aggregate budgets precede RNG
        creation and numerical work. No full native/CLR/RNG equivalence is claimed.
        """
        root_seed = _integer(seed, "seed", 0, 2**64 - 1)
        repetitions = _integer(trials, "trials", 1, 10_000)
        draws = _integer(
            max_unit_exponentials_per_trial, "max_unit_exponentials_per_trial", 1, 1_000_000
        )
        draw_budget = _integer(max_total_draws, "max_total_draws", 1, 1_000_000_000)
        exact_budget = _integer(max_total_exact_work, "max_total_exact_work", 1, 2_000_000_000)
        storage = _integer(max_storage_bytes, "max_storage_bytes", 1, 1_000_000_000)
        if not isinstance(show_potential_boundaries, (bool, np.bool_)):
            raise ValueError("show_potential_boundaries must be boolean")
        names = _validate_scenario_names(scenario_names, len(self.scenarios))
        cap, count = self.max_subjects, len(self.scenarios)
        timed = sum(s.simulation_enabled for s in self.scenarios)
        exact_work = sum((n + 1) ** 2 for n in range(1, cap + 1))
        if count and (exact_work > 10_000_000 or count * exact_work > exact_budget):
            raise ValueError(
                "model scenarios exceed max_total_exact_work or the per-scenario OC limit"
            )
        if timed * repetitions * (cap + draws) > draw_budget:
            raise ValueError("model scenarios exceed max_total_draws")
        required = count * (cap + 1) * 8 + cap * 1024 + 131072
        if timed:
            required += timed * repetitions * 64 + repetitions * 128 + draws * 10
        if required > storage:
            raise ValueError("model scenarios exceed max_storage_bytes")
        design = self.to_design()
        boundaries = multc_legacy_boundaries(design)
        rng = np.random.Generator(np.random.PCG64(root_seed))
        seeds = rng.integers(0, 2**64 - 1, size=count, dtype=np.uint64, endpoint=True)
        results = []
        for scenario, child_seed in zip(self.scenarios, seeds, strict=True):
            exact = design.operating_characteristics(scenario.joint_probabilities)
            simulation = (
                simulate_multc_legacy_duration(
                    design,
                    joint_probabilities=scenario.joint_probabilities,
                    mean_interarrival=scenario.mean_interarrival,
                    response_window=scenario.response_window,
                    trials=repetitions,
                    seed=int(child_seed),
                    max_unit_exponentials_per_trial=draws,
                    max_total_draws=draw_budget,
                    max_storage_bytes=storage,
                )
                if scenario.simulation_enabled
                else None
            )
            results.append(MultcLeanScenarioResult(exact, simulation, int(child_seed)))
        options = dict(
            seed=root_seed,
            trials=repetitions,
            max_unit_exponentials_per_trial=draws,
            max_total_draws=draw_budget,
            max_total_exact_work=exact_budget,
            max_storage_bytes=storage,
            scenario_names=list(names),
            show_potential_boundaries=bool(show_potential_boundaries),
        )
        # Store immutable option pairs rather than a mutable caller-owned dict.
        options["scenario_names"] = names
        return MultcLeanStudy(self, design, boundaries, tuple(results), tuple(options.items()))


def parse_multc_lean_model(text: str) -> MultcLeanModel:
    """Read bounded canonical desktop text, rejecting malformed or extra data.

    Leading '%' headers, CRLF/LF and a UTF-8 BOM are supported. Data columns use
    single ASCII spaces as in the native writer. No executable serialization or
    numeric expressions are accepted. Missing/invalid input raises an error,
    rather than silently resetting to the example as the desktop UI does.
    """
    if (
        not isinstance(text, str)
        or len(text) > _MAX_BYTES
        or len(text.encode("utf-8")) > _MAX_BYTES
    ):
        raise ValueError("model text must be no larger than 1 MiB")
    lines = re.split(r"\r\n|\r|\n", text.removeprefix("\ufeff"))
    if lines and lines[-1] == "":
        lines.pop()
    first = next((i for i, line in enumerate(lines) if not line.startswith("%")), len(lines))
    lines = lines[first:]
    if len(lines) < 4:
        raise ValueError("model needs two endpoint rows, controls and scenario count")

    def tokens(line: str, count: int) -> list[str]:
        values = line.split(" ")
        if len(values) != count or any(not value for value in values):
            raise ValueError(f"model data row needs exactly {count} single-space columns")
        return values

    def number(value: str) -> float:
        if not _NUMBER.fullmatch(value):
            raise ValueError("model numbers must use finite decimal/scientific notation")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("model numbers must be finite")
        return result

    endpoints = []
    for line in lines[:2]:
        v = tokens(line, 6)
        if v[4].lower() not in ("true", "false"):
            raise ValueError("native history flag must be True or False")
        endpoints.append(
            MultcLeanEndpointInput(
                number(v[0]),
                number(v[1]),
                number(v[2]),
                number(v[3]),
                v[4].lower() == "true",
                number(v[5]),
            )
        )
    control = tokens(lines[2], 7)
    if any(not re.fullmatch(r"[+]?[0-9]+", v) for v in (*control[:3], lines[3])):
        raise ValueError("model sample sizes and scenario count must be decimal integers")
    count = int(lines[3])
    if not 0 <= count <= 100 or len(lines) != 4 + count:
        raise ValueError("scenario count must be 0..100 and match exactly the remaining rows")
    scenarios = []
    for line in lines[4:]:
        values = [number(v) for v in tokens(line, 6)]
        scenarios.append(
            MultcLeanScenario((values[0], values[1], values[2], values[3]), values[4], values[5])
        )
    return MultcLeanModel(
        endpoints[0],
        endpoints[1],
        int(control[0]),
        int(control[1]),
        int(control[2]),
        number(control[3]),
        number(control[4]),
        number(control[5]),
        number(control[6]),
        tuple(scenarios),
    )


def read_multc_lean_model(path: str | Path) -> MultcLeanModel:
    with Path(path).open("rb") as stream:
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError("model file exceeds 1 MiB")
    return parse_multc_lean_model(data.decode("utf-8-sig"))


@dataclass(frozen=True)
class MultcLeanScenarioResult:
    exact: MultcOperatingCharacteristics
    simulation: MultcLegacySimulation | None
    seed: int


@dataclass(frozen=True)
class MultcLeanStudy:
    """Captured native model, named exact scenarios and legacy simulations."""

    model: MultcLeanModel
    design: MultcLeanDesign
    boundaries: MultcLegacyBoundaries
    results: tuple[MultcLeanScenarioResult, ...]
    options: tuple[tuple[str, Any], ...]

    def to_json(self) -> str:
        value = dict(
            format_version=1, model_text=self.model.to_model_text(), options=dict(self.options)
        )
        return json.dumps(value, indent=2, allow_nan=False) + "\n"

    def write_json(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_json())

    def protocol(self) -> str:
        """Editable Markdown with actual model/rules and captured study settings."""
        d, b = self.design, self.boundaries
        lines = [
            "# Multc Lean study protocol",
            "",
            f"Maximum enrollment: {d.max_subjects}; minimum look: {d.min_subjects}; "
            f"cohort: {d.cohort_size}.",
            f"Historical response: {self.model.response.historical}; experimental beta prior: "
            f"({d.response_prior.alpha}, {d.response_prior.beta}).",
            f"Historical toxicity: {self.model.toxicity.historical}; experimental beta prior: "
            f"({d.toxicity_prior.alpha}, {d.toxicity_prior.beta}).",
            f"Response rule: P(response < historical response + {d.response_margin}) "
            f"> {d.response_cutoff}.",
            f"Toxicity rule: P(toxicity > historical toxicity + {d.toxicity_margin}) "
            f"> {d.toxicity_cutoff}.",
            "Strict cutoff equality continues. The mandatory prior screen precedes enrollment.",
            f"Prior rejects response: {b.prior_response}; toxicity: {b.prior_toxicity}.",
            "At the enrollment cap, record cap completion separately from adverse stopping.",
            "",
            "| Enrolled | Stop if responses <= | Stop if toxicities >= |",
            "| --- | --- | --- |",
        ]
        bounds = d.stopping_bounds()
        for n, r, t in zip(
            bounds.looks, bounds.response_stop_max, bounds.toxicity_stop_min, strict=True
        ):
            if n == d.max_subjects:
                lines.append(f"| {int(n)} | cap completion | cap completion |")
            else:
                lines.append(
                    f"| {int(n)} | {int(r) if r >= 0 else 'never'} | "
                    f"{int(t) if t <= n else 'never'} |"
                )
        lines += [
            "",
            "Legacy duration simulation uses latent outcome counts, "
            "clipped/shared follow-up and balked arrivals.",
            "Time units must be identical for interarrival, follow-up window "
            "and reported duration.",
            "Exact operating characteristics and Monte Carlo timing estimates "
            "are reported separately.",
            "Python PCG64 streams replace the original clock-seeded Windows RNG.",
            "",
            "## Captured run inputs",
            "",
            "```json",
            self.to_json().strip(),
            "```",
            "",
        ]
        return "\n".join(lines)

    def write_protocol(self, path: str | Path) -> Path:
        return _atomic_write(path, self.protocol())

    def to_html(self) -> str:
        """Portable report of inputs, bounds, exact OCs and legacy MC timing."""
        options = dict(self.options)
        names = options["scenario_names"]
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Multc Lean study</title></head><body>",
            "<h1>Multc Lean study</h1><p>Exact count operating characteristics "
            "and legacy Monte Carlo timing are separate estimates.</p>",
            "<h2>Model and stopping rules</h2><pre>"
            + html.escape(self.protocol().split("## Captured run inputs")[0])
            + "</pre>",
        ]
        if options["show_potential_boundaries"]:
            potential = self.design.potential_boundaries()
            parts += [
                "<h2>Reachable stopping intervals</h2>",
                "<table><tr><th>N</th><th>Response interval</th><th>Toxicity interval</th></tr>",
            ]
            for n, r, t in zip(
                potential.looks, potential.response_stop, potential.toxicity_stop, strict=True
            ):
                if n == self.design.max_subjects:
                    parts.append(
                        f"<tr><td>{int(n)}</td><td>cap completion</td><td>cap completion</td></tr>"
                    )
                else:
                    parts.append(f"<tr><td>{int(n)}</td><td>{r}</td><td>{t}</td></tr>")
            parts.append("</table>")
        for name, scenario, result in zip(names, self.model.scenarios, self.results, strict=True):
            oc, simulation = result.exact, result.simulation
            parts += [
                "<h2>" + html.escape(name) + "</h2>",
                "<p>Joint truth [both, response only, toxicity only, neither]: "
                + str(scenario.joint_probabilities)
                + "</p>",
                "<h3>Exact count estimates</h3><dl>",
            ]
            for label, value in (
                ("Expected enrollment", oc.expected_sample_size),
                ("Enrollment SD", oc.sample_size_sd),
                ("Expected responses", oc.expected_responses),
                ("Expected toxicities", oc.expected_toxicities),
                ("Stop response only", oc.stop_response_only),
                ("Stop toxicity only", oc.stop_toxicity_only),
                ("Stop both", oc.stop_both),
                ("Cap completion", oc.cap_completion),
            ):
                parts.append(f"<dt>{label}</dt><dd>{float(value):.17g}</dd>")
            parts.append("</dl>")
            if simulation is not None:
                s = simulation.summary
                parts += [
                    "<h3>Legacy timing Monte Carlo estimates</h3>",
                    f"<p>Seed: {result.seed}; trials: {s.trials}; "
                    f"mean interarrival: {scenario.mean_interarrival:.17g}; "
                    f"window: {scenario.response_window:.17g}</p>",
                    f"<p>Mean duration: {s.mean_duration:.17g}; MCSE: {s.duration_mcse:.17g}; "
                    f"mean balks: {s.mean_balks:.17g}; balk MCSE: {s.balks_mcse:.17g}</p>",
                    "<p>Captured per-trial seeds (PCG64): "
                    + html.escape(str(simulation.trial_seeds.tolist()))
                    + "</p>",
                ]
            else:
                parts.append(
                    "<p>Duration simulation disabled: at least one timing field is zero.</p>"
                )
            parts += [
                "<h3>Exact stopping sample-size distribution</h3><table><tr><th>N</th>"
                "<th>P(N)</th><th>P(N &lt;= n)</th></tr>"
            ]
            cumulative = np.cumsum(oc.sample_size_probability)
            for n in (0, *map(int, self.design.looks)):
                parts.append(
                    f"<tr><td>{n}</td><td>{oc.sample_size_probability[n]:.17g}</td><td>{cumulative[n]:.17g}</td></tr>"
                )
            parts.append("</table>")
        parts += [
            "<h2>Replayable inputs</h2><pre>" + html.escape(self.to_json()) + "</pre>",
            "</body></html>",
        ]
        return "\n".join(parts) + "\n"

    def write_report(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_html())

    def plot(self, *, cumulative: bool = True) -> Any:
        """Plot exact CDFs or sample-size PMFs; requires the plotting extra."""
        if not isinstance(cumulative, (bool, np.bool_)):
            raise ValueError("cumulative must be boolean")
        if not self.results:
            raise ValueError("plot needs at least one scenario")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        for name, result in zip(dict(self.options)["scenario_names"], self.results, strict=True):
            probability = result.exact.sample_size_probability
            y = np.cumsum(probability) if cumulative else probability
            axis.step(np.arange(self.model.max_subjects + 1), y, where="post", label=name)
        axis.set(
            xlabel="Treated sample size",
            ylabel="Cumulative stopping probability" if cumulative else "Sample-size probability",
            ylim=(0, 1),
            xlim=(0, self.model.max_subjects),
        )
        axis.legend()
        return figure


def replay_multc_lean_study(text: str) -> MultcLeanStudy:
    """Read versioned community JSON inputs and recreate the native-model study."""
    if (
        not isinstance(text, str)
        or len(text) > _MAX_BYTES
        or len(text.encode("utf-8")) > _MAX_BYTES
    ):
        raise ValueError("study JSON exceeds 1 MiB")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate study JSON key")
            obj[key] = value
        return obj

    def constant(value: str) -> None:
        raise ValueError("study JSON must contain finite numbers")

    values = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
    if not isinstance(values, dict) or set(values) != {"format_version", "model_text", "options"}:
        raise ValueError("unexpected study JSON fields")
    if type(values["format_version"]) is not int or values["format_version"] != 1:
        raise ValueError("unsupported study format_version")
    options = values["options"]
    if not isinstance(options, dict) or set(options) != {
        "seed",
        "trials",
        "max_unit_exponentials_per_trial",
        "max_total_draws",
        "max_total_exact_work",
        "max_storage_bytes",
        "scenario_names",
        "show_potential_boundaries",
    }:
        raise ValueError("unexpected study options")
    return parse_multc_lean_model(values["model_text"]).run(**options)
