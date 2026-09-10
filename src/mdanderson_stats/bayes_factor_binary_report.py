"""Guide-compatible text input and standalone HTML for Bayes Factor Binary."""

from dataclasses import dataclass
from html import escape

import numpy as np

from ._validation import FloatArray
from .bayes_factor_binary import (
    BayesFactorBinaryDesign,
    BayesFactorBinaryOC,
    BayesFactorBinarySimulation,
    bayes_factor_binary_design,
)
from .beta_binomial import _owned


@dataclass(frozen=True)
class BayesFactorBinaryReport:
    job: "BayesFactorBinaryJob"
    exact: BayesFactorBinaryOC
    simulation: BayesFactorBinarySimulation

    def to_html(self) -> str:
        """Return HTML containing input provenance, exact and Monte Carlo results."""
        job, design = self.job, self.job.design
        lower_symbol = "<" if design.strict_thresholds else "<="
        upper_symbol = ">" if design.strict_thresholds else ">="
        parameters = {
            "Implementation": "mdanderson-stats independent Python implementation",
            "Random seed": job.seed,
            "Simulation repetitions": job.n_trials,
            "Maximum patients": design.max_subjects,
            "Null response rate": design.null_rate,
            "Alternative prior mode": design.alternative_mode,
            "iMOM prior": (
                f"k={design.imom_shape:g}, nu={2 * design.imom_shape:g}, "
                "normalized on (null rate, 1); equal prior model odds"
            ),
            f"Inferiority cutoff ({lower_symbol})": design.inferiority_cutoff,
            f"Superiority cutoff ({upper_symbol})": design.superiority_cutoff,
            "Analysis sample sizes": ", ".join(map(str, design.looks)),
        }

        def table(headers, rows):
            head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
            body = "".join(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                for row in rows
            )
            return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

        parts = [
            "<!doctype html><html lang='en'><meta charset='utf-8'>",
            "<title>Bayes Factor Binary</title><style>body{font:16px "
            "sans-serif;max-width:1200px;margin:2em auto}td,th{padding:.4em;border:1px solid "
            "#ccc}table{border-collapse:collapse;margin-bottom:2em}</style>",
            "<h1>Bayes Factor Binary</h1>",
            table(["Parameter", "Value"], parameters.items()),
        ]
        headers = [
            "Response rate",
            "P(superiority)",
            "P(inferiority)",
            "P(inconclusive)",
            "Mean N",
            "N quantiles: 10%, 25%, 50%, 75%, 90%",
        ]
        quantiles = np.stack(
            [self.exact.sample_size_quantile(p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]], axis=-1
        )
        exact_rows, simulated_rows = [], []
        for i, rate in enumerate(job.probabilities):
            exact_rows.append(
                [
                    f"{rate:g}",
                    f"{self.exact.superiority[i]:.8g}",
                    f"{self.exact.inferiority[i]:.8g}",
                    f"{self.exact.inconclusive[i]:.8g}",
                    f"{self.exact.expected_sample_size[i]:.8g}",
                    ", ".join(f"{q:g}" for q in quantiles[i]),
                ]
            )
            sizes = np.sort(self.simulation.sample_size[i])
            q = sizes[np.ceil(np.array([0.1, 0.25, 0.5, 0.75, 0.9]) * job.n_trials).astype(int) - 1]
            probabilities = self.simulation.decision_probability[i]
            simulated_rows.append(
                [
                    f"{rate:g}",
                    *(f"{probabilities[j]:.8g}" for j in [1, 0, 2]),
                    f"{self.simulation.expected_sample_size[i]:.8g}",
                    ", ".join(map(str, q)),
                ]
            )
        parts += [
            "<h2>Exact operating characteristics</h2>",
            table(headers, exact_rows),
            "<h2>Monte Carlo operating characteristics</h2>",
            table(headers, simulated_rows),
            "<p>Simulation uses NumPy's random generator. Identical seeds do not reproduce "
            "the original application's random stream. Exact probabilities do not contain "
            "simulation noise.</p>",
        ]
        if job.show_boundaries:
            rows = []
            for n, lo, hi in zip(
                design.looks, design.inferiority_max, design.superiority_min, strict=True
            ):
                rows.append(
                    [
                        n,
                        "none" if lo < 0 else f"0–{lo}",
                        "none" if hi <= lo + 1 else f"{lo + 1}–{hi - 1}",
                        "none" if hi > n else f"{hi}–{n}",
                    ]
                )
            parts += [
                "<h2>Stopping boundaries (inclusive counts)</h2>",
                table(
                    ["Patients", "Inferiority", "Continue / final inconclusive", "Superiority"],
                    rows,
                ),
                "<p>At maximum enrollment every trial stops; counts between the boundaries yield "
                "an inconclusive result.</p>",
            ]
        return "\n".join(parts) + "</html>"


@dataclass(frozen=True)
class BayesFactorBinaryJob:
    design: BayesFactorBinaryDesign
    probabilities: FloatArray
    n_trials: int
    seed: int
    show_boundaries: bool

    def run(self) -> BayesFactorBinaryReport:
        return BayesFactorBinaryReport(
            self,
            self.design.operating_characteristics(self.probabilities),
            self.design.simulate(self.probabilities, n_trials=self.n_trials, rng=self.seed),
        )


def parse_bayes_factor_binary_input(text: str) -> BayesFactorBinaryJob:
    """Parse the guide's ten settings followed by scenarios, ignoring # comments."""
    values = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    values = [value for value in values if value]
    if len(values) < 11 or any(len(value.split()) != 1 for value in values):
        raise ValueError("input needs ten single-value settings and at least one scenario")
    seed, n = int(values[0]), int(values[1])
    p0, mode, low, high = map(float, values[2:6])
    trials = int(values[6])
    if not 0 <= seed <= 4000000000 or not 1 <= trials <= 100000:
        raise ValueError("seed must be 0..4 billion and simulation repetitions 1..100000")
    if values[7].lower() not in ("yes", "no"):
        raise ValueError("boundary-table setting must be Yes or No")
    design = bayes_factor_binary_design(
        n,
        null_rate=p0,
        alternative_mode=mode,
        inferiority_cutoff=low,
        superiority_cutoff=high,
        min_subjects=int(values[8]),
        cohort_size=int(values[9]),
    )
    scenarios = np.array(list(map(float, values[10:])))
    if np.any(~np.isfinite(scenarios)) or np.any((scenarios < 0) | (scenarios > 1)):
        raise ValueError("scenario response rates must be finite and in [0,1]")
    return BayesFactorBinaryJob(design, _owned(scenarios), trials, seed, values[7].lower() == "yes")
