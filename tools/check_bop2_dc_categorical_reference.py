"""Manually compare the categorical design API with the independent base-R CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bop2_dc_categorical import bop2_dc_categorical_design
from mdanderson_stats.bop2_dc_paired import bop2_dc_paired_design


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TOL = 2e-7


def _vector(text: str) -> np.ndarray:
    return np.asarray([float(item) for item in text.split(";")])


def _matrix(text: str) -> np.ndarray:
    return np.asarray([[int(value) for value in row] for row in text.split("|")])


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=TOL, atol=TOL):
        raise AssertionError(f"{label}: actual={actual:.16g}, reference={expected:.16g}")


def _build(config: dict[str, str]):
    max_n = int(config["max_n"])
    interim_n = int(config["interim_n"])
    looks = [max_n] if interim_n == 0 else sorted({interim_n, max_n})
    kwargs = dict(
        max_subjects=max_n,
        indicators=_matrix(config["indicators"]),
        combination=config["combination"],
        directions=config["direction"].split(";"),
        lrv=_vector(config["lrv"]),
        cmv=_vector(config["cmv"]),
        prior=_vector(config["prior_experimental"]),
        looks=looks,
        lambda_lrv=float(config["lambda_lrv"]),
        lambda_cmv=float(config["lambda_cmv"]),
        gamma_lrv=float(config["gamma_lrv"]),
        gamma_cmv=float(config["gamma_cmv"]),
    )
    if config["arm"] == "randomized":
        assignments = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
        kwargs.update(
            control_prior=_vector(config["prior_control"]),
            arm_assignments=assignments,
        )
    return bop2_dc_categorical_design(**kwargs)


def _counts(config: dict[str, str], stage: str) -> np.ndarray:
    experimental = _vector(config[f"counts_experimental_{stage}"]).astype(int)
    if config["arm"] == "single":
        return experimental
    control = _vector(config[f"counts_control_{stage}"]).astype(int)
    return np.stack((control, experimental))


def _probability_rows() -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {
        (row["case"], row["stage"], row["endpoint"], row["criterion"]): row
        for row in _rows(FIXTURES / "bop2-dc-categorical-reference.csv")
    }


def main() -> None:
    configs = {
        row["case"]: row for row in _rows(FIXTURES / "bop2-dc-categorical-config.csv")
    }
    references = _probability_rows()
    decisions = {
        row["case"]: row for row in _rows(FIXTURES / "bop2-dc-categorical-decisions.csv")
    }
    compared = 0
    for case, config in configs.items():
        design = _build(config)
        for stage in ("final", "interim"):
            counts = _counts(config, "final" if stage == "final" else "interim")
            n = int(np.asarray(counts).sum())
            # The prior-only fixtures record final strict-equality decisions as a
            # source-rule check; monitor(0) is an interim state for max_n > 0.
            if stage == "final" and n < int(config["max_n"]):
                continue
            state = design.monitor(counts)
            if stage == "final":
                expected_action = decisions[case]["action_final"]
            else:
                expected_action = decisions[case]["action_interim"]
            action = str(np.asarray(state.decision).item())
            if action != expected_action:
                raise AssertionError(
                    f"{case}/{stage}: {action} != {expected_action}"
                )
            endpoint_names = {
                "single_any": ["endpoint_a", "endpoint_b", "endpoint_c"],
                "randomized_all": ["endpoint_a", "endpoint_b", "endpoint_c"],
                "two_endpoint_any": ["endpoint_a", "endpoint_b"],
                "prior_only_equal_single": ["endpoint_a"],
                "prior_only_equal_randomized": ["endpoint_a"],
            }[case]
            for j, endpoint in enumerate(endpoint_names):
                for q, criterion in enumerate(("LRV", "CMV")):
                    reference = references[(case, stage, endpoint, criterion)]
                    _close(
                        float(state.posterior_probability[j, q]),
                        float(reference["probability"]),
                        f"{case}/{stage}/{endpoint}/{criterion}",
                    )
                    compared += 1

    # The K=4 indicator mapping (both, first-only, second-only, neither)
    # must reproduce the established two-endpoint multiple-efficacy marginal.
    config = configs["two_endpoint_any"]
    counts = _counts(config, "final")
    categorical = _build(config).monitor(counts)
    paired = bop2_dc_paired_design(
        4,
        "multiple_efficacy",
        [.25, .40],
        [.45, .60],
        lambda_lrv=[.65, .65],
        lambda_cmv=[.45, .45],
        gamma_lrv=[.5, .5],
        gamma_cmv=[.5, .5],
        prior=_vector(config["prior_experimental"]),
        looks=[2, 4],
    ).monitor(counts)
    if not np.allclose(categorical.posterior_probability, paired.marginal_posterior, rtol=TOL, atol=TOL):
        raise AssertionError("K=4 categorical marginals disagree with paired two-endpoint design")
    if np.asarray(categorical.endpoint_decisions).tolist() != paired.endpoint_decision.tolist():
        raise AssertionError("K=4 categorical endpoint decisions disagree with paired design")

    randomized = _build(configs["randomized_all"])
    replay = randomized.replay(np.asarray([1, 4, 0, 2, 1, 3, 0, 4], dtype=int))
    if len(replay.states) != 1 or replay.states[0].total_n != 2:
        raise AssertionError("fixed replay tape should stop at its first look, n=2")
    replay_reference = _rows(FIXTURES / "bop2-dc-categorical-replay.csv")[0]
    expected_probabilities = np.asarray(
        [float(value) for value in replay_reference["posterior_probabilities"].split(";")]
    ).reshape(3, 2)
    if not np.allclose(
        replay.states[0].posterior_probability, expected_probabilities, rtol=TOL, atol=TOL
    ):
        raise AssertionError("first reached replay probabilities disagree with base-R reference")
    if np.asarray(replay.states[0].endpoint_decisions).tolist() != replay_reference[
        "endpoint_actions"
    ].split(";"):
        raise AssertionError("first reached replay endpoint actions disagree with base-R reference")
    if str(np.asarray(replay.terminal_decision).item()) != replay_reference["action"]:
        raise AssertionError("replay terminal action disagrees with base-R reference")
    print(f"BOP2 categorical reference check passed ({compared} posterior probabilities + paired reduction).")


if __name__ == "__main__":
    main()
