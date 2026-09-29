"""Compare randomized-paired Python APIs with independent base-R fixture output.

This checker is intentionally bounded: the core's posterior-tail calculations
are evaluated once per unique marginal-count state, in chunks of at most 200;
the exact candidate optimizer reuses its own cached tables. A few representative
public replays cover each terminal decision reached by selected candidates.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_ACTIONS = ("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
_ATOL = 2e-12
_TIE_FACTOR = 64.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _numbers(value: str) -> list[float]:
    return [float(part) for part in value.split(";")]


def _integers(value: str) -> list[int]:
    return [int(part) for part in value.split(";")]


def _pair_grid(value: str) -> np.ndarray:
    return np.asarray([[float(v) for v in row.split(",")] for row in value.split(";")])


def _close(a: float, b: float, n: int) -> bool:
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) <= _TIE_FACTOR * np.finfo(float).eps * n * scale)


def _select(
    metrics: list[dict[str, float]], feasible: list[bool], objective: str, n: int
) -> int | None:
    indices = [i for i, ok in enumerate(feasible) if ok]
    if not indices:
        return None
    if objective == "cgr":
        primary = [row["CGR"] for row in metrics]
        secondary = [-row["EN_futile"] for row in metrics]
        maximize = True
    else:
        primary = [row["EN_futile"] for row in metrics]
        secondary = [row["CGR"] for row in metrics]
        maximize = False
    best = indices[0]
    for i in indices[1:]:
        a, b = primary[i], primary[best]
        if _close(a, b, n):
            a2, b2 = secondary[i], secondary[best]
            if not _close(a2, b2, n) and a2 > b2:
                best = i
        elif (a > b) if maximize else (a < b):
            best = i
    return best


def _package_paths(main_src: Path, core_src: Path) -> None:
    sys.path.insert(0, str(main_src))
    import mdanderson_stats  # type: ignore[import-not-found]

    core_package = str(core_src / "mdanderson_stats")
    if core_package not in mdanderson_stats.__path__:
        mdanderson_stats.__path__.append(core_package)


def _truth_array(rows: list[dict[str, str]], scenario: str) -> np.ndarray:
    selected = next(row for row in rows if row["scenario"] == scenario)
    return np.asarray(
        [_numbers(selected["control_probability"]), _numbers(selected["treatment_probability"])],
        dtype=float,
    )


def _parse_grid_rows(row: dict[str, str]) -> tuple[np.ndarray, ...]:
    return tuple(
        _pair_grid(row[name])
        for name in ("lambda_lrv_grid", "lambda_cmv_grid", "gamma_lrv_grid", "gamma_cmv_grid")
    )


def _monitor_counts(path: str, arms: list[int], look: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((2, 4), dtype=np.int64)
    for arm, category_char in zip(arms[:look], path[:look], strict=True):
        counts[arm, int(category_char) - 1] += 1
    return counts[0], counts[1]


def _success_key_from_path(path: str, arms: list[int], look: int) -> tuple[int, int, int, int]:
    cc, tc = _monitor_counts(path, arms, look)
    return (int(cc[0] + cc[1]), int(cc[0] + cc[2]), int(tc[0] + tc[1]), int(tc[0] + tc[2]))


def _compare_unique_tails(
    design: Any,
    r_rows: list[dict[str, str]],
    config_id: str,
    path_by_id: dict[int, str],
    arms: list[int],
    candidate_designs: dict[int, Any],
) -> tuple[int, int, float]:
    """Cross-check unique posterior states and every path/look action in small batches."""
    states: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    row_states: list[tuple[dict[str, str], tuple[int, ...]]] = []
    for row in r_rows:
        look = int(row["look"])
        cc, tc = _monitor_counts(path_by_id[int(row["path_id"])], arms, look)
        cs = np.asarray([[cc[0] + cc[1], cc[0] + cc[2]]], dtype=np.int64)
        ts = np.asarray([[tc[0] + tc[1], tc[0] + tc[2]]], dtype=np.int64)
        key = (
            look,
            int(cc.sum()),
            int(tc.sum()),
            int(cs[0, 0]),
            int(cs[0, 1]),
            int(ts[0, 0]),
            int(ts[0, 1]),
        )
        states.setdefault(key, (cs, ts))
        row_states.append((row, key))

    # Batch unique count states by arm sample sizes to keep intermediate arrays small.
    grouped: dict[tuple[int, int], list[tuple[tuple[int, ...], tuple[np.ndarray, np.ndarray]]]] = (
        defaultdict(list)
    )
    for key, value in states.items():
        grouped[(key[1], key[2])].append((key, value))
    tails_by_state: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    for (control_n, treatment_n), items in grouped.items():
        for start in range(0, len(items), 200):
            batch = items[start : start + 200]
            cs = np.concatenate([item[1][0] for item in batch], axis=0)
            ts = np.concatenate([item[1][1] for item in batch], axis=0)
            tails, errors = design._posterior_tails_from_success_counts(
                control_n, treatment_n, cs, ts
            )
            for (key, _), tail, error in zip(batch, tails, errors, strict=True):
                tails_by_state[key] = (tail, error)

    checked = 0
    max_tail_difference = 0.0
    by_candidate_look: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row, key in row_states:
        tail, error = tails_by_state[key]
        expected = np.asarray(
            [
                [float(row["tail1_lrv"]), float(row["tail1_cmv"])],
                [float(row["tail2_lrv"]), float(row["tail2_cmv"])],
            ],
            dtype=float,
        )
        tail_difference = float(np.max(np.abs(tail - expected)))
        max_tail_difference = max(max_tail_difference, tail_difference)
        if np.any(np.abs(tail - expected) > error + 3e-12):
            raise AssertionError(f"posterior tails exceed quadrature error: {config_id} key={key}")
        by_candidate_look[(int(row["candidate_index"]), int(row["look"]))].append(row)
        checked += 1

    for (candidate, look), rows in by_candidate_look.items():
        candidate_design = candidate_designs[candidate]
        for start in range(0, len(rows), 200):
            batch = rows[start : start + 200]
            probabilities = np.asarray(
                [
                    tails_by_state[
                        (
                            look,
                            int(row["control_n"]),
                            int(row["treatment_n"]),
                            *_success_key_from_path(path_by_id[int(row["path_id"])], arms, look),
                        )
                    ][0]
                    for row in batch
                ]
            )
            errors = np.asarray(
                [
                    tails_by_state[
                        (
                            look,
                            int(row["control_n"]),
                            int(row["treatment_n"]),
                            *_success_key_from_path(path_by_id[int(row["path_id"])], arms, look),
                        )
                    ][1]
                    for row in batch
                ]
            )
            labels = candidate_design._paired_decisions_from_tails(
                probabilities, errors, total_n=look
            )
            for row, label in zip(batch, labels, strict=True):
                if str(label) != row["decision"]:
                    raise AssertionError(
                        f"path/look decision mismatch {config_id} candidate={candidate} "
                        f"path={row['path_id']} look={look}: {label} != {row['decision']}"
                    )
    return len(states), checked, max_tail_difference


def _truth_oc(
    result: Any, scenario_index: int, candidate: int
) -> tuple[np.ndarray, np.ndarray, float]:
    evidence = result.candidates
    return (
        evidence.decision_probability[scenario_index, candidate],
        evidence.sample_size_probability[scenario_index, candidate],
        float(evidence.expected_sample_size[scenario_index, candidate]),
    )


def check(prefix: Path, main_src: Path, core_src: Path) -> dict[str, int]:
    _package_paths(main_src, core_src)
    from mdanderson_stats.bop2_dc_randomized_paired import bop2_dc_randomized_paired_design
    from mdanderson_stats.bop2_dc_randomized_paired_optimization import (
        BOP2DCRandomizedPairedInfeasibleError,
        optimize_bop2_dc_randomized_paired,
    )

    settings = _read_csv(prefix / "settings.csv")
    truths = _read_csv(prefix / "truths.csv")
    candidate_rows = _read_csv(prefix / "candidate-metrics.csv")
    path_rows = _read_csv(prefix / "path-decisions.csv")
    monitor_rows = _read_csv(prefix / "monitor-paths.csv")
    oc_rows = _read_csv(prefix / "operating-characteristics.csv")
    selected_rows = _read_csv(prefix / "selected.csv")
    counts: dict[str, Any] = {
        "unique_posterior_states": 0,
        "candidate_summaries": 0,
        "oc_look_summaries": 0,
        "path_look_decisions": 0,
        "replays": 0,
        "max_tail_abs_difference": 0.0,
        "max_candidate_metric_abs_difference": 0.0,
        "max_oc_abs_difference": 0.0,
        "selected_indices": {},
    }

    for setting in settings:
        config_id = setting["config_id"]
        endpoint = setting["endpoint"]
        n = int(setting["max_subjects"])
        looks = _integers(setting["looks"])
        arms = [int(char) for char in setting["arm_assignments"]]
        lrv, cmv = _numbers(setting["lrv"]), _numbers(setting["cmv"])
        grids = _parse_grid_rows(setting)
        control_prior, treatment_prior = (
            _numbers(setting["control_prior"]),
            _numbers(setting["treatment_prior"]),
        )
        truth_config = [r for r in truths if r["config_id"] == config_id]
        futile = _truth_array(truth_config, "futile")
        effective = _truth_array(truth_config, "effective")
        kwargs = dict(
            max_subjects=n,
            endpoint=endpoint,
            lrv=lrv,
            cmv=cmv,
            futile_joint_probabilities=futile,
            effective_joint_probabilities=effective,
            arm_assignments=arms,
            lambda_lrv_grid=grids[0],
            lambda_cmv_grid=grids[1],
            gamma_lrv_grid=grids[2],
            gamma_cmv_grid=grids[3],
            control_prior=control_prior,
            treatment_prior=treatment_prior,
            looks=looks,
            false_go_limit=float(setting["false_go_limit"]),
            false_no_go_limit=float(setting["false_no_go_limit"]),
            false_consider_limit=float(setting["false_consider_limit"]),
            graduate_at_interim=setting["graduate_at_interim"] == "1",
            comparison_tolerance=1e-8,
        )
        is_infeasible = config_id == "multi_infeasible"
        if is_infeasible:
            for objective in ("cgr", "ess_futile"):
                try:
                    optimize_bop2_dc_randomized_paired(**kwargs, objective=objective)
                except BOP2DCRandomizedPairedInfeasibleError:
                    pass
                else:
                    raise AssertionError(
                        f"{config_id}/{objective}: expected infeasible-grid exception"
                    )
            # Obtain full candidate evidence separately with nonbinding limits.
            kwargs["false_go_limit"] = kwargs["false_no_go_limit"] = kwargs[
                "false_consider_limit"
            ] = 1.0
        result = optimize_bop2_dc_randomized_paired(**kwargs, objective="cgr")
        result_ess = optimize_bop2_dc_randomized_paired(**kwargs, objective="ess_futile")

        # Product/input order, including independent two-endpoint controls.
        combinations = itertools.product(*(range(grid.shape[0]) for grid in grids))
        expected_parameters = []
        for i_lrv, i_cmv, i_glrv, i_gcmv in combinations:
            expected_parameters.append(
                np.concatenate(
                    (grids[0][i_lrv], grids[1][i_cmv], grids[2][i_glrv], grids[3][i_gcmv])
                )
            )
        expected_parameters = np.asarray(expected_parameters)
        np.testing.assert_allclose(
            result.candidates.parameters,
            expected_parameters,
            rtol=0.0,
            atol=0.0,
            err_msg=f"grid order {config_id}",
        )
        if result.decision_labels != _ACTIONS:
            raise AssertionError(f"unexpected decision order {result.decision_labels}")

        expected_rows = [r for r in candidate_rows if r["config_id"] == config_id]
        if len(expected_rows) != len(expected_parameters):
            raise AssertionError(f"candidate-row count differs for {config_id}")
        expected_metrics: list[dict[str, float]] = []
        expected_feasible: list[bool] = []
        for candidate, row in enumerate(expected_rows):
            metric = {
                name: float(row[name])
                for name in ("FGR", "FNGR", "CGR", "FCR", "EN_futile", "EN_effective")
            }
            expected_metrics.append(metric)
            expected_feasible.append(row["feasible"].lower() == "true")
            candidate_parameter_row = result.candidates.parameters[candidate]
            candidate_param_cols = (
                "lambda_lrv_1",
                "lambda_lrv_2",
                "lambda_cmv_1",
                "lambda_cmv_2",
                "gamma_lrv_1",
                "gamma_lrv_2",
                "gamma_cmv_1",
                "gamma_cmv_2",
            )
            rparams = np.asarray([float(row[key]) for key in candidate_param_cols])
            np.testing.assert_allclose(candidate_parameter_row, rparams, rtol=0.0, atol=0.0)
            for attribute, field in (
                ("false_go_rate", "FGR"),
                ("false_no_go_rate", "FNGR"),
                ("correct_go_rate", "CGR"),
                ("false_consider_rate", "FCR"),
            ):
                counts["max_candidate_metric_abs_difference"] = max(
                    counts["max_candidate_metric_abs_difference"],
                    abs(float(getattr(result.candidates, attribute)[candidate]) - metric[field]),
                )
                np.testing.assert_allclose(
                    getattr(result.candidates, attribute)[candidate],
                    metric[field],
                    rtol=0.0,
                    atol=_ATOL,
                    err_msg=f"{config_id} candidate {candidate} {field}",
                )
            expected_en = np.asarray([metric["EN_futile"], metric["EN_effective"]])
            observed_en = result.candidates.expected_sample_size[:, candidate]
            counts["max_candidate_metric_abs_difference"] = max(
                counts["max_candidate_metric_abs_difference"],
                float(np.max(np.abs(observed_en - expected_en))),
            )
            np.testing.assert_allclose(observed_en, expected_en, rtol=0.0, atol=_ATOL)
            if (
                not is_infeasible
                and bool(result.candidates.feasible[candidate]) != expected_feasible[-1]
            ):
                raise AssertionError(f"feasibility mismatch {config_id} candidate {candidate}")
            counts["candidate_summaries"] += 1

        # Compare full per-truth/per-candidate/per-look decisions and sample-size mass.
        row_lookup = {
            (r["scenario"], int(r["candidate_index"]), int(r["look"])): r
            for r in oc_rows
            if r["config_id"] == config_id
        }
        decision_index = {label: i for i, label in enumerate(result.decision_labels)}
        for scenario_index, scenario in enumerate(("futile", "effective")):
            for candidate in range(len(expected_parameters)):
                decision_probability, ss_probability, en = _truth_oc(
                    result, scenario_index, candidate
                )
                for look_index, look in enumerate(looks):
                    r = row_lookup[(scenario, candidate, look)]
                    expected_actions = np.asarray(
                        [
                            float(r[key])
                            for key in (
                                "stop_no_go",
                                "graduate",
                                "final_go",
                                "final_consider",
                                "final_no_go",
                            )
                        ]
                    )
                    observed_actions = np.asarray(
                        [decision_probability[look_index, decision_index[a]] for a in _ACTIONS]
                    )
                    observed_ss = float(ss_probability[look_index])
                    expected_ss = float(r["sample_size_probability"])
                    expected_ess = float(r["expected_sample_size"])
                    oc_difference = max(
                        float(np.max(np.abs(observed_actions - expected_actions))),
                        abs(observed_ss - expected_ss),
                        abs(en - expected_ess),
                    )
                    counts["max_oc_abs_difference"] = max(
                        counts["max_oc_abs_difference"], oc_difference
                    )
                    np.testing.assert_allclose(
                        observed_actions,
                        expected_actions,
                        rtol=0.0,
                        atol=_ATOL,
                        err_msg=f"OC actions {config_id}/{scenario}/{candidate}/{look}",
                    )
                    np.testing.assert_allclose(observed_ss, expected_ss, rtol=0.0, atol=_ATOL)
                    np.testing.assert_allclose(en, expected_ess, rtol=0.0, atol=_ATOL)
                    counts["oc_look_summaries"] += 1

        # Independently derive both lexicographic objectives from complete candidate evidence.
        if is_infeasible:
            if any(expected_feasible):
                raise AssertionError("zero-limit fixture unexpectedly marked a candidate feasible")
        else:
            for objective in ("cgr", "ess_futile"):
                expected_index = _select(expected_metrics, expected_feasible, objective, n)
                row = next(
                    r
                    for r in selected_rows
                    if r["config_id"] == config_id and r["objective"] == objective
                )
                if expected_index != int(row["selected_index"]):
                    raise AssertionError(f"R {objective} index mismatch for {config_id}")
                python_index = (
                    result.selected_index if objective == "cgr" else result_ess.selected_index
                )
                if expected_index != python_index:
                    raise AssertionError(f"Python {objective} index mismatch for {config_id}")
                counts["selected_indices"][f"{config_id}/{objective}"] = python_index

        # Recompute every unique count-state posterior once, in bounded batches.
        # The design's candidate-0 cutoffs are sufficient; tails are cutoff-independent.
        first_grid_values = [grid[0] for grid in grids]
        design = bop2_dc_randomized_paired_design(
            n,
            endpoint,
            lrv,
            cmv,
            control_prior=control_prior,
            treatment_prior=treatment_prior,
            arm_assignments=arms,
            looks=looks,
            lambda_lrv=first_grid_values[0],
            lambda_cmv=first_grid_values[1],
            gamma_lrv=first_grid_values[2],
            gamma_cmv=first_grid_values[3],
            graduate_at_interim=setting["graduate_at_interim"] == "1",
            comparison_tolerance=1e-8,
        )
        state_rows = [r for r in monitor_rows if r["config_id"] == config_id]
        candidate_designs: dict[int, Any] = {}
        for candidate, parameter in enumerate(result.candidates.parameters):
            candidate_designs[candidate] = bop2_dc_randomized_paired_design(
                n,
                endpoint,
                lrv,
                cmv,
                control_prior=control_prior,
                treatment_prior=treatment_prior,
                arm_assignments=arms,
                looks=looks,
                lambda_lrv=parameter[0:2],
                lambda_cmv=parameter[2:4],
                gamma_lrv=parameter[4:6],
                gamma_cmv=parameter[6:8],
                graduate_at_interim=setting["graduate_at_interim"] == "1",
                comparison_tolerance=1e-8,
            )
        path_by_id = {
            int(r["path_id"]): r["path"] for r in path_rows if r["config_id"] == config_id
        }
        unique_count, classified_count, max_tail_difference = _compare_unique_tails(
            design, state_rows, config_id, path_by_id, arms, candidate_designs
        )
        counts["unique_posterior_states"] += unique_count
        counts["path_look_decisions"] += classified_count
        counts["max_tail_abs_difference"] = max(
            counts["max_tail_abs_difference"], max_tail_difference
        )

        # Public replay: one tape for each terminal category actually present at
        # the CGR-selected and ESS-selected candidates, avoiding repeated tails.
        if not is_infeasible:
            selected_by_obj = {
                r["objective"]: int(r["selected_index"])
                for r in selected_rows
                if r["config_id"] == config_id
            }
            for candidate in sorted(set(selected_by_obj.values())):
                rpaths = [
                    r
                    for r in path_rows
                    if r["config_id"] == config_id and int(r["candidate_index"]) == candidate
                ]
                chosen: dict[str, dict[str, str]] = {}
                for row in rpaths:
                    chosen.setdefault(row["terminal_decision"], row)
                parameter = result.candidates.parameters[candidate]
                design_selected = bop2_dc_randomized_paired_design(
                    n,
                    endpoint,
                    lrv,
                    cmv,
                    control_prior=control_prior,
                    treatment_prior=treatment_prior,
                    arm_assignments=arms,
                    looks=looks,
                    lambda_lrv=parameter[0:2],
                    lambda_cmv=parameter[2:4],
                    gamma_lrv=parameter[4:6],
                    gamma_cmv=parameter[6:8],
                    graduate_at_interim=setting["graduate_at_interim"] == "1",
                    comparison_tolerance=1e-8,
                )
                for label, row in chosen.items():
                    tape = np.asarray([int(c) - 1 for c in row["path"]], dtype=np.int64)
                    replay = design_selected.replay(tape)
                    if replay.terminal_decision != label:
                        raise AssertionError(
                            f"public replay {config_id} {candidate}: "
                            f"{label} != {replay.terminal_decision}"
                        )
                    terminal_n = int(row["terminal_look"])
                    assert len(replay.outcomes_observed) == terminal_n
                    assert [state.total_n for state in replay.states] == [
                        look for look in looks if look <= terminal_n
                    ]
                    counts["replays"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/bop2-dc-randomized-paired",
    )
    parser.add_argument(
        "--main-src", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    parser.add_argument(
        "--core-src", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    args = parser.parse_args()
    print(check(args.prefix, args.main_src, args.core_src))


if __name__ == "__main__":
    main()
