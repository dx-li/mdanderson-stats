"""Strict noncovariate ARAND desktop study reader with retained source XML.

This module maps simulation inputs. Persisted patient histories are preserved,
not silently used as newly simulated observations. Use an explicit clinical
history importer before making recommendations from such records.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .arand_native import (
    ArandNativeArmLimits,
    ArandNativeGates,
    ArandNativeMonitor,
    arand_native_batch_monitor,
)
from .arand_native_study import ArandNativeStudy
from .arand_native_trial import ArandNativeRules


@dataclass(frozen=True)
class ArandNativeDocument:
    source_xml: str
    name: str
    seed: int
    arm_names: tuple[str, ...]
    studies: tuple[ArandNativeStudy, ...]
    replications: tuple[int, ...]
    has_patient_history: bool


def read_arand_native_xml(text: str, *, max_candidates: int = 1000) -> ArandNativeDocument:
    """Read supported binary/TTE simulation inputs without dropping XML metadata.

    Missing enable flags are false; absent/unknown actions retain native code 2.
    Original XML and patient history remain available byte-for-byte as text.
    Simulation studies start fresh, as opposed to importing a clinical history.
    Unsupported model/group/endpoint branches fail explicitly.
    """
    if (
        not isinstance(text, str)
        or len(text) > 2_000_000
        or "<!DOCTYPE" in text.upper()
        or "<!ENTITY" in text.upper()
    ):
        raise ValueError("require bounded XML without entity or document declarations")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("invalid ARAND XML") from exc
    if root.tag != "simulation":
        raise ValueError("require a simulation root")

    def one(parent: ET.Element, tag: str) -> ET.Element:
        matches = parent.findall(tag)
        if len(matches) != 1:
            raise ValueError(f"require exactly one {tag}")
        return matches[0]

    def number(node: ET.Element, key: str, default: float | None = None) -> float:
        value = node.get(key)
        if value is None and default is None:
            raise ValueError(f"missing numeric attribute {key}")
        try:
            result = float(value) if value is not None else float(default)  # type: ignore[arg-type]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid numeric attribute {key}") from exc
        if not math.isfinite(result):
            raise ValueError(f"nonfinite numeric attribute {key}")
        return result

    def integer(node: ET.Element, key: str, default: int | None = None) -> int:
        value = number(node, key, default)
        if value != int(value):
            raise ValueError(f"{key} must be integral")
        return int(value)

    def flag(node: ET.Element, key: str) -> bool:
        value = node.get(key, "false")
        if value not in ("false", "true", "0", "1"):
            raise ValueError(f"invalid boolean attribute {key}")
        return value in ("true", "1")

    trial = one(root, "trial_details")
    if trial.findtext("time_type") != "Months":
        raise ValueError("this simulation profile requires explicit Months time units")
    calculator = one(trial, "calculator")
    model = calculator.get("name")
    if model not in ("Binary", "TTE") or trial.findall("covariate"):
        raise ValueError("only independent Binary and TTE models are supported")
    outcome = one(calculator, "calc_outcome")
    endpoint = one(trial, "outcome").get("name")
    if not endpoint or outcome.get("name") != endpoint:
        raise ValueError("require one consistently named endpoint")
    if outcome.get("prior_dist") != ("Beta" if model == "Binary" else "InverseGamma"):
        raise ValueError("unsupported prior family")
    arms = trial.findall("arm")
    names = tuple(sorted(a.get("name", "") for a in arms))
    if not 1 <= len(names) <= 10 or len(set(names)) != len(names) or not all(names):
        raise ValueError("require one to ten uniquely named arms")
    for arm in arms:
        if one(arm, "arm_group").get("name") != "No Groups":
            raise ValueError("only the advertised No Groups model is supported")
        if set(arm.attrib) - {"name", "description"}:
            raise ValueError("individual arm limits require an explicit supported mapping")
    prior_by_arm = {}
    for item in outcome.findall("calc_parameters"):
        name = item.get("arm_name")
        if item.get("group_name") != "No Groups" or name in prior_by_arm:
            raise ValueError("invalid prior group or duplicate arm")
        prior_by_arm[name] = (number(item, "param1"), number(item, "param2"))
    if set(prior_by_arm) != set(names):
        raise ValueError("require exactly one prior pair per arm")
    stopping = one(trial, "stopping_rule")
    gates = ArandNativeGates(
        group_minimum=(flag(stopping, "min_pats_per_group") or flag(stopping, "min_pat_per_group")),
        group_maximum=(flag(stopping, "max_pats_per_group") or flag(stopping, "max_pat_per_group")),
        arm_minimum=(flag(stopping, "min_pats_per_arm") or flag(stopping, "min_pat_per_arm")),
        arm_maximum=(flag(stopping, "max_pats_per_arm") or flag(stopping, "max_pat_per_arm")),
        trial_minimum=flag(stopping, "min_pats_in_trial") or flag(stopping, "min_pat_in_trial"),
        trial_maximum=flag(stopping, "max_pats_in_trial") or flag(stopping, "max_pat_in_trial"),
        duration_minimum=flag(stopping, "min_trial_length"),
        duration_maximum=flag(stopping, "max_trial_length"),
    )

    def limit(key: str, enabled: bool, default: int) -> int:
        return integer(trial, key) if enabled else default

    limits = ArandNativeArmLimits(
        limit("min_pats_per_arm", gates.arm_minimum, 0),
        limit("max_pats_per_arm", gates.arm_maximum, 2**31 - 1),
        limit("min_pats_per_group", gates.group_minimum, 0),
        limit("max_pats_per_group", gates.group_maximum, 2**31 - 1),
    )
    arm_rules = stopping.findall("stopping_rule_arm")
    mapping = {a.get("name"): a for a in arm_rules}
    if arm_rules and (len(mapping) != len(arm_rules) or set(mapping) != set(names)):
        raise ValueError("require exactly one stopping rule per arm")
    for rule in arm_rules:
        if list(rule) or flag(rule, "enforce_min_pred_prob") or flag(rule, "enforce_max_pred_prob"):
            raise ValueError("predictive or per-group stopping is outside this profile")
    rule_nodes = [mapping[n] for n in names] if arm_rules else [stopping] * len(names)
    lower_key, upper_key, final_key = (
        ("pl", "pu", "pu_final")
        if arm_rules
        else ("min_prob_best", "max_prob_best", "final_max_prob_best")
    )
    thresholds = tuple(
        number(r, "comp_min_param_value") if flag(r, "enforce_comp_min_param_value") else 0
        for r in rule_nodes
    )
    randomizer = one(trial, "randomizer")
    if flag(randomizer, "use_ballancing") or randomizer.get("treatment"):
        raise ValueError("fixed treatment and covariate balancing are outside this profile")
    action = {"open": 0, "tmp_close": 1, "close_early": 2, "close": 3}
    rules = ArandNativeRules(
        arms=(limits,) * len(names),
        gates=gates,
        minimum_enrollment=limit("min_qty_pats_in_trial", gates.trial_minimum, 0),
        maximum_enrollment=limit("max_qty_pats_in_trial", gates.trial_maximum, 10000),
        minimum_duration=number(trial, "min_trial_length") if gates.duration_minimum else 0,
        maximum_duration=number(trial, "max_trial_length") if gates.duration_maximum else 1e300,
        final_followup=number(stopping, "additional_fu", 0),
        stop_when_one_arm=flag(stopping, "stop_when_one_treat"),
        lower=tuple(number(r, lower_key, None if arm_rules else 0) for r in rule_nodes),
        upper=tuple(number(r, upper_key, None if arm_rules else 1) for r in rule_nodes),
        final_cutoff=tuple(number(r, final_key, None if arm_rules else 0) for r in rule_nodes),
        futility_cutoff=tuple(
            number(r, "comp_min_param_prob") if flag(r, "enforce_comp_min_param_value") else 0
            for r in rule_nodes
        ),
        min_action=action.get(stopping.get("min_prob_action", ""), 2),
        max_action=action.get(stopping.get("max_prob_action", ""), 2),
        power=number(randomizer, "tunning_parameter", 1),
        floor=number(randomizer, "min_rand_prob", 0),
        runin=integer(randomizer, "qty_pats_enroll_b4_adapt", 0),
        force_equal=flag(randomizer, "force_equal_qty_pats"),
    )
    mode = trial.get("monitoring_type", "continuous")
    if mode == "continuous":
        monitor = ArandNativeMonitor(0)
    elif mode == "once per unit time":
        monitor = ArandNativeMonitor(1)
    elif mode == "specified number of patients":
        step = integer(trial, "qty_pats_betw_eval", 1)
        monitor = ArandNativeMonitor(3, next_enrollment=step, step=step)
    elif mode == "exponential":
        monitor = arand_native_batch_monitor(
            number(trial, "a", 1),
            number(trial, "b", 0),
            integer(trial, "max_qty_pats_in_trial"),
            reset=True,
        )
    elif mode in ("discrete", "specified number of events"):
        try:
            targets = tuple(
                json.loads(
                    trial.get("monitor_times" if mode == "discrete" else "number_events", "")
                )
            )
        except (ValueError, TypeError) as exc:
            raise ValueError("monitor targets must use a numeric bracketed list") from exc
        if not targets:
            raise ValueError("monitor targets cannot be empty")
        monitor = ArandNativeMonitor(
            2 if mode == "discrete" else 5,
            targets=targets,
            next_time=targets[0],
            next_outcome=int(targets[0]),
        )
    else:
        raise ValueError("unsupported monitoring type")
    direction = calculator.get("max_or_min", "max")
    if direction not in ("max", "min") or flag(calculator, "use_delta_futility"):
        raise ValueError("unsupported objective or delta futility")
    parameter = calculator.get("calc_tte_meth", "Mean").lower() if model == "TTE" else "mean"
    if parameter not in ("mean", "median"):
        raise ValueError("unsupported TTE parameter")
    studies, replications = [], []
    for scenario in root.findall("scenario"):
        details = one(scenario, "sim_details")
        accrual = one(one(details, "accrual"), "accrual_group")
        simulated = one(details, "sim_outcome")
        if (
            accrual.get("name") != "No Groups"
            or simulated.get("name") != endpoint
            or simulated.get("sim_type") != model
        ):
            raise ValueError("scenario does not match the noncovariate model")
        parameters = {}
        for item in simulated.findall("param_set"):
            name = item.get("arm")
            if item.get("group") != "No Groups" or name in parameters:
                raise ValueError("invalid scenario group or duplicate arm")
            values = {p.get("name"): number(p, "value") for p in item.findall("param")}
            if len(values) != len(item.findall("param")):
                raise ValueError("duplicate scenario parameter")
            required = (
                {"ProbSuccess", "TimeBetweenEvals"} if model == "Binary" else {parameter.title()}
            )
            if set(values) != required:
                raise ValueError("unsupported or missing scenario parameter")
            parameters[name] = values
        if set(parameters) != set(names):
            raise ValueError("scenario requires each declared arm")
        studies.append(
            ArandNativeStudy(
                rules=rules,
                monitor=monitor,
                prior=tuple(prior_by_arm[n] for n in names),
                true_parameters=tuple(
                    parameters[n]["ProbSuccess" if model == "Binary" else parameter.title()]
                    for n in names
                ),
                accrual_rate=number(accrual, "acc"),
                family="binary" if model == "Binary" else "exponential",
                parameter=parameter,
                maximize=direction == "max",
                futility_threshold=thresholds,
                binary_delay=tuple(parameters[n]["TimeBetweenEvals"] for n in names)
                if model == "Binary"
                else 1,
                max_candidates=max_candidates,
            )
        )
        reps = integer(scenario, "qty_reps")
        if reps < 1:
            raise ValueError("replications must be positive")
        replications.append(reps)
    if not studies:
        raise ValueError("require at least one scenario")
    seed = integer(root, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    return ArandNativeDocument(
        text,
        root.get("name", ""),
        seed,
        names,
        tuple(studies),
        tuple(replications),
        bool(trial.findall("./current_patients/patient")),
    )
