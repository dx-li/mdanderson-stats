"""Compare explicit-tape Python GAO trial with independent base-R oracle."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.u2oet_decision import U2OETCriteria
from mdanderson_stats.u2oet_gao_fit import u2oet_gao_parameter_names
from mdanderson_stats.u2oet_gao_simulation import simulate_u2oet_gao_trial
from mdanderson_stats.u2oet_scenario import U2OETScenario

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/u2oet-gao-calendar"


def read(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    marg = read("marginals.csv")
    jrows = read("joint.csv")
    joint = np.zeros((2, 2, 2, 2))
    e = np.zeros((2, 2, 2))
    t = np.zeros_like(e)
    for r in marg:
        i, j = int(r["dose1"]), int(r["dose2"])
        e[i, j] = [float(r["efficacy0"]), float(r["efficacy1"])]
        t[i, j] = [float(r["toxicity0"]), float(r["toxicity1"])]
    for r in jrows:
        joint[int(r["dose1"]), int(r["dose2"]), int(r["efficacy"]), int(r["toxicity"])] = float(
            r["probability"]
        )
    scenario = U2OETScenario(e, t, 0.25, joint, np.zeros_like(joint))
    pars = {
        "efficacy.intercept.1.agent1": -1.2,
        "efficacy.intercept.1.agent2": -0.8,
        "efficacy.slope.1.agent1": 0.35,
        "efficacy.slope.1.agent2": 0.25,
        "efficacy.log_lambda": np.log(0.8),
        "toxicity.intercept.1.agent1": -2.0,
        "toxicity.intercept.1.agent2": -1.6,
        "toxicity.slope.1.agent1": 0.45,
        "toxicity.slope.1.agent2": 0.35,
        "toxicity.log_lambda": np.log(0.7),
        "shared.log_kappa": np.log(0.3),
        "association.fisher_z": np.arctanh(0.25),
    }
    names = u2oet_gao_parameter_names(2, 2)
    mean = np.array([pars[n] for n in names])
    tape = read("calendar_tape.csv")
    summaries = {r["case"]: r for r in read("calendar_summary.csv")}
    patients = read("patients.csv")
    snaps = read("snapshots.csv")
    finals = read("final_snapshots.csv")
    maxerr = 0.0
    checked = 0
    for case, summary in summaries.items():
        rows = [r for r in tape if r["case"] == case]
        n = len(rows)
        uniforms = np.array(
            [
                [float(r[k]) for k in ("u_alloc", "u_cell", "u_eff_delay", "u_tox_delay")]
                for r in rows
            ]
        )
        arrivals = np.array([float(r["arrival"]) for r in rows])
        criteria = U2OETCriteria(
            efficacy_level=1,
            toxicity_level=1,
            min_efficacy=0.95 if case == "stop_no_pair" else 0.25,
            max_toxicity=0.01 if case == "stop_no_pair" else 0.8,
            inefficacy_cutoff=0.9,
            toxicity_cutoff=0.9,
        )
        result = simulate_u2oet_gao_trial(
            [1, 2],
            [1, 2],
            scenario,
            [[0.1, 0], [1, 0.7]],
            prior_mean=mean,
            prior_sd=np.zeros_like(mean),
            initial=(1, 1) if case == "ar2" else (0, 0),
            criteria=criteria,
            max_patients=n,
            cohort_size=2,
            surplus=2,
            top=2,
            efficacy_window=(0, 6),
            toxicity_window=(0, 6),
            final_scope="tried" if case == "scope_tried" else "acceptable",
            draws=8,
            warmup=0,
            chains=2,
            arrival_times=arrivals,
            data_uniforms=uniforms,
            rng=np.random.default_rng(531),
        )
        expected_early = summary["early"].lower() == "true"
        if result.stopped_early != expected_early:
            raise AssertionError(f"{case}: early-stop mismatch")
        for field, actual in [
            ("stop_time", result.stop_time),
            ("analysis_time", result.analysis_time),
        ]:
            err = abs(float(summary[field]) - float(actual))
            maxerr = max(maxerr, err)
            if err > 1e-12:
                raise AssertionError(f"{case}: {field} mismatch")
        sel = summary["selected"]
        expected_sel = None if sel in ("NA", "") else ((int(sel) - 1) // 2, (int(sel) - 1) % 2)
        if result.selected != expected_sel:
            raise AssertionError(f"{case}: selected {result.selected} != {expected_sel}")
        got = result.patients.records
        want = [r for r in patients if r["case"] == case]
        if got.shape[0] != len(want):
            raise AssertionError(f"{case}: patient count mismatch")
        for i, r in enumerate(want):
            wr = np.array(
                [int(r[k]) for k in ("patient", "dose1", "dose2", "efficacy", "toxicity")]
            )
            if not np.array_equal(got[i], wr):
                raise AssertionError(f"{case}: patient row {i + 1} differs {got[i]} != {wr}")
            err = float(
                np.max(
                    np.abs(
                        result.outcome_times[i]
                        - [float(r["efficacy_time"]), float(r["toxicity_time"])]
                    )
                )
            )
            maxerr = max(maxerr, err)
            if err > 1e-12:
                raise AssertionError(f"{case}: event times differ")
        actual = list(result.decisions)
        expected = [r for r in snaps if r["case"] == case and float(r["time"]) > 0]
        if len(actual) != len(expected):
            raise AssertionError(f"{case}: decision count differs {len(actual)} != {len(expected)}")
        for a, w in zip(actual, expected, strict=True):
            pairs = [
                ("time", a.time),
                ("assigned", a.complete_patients + a.toxicity_only_patients + a.ignored_patients),
                ("complete", a.complete_patients),
                ("toxicity_only", a.toxicity_only_patients),
                ("ignored", a.ignored_patients),
            ]
            for k, v in pairs:
                err = abs(float(w[k]) - float(v))
                maxerr = max(maxerr, err)
                if err > 1e-12:
                    raise AssertionError(f"{case}: snapshot {k} differs")
        f = next(r for r in finals if r["case"] == case)
        gotdata = result.patients
        for k, v in [
            ("assigned", gotdata.records.shape[0]),
            ("complete", gotdata.complete.sum()),
            ("toxicity_only", gotdata.toxicity_only.sum()),
            ("ignored", gotdata.ignored_outcomes),
        ]:
            err = abs(float(f[k]) - float(v))
            maxerr = max(maxerr, err)
            if err > 1e-12:
                raise AssertionError(f"{case}: final {k} differs")
        for row in read("model.csv"):
            i, j = int(row["dose1"]), int(row["dose2"])
            error = abs(result.final_posterior.mean_utility[i, j] - float(row["mean_utility"]))
            maxerr = max(maxerr, error)
            assert error < 2e-12, (case, "posterior utility", error)
        checked += 1
        print(
            f"{case}: patients={len(want)} decisions={len(expected)} selected={result.selected} "
            f"stop={result.stop_time:g} analysis={result.analysis_time:g}"
        )
    print(f"PASS cases={checked} max_calendar_absolute_error={maxerr:.3g}")


if __name__ == "__main__":
    main()
