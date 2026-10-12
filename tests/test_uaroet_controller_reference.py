"""Independent reconstruction of bounded native cohort and parser references."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent / "fixtures"
CONTROLLER = json.loads((ROOT / "uaroet-desktop-controller.json").read_text())
KEYWORDS = json.loads((ROOT / "uaroet-desktop-keyword.json").read_text())


@pytest.mark.parametrize("case", CONTROLLER["cases"])
def test_native_cohort_trace(case):
    x = case["inputs"]
    seen = [0] * 3
    n = 0
    new_open = old_open = False
    new_start = old_start = 0
    last = 0
    trace = []
    for utility, status in zip(x["utility_sequence"], x["status_sequence"]):
        eligible = [i for i, v in enumerate(status) if v == 1]
        if n < x["first"]:
            selected = 0
        elif not eligible:
            trace.append(dict(selected=-1, stop=True, new_open=new_open, old_open=old_open))
            break
        elif (new_open or old_open) and status[last] == 1:
            selected = last
        else:
            selected = max(eligible, key=lambda i: (utility[i], -i))
        last = selected
        n += 1
        if seen[selected] == 0:
            seen[selected] = n
            new_start = n
            if x["new"] > 1:
                new_open = True
        elif not new_open and not old_open and x["old"] > 1:
            old_open = True
            old_start = n
        if new_open:
            if n >= new_start + x["new"] - 1:
                new_open = False
        elif old_open and n >= old_start + x["new"] - 1:
            # Native old-cohort closure reads the NEW cohort size.
            old_open = False
        trace.append(dict(selected=selected, stop=False, new_open=new_open, old_open=old_open))
    assert trace == case["trace"]


@pytest.mark.parametrize("case", KEYWORDS["cases"])
def test_native_keyword_spelling(case):
    matched = case["word"] in ("UseHybridGreedyArandAssignment", "UseHybridGreedyARandAssignment")
    assert case["result"] == dict(
        matched=matched, flag=bool(case["value"]) if matched else case["initial_flag"]
    )


ESS = json.loads((ROOT / "uaroet-desktop-ess.json").read_text())


@pytest.mark.parametrize("case", ESS["cases"])
def test_native_beta_moment_reduction(case):
    import numpy as np

    values = []
    endpoint_means = []
    for endpoint in ("efficacy", "toxicity"):
        draws = np.asarray(case[endpoint])
        mean = draws.mean(axis=0)
        expected = mean * (1 - mean) / draws.var(axis=0, ddof=1) - 1
        np.testing.assert_allclose(case["result"][endpoint], expected, atol=3e-13, rtol=3e-14)
        values.extend(expected.ravel())
        endpoint_means.append(expected.mean())
    np.testing.assert_allclose(
        case["result"]["summary"],
        [*endpoint_means, np.mean(values), min(values), max(values)],
        atol=3e-13,
        rtol=3e-14,
    )


SUMMARY = json.loads((ROOT / "uaroet-desktop-summary.json").read_text())


@pytest.mark.parametrize("case", SUMMARY["cases"])
def test_native_per_draw_summary_and_distinct_tie_rules(case):
    import numpy as np

    x = case["inputs"]
    joint = np.asarray(x["joint"])
    efficacy = joint.sum(2)
    toxicity = joint.sum(1)
    D, E, T = joint.shape
    expected = np.zeros((D, 13))
    expected[:, : E - 1] = efficacy[:, 1:]
    expected[:, 3 : 3 + T - 1] = toxicity[:, 1:]
    expected[:, 6] = x["utility"]
    # Per-draw best probability credits the first tie, whereas the hybrid
    # controller's greedy mean-utility tie chooses the last eligible dose.
    expected[np.argmax(x["utility"]), 7] = 1.0
    for d in range(D):
        expected[d, 8] = sum(toxicity[d, x["bad_tox"] :]) > x["tox_limit"]
        expected[d, 9] = sum(efficacy[d, x["good_eff"] :]) < x["eff_limit"]
    expected[:, 10] = (joint * np.asarray(x["good_mask"])).sum(axis=(1, 2))
    expected[:, 11] = efficacy[:, 0]
    expected[:, 12] = toxicity[:, 0]
    np.testing.assert_allclose(case["result"], expected, atol=1e-14, rtol=1e-14)


COUNTS = json.loads((ROOT / "uaroet-desktop-counts.json").read_text())


@pytest.mark.parametrize("case", COUNTS["cases"])
def test_native_complete_counts_and_likelihood_escalation(case):
    D, E, T = case["doses"], case["efficacy_levels"], case["toxicity_levels"]
    assigned, complete = [0] * D, [0] * D
    efficacy, toxicity = [[0] * E for _ in range(D)], [[0] * T for _ in range(D)]
    high = 0
    for _, dose, eff, tox in case["patients"]:
        assigned[dose] += 1
        high = max(high, dose)
        if eff >= 0 and tox >= 0:
            complete[dose] += 1
            efficacy[dose][eff] += 1
            toxicity[dose][tox] += 1
    used = case["likelihood_counts"]
    assert case["result"] == dict(
        assigned=assigned,
        complete=complete,
        likelihood=used,
        efficacy=efficacy,
        toxicity=toxicity,
        highest_assigned=high,
        highest_allowed=min(D - 1, high + 1) if used[high] else high,
    )


FINAL = json.loads((ROOT / "uaroet-desktop-final.json").read_text())


@pytest.mark.parametrize("case", FINAL["cases"])
def test_native_final_recomputes_ceiling_and_keeps_first_greedy_tie(case):
    x = case["inputs"]
    highest = max(row[1] for row in x["patients"])
    used = any(d == highest and t >= 0 for _, d, e, t in x["patients"])
    ceiling = min(2, highest + 1) if used else highest
    status = []
    lower_toxic = False
    for dose in range(3):
        bad_tox = any(d == dose and e >= 0 and t == 3 for _, d, e, t in x["patients"])
        bad_eff = any(d == dose and e == 0 and t >= 0 for _, d, e, t in x["patients"])
        if dose > ceiling:
            value = -5
        elif x["tox_tails"][dose] > 0.8 and (bad_tox or lower_toxic):
            value = -2
            lower_toxic = True
        elif x["eff_tails"][dose] > 0.8 and bad_eff:
            value = -1
        else:
            value = 1
        status.append(value)
    eligible = [i for i, v in enumerate(status) if v == 1]
    selected = max(eligible, key=lambda i: (x["utility"][i], -i)) if eligible else -1
    assert case["result"] == dict(selected=selected, highest_allowed=ceiling, status=status)


REPLAY = json.loads((ROOT / "uaroet-desktop-replay.json").read_text())


@pytest.mark.parametrize("case", REPLAY["cases"])
def test_native_saved_history_reconstruction_uses_last_record_newness(case):
    doses = case["doses"]
    assigned, first_seen = [0] * 3, [0] * 3
    trailing = 0
    last = -1
    new = False
    for index, dose in enumerate(doses, 1):
        assigned[dose] += 1
        new = first_seen[dose] == 0
        if new:
            first_seen[dose] = index
        trailing = trailing + 1 if dose == last else 1
        last = dose
    size = case["new"] if new else case["old"]
    incomplete = bool(doses) and size > 1 and trailing % size != 0
    assert case["result"] == dict(
        assigned=assigned,
        first_seen=first_seen,
        total=len(doses),
        last=last if incomplete else 0,
        new_open=incomplete and new,
        old_open=incomplete and not new,
        incomplete=incomplete,
    )
