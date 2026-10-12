"""Compare per-arm desktop cutoffs with the original native instruction paths."""

import json
import random
from pathlib import Path

import arand_final_reference as final
import arand_groupfinal_reference as group
import arand_status_reference as status

from mdanderson_stats.arand_native import arand_native_select, arand_native_statuses

rng = random.Random(921)
fixtures = []
for index in range(600):
    k = rng.randint(1, 4)
    p = [rng.choice([0, 0.05, 0.1, 0.5, 0.9, 0.95, 1]) for _ in range(k)]
    states = [rng.choice([0, 1, 2, 3, 4, 5, 9]) for _ in range(k)]
    f = [rng.choice([0, 0.1, 0.5, 1]) for _ in range(k)]
    arms = [
        dict(
            upper=rng.choice([0.5, 0.95, 1]),
            lower=rng.choice([0, 0.1, 0.5]),
            cutoff=rng.choice([0, 0.5, 1]),
            threshold=0.5,
            futility_cutoff=rng.choice([0, 0.2, 1]),
            enforce=rng.randrange(2),
        )
        for _ in range(k)
    ]
    low = tuple(a["lower"] for a in arms)
    high = tuple(a["upper"] for a in arms)
    cutoff = tuple(a["cutoff"] for a in arms)
    futility = tuple(a["futility_cutoff"] if a["enforce"] else 0 for a in arms)
    min_action, max_action = rng.randrange(4), rng.randrange(4)
    expected = status.run(
        p, states, futility=f, min_action=min_action, max_action=max_action, arm_rules=arms
    )["statuses"]
    actual = arand_native_statuses(
        p,
        tuple(states),
        lower=low,
        upper=high,
        futility=f,
        futility_cutoff=futility,
        min_action=min_action,
        max_action=max_action,
    )
    assert tuple(expected) == actual, (index, p, states, arms, expected, actual)
    completed = bool(rng.randrange(2))
    selections = []
    for grouped, module in [(False, final), (True, group)]:
        calls = module.run(p, states, completed=completed, arm_rules=arms)
        winner = next(c for c in calls if c[0] == "winner")
        expected_selection = [None if winner[2] == "None" else int(winner[2]), bool(winner[3])]
        actual_selection = arand_native_select(
            p, tuple(states), cutoff=cutoff, upper=high, completed=completed, group=grouped
        )
        assert [actual_selection.winner, actual_selection.early] == expected_selection, (
            index,
            calls,
            actual_selection,
        )
        selections.append(expected_selection)
    if index < 60:
        fixtures.append(
            dict(
                probabilities=p,
                statuses=states,
                arm_rules=arms,
                futility=f,
                min_action=min_action,
                max_action=max_action,
                completed=completed,
                expected_statuses=expected,
                expected_selections=selections,
            )
        )
path = Path("research/fixtures/arand-native/per-arm-rules.json")
path.write_text(json.dumps(fixtures, indent=2) + "\n")
print("1800 original per-arm status/whole/group selector comparisons passed; 60 portable cases")
