import json
from pathlib import Path

import pytest

from mdanderson_stats import (
    RanlistSession,
    RanlistSpecification,
    ranlist_parameter_text,
    ranlist_seeds,
    ranlist_starting_seeds,
    read_ranlist_parameters,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranlist_creation.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_list_creation(case):
    if not case["restricted"] and "*" in case["parameters"].splitlines()[-3]:
        with pytest.raises(ValueError, match="balance"):
            read_ranlist_parameters(case["parameters"])
    native = read_ranlist_parameters(case["parameters"], repair_unrestricted_balance=True)
    phrase = case["phrase"]
    seed = (1, 1) if phrase is None else ranlist_starting_seeds(phrase)
    spec = RanlistSpecification(
        (1, 2),
        restricted=case["restricted"],
        legacy=True,
        balance=(1, 3) if case["restricted"] else (1, 1),
        seed=seed,
        phrase="" if phrase is None else phrase[:31],
        strata=("North", "South"),
        treatments=("Control", "Experimental"),
        title=("Native creation",),
    )
    assert native == RanlistSession(spec)
    assert read_ranlist_parameters(ranlist_parameter_text(native)) == native


def test_starting_phrase_retains_hash_api_and_truncates_explicitly():
    phrase = "a phrase longer than thirty one characters"
    assert ranlist_starting_seeds(phrase) == ranlist_seeds(phrase[:31])
    assert ranlist_starting_seeds(phrase) != ranlist_seeds(phrase)
    assert ranlist_starting_seeds("") == ranlist_seeds("")


@pytest.mark.parametrize("phrase", [None, 1, "é"])
def test_invalid_starting_phrase(phrase):
    with pytest.raises(ValueError, match="ASCII"):
        ranlist_starting_seeds(phrase)


def test_repair_never_ignores_restricted_balance():
    text = next(c["parameters"] for c in FIXTURE["cases"] if c["restricted"])
    lines = text.splitlines()
    lines[-3] = "******"
    with pytest.raises(ValueError, match="balance"):
        read_ranlist_parameters("\n".join(lines), repair_unrestricted_balance=True)
