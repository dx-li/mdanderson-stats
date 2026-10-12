"""Published BLESS scope: independent Decimal source references and workflow contracts."""

import csv
import hashlib
import io
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import BLESS_TIME_DAYS, BLESSPublishedStudy, bless_published_prediction

FIXTURE = json.loads((Path(__file__).parent / "fixtures/bless-published-decimal.json").read_text())
CASES = FIXTURE["cases"]
EXAMPLES = {
    m: next(c for c in CASES if c["model"] == m) for m in ("lung", "breast", "hematologic", "gcmm")
}


@pytest.mark.parametrize("case", CASES)
def test_independent_decimal(case):
    prediction = bless_published_prediction(case["model"], case["predictors"])
    assert prediction.linear_predictor == pytest.approx(float(case["eta"]), abs=3e-14)
    np.testing.assert_allclose(
        prediction.survival, [float(s) for s in case["survival"]], rtol=3e-14, atol=2e-16
    )
    assert all(0 <= s <= 1 for s in prediction.survival)
    assert all(b <= a for a, b in zip(prediction.survival, prediction.survival[1:]))


@pytest.mark.parametrize("model", EXAMPLES)
def test_reference_identity_and_workflow(model):
    case = EXAMPLES[model]
    study = BLESSPublishedStudy(model, case["predictors"])
    assert study == BLESSPublishedStudy.from_json(study.to_json())
    result = study.predict()
    assert result.linear_predictor == pytest.approx(0, abs=1e-14)
    assert result.survival == pytest.approx(result.reference_survival, abs=2e-16)
    report = json.loads(result.to_json())
    assert report["source_warnings"] and report["limitations"]
    assert report["study"] == json.loads(study.to_json())
    rows = list(csv.DictReader(io.StringIO(result.to_csv())))
    assert [int(row["day"]) for row in rows] == list(BLESS_TIME_DAYS)
    assert [float(row["survival"]) for row in rows] == list(result.survival)
    subset = bless_published_prediction(model, case["predictors"], time_days=[30, 180])
    assert subset.survival == (result.survival[1], result.survival[-1])
    with pytest.raises(TypeError):
        study.predictors["ecog"] = 4
    with pytest.raises(FrozenInstanceError):
        study.model = "other"


def test_transcription_identity():
    path = Path(__file__).parents[1] / "research/bless-published-parameters-audit.json"
    if not path.exists():
        pytest.skip("source audit is in the repository, not the installed package")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE["source_audit_sha256"]


@pytest.mark.parametrize("model", EXAMPLES)
def test_missing_extra_and_malformed_predictors(model):
    x = EXAMPLES[model]["predictors"]
    for key in x:
        missing = dict(x)
        del missing[key]
        with pytest.raises(ValueError):
            bless_published_prediction(model, missing)
        for invalid in (None, [], {}, "unknown", float("nan"), float("inf"), -1):
            bad = dict(x, **{key: invalid})
            with pytest.raises(ValueError):
                bless_published_prediction(model, bad)
    with pytest.raises(ValueError):
        bless_published_prediction(model, dict(x, age=61))


@pytest.mark.parametrize(
    "times", [[], [1], [181], [-30], [30, 0], [30, 30], [True], [30.0], None, "30"]
)
def test_grid_is_exact(times):
    with pytest.raises(ValueError):
        bless_published_prediction("lung", EXAMPLES["lung"]["predictors"], time_days=times)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ecog", True),
        ("ecog", 1.0),
        ("ecog", 5),
        ("bilateral", 0.5),
        ("bilateral", 1.0),
        ("ldh_u_l", True),
    ],
)
def test_strict_types(key, value):
    with pytest.raises(ValueError):
        bless_published_prediction("lung", dict(EXAMPLES["lung"]["predictors"], **{key: value}))


@pytest.mark.parametrize(
    "text", ["{}", "[]", "null", "{", '{"profile":1,"profile":2}', "[" * 1100, " " * 16385]
)
def test_invalid_json(text):
    with pytest.raises(ValueError):
        BLESSPublishedStudy.from_json(text)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("profile", "daily"),
        ("model", "unknown"),
        ("time_days", None),
        ("predictors", []),
    ],
)
def test_schema_validation(key, value):
    doc = json.loads(BLESSPublishedStudy("lung", EXAMPLES["lung"]["predictors"]).to_json())
    doc[key] = value
    with pytest.raises(ValueError):
        BLESSPublishedStudy.from_json(json.dumps(doc))


def test_source_discrepancies_remain_visible():
    example = next(c for c in CASES if c["model"] == "breast" and c["predictors"]["nlr"] == 10)
    result = bless_published_prediction("breast", example["predictors"])
    assert result.survival[1] == pytest.approx(0.6312601492191469)
    assert abs(result.survival[1] - 0.644) > 0.01
    assert "conflicts" in result.to_json()
    hematologic = next(
        c for c in CASES if c["model"] == "hematologic" and c["predictors"]["protein_g_dl"] == 3
    )
    assert bless_published_prediction(
        "hematologic", hematologic["predictors"]
    ).linear_predictor == pytest.approx(-0.502)


def test_extreme_finite_predictors_without_overflow():
    x = dict(EXAMPLES["lung"]["predictors"], ldh_u_l=1e308)
    assert bless_published_prediction("lung", x).survival == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    x = dict(EXAMPLES["lung"]["predictors"], protein_g_dl=1e308)
    assert bless_published_prediction("lung", x).survival == (1.0,) * 7
    with pytest.raises(ValueError):
        bless_published_prediction("lung", dict(x, ldh_u_l=10**400))


def test_input_mutation_cannot_change_saved_study():
    x = dict(EXAMPLES["lung"]["predictors"])
    study = BLESSPublishedStudy("lung", x)
    x["ecog"] = 4
    assert study.predictors["ecog"] == 0
