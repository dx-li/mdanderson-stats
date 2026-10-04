from importlib import import_module

import pytest

from mdanderson_stats.cid2bp_session import CID2BPRequest, cid2bp_session

session_module = import_module("mdanderson_stats.cid2bp_session")


def test_ordered_repeated_methods_entry_modes_and_report(tmp_path):
    result = cid2bp_session(
        [
            CID2BPRequest(7, 5, 1, 6, entry="failures", methods=("auto", "exact")),
            CID2BPRequest(
                10,
                10,
                0,
                10,
                entry="trials",
                confidence=0.99,
                methods=("wald",),
            ),
        ]
    )
    assert len(result.records) == 3
    first, second, third = result.records
    assert (first.comparison_index, first.trials1, first.failures1) == (1, 12, 5)
    assert (first.trials2, first.failures2, first.requested_method) == (7, 6, "auto")
    assert first.result.method == "weighted_likelihood"
    assert second.requested_method == second.result.method == "exact"
    assert (third.comparison_index, third.trials1, third.failures1) == (2, 10, 0)
    assert third.confidence == 0.99

    report = result.report()
    assert "Requested method\tMethod" in report
    assert "auto\tweighted_likelihood" in report
    assert "exact\texact" in report
    destination = tmp_path / "session.tsv"
    assert result.write_report(destination) == destination
    assert destination.read_text(encoding="utf-8") == report
    with pytest.raises(ValueError, match="digits"):
        result.write_report(destination, digits=0)
    assert destination.read_text(encoding="utf-8") == report


def test_all_nine_methods_match_native_reference_example():
    methods = (
        "wald",
        "continuity_corrected",
        "yates",
        "peskun_native",
        "cox_snell",
        "weighted_mid_p",
        "weighted_likelihood",
        "auto",
        "exact",
    )
    result = cid2bp_session([CID2BPRequest(7, 12, 1, 7, methods=methods)])
    expected = {
        "wald": (0.0597, 0.8213),
        "continuity_corrected": (-0.0350, 0.9160),
        "yates": (-0.0534, 0.9344),
        "peskun_native": (-0.0313, 0.7663),
        "cox_snell": (-0.0013, 0.7523),
        "weighted_mid_p": (-0.0069, 0.7551),
        "weighted_likelihood": (-0.0247, 0.7645),
        "auto": (-0.0247, 0.7645),
        "exact": (-0.0469, 0.8159),
    }
    assert len(result.records) == 9
    for record in result.records:
        lower, upper = expected[record.requested_method]
        assert record.result.method == (
            "weighted_likelihood" if record.requested_method == "auto" else record.requested_method
        )
        assert record.result.lower == pytest.approx(lower, abs=0.00015)
        assert record.result.upper == pytest.approx(upper, abs=0.00015)


def test_preflight_rejects_whole_invalid_or_oversized_session_before_calculation(
    tmp_path, monkeypatch
):
    destination = tmp_path / "existing.txt"
    destination.write_text("keep this report", encoding="utf-8")

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("preflight must finish before interval calculations")

    monkeypatch.setattr(session_module, "cid2bp_interval", unexpected_calculation)
    with pytest.raises(ValueError, match="successes must lie"):
        cid2bp_session(
            [
                CID2BPRequest(1, 2, 0, 2, methods=("wald",)),
                CID2BPRequest(3, 2, 0, 2),
            ]
        )
    assert destination.read_text(encoding="utf-8") == "keep this report"

    many_methods = ("wald",) * 600
    with pytest.raises(ValueError, match="1000-calculation limit"):
        cid2bp_session(
            [
                CID2BPRequest(1, 2, 0, 2, methods=many_methods),
                CID2BPRequest(1, 2, 0, 2, methods=many_methods),
            ]
        )
    assert destination.read_text(encoding="utf-8") == "keep this report"

    with pytest.raises(ValueError, match="2 million grid points"):
        cid2bp_session(
            [
                CID2BPRequest(
                    0,
                    1500,
                    0,
                    1500,
                    methods=("wald", "peskun_native"),
                )
            ]
        )
    assert destination.read_text(encoding="utf-8") == "keep this report"
