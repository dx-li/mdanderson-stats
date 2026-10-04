"""Contract checks for readable decentralized-trial planning reports."""

import pytest

from mdanderson_stats.dct_report import dct_sample_size_report


def test_continuous_report_records_defaults_rounding_and_citation(tmp_path):
    report = dct_sample_size_report(
        "continuous",
        allocation_unit="participants",
        effect=10,
        onsite_sd=20,
        offsite_sd=20,
        offsite_fraction=1,
    )
    assert report.allocation == ((0, 0), (63, 63))
    assert report.rounded_total == 126
    assert report.unrounded_total < 126
    assert report.achieved_power >= report.target_power
    assert dict(report.settings)["onsite_experimental_sd"] == 20
    assert dict(report.settings)["offsite_experimental_sd"] == 20
    text = report.report()
    assert "Allocation unit: participants" in text
    assert "Rounded total allocation: 126 participants" in text
    assert "independent upward rounding" in text
    assert "Native help lists 128" in text
    assert "dyaf053" in text

    path = report.write_report(tmp_path / "dct.txt")
    assert path.read_text(encoding="utf-8") == text


def test_binary_report_explicitly_reports_cluster_units():
    report = dct_sample_size_report(
        "binary",
        allocation_unit="clusters",
        onsite_control=0.2,
        onsite_experimental=0.4,
        offsite_control=0.25,
        offsite_experimental=0.4,
        onsite_repeats=4,
        offsite_repeats=6,
    )
    text = report.report()
    assert report.allocation_unit == "clusters"
    assert "Allocation unit: clusters" in text
    assert "clusters" in text
    assert "participant totals are not inferred from cluster counts" in text
    assert "Endpoint: binary" in text
    assert "weighted difference-in-proportions z-test normal approximation" in text


def test_invalid_settings_fail_before_report_and_failed_write_preserves_file(tmp_path, monkeypatch):
    with pytest.raises(TypeError, match="scalar setting"):
        dct_sample_size_report(
            "continuous",
            allocation_unit="participants",
            effect=[10, 11],
            onsite_sd=20,
            offsite_sd=20,
        )
    with pytest.raises(ValueError, match="allocation_unit"):
        dct_sample_size_report(
            "continuous", allocation_unit="unspecified", effect=10, onsite_sd=20, offsite_sd=20
        )

    report = dct_sample_size_report(
        "continuous", allocation_unit="participants", effect=10, onsite_sd=20, offsite_sd=25
    )
    path = tmp_path / "existing.txt"
    path.write_text("original", encoding="utf-8")

    def fail_render(self):
        raise ValueError("render failed")

    monkeypatch.setattr(type(report), "report", fail_render)
    with pytest.raises(ValueError, match="render failed"):
        report.write_report(path)
    assert path.read_text(encoding="utf-8") == "original"
