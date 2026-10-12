"""Methods include saved settings and distinguish complete from pending outcomes."""

import pytest
from test_boin_desktop_univariate_simulation import scenario_study

from mdanderson_stats import boin_desktop_univariate_protocol


@pytest.mark.parametrize("tite", [False, True])
@pytest.mark.parametrize("language", ["en", "zh"])
def test_saved_univariate_methods(tite, language):
    study = scenario_study(tite)
    report = boin_desktop_univariate_protocol(study, language=language)
    assert "12 |" in report
    if tite:
        assert "30.436666666666667" in report
        assert "STFT" in report
    if language == "en":
        assert "not a complete clinical" in report
        assert "Run-in modification active" in report
        assert "Complete-outcome integer boundaries" in report
        if not tite:
            assert "min(cap+1" in report


def test_unsupported_language():
    with pytest.raises(ValueError, match="language"):
        boin_desktop_univariate_protocol(scenario_study(False), language="fr")
