import json
from pathlib import Path

import pytest

from mdanderson_stats import sortf90


def test_original_perl_output_and_idempotence():
    cases = json.loads((Path(__file__).parent / "fixtures/sortf90-native.json").read_text())
    for case in cases:
        result = sortf90(case["source"])
        assert result == case["expected"]
        assert sortf90(result) == result


def test_reject_incomplete_or_ambiguous_source_and_preserve_string_literals():
    for source in (
        "module test\n",
        "module test\nend program test\n",
        "module test\ncontains\nsubroutine a()\nend subroutine\n"
        "subroutine A()\nend subroutine\nend module\n",
        "module test\ntype :: data\nend type\nend module\n",
        "program test\nend\n",
        "program test; print *, 1\nend program\n",
    ):
        with pytest.raises(ValueError):
            sortf90(source)
    source = 'program test\nprint *, "! function bogus()"\nend program test\n'
    assert sortf90(source) == source
