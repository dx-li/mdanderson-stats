import json
from decimal import Inexact, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import format_number


def test_native_integer_real_double_formatting():
    cases = json.loads((Path(__file__).parent / "fixtures/misclib-format-native.json").read_text())
    for c in cases:
        x = (
            int(c["x"])
            if c["mode"] == 1
            else np.float32(c["x"])
            if c["mode"] == 2
            else float(c["x"])
        )
        result = format_number(x, **c["options"])
        assert (result.text, result.width, result.fit) == (c["text"], c["width"], c["fit"])


def test_fit_large_integers_rounding_carry_and_extreme_exponents():
    assert format_number(2**60 + 1, justi=-1).text == str(2**60 + 1)
    assert format_number(-(2**31), justi=-1).text == "-2147483648"
    result = format_number(12, width=1)
    assert not result.fit and result.text is None and result.width == 1
    # Native checks the padded width before trimming zeros; preserve that rule.
    assert not format_number(1.0, width=3, ndecf=4, qpad=False).fit
    # Native F field allocation before carry can produce stars yet report fit.
    assert format_number(9.999, width=5, justi=-1, ndecf=2).text == "10.00"
    assert not format_number(9.999, width=4, ndecf=2).fit
    for value in (1e300, 1e-300, float(np.nextafter(0, 1))):
        text = format_number(value, justi=-1, ndece=8).text
        assert text is not None and "D" in text and "*" not in text
        assert float(text.replace("D", "E")) == pytest.approx(value, rel=1e-8, abs=0)
    assert format_number(-0.0, justi=-1, ndecf=2).text == "0.00"
    assert format_number(1.25, justi=-1, ndecf=1).text == "1.2"
    assert format_number(1.0, justi=-1, ndecf=0).text == "1."
    for options in ({"width": -1}, {"npe": -4}, {"minf": -1}, {"qpad": 1}):
        with pytest.raises(ValueError):
            format_number(1.0, **options)
    with pytest.raises(ValueError):
        format_number(float("inf"))


def test_formatting_does_not_inherit_callers_decimal_traps():
    with localcontext() as context:
        context.prec = 3
        context.traps[Inexact] = True
        assert format_number(1.2345, justi=-1, ndecf=2).text == "1.23"
