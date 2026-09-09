"""Public constant and floating-model contracts from the original compiler."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdflib_constants

REFERENCE = json.loads((Path(__file__).parent / "fixtures/cdflib_constants.json").read_text())


@pytest.mark.parametrize("name", REFERENCE["real_names"])
def test_binary64_constants_match_compiled_source(name):
    actual = getattr(cdflib_constants, name)
    assert isinstance(actual, float)
    assert struct.pack(">d", actual) == struct.pack(">d", REFERENCE["values"][name])


@pytest.mark.parametrize("name", REFERENCE["integer_names"])
def test_legacy_identifiers_match_compiled_source(name):
    actual = getattr(cdflib_constants, name)
    assert isinstance(actual, int)
    assert actual == REFERENCE["values"][name]


@pytest.mark.parametrize("kind", ["dpkind", "spkind"])
def test_native_floating_model_matches_numpy_representation(kind):
    model = REFERENCE["models"][kind]
    dtype = np.dtype(f"f{model['bits'] // 8}")
    info = np.finfo(dtype)
    assert model["radix"] == 2
    assert info.bits == model["bits"]
    assert info.nmant + 1 == model["digits"]
    # Fortran normalizes significands to [1/radix,1); NumPy uses [1,radix).
    assert info.minexp + 1 == model["minexponent"]
    assert info.maxexp == model["maxexponent"]
    assert float(info.eps) == model["epsilon"]
    assert float(info.tiny) == model["tiny"]
    assert float(info.max) == model["huge"]
