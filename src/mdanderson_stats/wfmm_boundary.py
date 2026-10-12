"""Native-convention one-dimensional extended Daubechies matrix pairs."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from importlib.resources import files

import numpy as np

from ._cdflib import _freeze
from .pinnacle_wavelet import pinnacle_daubechies_filter
from .wfmm_basis import WFMMBasis, _frozen_int, _setting


def wfmm_extended_wavelet_basis(
    time_count: int,
    *,
    levels: int = 1,
    filter_length: int = 8,
    boundary: str = "periodic",
    max_work: int = 200_000_000,
) -> WFMMBasis:
    """Construct a bounded native extended-mode db1..db10 analysis/synthesis pair.

    ``periodic`` wraps indices; ``reflection`` repeats the endpoint samples;
    ``zero`` pads with zeros. Each level has floor((n + filter_length - 1)/2)
    coefficients per band, packed [a_J, d_J, ..., d_1]. Synthesis uses cropped
    overlap-add, not the analysis transpose. Odd input lengths are preserved.
    This redundant coefficient representation is a working statistical model;
    independent transformed errors are not implied by exact reconstruction.

    Dense matrix storage bounds this constructor to 512 time points and two
    million live matrix cells. Legacy periodic-decimation bases are unchanged.
    """
    length = _setting(filter_length, "filter_length", 2, 20)
    if length % 2:
        raise ValueError("filter_length must be even")
    low, high = pinnacle_daubechies_filter(length)
    return _extended_pair(time_count, levels, low, high, boundary, max_work)


@lru_cache(maxsize=12)
def _named_filters(name: str):
    data = json.loads(files("mdanderson_stats").joinpath("_wfmm_filters.json").read_text())
    low = np.array(data["filters"][name], dtype=float)
    high = low[::-1] * ((-1.0) ** np.arange(low.size))
    return _freeze(low), _freeze(high)


def wfmm_named_wavelet_basis(
    time_count: int,
    *,
    wavelet: str = "db4",
    levels: int = 1,
    boundary: str = "periodic",
    max_work: int = 200_000_000,
) -> WFMMBasis:
    """Native-convention db1..db10, sym2..sym8 or coif1..coif5 matrix pairs.

    Manual aliases haar, d4..d20, s4..s16 and c6..c30 are accepted. Named
    Symlet/Coiflet taps are factual native impulse-response measurements;
    finite precision, especially coif5, limits orthogonality to about 1e-8
    in tested two-level cases. No exact arithmetic or clinical claim follows.
    """
    aliases = {"haar": "db1"}
    aliases.update({f"d{2 * i}": f"db{i}" for i in range(2, 11)})
    aliases.update({f"s{2 * i}": f"sym{i}" for i in range(2, 9)})
    aliases.update({f"c{6 * i}": f"coif{i}" for i in range(1, 6)})
    allowed = {
        *(f"db{i}" for i in range(1, 11)),
        *(f"sym{i}" for i in range(2, 9)),
        *(f"coif{i}" for i in range(1, 6)),
    }
    if not isinstance(wavelet, str):
        raise ValueError("wavelet must name a supported family")
    name = aliases.get(wavelet, wavelet)
    if name not in allowed:
        raise ValueError("wavelet must be db1..db10, sym2..sym8 or coif1..coif5")
    if name.startswith("db"):
        result = wfmm_extended_wavelet_basis(
            time_count,
            levels=levels,
            filter_length=2 * int(name[2:]),
            boundary=boundary,
            max_work=max_work,
        )
    else:
        low, high = _named_filters(name)
        result = _extended_pair(time_count, levels, low, high, boundary, max_work)
    return replace(result, wavelet_name=name)


def _extended_pair(time_count, levels, low, high, boundary, max_work):
    n = _setting(time_count, "time_count", 2, 512)
    depth = _setting(levels, "levels", 1, 12)
    length = len(low)
    budget = _setting(max_work, "max_work", 1, 200_000_000)
    if boundary not in ("periodic", "zero", "reflection"):
        raise ValueError("boundary must be periodic, zero or reflection")
    widths = [n]
    for _ in range(depth):
        widths.append((widths[-1] + length - 1) // 2)
    k = widths[-1] + sum(widths[1:])
    if 6 * max(n, k) ** 2 > 2_000_000 or 4 * (n + k) * k * length > budget:
        raise ValueError("extended basis exceeds bounded matrix/work limits")
    current = np.eye(n)
    bands = []
    for old, width in zip(widths[:-1], widths[1:], strict=True):
        approximation = np.zeros((n, width))
        detail = np.zeros_like(approximation)
        for tap, (lo, hi) in enumerate(zip(low, high, strict=True)):
            index = 2 * np.arange(width) + tap - (length - 2)
            valid = (index >= 0) & (index < old)
            if boundary == "periodic":
                mapped = index % old
            elif boundary == "reflection":
                folded = index % (2 * old)
                mapped = np.where(folded < old, folded, 2 * old - 1 - folded)
            else:
                mapped = np.clip(index, 0, old - 1)
            values = current[:, mapped]
            if boundary == "zero":
                values = values * valid
            approximation += float(lo) * values
            detail += float(hi) * values
        current = approximation
        bands.append(detail)
    analysis = np.concatenate((current, *bands[::-1]), axis=1)
    units = np.eye(k)
    current = units[:, : widths[-1]]
    offset = widths[-1]
    for level in range(depth, 0, -1):
        width = widths[level]
        old = widths[level - 1]
        detail = units[:, offset : offset + width]
        offset += width
        output = np.zeros((k, old))
        for tap, (lo, hi) in enumerate(zip(low, high, strict=True)):
            index = 2 * np.arange(width) + tap - (length - 2)
            valid = (index >= 0) & (index < old)
            output[:, index[valid]] += float(lo) * current[:, valid] + float(hi) * detail[:, valid]
        current = output
    partitions = np.concatenate(
        [np.full(width, i, dtype=np.int64) for i, width in enumerate([widths[-1], *widths[:0:-1]])]
    )
    scales = np.concatenate(
        [
            np.full(widths[-1], depth, dtype=np.int64),
            *[np.full(widths[j], j, dtype=np.int64) for j in range(depth, 0, -1)],
        ]
    )
    return WFMMBasis(
        "custom",
        n,
        depth,
        length,
        low,
        high,
        _freeze(analysis),
        _frozen_int(scales),
        _frozen_int(partitions),
        budget,
        _freeze(current),
        boundary,
    )
