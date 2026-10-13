"""Bounded separable extended wavelets for column-stacked images and volumes."""

from __future__ import annotations

from itertools import product
from math import prod

import numpy as np

from ._cdflib import _freeze
from .wfmm_basis import WFMMBasis, _frozen_int, _setting
from .wfmm_boundary import wfmm_named_wavelet_basis


def wfmm_spatial_wavelet_basis(
    shape,
    *,
    levels=1,
    wavelets="db1",
    boundaries="periodic",
    layout="pyramid",
    max_work=200_000_000,
) -> WFMMBasis:
    """Construct extended 2D/3D pyramid or rectangular 2D wavelet matrix pairs.

    Inputs are column-stacked: the first spatial index varies fastest. Pyramid
    decomposition recurses only into the joint approximation and requires common
    levels, wavelet and boundary on all axes. Rectangular 2D decomposition applies
    complete, independently configured transforms along both axes. Scalar axis
    settings broadcast; tuples specify each axis. Each contiguous subband is a
    separate shrinkage partition. No native fitted-file or RNG parity is implied.

    Dense storage is limited to 512 input samples and two million estimated live
    matrix cells. Every decomposed axis must have at least two samples. Extended
    redundant coordinates define a working model, not independent spatial errors.
    """
    if not isinstance(shape, (tuple, list)) or len(shape) not in (2, 3):
        raise ValueError("shape must contain two or three spatial dimensions")
    dims = tuple(_setting(x, "shape", 2, 512) for x in shape)
    n = prod(dims)
    if n > 512:
        raise ValueError("spatial basis exceeds 512 input samples")
    budget = _setting(max_work, "max_work", 1, 200_000_000)
    if layout not in ("pyramid", "rectangular") or (layout == "rectangular" and len(dims) != 2):
        raise ValueError("layout must be pyramid (2D/3D) or rectangular (2D)")

    def axes(value, name):
        values = tuple(value) if isinstance(value, (tuple, list)) else (value,) * len(dims)
        if len(values) != len(dims):
            raise ValueError(name + " must have one setting per axis")
        return values

    depths = tuple(_setting(x, "levels", 1, 12) for x in axes(levels, "levels"))
    names = axes(wavelets, "wavelets")
    edges = axes(boundaries, "boundaries")
    # Validate names/boundaries and canonicalize aliases before common-axis checks.
    prototypes = [
        wfmm_named_wavelet_basis(2, wavelet=name, boundary=edge)
        for name, edge in zip(names, edges, strict=True)
    ]
    canonical = tuple(b.wavelet_name for b in prototypes)
    if layout == "pyramid" and (
        len(set(depths)) != 1 or len(set(canonical)) != 1 or len(set(edges)) != 1
    ):
        raise ValueError("pyramid requires common levels, wavelets and boundaries")

    def bounded(k):
        if 6 * max(n, k) ** 2 > 2_000_000:
            raise ValueError("spatial basis exceeds bounded matrix storage")

    if layout == "rectangular":
        bases = [
            wfmm_named_wavelet_basis(d, levels=j, wavelet=w, boundary=e, max_work=budget)
            for d, j, w, e in zip(dims, depths, names, edges, strict=True)
        ]
        k = prod(b.coefficient_count for b in bases)
        bounded(k)
        if 2 * n * k > budget:
            raise ValueError("spatial basis exceeds bounded work")
        assert bases[0].custom_matrix is not None and bases[1].custom_matrix is not None
        a = np.kron(bases[1].custom_matrix, bases[0].custom_matrix)
        b = np.kron(bases[1].synthesis_matrix, bases[0].synthesis_matrix)
        groups = sorted(
            product(range(depths[0] + 1), range(depths[1] + 1)), key=lambda x: (max(x), *x)
        )
        order, labels, scales = [], [], []
        for label, (i, j) in enumerate(groups):
            ix = np.flatnonzero(bases[0].coefficient_partition == i)
            iy = np.flatnonzero(bases[1].coefficient_partition == j)
            tile = (iy[:, None] * bases[0].coefficient_count + ix).ravel()
            order.extend(tile.tolist())
            labels.extend([label] * len(tile))
            scales.extend([min(depths[0] - max(0, i - 1), depths[1] - max(0, j - 1))] * len(tile))
        analysis, synthesis = a[:, order], b[order]
    else:
        widths = dims
        bands, inverse_bands = [], []
        current = inverse = None
        work = 0
        band_order = (0, 1, 2, 3) if len(dims) == 2 else (0, 2, 1, 4, 3, 6, 5, 7)
        total_details = 0
        for _ in range(depths[0]):
            if min(widths) < 2:
                raise ValueError("each decomposed spatial axis must have at least two samples")
            bases = [
                wfmm_named_wavelet_basis(d, wavelet=w, boundary=e, max_work=budget)
                for d, w, e in zip(widths, names, edges, strict=True)
            ]
            next_widths = tuple(b.coefficient_count // 2 for b in bases)
            count = prod(next_widths)
            total_details += (len(band_order) - 1) * count
            bounded(total_details + count)
            work += 2 * n * count * len(band_order) * (1 if current is None else prod(widths))
            if work > budget:
                raise ValueError("spatial basis exceeds bounded work")
            aa, bb = [], []
            for orientation in band_order:
                a = b = np.ones((1, 1))
                for axis, base in enumerate(bases):
                    start = ((orientation >> axis) & 1) * next_widths[axis]
                    section = slice(start, start + next_widths[axis])
                    assert base.custom_matrix is not None
                    a = np.kron(base.custom_matrix[:, section], a)
                    b = np.kron(base.synthesis_matrix[section], b)
                aa.append(a if current is None else current @ a)
                bb.append(b if inverse is None else b @ inverse)
            current, inverse = aa[0], bb[0]
            bands.append(aa[1:])
            inverse_bands.append(bb[1:])
            widths = next_widths
        assert current is not None and inverse is not None
        matrices = [current, *[a for group in bands[::-1] for a in group]]
        analysis = np.concatenate(matrices, axis=1)
        synthesis = np.concatenate([inverse, *[b for group in inverse_bands[::-1] for b in group]])
        labels = [i for i, a in enumerate(matrices) for _ in range(a.shape[1])]
        scale_by_band = [depths[0], *[j for j in range(depths[0], 0, -1) for _ in band_order[1:]]]
        scales = [
            j for j, a in zip(scale_by_band, matrices, strict=True) for _ in range(a.shape[1])
        ]
    return WFMMBasis(
        "custom",
        n,
        max(depths),
        None,
        _freeze(np.array([])),
        _freeze(np.array([])),
        _freeze(analysis),
        _frozen_int(scales),
        _frozen_int(labels),
        budget,
        _freeze(synthesis),
        spatial_shape=dims,
        spatial_layout=layout,
    )
