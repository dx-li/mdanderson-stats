"""Compare seeded OOB permutation importance with pinned native C kernels.

Reconstructs one-feature trees independently using native split and leaf
routines, then independently permutes OOB rows and calls native concordance.
Requires the ignored pinned source cache described in the forest audit.
Run from the repository with PYTHONPATH=src.
"""

import json
import resource
import sys
import time

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from reference_random_survival_forest import doubles, library, reference_tree

from mdanderson_stats.random_survival_forest import fit_random_survival_forest
from mdanderson_stats.random_survival_forest_vimp import (
    permutation_random_survival_forest_importance,
)

start = time.perf_counter()
lib = library()
t = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8.0])
e = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0.0])
x0 = np.array([0.2, -1, 0.5, 2, -0.8, 1.5, 0, 1, 3, -0.5, 2.5, 4.0])
x = np.column_stack((x0, np.ones(12)))
fit = fit_random_survival_forest(
    t,
    e,
    x,
    n_trees=7,
    nodesize=1,
    mtry=2,
    nsplit=0,
    replace=False,
    sample_fraction=0.75,
    ntime=4,
    random_state=411,
    compute_oob=True,
)
n = t.size
grid = fit.time_grid
refs = []
for packed in fit.inbag_membership:
    inbag = np.unpackbits(packed, bitorder="little")[:n].astype(bool)
    included = np.flatnonzero(inbag)
    case = dict(
        name="vimp",
        time=t[included].tolist(),
        event=e[included].tolist(),
        x=x0[included].tolist(),
        times=grid.tolist(),
    )
    refs.append((reference_tree(lib, case, nodesize=1)["nodes"], np.flatnonzero(~inbag)))


def mortality(tree, value):
    node = tree[0]
    while node["cut"] is not None:
        node = tree[node["left"] if value <= node["cut"] else node["right"]]
    return sum(node["cumulative_hazard"])


def error(values, counts):
    return lib.getConcordanceIndex(
        1,
        n,
        doubles(t.tolist()),
        doubles(e.tolist()),
        doubles(values.tolist()),
        doubles(counts.tolist()),
    )


checks = []
for size in (7, 3, 1):
    output = permutation_random_survival_forest_importance(
        fit, t, e, x, block_size=size, random_state=1772
    )
    rng = np.random.default_rng(1772)
    blocks = 7 // size
    baseline = np.zeros(blocks)
    perturbed = np.zeros((2, blocks))
    for block in range(blocks):
        selected = refs[block * size : (block + 1) * size]
        counts = np.zeros(n, dtype=int)
        sums = np.zeros(n)
        for tree, oob in selected:
            for i in oob:
                sums[i] += mortality(tree, x0[i])
                counts[i] += 1
        mean = np.full(n, np.nan)
        np.divide(sums, counts, out=mean, where=counts > 0)
        baseline[block] = error(mean, counts)
        for feature in (0, 1):
            sums[:] = 0
            for tree, oob in selected:
                if oob.size == 0:
                    continue
                permutation = rng.permutation(oob.size)
                for k, i in enumerate(oob):
                    sums[i] += mortality(tree, x0[oob[permutation[k]]] if feature == 0 else x0[i])
            mean[:] = np.nan
            np.divide(sums, counts, out=mean, where=counts > 0)
            perturbed[feature, block] = error(mean, counts)
    difference = perturbed - baseline[None, :]
    valid = np.isfinite(difference)
    vcount = valid.sum(axis=1)
    expected = np.full(2, np.nan)
    np.divide(np.nansum(difference, axis=1), vcount, out=expected, where=vcount > 0)
    assert_allclose(output.baseline_error, baseline, rtol=0, atol=0, equal_nan=True)
    assert_allclose(output.perturbed_error, perturbed, rtol=0, atol=0, equal_nan=True)
    assert_allclose(output.block_importance, difference, rtol=0, atol=0, equal_nan=True)
    assert_allclose(output.importance, expected, rtol=0, atol=0, equal_nan=True)
    assert_array_equal(output.valid_block_count, vcount)
    assert_array_equal(output.ignored_tree_indices, np.arange(blocks * size, 7))
    assert output.importance[1] == 0
    checks.append(
        dict(
            block_size=size,
            blocks=blocks,
            ignored=output.ignored_tree_indices.tolist(),
            valid_blocks=vcount.tolist(),
            importance=output.importance.tolist(),
        )
    )
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            checks=checks,
            elapsed_seconds=time.perf_counter() - start,
            peak_mib=usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
