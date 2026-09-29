"""Check exact paired calibration against independent enumeration of 4**4 paths."""

import csv
import itertools
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_paired_optimization import optimize_bop2_dc_paired

root = Path(__file__).resolve().parents[1]
groups = defaultdict(list)
with (root / "tests/fixtures/bop2-dc-paired-calibration.csv").open() as stream:
    for row in csv.DictReader(stream):
        groups[(row["endpoint"], row["association_id"])].append(row)


def joint(margins, both):
    first, second = margins
    return [both, first - both, second - both, 1 - first - second + both]


grids = dict(
    lambda_lrv_grid=[[0.4, 0.6], [0.8, 0.9]],
    lambda_cmv_grid=[[0.15, 0.25], [0.45, 0.55]],
    gamma_lrv_grid=[[0, 0.5], [0.5, 0]],
    gamma_cmv_grid=[[0.5, 0.5]],
)
candidate_parameters = list(itertools.product(*grids.values()))
checked = 0
selected = {}
association_go = defaultdict(list)
start = time.monotonic()
for (endpoint, association), rows in groups.items():
    first = rows[0]
    if endpoint == "multiple_efficacy":
        lrv, cmv = [0.2, 0.15], [0.5, 0.45]
        futile_margin, effective_margin = [0.2, 0.2], [0.6, 0.5]
    else:
        lrv, cmv = [0.2, 0.5], [0.5, 0.2]
        futile_margin, effective_margin = [0.2, 0.5], [0.6, 0.15]
    settings = dict(
        max_subjects=4,
        endpoint=endpoint,
        lrv=lrv,
        cmv=cmv,
        futile_probabilities=joint(futile_margin, float(first["futile_p11"])),
        effective_probabilities=joint(effective_margin, float(first["effective_p11"])),
        prior=[0.3, 0.2, 0.4, 0.1],
        looks=[2, 4],
        **grids,
        **{
            name: float(first[name])
            for name in ("false_go_limit", "false_no_go_limit", "false_consider_limit")
        },
    )
    for objective in ("cgr", "ess_futile"):
        result = optimize_bop2_dc_paired(**settings, objective=objective)
        selected_column = "selected_cgr_index" if objective == "cgr" else "selected_ess_index"
        assert result.selected_index == int(first[selected_column]), (
            endpoint,
            association,
            objective,
            result.selected_index,
            first[selected_column],
        )
        selected[f"{endpoint}_{association}_{objective}"] = result.selected_index
        assert_array_equal(
            [
                result.design.lambda_lrv,
                result.design.lambda_cmv,
                result.design.gamma_lrv,
                result.design.gamma_cmv,
            ],
            candidate_parameters[result.selected_index],
        )
        for row in rows:
            i = int(row["candidate_index"])
            for truth in ("futile", "effective"):
                oc = getattr(result, truth + "_oc")
                for field, suffix in (
                    ("final_go", "final_go"),
                    ("final_consider", "final_consider"),
                    ("final_no_go", "final_no_go"),
                    ("expected_sample_size", "ess"),
                ):
                    assert_allclose(
                        getattr(oc, field)[i],
                        float(row[truth + "_" + suffix]),
                        rtol=4e-14,
                        atol=2e-15,
                    )
                    checked += 1
                for field, prefix in (
                    ("stop_no_go", "stop_n"),
                    ("sample_size_probability", "sample_size_p_n"),
                ):
                    assert_allclose(
                        getattr(oc, field)[i],
                        [float(row[f"{truth}_{prefix}{n}"]) for n in (2, 4)],
                        rtol=4e-14,
                        atol=2e-15,
                    )
                    checked += 2
            for field, column in (
                ("false_go_rate", "fgr"),
                ("false_no_go_rate", "fngr"),
                ("correct_go_rate", "cgr"),
                ("false_consider_rate", "fcr"),
            ):
                assert_allclose(
                    getattr(result, field)[i], float(row[column]), rtol=4e-14, atol=2e-15
                )
                checked += 1
            assert bool(result.feasible[i]) == (row["feasible"] == "TRUE")
        assert np.any(result.feasible) and not np.all(result.feasible)
    association_go[endpoint].append(result.futile_oc.final_go.copy())

# Matching marginal rates must not silently erase joint association effects.
for vectors in association_go.values():
    assert not np.allclose(vectors[0], vectors[-1])
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            scenarios=len(groups),
            candidates_per_scenario=len(candidate_parameters),
            enumerated_paths_per_truth=4**4,
            checked_summaries=checked,
            selected=selected,
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
