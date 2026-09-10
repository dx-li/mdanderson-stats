import hashlib
import json
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import catbub_compare, catbub_operating_characteristics, catbub_thresholds

raw = np.loadtxt("research/raw/CATBUB/common-trials.csv", delimiter=",")
t0 = time.perf_counter()
p = catbub_compare(raw[:, 2:6].reshape(-1, 2, 2), [100, 0]).probability.reshape(500, 2, 2)
error = float(np.max(np.abs(p.reshape(-1, 2) - raw[:, 6:8])))
cuts = catbub_thresholds(p[:250], [0.5, 1])
null = catbub_operating_characteristics(p[:250], [20, 40], cuts)
alt = catbub_operating_characteristics(p[250:], [20, 40], cuts)
np.testing.assert_allclose(cuts, [0.9906, 0.9704], atol=0, rtol=0)
np.testing.assert_allclose(null.stopping_probability, [[0.008, 0], [0.020, 0.024]], atol=1e-15)
np.testing.assert_allclose(alt.stopping_probability, [[0.324, 0], [0.46, 0]], atol=1e-15)
assert error < 1e-7
result = {
    "source_url": (
        "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/"
        "BUBDesign/CATBUBDesign.zip"
    ),
    "sha256": hashlib.sha256(Path("research/raw/CATBUB/archive.zip").read_bytes()).hexdigest(),
    "r_seed": 9703,
    "trials_null": 250,
    "trials_target": 250,
    "sample_sizes_per_arm": [20, 40],
    "cutoffs": cuts.tolist(),
    "max_posterior_difference_vs_original_r": error,
    "null_first_stop_probabilities": null.stopping_probability.tolist(),
    "target_first_stop_probabilities": alt.stopping_probability.tolist(),
    "null_mean_n_per_arm": null.mean_sample_size.tolist(),
    "target_mean_n_per_arm": alt.mean_sample_size.tolist(),
    "elapsed_seconds": time.perf_counter() - t0,
    "interpretation": (
        "Identical source-generated data; small validation run, "
        "not a precision calibration for clinical use."
    ),
}
Path("docs/catbub-reference.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
