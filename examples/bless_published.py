"""Reopen and report the publication's breast example at published precision."""

import json

from mdanderson_stats import BLESSPublishedStudy

study = BLESSPublishedStudy(
    "breast",
    {"ecog": 3, "benign_cytology": 0, "nlr": 10, "ldh_u_l": 900, "protein_g_dl": 2.8},
)
result = BLESSPublishedStudy.from_json(study.to_json()).predict()
report = json.loads(result.to_json())
assert abs(result.survival[1] - 0.6312601492191469) < 1e-14
assert "conflicts" in " ".join(report["source_warnings"])
print(result.to_json())
print(result.to_csv())
