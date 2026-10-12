import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import mdanderson_stats as m

parser = argparse.ArgumentParser(
    description="Installed BOIN desktop scenario quality check; originals remain external."
)
parser.add_argument("datafiles", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
start = time.monotonic()
records = []
root = args.datafiles
for file in ["DefaultSingleAgent.html", "DefaultTITE.html"]:
    original = m.read_boin_desktop_project((root / file).read_text(encoding="utf-8-sig"))
    for distribution in (
        ["eUniform"] if original.mode == "single" else ["eUniform", "eWeibull", "eLogLogistic"]
    ):
        tree = ET.fromstring(original.xml)
        sim = tree.find("SimDat")
        if original.mode == "tite":
            sim.find("TimeToToxDistribution").text = distribution
        flag = "Simulate_3Plus3Design" if original.mode == "single" else "Simulate_Rolling6Design"
        sim.find(flag).text = "true"
        study = m.read_boin_desktop_project(ET.tostring(tree, encoding="unicode"))
        workflow = m.simulate_boin_desktop_workflow(study, batches=2, trials=500, seed=712)
        assert m.read_boin_desktop_project(m.boin_desktop_workflow_html(workflow)) == study
        for batch in [workflow.primary, workflow.comparator]:
            for r in batch.results:
                assert not r.selected_eliminated.any()
                assert not np.any(r.selected & (r.actual_patients == 0))
                assert np.array_equal(r.patients, r.actual_patients)
                assert (r.toxicities <= r.patients).all()
                if r.method == "BOIN":
                    assert (r.patients.sum(axis=(1, 2)) <= study.maximum_patients).all()
                if r.unsafe_assignment_count is not None:
                    assert not r.unsafe_assignment_count.any()
                if r.actual_duration is not None:
                    assert (r.actual_duration >= r.reported_duration).all()
            for index in range(len(study.scenarios)):
                summary = batch.aggregate(index)
                records.append(
                    dict(
                        file=file,
                        distribution=distribution,
                        method=batch.results[0].method,
                        scenario=index,
                        trials=summary["trials"],
                        selection=summary["selection_probability"].tolist(),
                        max_mcse=float(summary["selection_mcse"].max()),
                        mean_patients=summary["mean_actual_patients"].tolist(),
                    )
                )
        print(file, distribution, "passed", flush=True)
result = dict(
    scope="synthetic operating-characteristic precision and workflow invariants, "
    "not clinical validity",
    import_location=m.__file__,
    seconds=time.monotonic() - start,
    master_seed=712,
    batches=2,
    trials_per_batch=500,
    total_trials=sum(r["trials"] for r in records),
    records=records,
)
result["input_sha256"] = {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in ("DefaultSingleAgent.html", "DefaultTITE.html")
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
