"""Import PRT patient records, inspect counts and reopen a versioned study."""

from mdanderson_stats import import_prt_conduct, prt_study_counts, read_prt_study

study = import_prt_conduct(
    "3\n2\n0.3\n0.05\n0.3\n0.9\n-14\n28\n",
    "1 0 1\n1 3 1\n2 2 0\n2 1 1\n",
)
reopened = read_prt_study(study.to_json())
assert reopened == study
survived, events, assigned, assessed, pending = prt_study_counts(reopened)
print("Assigned:", assigned)
print("Assessed:", assessed)
print("Events:", events.sum(axis=0))
print("Pending:", pending.sum(axis=0))
# This example checks input semantics; it makes no treatment recommendation.
