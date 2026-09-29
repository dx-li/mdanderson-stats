# Random-routing VIMP source reference

This helper verifies the pinned `randomForestSRC` C blob before extracting
`randomMembershipGeneric`, compiles a tiny deterministic wrapper, and writes
expected route/draw results for the input case CSV. It does not install or run
the R package and does not claim native RNG-stream parity.

From the repository root, after making the pinned source available under
`research/raw/randomForestSRC`, run:

```sh
python tools/reference_random_survival_forest_random_vimp.py \
  --root . \
  --cases tests/fixtures/random-survival-forest-random-routing.csv \
  --output tests/fixtures/random-survival-forest-random-routing-native.csv
```

The reference is limited to the unchanged route kernel. The companion fixture
includes the deterministic alpha tapes, ordinary branches, represented child
counts, native terminal nodes and draw counts. Two source cases for grouped
importance flags are retained in the fixture but are outside the public
per-feature Python API.
