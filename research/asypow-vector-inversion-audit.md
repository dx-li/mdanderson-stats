# ASYPOW vector sample-size inversion contract

The original manual documents the three-way power/sample-size/significance
workflow. It says exactly two of significance, power, and sample size are
supplied; the remaining value is calculated, and the two supplied values may be
vectors for multiple calculations (`research/raw/ASYPOW/original/asypow/docs/asypow.txt:234-248`).
The LR inverse `S/lr.n.s:27-32` and SMO inverse `S/self.n.s:44-51` implement
pairwise lengths: either input may be scalar and expand to the other vector's
length, while unequal non-scalar lengths fail. `self.n.s:53-75` preserves the
chosen `nu = n*w - df` or `nu = n*w` convention when converting each target
noncentrality into sample size.

Python already exposes separate power, significance, and sample-size methods.
Power and significance accept broadcastable arrays, but LR
`AsymptoticPower.sample_size` and SMO `SMOPower.sample_size` previously forced
scalar targets (`src/mdanderson_stats/asypow.py` and
`src/mdanderson_stats/asypow_smo.py`). The vector extension keeps scalar calls
returning floats, accepts scalar/singleton expansion or equal-length vectors,
returns an owned read-only vector, rejects outer-product/multidimensional inputs,
and caps a call at 10,000 inversions. It does not add a new hypothesis model or
native prompt/retry/report layer.

The existing Python numerical contracts remain in force: LR targets no greater
than significance map to sample size zero and null alternatives cannot reach
higher power; SMO requires power strictly above significance and a non-null
alternative; the SMO `subtract_df` choice is preserved. Each inversion keeps
the existing noncentrality cap. Independent R reference values and forward
power checks are in `tests/test_asypow_vector_sample_size.py`.
