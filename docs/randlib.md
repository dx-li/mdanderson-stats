# RANDLIB

Catalog entry 27 is partial. The 32-stream generator bank and state controls are
implemented; distribution samplers, random permutations and the final archive
coverage/performance audit remain pending.

The [official entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/27)
lists version 90, modified September 27, 2002. RANDLIB_V90.tar.gz contains
Fortran 77, C and Fortran 95 implementations and their documentation. The
Fortran 77 readme identifies version 1.3, August 1997. Original archives, source
and reference executables remain local research material. Retained legal terms
and attribution are linked from THIRD_PARTY_NOTICES.md.

## Explicit generator banks

```python
from mdanderson_stats import RandlibGenerator, ranlist_seeds

bank = RandlibGenerator(seed=ranlist_seeds("example"))
raw = bank.integers(1000)
u = bank.uniform(1000)
bank.select(2)
bank.set_antithetic(True)
paired = bank.uniform(1000)
bank.reinitialize(1)  # First draw of the next 2**30-draw block is next
current_pair = bank.get_seeds()
```

Each `RandlibGenerator` owns 32 streams with independent current, initial and
block-start states and antithetic flags. Stream starts are separated by 2^50
draws. Default seeds are (1234567890, 123456789); explicit seeds must be integers
in 1..2147483562 and 1..2147483398. The recurrence and vectorized modular-power
helper are shared with the already validated RANLIST implementation.

`integers(size=1)` returns a read-only vector of consecutive raw IGNLGI draws
in 1..2147483562, advancing the selected stream exactly that many times.
`uniform(size=1)` divides those integers by 2147483563 in double precision.
`legacy=True` instead applies RANF's single-precision integer conversion and
scale constant, returning the resulting float32 values as float64. Empty draws
leave state unchanged. `max_draws=1_000_000` bounds a batch before allocation;
a different positive limit may be supplied to the constructor. Invalid requests
do not change state. A bank is mutable; concurrent callers need coordination
or separate banks. Separate objects share no mutable RNG state.

| Method | Original routine | Effect |
| --- | --- | --- |
| `select(stream)` / `stream` | SETCGN / GETCGN | Select or inspect stream 1..32 without restarting it |
| `get_seeds()` | GETSD | Return current component states of selected stream |
| `set_seeds(pair)` | SETSD | Replace selected stream's initial, block and current states |
| `set_all_seeds(pair)` | SETALL | Rebuild all streams from the supplied stream-one seed, retaining selection and antithetic flags |
| `set_antithetic(enabled)` | SETANT | Complement future raw draws as M1-z; internal component states are unchanged |
| `reinitialize(-1)` | INITGN(-1) | Reset block and current states to selected stream's initial state |
| `reinitialize(0)` | INITGN(0) | Return to current block's start |
| `reinitialize(1)` | INITGN(1) | Move block start forward by 2^30 draws, then reset current state to that start |
| `advance_state(k)` | ADVNST(k) | Advance current state by 2^k draws and adopt it as the new initial/block/current state |

Block advancement is measured from the block start, not from the last consumed
draw. ADVNST differs: it advances from the current state and replaces the reset
anchor. Exponents are nonnegative integers below 2^53. Prime-modulus exponent
reduction makes even large advances practical without constructing 2^k or
replaying the intervening sequence. The raw stream recurrence and state remain
identical regardless of antithetic mode. Modern ordinary/antithetic uniforms
sum to one to floating-point precision; legacy float32 rounding need not obey
that identity exactly.

The archived Fortran SETALL implementations fail to select stream one before
its reset. Calling them while another stream is selected can leave stream one's
current/block state stale. Python initializes every stream explicitly, so the
selected stream cannot affect seed initialization. Tests cover this correction.
Native comparison sequences reseed all streams while stream one is selected.
Invalid stream zero, invalid seeds and negative advance exponents are rejected
rather than preserving unsafe or accidental source behavior.

`get_seeds` is a snapshot of the selected current pair, not a serialized bank;
restoring it with `set_seeds` intentionally replaces the reset anchors. Full bank
serialization and non-uniform distribution sampling are not yet implemented.

## Validation

`tools/reference_randlib_streams.py` compiles unchanged Fortran 77 base routines
and the Fortran 95 ecuyer_cote_mod independently. Sixty-four cases cover two
seed pairs, four streams and advance exponents 0/5/30/80. Each records fourteen
integer draws and their current component states through antithetic changes,
block resets, advancement, seed replacement and stream switching. Python matches
every value exactly. Fixtures include source hashes, compiler and flags.

Independent tests check split versus combined batches, interleaved streams,
antithetic complements, block-start semantics, ADVNST anchor replacement,
prime-modulus assumptions, large powers, read-only outputs and invalid-operation
state preservation. These validate the bank foundation only; the C distribution
library and higher-level samplers remain pending.
