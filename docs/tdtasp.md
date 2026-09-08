# TDTASP genetic model

TDTASP version 1.1 (April 2003), by Barry W. Brown and Dan Serachitopol,
plans transmission-disequilibrium (TDT) and affected-sibling-pair (ASP) studies.
This catalog entry is **partial**. The genetic layer is implemented;
ascertainment, sample-size/power calculations, template files, reports and the
complete archive audit remain outstanding.

Source: [TDTASP catalog entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/20)
and its [version 1 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TDTASP/TDTASP%20%20_V1.tar.gz).
Despite the archive filename, the enclosed source and manual are version 1.1.
Original [legal notices](../notices/mdanderson-tdtasp-LEGALITIES.txt) are retained.
Original source, executables and the archived ACM numerical routines are not
bundled with the Python package.

```python
from mdanderson_stats import tdtasp_genetics, tdtasp_haplotype_frequencies

frequencies = tdtasp_haplotype_frequencies(
    marker_frequency=0.3, disease_frequency=0.2, relative_disequilibrium=0.7
)
model = tdtasp_genetics(frequencies, penetrance=[0.8, 0.3, 0.01], recombination=0.1)
```

Haplotype probabilities use **AD, Ad, BD, Bd** order; penetrances use **DD, Dd,
dd** order. A/B is the marker locus and D/d the disease locus. The frequency
vector must sum to one within 1e-12 and is then normalized. Penetrance values
lie in [0, 1] and recombination lies in [0, 0.5]. These strict probability
checks replace the console's approximate frequency-sum acceptance window.

`tdtasp_haplotype_frequencies` accepts population frequencies m of A and p of D,
and relative disequilibrium D prime. The absolute disequilibrium is bounded by
`-min(m*p, (1-m)*(1-p))` and `min(m*(1-p), (1-m)*p)`. D prime scales the bound
of the corresponding sign. Negative D prime and degenerate population allele
frequencies are supported for genetic calculations; the original one-sided
planning interface assumes positive association between A and D.

`TDTASPGenetics` contains immutable arrays for all 256 ordered parental
combinations. `parents` columns are father's two haplotypes and mother's two
haplotypes; codes are **0=Bd, 1=BD, 2=Ad, 3=AD**, matching the Fortran enumeration.
The rows are in lexicographic order. Outputs include:

- Each family's population probability under independent haplotype sampling.
- Father/mother marker-heterozygosity flags.
- The probability a child is affected, conditional on that parental family.
- TDT probability of transmitting A to an affected child, averaged over the
  marker-heterozygous parents in that family.
- ASP probability of transmitting the same marker allele to two independently
  generated affected siblings, averaged over those informative parents.
- Population prevalence, weighting affected-child probabilities by family mass.

A family with no informative parent has transmission/sharing zero. If a family
cannot have an affected child, its conditional transmission/sharing outputs are
also zero, following the source convention; `affected_probability` identifies
those rows. The zero is a sentinel for an uninformative/impossible conditioning
event, not a defined conditional probability. Families with zero population
mass remain present so downstream ascertainment can use the original indexing.

## Corrected and original ASP calculations

The original `offspring_mod.f90` first multiplies penetrance by offspring
transmission probability. Its `calc_exp_p_same` then multiplies both offspring
probabilities by their transmission probabilities again. That extra weighting
changes the conditional distribution when recombination is between zero and
one half. The default Python calculation uses each transmission probability
once. Given parental marker-A transmission probabilities q conditional on an
affected child, sibling sharing is `q*q + (1-q)*(1-q)`, averaged over informative
parents. This follows directly from independent sibling transmissions.

`legacy_asp=True` reproduces the archived duplicate path weighting and its
`1e-100` cutoff on pair contributions to the numerator. The denominator still
includes those tiny pairs, as in the source. Keeping transmission paths distinct
is necessary for that compatibility calculation, even when they encode the
same gamete. If positive affected probability produces a zero legacy pair
denominator through underflow, Python raises `ArithmeticError`; it does not
return a native NaN. The corrected calculation conditions before forming sibling
probabilities and avoids that unnecessary squaring of small masses. Both modes
use floating-point arithmetic, not arbitrary precision.

## Validation

`tools/reference_tdtasp_genetics.py` compiles the **unmodified** original
`offspring_mod.f90` with a separate driver. The fixture records archive/source
hashes, compiler, build command, driver and output column definitions. Twenty-one
models combine seven penetrance patterns and recombination 0, 0.1 and 0.5,
covering **5,376 parental-family rows**. Tests compare heterozygosity counts,
affected-child and transmission probabilities, and legacy ASP sharing.

Independent tests enumerate Mendelian gametes and all affected-sibling pairs
for every parental family at three recombination settings. These validate the
corrected ASP result separately from the archived defect. Other checks cover
analytic dominant/recessive examples, population prevalence under random mating,
allele marginals and disequilibrium bounds, impossible conditioning events,
legacy tiny-probability behavior, input validation and immutable results.

The numerical layer evaluates all families and offspring paths with NumPy.
The default sharing calculation uses conditional marginals instead of explicitly
enumerating sibling pairs. Compatibility mode uses a bounded 256-by-16-by-16
array to reproduce the original cutoff. Performance has not yet been benchmarked.
