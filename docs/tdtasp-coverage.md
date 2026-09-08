# TDTASP archive coverage

The catalog's version-1 download contains TDTASP **1.1 (April 2003)**, with
44 regular files: 30 Fortran source files, two build files, one Windows executable
and 11 documentation/notice files. `tools/audit_tdtasp.py` compares each extracted
file with its archive bytes, records hashes and routine names, maps source files
to replacements, and rejects unknown members. See [the inventory](tdtasp-archive.json).
The original source and executable remain outside the Python distribution.

| Original responsibility | Python implementation and validation |
|---|---|
| `tdtasp`, `statistics_mod`: complete study, power/sample size, fixed comparison | `tdtasp_study`, immutable `TDTASPStudy`, `format_tdtasp_study`; real-component integration, first-qualifying designs and file/report tests |
| `genetics_mod`, `families_mod`: haplotypes, disequilibrium, random mating, population and parental-family probabilities | `tdtasp_haplotype_frequencies`, `tdtasp_genetics`; allele marginals, analytic genetic cases, all 256 ordered parental families and 5,376 native family rows |
| `offspring_mod`: offspring paths, disease penetrance, transmission and sibling sharing | Vectorized Mendelian paths, corrected conditional sharing and explicit source double-weight/cutoff option; independent gamete/sibling enumeration and native comparisons |
| `ascertain_mod`: family/individual sampling, affected-count and parent criteria, selected moments and contributions | `tdtasp_ascertainment`; 18 native studies and independent joint family/Poisson-count sums, all criteria and corrected individual size bias |
| `tail_poisson_mod`, standalone `exp_n_given_k`: tail masses and truncated means | Stable positive series, logarithmic masses and ratios; tiny-probability and zero-affected-family tests |
| `power_bin1_mod`: fixed-binomial critical values, actual size and power | `tdtasp_fixed_power`; rational enumeration, 52 successful native cases, and 11 native empty-region failures repaired explicitly |
| `power_bin1_vn_mod`, `normal_integrate`: random usable-family count, short-tail sums and Hermite approximation | `tdtasp_power`; complete discrete binomial averaging with explicit resource limits, rounding and independently enumerated mixtures |
| `bin1_ss_pow_mod`, `bin1_vn_ss_pow_mod`, `zero_finder`: requested-power designs | First qualifying integer searches, conditional-curve cache and monotone upper bound; exhaustive bounded forward comparisons, power oscillations and search-limit tests |
| `cdf_aux_mod`, `cdf_beta_mod`, `beta_gamma_mod`, `cdf_binomial_mod`, `cdf_normal_mod`: numerical support | SciPy numerical kernels and integer rejection regions replace internal CDF/inversion support; the normal approximation is eliminated, and archived ACM code is not bundled |
| `param_defn_mod`, `param_io`, `get_numbers_mod`: parameter values, ranges and prompts | Explicit component validation and immutable `TDTASPTemplate`; active-value, count-precision, ownership and invalid-model tests |
| `parser_mod`, `user_interface_mod`: template tokens, comments, lists, labels and choices | `parse_tdtasp_template`; 49 original-reader cases, multiline/named/positional forms, inactive fields, aliases and malformed-input tests |
| `create_write_form_mod`: blank forms and filled forms | `format_tdtasp_template`; all 17 original fields, source ordering, placeholders and high-precision round trips |
| `user_dialogues_mod`, `open_file_mod`, `print_it`, `print_array`: menus, file dialogs, reports and terminal output | Python calls, exceptions, complete returned summaries and full immutable family arrays; caller-managed printing and file reading/writing |
| `intro_mod`: TDT, ASP, ascertainment and exact-binomial tutorial | [Method and usage notes](tdtasp.md), including the statistical assumptions and original numerical defects |

The main program's keyboard, template, tutorial, blank-form, report-file,
filled-form and repeat-study workflows are represented by Python calls and
files rather than a recreation of its terminal menus. Each call owns its state;
no global Fortran parameter table, terminal pagination or overwrite prompt is
needed. Unix/Windows manuals, duplicated notices, build scripts and the Windows
binary are all accounted for. Original legal terms are retained in the wheel.
Generic numerical inversions and prompt/formatting helpers that support the
application are not separate user-facing statistical products.

The report includes the genetic parameters, allele marginals and disequilibrium,
separate father/mother heterozygosity, ascertainment and screening quantities,
selected moments, mean contribution used, screened/expected eligible counts,
actual significance and power. Search output includes the target, bounds,
first-crossing diagnostics and fixed-observation comparison. Rare screening
factors are shown on the log scale. This retains substantive information while
making the source's population-average moment distinct from a selected-family
expectation. Individual-list sampling assumes repeated selection of the same
family is negligible.

## Statistical scope and corrections

The scientific target remains TDT/ASP study planning under the archived
**mean-contribution approximation**. Full discrete averaging over eligible-family
counts does not turn it into an exact joint model of dependent within-family
transmissions. All assumptions, numerical bounds and compatibility choices are
explained in the method notes and exposed in results or reports.

The Python defaults correct duplicate ASP path weighting, individual-sampling
moments, the screening-ratio conditioning, all-child contribution scaling,
one-tail-only two-sided power, empty critical regions and unreliable continuous
sample-size inversion. Separate `legacy_asp`, `legacy_moments`, `legacy_scale`
and `legacy_two_sided` flags allow the identified source conventions to be
examined. Old floating-point cancellation, invalid states, arbitrary cutoffs,
normal/Hermite integration and local root heuristics are not reproduced as
silent substitutes for valid calculations.

Native fixtures record original file hashes, compiler/build commands, drivers,
and every accessibility or arithmetic adaptation. Genetic and fixed-power
routine bodies are unmodified. Ascertainment/template references only expose
private routines/state with PUBLIC declarations. Independent probability
calculations validate corrected behavior; native failures are recorded as
failures, not accepted as statistical reference values.

The template interpreter deliberately improves input handling while retaining
active/inactive field conventions. It accepts mathematical domains already
supported by the Python model, rejects fractional counts instead of truncating
them, and requires unambiguous choices. The method notes list format extensions
and restrictions, so a successful parse is not a claim that the old executable
accepts every extended Python input.

## Performance evidence

Genetics and ascertainment operate on bounded NumPy arrays. Fixed-count power
and search batches reuse compiled SciPy kernels. Family design search caches
conditional powers and uses a monotone upper bound to skip counts that cannot
qualify, without assuming actual discrete power increases monotonically.
[Recorded median-of-three benchmarks](tdtasp-search-benchmark.json) compare this
search with exhaustive repeated forward calls and verify identical first
qualifying designs. The measured improvement is about 42–80 times across three
parent-eligibility workloads. These are comparisons between Python algorithms,
not native Fortran benchmarks or guarantees for other models/hardware.
