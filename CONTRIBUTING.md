# Contributing

The goal is usable, numerically validated Python workflows for every application
in the MD Anderson biostatistics catalog. Start with [software status](docs/software-status.md)
and the [remaining statistical contracts](research/statistical-coverage-priority-2026-10-04.md).
An entry's label reflects its documented scope; it is not a percentage of code
or a claim of native interface or clinical equivalence.

## Set up and check a change

Use Python 3.12 or newer and install uv. From the checkout:

```sh
uv sync --locked --group dev --extra plot
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
uv run --extra plot pytest -W error
```

During development run the affected tests first. Full regression tests may take
several minutes. Optional TIFF workflows require `uv sync --locked --group dev
--extra plot --extra image`. Use a writable `UV_CACHE_DIR` in a restricted
environment; do not change global environment settings to run tests.

## Implement and validate a statistical workflow

- Recover the source contract before implementing it: primary paper, public
  manual, lawful source archive, published examples, or reference-program outputs.
  Record URLs, versions, dates and hashes in a guide or research audit.
- Keep original downloads under ignored `research/raw/`. Do not commit original
  binaries, confidential data, credentials, or source without verified permission.
- Distinguish a native convention from an explicit Python policy. Missing fitted
  coefficients, priors or baseline hazards cannot be replaced with guesses while
  claiming equivalence. Record defects and corrected behavior separately.
- Validate against an independent mathematical calculation or executable source
  reference, with reproducible fixtures, tolerances and uncertainty. A second
  implementation of the same mistaken assumption is insufficient evidence.
- Document units, parameter domains, defaults, seed behavior, result fields and
  remaining limitations. Provide a short example that runs from an installed
  wheel. Public arrays should follow the existing immutable-result conventions.
- Keep `catalog.json`, guides and `docs/software-status.md` consistent. Promote
  partial status only after all declared calculations and workflows have evidence,
  tests and usable interfaces; keep unresolved source contracts visible.

## Packaging and review

Build both wheel and sdist and test an isolated wheel install outside the source
directory, including changed optional features. Confirm licenses/notices are
included and original research artifacts are excluded. Describe what changed,
the exact validation run, and limitations in the proposed pull request.

Read [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Original contributions are MIT; some adaptations retain other terms, including
commercial-use restrictions. Do not remove notices, relicense upstream code or
describe the current complete distribution as unrestricted open source. The
[licensing-path review](research/community-license-path-2026-10-12.md) identifies
the current separation/reimplementation work.

Feature-branch pushes do not automatically run the active GitHub workflow.
Pull requests targeting main/master and permitted manual dispatches do. A local
test result is not GitHub CI evidence. Creating a contribution does not authorize
merging, publishing a registry release, or changing repository access/settings.
