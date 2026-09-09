"""Run the STATTAB console with ``python -m mdanderson_stats.stattab``."""

from .cdflib_console import CDFConsoleError
from .stattab_console import run_stattab


def main() -> None:
    """Interactive entry point, including the optional report-file dialogue."""
    try:
        run_stattab(ask_report=True)
    except (CDFConsoleError, OSError) as error:
        raise SystemExit(str(error)) from error
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
