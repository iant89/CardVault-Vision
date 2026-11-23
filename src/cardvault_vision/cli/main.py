"""Argument parsing and dispatch.

Uses ``argparse`` deliberately: the CLI must work on a bare Python install with
no extra packages, because it is the entry point for data generation on CPU
workers where only numpy/Pillow are available.

Error handling contract: any :class:`CardVaultVisionError` prints a concise
message and exits ``2``; unexpected exceptions print a traceback and exit ``1``.
That makes CI failures distinguishable from bugs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from ..errors import CardVaultVisionError, MissingDependencyError
from ..logging_utils import setup_logging
from ..version import __version__
from . import (
    commands_benchmark,
    commands_dataset,
    commands_eval,
    commands_export,
    commands_models,
    commands_synthetic,
    commands_system,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision",
        description="CardVault Vision - the CardVault model factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  vision dataset bootstrap --count 40 --out datasets/source/demo\n"
            "  vision synthetic generate --source datasets/source/demo \\\n"
            "      --profile realistic --count 5000 --seed 12345 --out datasets/synthetic/v1\n"
            "  vision synthetic generate --source datasets/source/demo \\\n"
            "      --profile phone --problem glare:3 --count 2000 --out datasets/synthetic/glare\n"
            "  vision dataset split --manifest datasets/synthetic/v1 --seed 7\n"
            "  vision synthetic preview --source datasets/source/demo --count 12\n"
            "  vision train name-ocr --dataset datasets/synthetic/v1 --epochs 10\n"
            "  vision test name-ocr --checkpoint experiments/name-ocr/best.pt \\\n"
            "      --dataset datasets/synthetic/v1 --split test\n"
            "  vision benchmark --all\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"cardvault-vision {__version__}")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="console verbosity")
    parser.add_argument("--log-file", default=None, help="also write logs to this file")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    commands_dataset.register(sub)
    commands_synthetic.register(sub)
    commands_models.register(sub)
    commands_eval.register(sub)
    commands_benchmark.register(sub)
    commands_export.register(sub)
    commands_system.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, log_file=args.log_file, force=True)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - every registered group sets one
        parser.error(f"No handler for command {args.command!r}")

    log = logging.getLogger("cardvault_vision.cli")
    try:
        return int(handler(args) or 0)
    except MissingDependencyError as exc:
        log.error("%s", exc)
        return 2
    except CardVaultVisionError as exc:
        log.error("%s", exc)
        if args.log_level == "DEBUG":
            raise
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        log.warning("Interrupted")
        return 130
    except BrokenPipeError:  # pragma: no cover - piping into head
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
