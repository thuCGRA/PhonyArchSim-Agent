#!/usr/bin/env python3
"""Create the ``OPENAI_API_KEY`` environment variable for the current terminal.

This is the book's credential step, adapted to a real Bash/Zsh session.  The book
shows two ways of setting the key:

    export OPENAI_API_KEY="your_openai_api_key"          # in the terminal
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # in a Python script

The second form only affects the running process and its children -- a Python
script can never mutate the environment of the shell that launched it.  So this
script *prints* the shell statement and the caller evaluates it:

    eval "$(python configs/api_keys.py)"                # ask for the key (hidden input)
    eval "$(python configs/api_keys.py --key sk-...)"   # pass it explicitly
    eval "$(python configs/api_keys.py)"                # reuse the one already exported

Put the key in ``API_KEY`` below if you prefer the book's "write it once in the
file" style; an explicit ``--key`` still wins.  This file is gitignored
(``.gitignore``) precisely because that slot may hold a real secret.

Inspection helpers (no secret is printed in full):

    python configs/api_keys.py --check    # exit 0 if set, 1 if not; prints masked value
    python configs/api_keys.py --masked   # just the masked value

Removing it again:

    eval "$(python configs/api_keys.py --unset)"

Design notes:
* Every human-readable hint goes to **stderr**, so ``$(...)`` captures nothing but
  the shell statement.  Otherwise ``eval`` would try to execute the hints.
* Only ``export``/``unset``/``false`` ever reach stdout.  Error paths emit
  ``false`` when stdout is captured, so ``eval "$(...)"`` fails loudly instead of
  silently evaluating an empty string.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import sys

ENV_NAME = "OPENAI_API_KEY"

# ---- the book's line: put your key here, or leave None and use --key / the prompt.
API_KEY: str | None = None

PLACEHOLDERS = {"your_openai_api_key", "sk-...", "...", "xxx"}


# --------------------------------------------------------------------------- #
# key resolution
# --------------------------------------------------------------------------- #
def mask(key: str) -> str:
    """Render a key safely for logs: ``sk-a...wxyz (len=51)``."""
    if len(key) <= 8:
        return f"{'*' * len(key)} (len={len(key)})"
    return f"{key[:4]}...{key[-4:]} (len={len(key)})"


def _looks_malformed(key: str) -> str | None:
    """Return a hint string when the key looks like a placeholder / corrupted."""
    if key in PLACEHOLDERS:
        return f"'{key}' is still the book's placeholder, not a real key"
    if any(ch.isspace() for ch in key):
        return "the value contains whitespace; quote it or re-copy it"
    return None


def resolve_key(explicit: str | None) -> str:
    """Explicit ``--key`` > ``API_KEY`` constant > already-exported value > prompt."""
    for source, value in (
        ("--key", explicit),
        ("API_KEY in this file", API_KEY),
        (f"${ENV_NAME} (already set)", os.environ.get(ENV_NAME)),
    ):
        if value and value.strip():
            key = value.strip()
            if key != value:
                # A trailing newline from `$(cat key.txt)` is the usual culprit.
                _hint("stripped surrounding whitespace from the key")
            if source != "--key":
                _hint(f"using the key from {source}")
            return key

    if not sys.stdin.isatty():
        raise SystemExit(
            f"error: {ENV_NAME} is not set and there is no terminal to prompt on.\n"
            f"       pass it explicitly:  eval \"$(python configs/api_keys.py --key sk-...)\""
        )

    # getpass keeps the key off the screen and out of shell history.
    try:
        key = getpass.getpass(f"Paste your {ENV_NAME} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        # Ctrl-C / closed stdin must not escape as a traceback.
        raise SystemExit(f"error: input cancelled; {ENV_NAME} was not created.")
    if not key:
        raise SystemExit(f"error: no key entered; {ENV_NAME} was not created.")
    return key


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def _hint(message: str) -> None:
    """Diagnostics for the human; never captured by ``$(...)``."""
    print(f"[api_keys] {message}", file=sys.stderr)


def _report_failure(message: str) -> None:
    """Tell the user, and make a wrapping ``eval`` return non-zero too."""
    print(f"error: {message}", file=sys.stderr)
    if not sys.stdout.isatty():
        # stdout is being captured by $(...) -- hand eval a failing command so the
        # caller does not believe the variable was created.
        print("false")


def export_line(key: str) -> str:
    """The single shell statement that creates the variable."""
    return f"export {ENV_NAME}={shlex.quote(key)}"


def apply_in_process(key: str) -> None:
    """The book's in-script form, for code that imports this module."""
    os.environ[ENV_NAME] = key


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="api_keys.py",
        description=f"Create/refresh the {ENV_NAME} environment variable in this terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical use:\n"
            '  eval "$(python configs/api_keys.py)"\n'
            '  eval "$(python configs/api_keys.py --key sk-...)"\n'
            "  python configs/api_keys.py --check\n"
            '  eval "$(python configs/api_keys.py --unset)"\n'
        ),
    )
    parser.add_argument("--key", default=None,
                        help="Key value. Omit to use API_KEY in this file, the already "
                             "exported value, or an interactive hidden prompt.")
    parser.add_argument("--check", action="store_true",
                        help="Print the masked status and exit 0 if set, 1 if not.")
    parser.add_argument("--masked", action="store_true",
                        help="Print only the masked value of the configured key.")
    parser.add_argument("--unset", action="store_true",
                        help="Emit 'unset' instead of 'export' (use with eval).")
    parser.add_argument("--quiet", action="store_true", help="Suppress the stderr hints.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.check:
        current = os.environ.get(ENV_NAME, "").strip()
        if current:
            print(f"{ENV_NAME} is set: {mask(current)}")
            return 0
        print(f"{ENV_NAME} is NOT set")
        return 1

    if args.masked:
        current = (args.key or API_KEY or os.environ.get(ENV_NAME, "")).strip()
        print(mask(current) if current else f"{ENV_NAME} is NOT set")
        return 0 if current else 1

    if args.unset:
        print(f"unset {ENV_NAME}")
        if not args.quiet:
            _hint("emitted 'unset'; run it with: eval \"$(python configs/api_keys.py --unset)\"")
        return 0

    try:
        key = resolve_key(args.key)
    except SystemExit as exc:  # only raised for missing-input paths
        # Failures are never silenced: --quiet only mutes the routine hints.
        _report_failure(str(exc).replace("error: ", "", 1))
        return 1

    warning = _looks_malformed(key)
    if warning and not args.quiet:
        _hint(f"warning: {warning}")

    print(export_line(key))
    if not args.quiet:
        _hint(f"exported {ENV_NAME}={mask(key)} for this shell")
        _hint('apply it with:  eval "$(python configs/api_keys.py)"')
        _hint("verify with:    python configs/api_keys.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
