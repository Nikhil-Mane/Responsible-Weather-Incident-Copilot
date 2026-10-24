"""Repeatable evaluation command.

Run: python eval/run_eval.py [--report PATH] [--min-accuracy 0.9]
Exit code 0 when every gate passes, 8 when one fails (see `copilot.cli`).
"""

from __future__ import annotations

import sys

from copilot.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
