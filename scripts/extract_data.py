"""Convert the supplied PDFs in data/ to JSON in data/generated/.

Run: python scripts/extract_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from copilot.core.config import get_settings
from copilot.core.exceptions import CopilotError
from copilot.data.extract import extract_all


def main() -> int:
    """Extract all PDFs. Returns a process exit code."""
    try:
        settings = get_settings()
        manifest = extract_all(settings.data_dir, settings.generated_dir)
    except CopilotError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    for name, info in manifest["files"].items():
        sys.stdout.write(f"{name}: {info['records']} record(s) from {info['source']}\n")
    sys.stdout.write(f"written to {Path(settings.generated_dir)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
