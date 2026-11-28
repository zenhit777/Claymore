#!/usr/bin/env python3
"""Turn an M11.4 upstream-differential test log into audit tables.

The differential test prints one `diff,<stage>,...` line per recorded stage on
every pass boundary. This script keeps the *worst* row seen for each stage and
renders a Markdown table, so the audit document is generated from observed test
output instead of being retyped by hand.

Usage:
    python3 tools/m11/summarize_differential_log.py \
        docs/audits/m11.4-final-2026-11-27-differential.log.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"^diff,(?P<stage>[^,]+),(?P<checked>\d+),(?P<worst_abs>[0-9.eE+-]+),"
    r"(?P<worst_rel>[0-9.eE+-]+),(?P<worst_rel_l2>[0-9.eE+-]+),(?P<fails_6>\d+),"
    r"(?P<fails_8>\d+),(?P<violations>\d+)$"
)

# The test prints a header before each batch of rows: `diff,<label>,checked,...`.
# The label carries the comparison mode, so rows are attributed to that section.
HEADER = re.compile(r"^diff,(?P<label>[^,]+),checked,")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--skip-layers",
        action="store_true",
        help="drop the per-layer '#L<nn>' breakdown and keep the aggregates",
    )
    args = parser.parse_args()

    worst: dict[str, dict[str, float]] = {}
    section = ""
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        header = HEADER.match(line)
        if header:
            # `<mode>/after-<pass>` -> `<mode>`
            section = header.group("label").split("/")[0] + "/"
            continue
        match = LINE.match(line)
        if not match:
            continue
        record = match.groupdict()
        stage = section + record["stage"]
        if args.skip_layers and "#L" in stage:
            continue
        previous = worst.get(stage)
        if previous is None or float(record["worst_rel"]) > float(previous["worst_rel"]):
            worst[stage] = record

    if not worst:
        print(f"no differential rows found in {args.log}", file=sys.stderr)
        return 1

    print(
        "| stage | comparés | pire absolu | pire relatif | pire relatif L2 | "
        "> 2⁻⁸ | > 2⁻⁶ | violations |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stage in sorted(worst):
        record = worst[stage]
        checked = max(int(record["checked"]), 1)
        fails_6 = int(record["fails_6"])
        fails_8 = int(record["fails_8"])
        print(
            f"| `{stage}` | {record['checked']} | {float(record['worst_abs']):.3e} | "
            f"{float(record['worst_rel']):.3e} | {float(record['worst_rel_l2']):.3e} | "
            f"{fails_8} ({100.0 * fails_8 / checked:.1f}%) | "
            f"{fails_6} ({100.0 * fails_6 / checked:.1f}%) | {record['violations']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
