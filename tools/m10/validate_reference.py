#!/usr/bin/env python3
"""Validate a regenerated M10 reference directory against a pinned manifest.

Criterion 2 of `docs/audits/m10-cpu-finalization-contract.md`: control the
manifest, the sizes, the shapes, the axis order, the finite values and the
SHA-256 of every artifact. This validator never writes into either directory;
it only compares `--dir` artifacts against the entries of `--against`
(a `manifest.json` produced by `gen_reference.py`).

Environment differences (torch/transformers/libc/threads) are reported and,
with `--strict-env`, fail the validation. Artifact mismatches always fail.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--against", required=True, help="pinned manifest.json")
    parser.add_argument("--dir", required=True, help="regenerated artifact directory")
    parser.add_argument("--strict-env", action="store_true", help="environment fingerprint drift fails")
    args = parser.parse_args()

    ref = json.loads(Path(args.against).read_text())
    root = Path(args.dir)
    failures, checked = [], 0
    for name, meta in sorted(ref["artifacts"].items()):
        path = root / meta["file"]
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        import numpy as np

        flat = np.fromfile(path, dtype="<f4")
        numel = 1
        for dim in meta["shape"]:
            numel *= dim
        if flat.size != numel:
            failures.append(f"{name}: size {flat.size} != prod(shape) {numel}")
            continue
        if not bool(np.isfinite(flat).all()):
            failures.append(f"{name}: non-finite values")
            continue
        if meta["dtype"] != "float32-le":
            failures.append(f"{name}: unexpected dtype {meta['dtype']}")
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            failures.append(f"{name}: sha256 {digest[:16]}… != pinned {meta['sha256'][:16]}…")
            continue
        checked += 1

    env_a, env_b = ref.get("environment", {}), None
    local_manifest = root / "manifest.json"
    if local_manifest.is_file():
        env_b = json.loads(local_manifest.read_text()).get("environment", {})
    drift = []
    if env_b:
        for key in sorted(set(env_a) | set(env_b)):
            if env_a.get(key) != env_b.get(key):
                drift.append(f"{key}: {env_a.get(key)!r} vs {env_b.get(key)!r}")

    print(f"artifacts validated: {checked}/{len(ref['artifacts'])}")
    for line in failures:
        print("FAIL", line)
    for line in drift:
        print("env-drift" if args.strict_env else "env-note", line)
    if failures or (drift and args.strict_env):
        print("VALIDATE_FAILED")
        return 1
    print("VALIDATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
