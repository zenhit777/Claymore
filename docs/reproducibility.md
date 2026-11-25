# Reproducibility policy

Every validation claim in this repository — a golden fixture, an audit log, a
hardware-report number — must be replayable from a clean checkout by someone
who was not in the original session. Concretely:

## 1. Nothing a test needs is a mystery path

Contract tests may depend on exactly two kinds of external input:

* **committed fixtures** under the test crate (small enough to review, hashed
  in `tests/*` via a pinned manifest digest), and
* **pinned large assets** (model checkpoints, historical golden sets) read
  from documented environment variables (e.g. `CLAYMORE_M10_CHECKPOINT`), with
  pinned upstream revision + file digests recorded in the docs and in every
  generated manifest.

No test may hardcode a session-scoped path (`/tmp/opencode/…`) as its only
source of truth. A clean checkout plus the env var is the contract.

## 2. Goldens are immutable history

Regenerating a reference **never overwrites** an approved set: generators write
into a fresh, empty directory (and refuse otherwise), and gate the new output
bit-exactly against the historical set. Changing a pinned digest is a
deliberate, documented commit with the reason, the new environment fingerprint,
and the full validation results (`tools/m10/README.md` documents the M10
instance of this procedure).

## 3. Generators are versioned, deterministic, and fingerprinted

Throwaway campaign scripts are replaced by `tools/` versions that (a) pin their
dependency environment (`requirements.*.lock.txt` + setup scripts that assert
the versions), (b) pin the upstream commit they were run against and hash the
upstream module files actually imported, (c) record the full environment
fingerprint (libc, CPU model, thread counts, versions) into a manifest, and
(d) record internal consistency checks so a silent semantic change cannot pass.

## 4. Campaigns leave raw evidence

Hardware or long-validation campaigns (M8–M10 style) produce, under
`docs/audits/`: the exact commands with exit codes, tool versions, the raw logs
(`*.log.txt`), and the measured tables the numeric bounds were fixed from.
Bounds are set from measurement, once, never nudged after the fact; a re-run
that exceeds a bound is an incident to investigate, not a threshold to edit.

## 5. CI runs the non-hardware contract

Everything that does not need a GPU must run in default CI from a clean
checkout, including fixture validation (`m10_reference.rs`) — the CPU contract
is machine-verifiable; only pinned-checkpoint tests and hardware tests are
`#[ignore]`d, each documenting its exact replay command.
