# Memory Orchestrator v0.3

The memory crate is a deterministic host-side policy layer.

## Implemented

- Typed VRAM, host RAM, and pinned-host budget validation.
- Allocation accounting by memory tier.
- Admission planning with deterministic eviction candidates.
- LRU, LFU, and adaptive eviction policy modes.
- Logical event-horizon protection: allocations with `safe_after_tick > now_tick` are not candidates for eviction.
- Explicit eviction confirmation, so accounting changes only after a backend confirms that memory was freed.
- Expert-cache usage, target, and maximum counters.

## Backend contract

The orchestrator does **not** call CUDA. A future CUDA backend must:

1. Query actual device and host memory availability.
2. Ask the orchestrator for an admission plan.
3. Wait for CUDA events associated with planned victims.
4. Free only allocations whose events have completed.
5. Confirm each successful free through `confirm_eviction`.
6. Perform the allocation and register it with `record_allocation` only after success.

Never treat a planned eviction as a completed free. The current logical tick is an abstraction for tests and policy integration; the CUDA backend should replace it with event-backed readiness.

## Validation

Run locally:

```bash
cargo fmt --all
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The repository changes have not been compiled in this environment yet.
