# CUDA backend v0.4

## Scope

The `claymore-cuda` crate has a hardware-independent default build and an opt-in CUDA Driver API backend.

Enable it with:

```bash
cargo check -p claymore-cuda --features cuda-driver
```

The feature links against the NVIDIA driver library (`libcuda.so.1`). It requires a compatible NVIDIA driver exposed inside Linux/WSL2. The CUDA toolkit is not required for Driver API linking if the driver library and linker symlink are available.

## Implemented

- Initialize the CUDA driver and select a device ordinal.
- Read device name and total memory.
- Query live free/total device memory.
- Allocate and free device buffers.
- Allocate and free portable page-locked host buffers.
- Create and synchronize CUDA streams.
- Create, record, and synchronize CUDA events.
- Enqueue asynchronous H2D and D2H transfers.

## Safety contract

Async copy methods are unsafe by design. Callers must keep both allocations alive and prevent conflicting access until the stream completes. The memory crate's logical event horizon is not automatically wired to CUDA events yet; that adapter is future work. A buffer must not be dropped until all work using it has completed.

## Known limitations

- Compute capability is not queried yet.
- Device selection is ordinal-based.
- No allocator pool, defragmentation, telemetry, or automatic eviction integration yet.
- No WSL2 runtime test has been performed from this development environment.
- CUDA feature compilation requires the NVIDIA driver library; default CI tests without hardware.
