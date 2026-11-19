# WSL2 notes

Suggested starting point for `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=115GB
processors=16
swap=16GB
```

This sets the WSL2 VM memory ceiling to 115 GB. On the 128 GiB reference machine, roughly 13 GiB remain outside the VM for Windows and host-side processes (128 - 115, subject to the GB/GiB conversion and the actual Windows memory report). Claymore's 88 GiB application budget is inside the VM ceiling; the remaining VM capacity is intended for Linux, page cache, development tools, and runtime overhead. It is not a guaranteed free-memory reserve: measure actual Linux and Windows usage under the target workload before increasing application limits. This deliberately prioritizes a large WSL2 working set over a 24 GiB Windows-side reserve.

On this WSL2 setup, the CUDA Driver API library is exposed under `/usr/lib/wsl/lib`. When the linker does not find `libcuda.so`, pass `RUSTFLAGS="-L native=/usr/lib/wsl/lib"` to Cargo, for example:

```bash
RUSTFLAGS="-L native=/usr/lib/wsl/lib" cargo check -p claymore-cuda --features cuda-driver
RUSTFLAGS="-L native=/usr/lib/wsl/lib" cargo test -p claymore-memory --features cuda-driver -- --ignored
```

Restart WSL after changing the file (`wsl --shutdown`). Keep swap out of the model-weight tier; use it only as an emergency system mechanism.
