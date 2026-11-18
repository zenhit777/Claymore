# WSL2 notes

Suggested starting point for `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=96GB
processors=16
swap=16GB
```

This gives the WSL2 VM a 96 GiB memory ceiling. The Claymore application budget is 88 GiB, leaving approximately 8 GiB for Linux and page cache; this headroom is an estimate and must be measured on the target machine. The physical machine has 128 GiB of RAM, so about 32 GiB remain outside the VM (128 - 96); the 24 GiB system reserve of the planning profile is an indicative allocation within that Windows-side memory.

Restart WSL after changing the file (`wsl --shutdown`). Keep swap out of the model-weight tier; use it only as an emergency system mechanism.
