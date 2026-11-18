# RTX 5090 memory profile

Initial planning values (GiB):

- VRAM total: 32
- Soft budget: 26
- Hard ceiling: 28
- Safety reserve: 4
- Model/expert working set: 18
- State/KV/canvas: 4
- CUDA workspace: 2
- Transfer: 1
- Allocator slack: 1
- Expert cache target: 14; maximum: 16 (included in the 18 GiB working set)

Host RAM planning:
- Claymore budget: 88 GiB
- Weight pool target: 64 GiB
- Pinned-memory ceiling: 8 GiB
- Staging/state: 16 GiB
- System reserve: 24 GiB

These are nested planning budgets, not additive independent allocations. Measure actual usage on the target machine before raising limits.
