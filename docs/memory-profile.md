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

Host RAM planning (reference frame: 128 GiB physical machine):
- Claymore budget: 88 GiB, made of three disjoint envelopes:
  - Weight pool target: 64 GiB (tier `Host`)
  - Staging/state: 16 GiB (tier `Host`)
  - Pinned-memory ceiling: 8 GiB (tier `PinnedHost`)
- System reserve: 24 GiB (outside the Claymore budget)

The `Host` tier limit is therefore 80 GiB (64 + 16) and the `PinnedHost` tier limit is 8 GiB, so the total never exceeds 88 GiB.

In VRAM, the expert cache is nested inside the working set (not additive). In host RAM, the envelopes above are disjoint and add up to the Claymore budget. Measure actual usage on the target machine before raising limits. See `Claymore.md` sections 5 and 7.
