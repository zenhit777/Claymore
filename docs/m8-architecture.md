# M8 — Claymore tensor engine architecture

## Status and scope

This document began as the architectural preparation deliverable produced during M7.1. M7.1 is now merged. M8 implementation is in progress; the first proposed increment introduces `claymore-tensor` metadata and storage ownership on the CPU reference path. This remains a partial foundation, not a complete tensor engine.

M8 establishes Claymore's native tensor execution foundation. Claymore owns the tensor model, storage and lifetime rules, operation contracts, dispatch policy, and integration with its scheduler and memory manager. CUDA provides device execution primitives; it must not become the owner of Claymore tensor semantics or orchestration.

The architecture must not assume that a tensor framework's abstractions, execution model, or model-specific runtime are interchangeable with Claymore's. Model families with materially different execution paradigms must retain separate model adapters and execution strategies over shared low-level tensor/device contracts.

## Layer boundaries

1. **Tensor core ('claymore-tensor', planned):** tensor metadata, shape/stride validation, layouts, element types, views, storage handles, and ownership/lifetime contracts. No CUDA FFI in this layer.
2. **Device abstraction ('claymore-device', evolve existing crate):** device identity, capabilities, allocation handles, host/device copies, synchronization primitives, and backend error mapping. CPU is the deterministic reference backend.
3. **Operation layer ('claymore-ops', planned):** validated operation descriptors and semantic contracts. Operations must define supported dtypes/layouts, shape constraints, aliasing rules, and numerical behavior.
4. **Kernel dispatch ('claymore-kernels', planned):** capability- and dtype-aware selection of CPU or CUDA kernels. Dispatch must be deterministic and explain unsupported combinations; it must not silently change precision.
5. **Runtime integration:** scheduler dependencies, transfer events, memory admission, cancellation, and resource retirement. Tensor operations may be submitted as work, but the scheduler does not own tensor storage.
6. **Model adapters:** model-specific graph/state, attention, routing, denoising/autoregressive control flow, and checkpoint interpretation. Model-specific behavior must not leak into the generic tensor core.

## Model execution paradigms

The shared tensor/device layer is deliberately below model execution semantics. It must not impose a conventional autoregressive, next-token decoding loop on every model.

- **DiffusionGemma adapter:** owns the diffusion-oriented generation loop, canvas/state representation, denoising progression, and the model's attention/state semantics.
- **Clover adapter:** owns its block-diffusion/GDN execution path, MoE routing, expert residency, and the scheduling of expert transfers.
- **Shared services:** tensor storage, validated primitive operations, backend capability discovery, memory admission, event-backed transfers, and task dependencies.

These adapters may share tensor primitives and memory services without sharing a single model graph, generation loop, state machine, or scheduler policy. M8 must first inspect the actual model graph and checkpoint contracts before selecting its initial operation set. The generic tensor engine must not encode assumptions borrowed from unrelated runtimes or frameworks.

## Tensor contract

A tensor descriptor must make the following explicit:

- **Shape:** rank and dimensions, with checked element-count multiplication and explicit handling of zero-sized dimensions.
- **Strides:** element or byte unit must be unambiguous; address-span calculations must be checked for overflow.
- **Element type:** an explicit closed enum for supported scalar representations. Storage byte width is derived from the type, never supplied independently.
- **Layout:** contiguous and strided layouts are distinct. Any specialized layout must be named and validated rather than inferred from shape alone.
- **Storage:** a tensor owns or shares a storage handle with an explicit device and byte capacity. A view references storage and carries an offset/shape/stride descriptor; it does not pretend to own an independent allocation.
- **Lifetime:** storage remains alive until all scheduled operations that reference it have completed. An in-flight tensor/storage cannot be released or evicted.
- **Mutability and aliasing:** read/write access and overlap rules are explicit per operation. Safe Rust APIs must not expose mutable aliases to overlapping storage.
- **Transfer:** host/device copies are explicit operations with completion tokens or scheduler dependencies. A successful enqueue is not equivalent to completed transfer.

The first implementation should prefer a small, auditable set of layouts and dtypes over broad but weakly specified support.

## Device and backend contract

A device reports stable identity and capabilities (supported dtypes, layouts, operation families, memory tiers, alignment constraints, and relevant limits). Capability discovery is separate from kernel selection.

The CPU backend is the semantic reference and must support deterministic tests without a GPU. The CUDA backend is optional and feature-gated. CUDA context/thread affinity, stream ownership, event completion, and driver errors remain encapsulated in the backend. A backend may expose opaque allocation and event handles, but callers must not manufacture raw CUDA handles.

Device memory allocation and release are coordinated with the memory manager. A backend release failure retains ownership and accounting; the system must never report a buffer freed before the backend confirms release.

## Operation and kernel contract

Each operation descriptor defines:

- input/output tensor requirements and shape inference;
- dtype and layout support, including explicit promotion/cast rules;
- output allocation requirements and permitted input/output aliasing;
- synchronous completion or asynchronous completion-token semantics;
- numerical tolerance and deterministic-mode behavior;
- backend capability requirements and fallback policy.

Kernel dispatch follows a strict order: validate semantic requirements, query backend capabilities, select a compatible kernel, reserve required memory, schedule dependencies/transfers, then execute. An unsupported operation returns a typed error. No implicit CPU fallback, precision downgrade, or hidden host round-trip is permitted unless explicitly requested by the caller or model strategy.

## Runtime and memory integration

- The scheduler owns dependency and task state, not tensor data.
- Transfer completion is established by the backend event, never by enqueue success or elapsed time.
- Memory admission considers the complete allocation set and all applicable budgets before evicting.
- Every allocation referenced by an in-flight operation remains guarded.
- Cancellation of queued work discards its payload; cancellation of running work requires backend-confirmed quiescence before resources can be released.
- Execution failures and panics transition tasks to terminal failure and propagate cancellation to dependents.
- Metrics distinguish queue delay, host execution, kernel execution, transfer enqueue, transfer completion, and allocation/eviction costs.

## M8 delivery sequence

1. **In progress (PR M8.1):** establish tensor metadata and storage ownership types; checked shape/stride arithmetic and CPU tests are proposed in `claymore-tensor`. The existing `claymore-device::Tensor` remains for compatibility during this increment; migration/removal requires a separate reviewed change.

2. Implement CPU storage and a minimal reference operation set with exhaustive unit tests.
3. Add operation descriptors, shape inference, typed validation errors, and explicit casting.
4. Add the backend capability interface and adapt the existing CPU device.
5. Add kernel dispatch with deterministic selection and unsupported-operation behavior.
6. Integrate transfer dependencies and memory lifetime guards.
7. Add the CUDA backend implementation behind the existing optional feature, with CPU-only CI coverage and separate RTX 5090 validation.
8. Add model-adapter integration only after generic tensor and operation contracts are stable.

The exact initial operation set is to be selected from the first M8 model execution path after reviewing its actual graph and numerical requirements; it must not be guessed from a generic framework checklist.

## Required validation gates

**CPU/CI:** tensor metadata and overflow tests; storage/view aliasing tests; operation shape/dtype/layout validation; deterministic CPU kernels; scheduler cancellation/failure tests; memory-accounting and release-failure tests; rustfmt, workspace tests, Clippy, MSRV, and CUDA-feature compilation without a physical GPU.

**RTX 5090:** CUDA context and stream lifecycle; allocations and transfers; event completion; kernel correctness against CPU reference within operation-specific tolerances; numerical stability at supported precisions; memory-pressure and eviction behavior; representative end-to-end workload and performance measurements. Hardware results must record the exact commit SHA, driver/runtime versions, device identity, and test commands.

## Explicit non-goals for M8 foundation

- Do not make a third-party tensor framework the owner of Claymore's tensor contracts.
- Do not merge distinct model paradigms into one generic model execution loop.
- Do not implement model-specific attention, routing, denoising, or autoregressive control flow inside the tensor core.
- Do not claim performance or hardware readiness from CPU simulation or feature compilation alone.
