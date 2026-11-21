//! Hardware validation of `CudaMemoryManager` contracts M4/M5 on a real GPU.
//!
//! These tests are ignored by default so the standard CI stays
//! hardware-independent. Tests that assert an event is still pending immediately
//! after enqueue must queue enough DMA work to make that observation robust across
//! GPU load and host scheduling. Run them with:
//!
//! ```text
//! RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib \
//! cargo test -p claymore-memory --features cuda-driver -- --ignored --test-threads=1
//! ```

use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use claymore_core::MemoryTier;
use claymore_memory::{
    CudaMemoryError, CudaMemoryManager, CudaMemoryTask, CudaTransferExecutor,
    EvictionPolicy, MemoryBudgets, MemoryError,
};
use claymore_scheduler::{
    ExecutionOutcome, QueueRunner, Scheduler, SchedulerConfig, TaskKind, TaskSpec,
};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn budgets(vram_bytes: u64, pinned_bytes: u64) -> MemoryBudgets {
    let host_bytes = pinned_bytes.saturating_mul(2).max(2 * 1024 * 1024);
    MemoryBudgets {
        vram_soft_bytes: vram_bytes,
        vram_hard_bytes: vram_bytes,
        host_bytes,
        pinned_host_bytes: pinned_bytes,
        expert_cache_target_bytes: 0,
        expert_cache_max_bytes: 0,
    }
}

fn manager(vram_bytes: u64, pinned_bytes: u64) -> CudaMemoryManager {
    CudaMemoryManager::new(0, budgets(vram_bytes, pinned_bytes), EvictionPolicy::Lru)
        .expect("CUDA memory manager")
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    let mut state = u32::from(seed) | 1;
    let mut bytes = Vec::with_capacity(len);
    for index in 0..len {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        bytes.push((state >> 24) as u8 ^ (index as u8));
    }
    bytes
}

struct Ticker(u64);

impl Ticker {
    fn new() -> Self {
        Self(1)
    }
    fn tick(&mut self) -> u64 {
        let tick = self.0;
        self.0 += 1;
        tick
    }
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c2_03_manager_release_semantics() {
    let _serial = serial();
    let mut manager = manager(2 * 1024 * 1024, 1024 * 1024);
    let mut tick = Ticker::new();

    let host_total = 128_u64 * 1024 * 1024 * 1024;
    let host_available = 100_u64 * 1024 * 1024 * 1024;
    let snapshot = manager
        .snapshot(host_total, host_available)
        .expect("memory snapshot");
    assert!(snapshot.vram_free_bytes <= snapshot.vram_total_bytes);
    assert!(snapshot.vram_total_bytes > 0);
    assert_eq!(snapshot.host_total_bytes, host_total);
    assert_eq!(snapshot.host_available_bytes, host_available);
    assert!(!manager.device_info().name.is_empty());

    manager
        .allocate_pinned("host", 4096, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("device", 4096, tick.tick(), false)
        .expect("device allocation");
    let data = pattern(4096, 3);
    manager
        .write_pinned("host", 0, &data, tick.tick())
        .expect("pinned write");
    let transfer = manager
        .copy_h2d("host", "device", data.len(), tick.tick())
        .expect("H2D enqueue");
    manager
        .wait_for_transfer(&transfer, tick.tick())
        .expect("H2D completion");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 4096);
    assert_eq!(manager.usage_bytes(MemoryTier::PinnedHost), 4096);

    let released = manager
        .release("device", tick.tick())
        .expect("explicit release");
    assert_eq!(released.id, "device");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 0);
    assert!(matches!(
        manager.release("device", tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "device"
    ));
    assert!(matches!(
        manager.copy_h2d("host", "device", 1, tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "device"
    ));
    assert!(matches!(
        manager.copy_d2h("device", "host", 1, tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "device"
    ));

    manager
        .allocate_device("device", 4096, tick.tick(), false)
        .expect("device reallocation");
    assert!(matches!(
        manager.allocate_device("device", 4096, tick.tick(), false),
        Err(CudaMemoryError::Policy(MemoryError::DuplicateAllocation(id))) if id == "device"
    ));
    assert!(matches!(
        manager.allocate_device("zero", 0, tick.tick(), false),
        Err(CudaMemoryError::ZeroSizedAllocation)
    ));

    manager
        .release("device", tick.tick())
        .expect("device release");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 0);
    assert_eq!(manager.usage_bytes(MemoryTier::PinnedHost), 0);
    println!("HW-C2-03 explicit release, accounting and post-release errors verified");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c3_01_transfer_sizes_are_byte_exact() {
    let _serial = serial();
    let max = 64 * 1024 * 1024;
    let mut manager = manager(128 * 1024 * 1024, 128 * 1024 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", max as u64, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("device", max as u64, tick.tick(), false)
        .expect("device allocation");

    for (index, len) in [1_usize, 37, 4096, 1024 * 1024, max]
        .into_iter()
        .enumerate()
    {
        let expected = pattern(len, index as u8);
        manager
            .write_pinned("host", 0, &expected, tick.tick())
            .expect("pinned write");
        let h2d = manager
            .copy_h2d("host", "device", len, tick.tick())
            .expect("H2D enqueue");
        manager
            .wait_for_transfer(&h2d, tick.tick())
            .expect("H2D completion");

        manager
            .write_pinned("host", 0, &vec![0_u8; len], tick.tick())
            .expect("pinned clear");
        let d2h = manager
            .copy_d2h("device", "host", len, tick.tick())
            .expect("D2H enqueue");
        manager
            .wait_for_transfer(&d2h, tick.tick())
            .expect("D2H completion");
        let copied = manager
            .read_pinned("host", 0, len, tick.tick())
            .expect("pinned read");
        assert_eq!(copied, expected, "integrity failure for {len} bytes");
        println!("HW-C3-01 size={len} byte_exact=yes");
    }

    manager
        .release("device", tick.tick())
        .expect("device release");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c3_02_multiple_allocations_and_successive_transfers() {
    let _serial = serial();
    let len = 1024 * 1024;
    let mut manager = manager(4 * len as u64, 4 * len as u64);
    let mut tick = Ticker::new();
    for id in ["host-a", "host-b"] {
        manager
            .allocate_pinned(id, len as u64, tick.tick())
            .expect("pinned allocation");
    }
    for id in ["device-a", "device-b"] {
        manager
            .allocate_device(id, len as u64, tick.tick(), false)
            .expect("device allocation");
    }

    let first = pattern(len, 11);
    let second = pattern(len, 29);
    manager
        .write_pinned("host-a", 0, &first, tick.tick())
        .expect("pinned write A");
    manager
        .write_pinned("host-b", 0, &second, tick.tick())
        .expect("pinned write B");

    let h2d_a = manager
        .copy_h2d("host-a", "device-a", len, tick.tick())
        .expect("H2D A");
    let h2d_b = manager
        .copy_h2d("host-b", "device-b", len, tick.tick())
        .expect("H2D B");
    manager
        .wait_for_transfer(&h2d_a, tick.tick())
        .expect("H2D A completion");
    manager
        .wait_for_transfer(&h2d_b, tick.tick())
        .expect("H2D B completion");

    manager
        .write_pinned("host-a", 0, &vec![0_u8; len], tick.tick())
        .expect("clear A");
    manager
        .write_pinned("host-b", 0, &vec![0_u8; len], tick.tick())
        .expect("clear B");
    let d2h_a = manager
        .copy_d2h("device-a", "host-a", len, tick.tick())
        .expect("D2H A");
    let d2h_b = manager
        .copy_d2h("device-b", "host-b", len, tick.tick())
        .expect("D2H B");
    manager
        .wait_for_transfer(&d2h_a, tick.tick())
        .expect("D2H A completion");
    manager
        .wait_for_transfer(&d2h_b, tick.tick())
        .expect("D2H B completion");

    assert_eq!(
        manager
            .read_pinned("host-a", 0, len, tick.tick())
            .expect("read A"),
        first
    );
    assert_eq!(
        manager
            .read_pinned("host-b", 0, len, tick.tick())
            .expect("read B"),
        second
    );
    for id in ["device-a", "device-b"] {
        manager.release(id, tick.tick()).expect("device release");
    }
    for id in ["host-a", "host-b"] {
        manager.release(id, tick.tick()).expect("pinned release");
    }
    println!("HW-C3-02 two allocation pairs transferred successively and verified");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c4_01_nonblocking_polling_and_completion() {
    let _serial = serial();
    let len = 256 * 1024 * 1024;
    let mut manager = manager(1024 * 1024 * 1024, 1024 * 1024 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", len as u64, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("device", len as u64, tick.tick(), false)
        .expect("device allocation");
    let expected = pattern(len, 5);
    manager
        .write_pinned("host", 0, &expected, tick.tick())
        .expect("pinned write");

    let mut tokens = Vec::new();
    for _ in 0..8 {
        tokens.push(
            manager
                .copy_h2d("host", "device", len, tick.tick())
                .expect("H2D enqueue"),
        );
    }
    assert!(
        !manager
            .is_transfer_complete(tokens.last().expect("token"))
            .expect("event query"),
        "the last transfer must still be pending immediately after enqueue"
    );
    let retired = manager
        .poll_completed_transfers(tick.tick())
        .expect("non-blocking poll");
    assert!(
        retired < tokens.len(),
        "NOT_READY must not be reported as completion"
    );
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), len as u64);

    for token in &tokens {
        manager
            .wait_for_transfer(token, tick.tick())
            .expect("transfer completion");
    }
    for token in &tokens {
        assert!(manager
            .is_transfer_complete(token)
            .expect("post-wait query"));
    }
    assert_eq!(
        manager
            .poll_completed_transfers(tick.tick())
            .expect("poll after waits"),
        0,
        "completed events must not be retired twice"
    );

    manager
        .write_pinned("host", 0, &vec![0_u8; len], tick.tick())
        .expect("pinned clear");
    let d2h = manager
        .copy_d2h("device", "host", len, tick.tick())
        .expect("D2H enqueue");
    manager
        .wait_for_transfer(&d2h, tick.tick())
        .expect("D2H completion");
    assert_eq!(
        manager
            .read_pinned("host", 0, len, tick.tick())
            .expect("pinned read"),
        expected
    );
    manager
        .release("device", tick.tick())
        .expect("device release");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    println!("HW-C4-01 retired={retired}/8 before wait; pending and complete states coherent");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c4_02_wait_for_allocation_is_scoped() {
    let _serial = serial();
    let small = 4096;
    let large = 256 * 1024 * 1024;
    let mut manager = manager(1024 * 1024 * 1024, 1024 * 1024 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host-small", small as u64, tick.tick())
        .expect("small pinned allocation");
    manager
        .allocate_device("device-small", small as u64, tick.tick(), false)
        .expect("small device allocation");
    manager
        .allocate_pinned("host-large", large as u64, tick.tick())
        .expect("large pinned allocation");
    manager
        .allocate_device("device-large", large as u64, tick.tick(), false)
        .expect("large device allocation");

    manager
        .write_pinned("host-small", 0, &pattern(small, 1), tick.tick())
        .expect("small pinned write");
    manager
        .write_pinned("host-large", 0, &pattern(large, 2), tick.tick())
        .expect("large pinned write");
    let small_token = manager
        .copy_h2d("host-small", "device-small", small, tick.tick())
        .expect("small H2D");
    let large_token = manager
        .copy_h2d("host-large", "device-large", large, tick.tick())
        .expect("large H2D");

    manager
        .wait_for_allocation("device-small", tick.tick())
        .expect("wait for small allocation");
    assert!(manager
        .is_transfer_complete(&small_token)
        .expect("small token query"));
    assert!(
        !manager
            .is_transfer_complete(&large_token)
            .expect("large token query"),
        "waiting for one allocation must not retire an unrelated large transfer"
    );
    manager
        .wait_for_allocation("device-large", tick.tick())
        .expect("wait for large allocation");
    assert!(manager
        .is_transfer_complete(&large_token)
        .expect("large token query"));

    manager
        .release("device-small", tick.tick())
        .expect("small device release");
    manager
        .release("device-large", tick.tick())
        .expect("large device release");
    manager
        .release("host-small", tick.tick())
        .expect("small pinned release");
    manager
        .release("host-large", tick.tick())
        .expect("large pinned release");
    println!("HW-C4-02 wait_for_allocation retired only transfers touching its target");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c4_03_tokens_are_manager_scoped() {
    let _serial = serial();
    let mut owner = manager(1024 * 1024, 1024 * 1024);
    let mut other = manager(1024 * 1024, 1024 * 1024);
    let mut tick = Ticker::new();
    owner
        .allocate_pinned("host", 4096, tick.tick())
        .expect("pinned allocation");
    owner
        .allocate_device("device", 4096, tick.tick(), false)
        .expect("device allocation");
    owner
        .write_pinned("host", 0, &pattern(4096, 9), tick.tick())
        .expect("pinned write");
    let token = owner
        .copy_h2d("host", "device", 4096, tick.tick())
        .expect("H2D enqueue");

    assert!(matches!(
        other.is_transfer_complete(&token),
        Err(CudaMemoryError::UnknownTransferToken(id)) if id == token.id()
    ));
    assert!(matches!(
        other.wait_for_transfer(&token, tick.tick()),
        Err(CudaMemoryError::UnknownTransferToken(id)) if id == token.id()
    ));
    owner
        .wait_for_transfer(&token, tick.tick())
        .expect("owner wait");
    owner
        .release("device", tick.tick())
        .expect("device release");
    owner.release("host", tick.tick()).expect("pinned release");
    println!("HW-C4-03 foreign transfer token rejected by another manager");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c5_01_in_flight_allocation_cannot_be_evicted() {
    let _serial = serial();
    let len = 512 * 1024 * 1024;
    let mut manager = manager(len as u64, len as u64);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", len as u64, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("device", len as u64, tick.tick(), false)
        .expect("device allocation");
    let expected = pattern(len, 13);
    manager
        .write_pinned("host", 0, &expected, tick.tick())
        .expect("pinned write");

    let mut tokens = Vec::new();
    for _ in 0..8 {
        tokens.push(
            manager
                .copy_h2d("host", "device", len, tick.tick())
                .expect("H2D enqueue"),
        );
    }
    assert!(
        !manager
            .is_transfer_complete(tokens.last().expect("token"))
            .expect("event query"),
        "at least the last of 4 GiB queued must still be pending"
    );

    let error = manager
        .allocate_device("intruder", 64 * 1024 * 1024, tick.tick(), false)
        .expect_err("a protected allocation must not be evicted");
    assert!(matches!(
        error,
        CudaMemoryError::Policy(MemoryError::InsufficientReclaimable {
            tier: MemoryTier::Vram,
            ..
        })
    ));
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), len as u64);

    let retired = manager
        .poll_completed_transfers(tick.tick())
        .expect("non-blocking poll");
    assert!(retired < tokens.len(), "pending work must remain protected");
    assert_eq!(
        manager.usage_bytes(MemoryTier::Vram),
        len as u64,
        "NOT_READY must not release the allocation guard"
    );

    manager
        .wait_for_allocation("device", tick.tick())
        .expect("wait for device transfers");
    manager
        .allocate_device("intruder", 64 * 1024 * 1024, tick.tick(), false)
        .expect("eviction becomes possible after completion");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 64 * 1024 * 1024);
    assert!(matches!(
        manager.release("device", tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "device"
    ));
    manager
        .release("intruder", tick.tick())
        .expect("intruder release");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    println!(
        "HW-C5-01 protected during {retired}/8 completed polls; evicted after confirmed completion"
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c5_02_shared_allocation_stays_protected_by_remaining_transfer() {
    let _serial = serial();
    let len = 256 * 1024 * 1024;
    let mut manager = manager(len as u64, 2 * len as u64);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host-a", len as u64, tick.tick())
        .expect("host A allocation");
    manager
        .allocate_pinned("host-b", len as u64, tick.tick())
        .expect("host B allocation");
    manager
        .allocate_device("shared", len as u64, tick.tick(), false)
        .expect("shared device allocation");
    manager
        .write_pinned("host-a", 0, &pattern(len, 41), tick.tick())
        .expect("host A write");
    manager
        .write_pinned("host-b", 0, &pattern(len, 43), tick.tick())
        .expect("host B write");

    let first = manager
        .copy_h2d("host-a", "shared", len, tick.tick())
        .expect("first transfer");
    let mut remaining = Vec::new();
    for _ in 0..8 {
        remaining.push(
            manager
                .copy_h2d("host-b", "shared", len, tick.tick())
                .expect("remaining transfer"),
        );
    }
    manager
        .wait_for_transfer(&first, tick.tick())
        .expect("first transfer completion");
    assert!(
        !manager
            .is_transfer_complete(remaining.last().expect("token"))
            .expect("remaining query"),
        "remaining transfers must still reference the shared allocation"
    );

    let error = manager
        .allocate_device("intruder", 64 * 1024 * 1024, tick.tick(), false)
        .expect_err("shared allocation is still protected by a pending event");
    assert!(matches!(
        error,
        CudaMemoryError::Policy(MemoryError::InsufficientReclaimable { .. })
    ));
    manager
        .wait_for_allocation("shared", tick.tick())
        .expect("wait for shared allocation");
    manager
        .allocate_device("intruder", 64 * 1024 * 1024, tick.tick(), false)
        .expect("eviction after the last reference completed");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 64 * 1024 * 1024);
    manager
        .release("intruder", tick.tick())
        .expect("intruder release");
    manager
        .release("host-a", tick.tick())
        .expect("host A release");
    manager
        .release("host-b", tick.tick())
        .expect("host B release");
    println!("HW-C5-02 shared allocation stayed protected until its last event completed");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c5_03_eviction_skips_protected_candidate() {
    let _serial = serial();
    // Queue 8 x 256 MiB (2 GiB total): a 512 KiB transfer was too short to
    // make the immediate pending-state assertion reliable across machines.
    let len = 256 * 1024 * 1024;
    let mut manager = manager(2 * len as u64, len as u64);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", len as u64, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("protected", len as u64, tick.tick(), false)
        .expect("protected allocation");
    manager
        .allocate_device("candidate", len as u64, tick.tick(), false)
        .expect("candidate allocation");
    manager
        .write_pinned("host", 0, &pattern(len, 47), tick.tick())
        .expect("pinned write");
    let mut transfers = Vec::new();
    for _ in 0..8 {
        transfers.push(
            manager
                .copy_h2d("host", "protected", len, tick.tick())
                .expect("protected transfer"),
        );
    }
    assert!(
        !manager
            .is_transfer_complete(transfers.last().expect("token"))
            .expect("protected query"),
        "the protected allocation must still have pending work after 2 GiB is queued"
    );

    manager
        .allocate_device("replacement", len as u64, tick.tick(), false)
        .expect("eviction of the unprotected candidate");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 2 * len as u64);
    assert!(matches!(
        manager.release("candidate", tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "candidate"
    ));
    manager
        .release("replacement", tick.tick())
        .expect("replacement release");
    manager
        .release("protected", tick.tick())
        .expect("protected allocation survived eviction");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 0);
    println!("HW-C5-03 eviction skipped the protected allocation and freed the idle candidate");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c6_01_polling_precedes_eviction_planning() {
    let _serial = serial();
    let mut manager = manager(1024 * 1024, 64 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", 4096, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("victim", 768 * 1024, tick.tick(), false)
        .expect("victim allocation");
    manager
        .write_pinned("host", 0, &pattern(4096, 17), tick.tick())
        .expect("pinned write");
    let token = manager
        .copy_h2d("host", "victim", 4096, tick.tick())
        .expect("H2D enqueue");

    let mut spins = 0_u32;
    while !manager.is_transfer_complete(&token).expect("event query") {
        spins += 1;
        assert!(spins < 5_000_000, "event did not complete");
        std::thread::yield_now();
    }
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 768 * 1024);

    // The event is complete but still tracked. Admission must poll it before
    // planning, otherwise the victim would still be considered in flight.
    manager
        .allocate_device("replacement", 512 * 1024, tick.tick(), false)
        .expect("admission after internal polling");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 512 * 1024);
    assert!(matches!(
        manager.release("victim", tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "victim"
    ));
    manager
        .release("replacement", tick.tick())
        .expect("replacement release");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    println!("HW-C6-01 completed event polled before eviction after {spins} queries");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c6_02_equal_priority_eviction_is_deterministic() {
    let _serial = serial();
    for round in 0..12 {
        let mut manager = manager(3 * 1024 * 1024, 64 * 1024);
        let mut tick = Ticker::new();
        let equal_priority_tick = tick.tick();
        for id in ["gamma", "alpha", "beta"] {
            manager
                .allocate_device(id, 1024 * 1024, equal_priority_tick, false)
                .expect("candidate allocation");
        }
        manager
            .allocate_device("delta", 1024 * 1024, tick.tick(), false)
            .expect("admission with eviction");
        assert_eq!(manager.usage_bytes(MemoryTier::Vram), 3 * 1024 * 1024);
        assert!(
            matches!(
                manager.release("alpha", tick.tick()),
                Err(CudaMemoryError::UnknownAllocation(id)) if id == "alpha"
            ),
            "round {round}: lexicographically first equal-priority id must be evicted"
        );
        for id in ["beta", "gamma", "delta"] {
            manager.release(id, tick.tick()).expect("remaining release");
        }
        assert_eq!(manager.usage_bytes(MemoryTier::Vram), 0);
    }
    println!("HW-C6-02 equal-priority eviction selected alpha in 12/12 rounds");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c6_03_accounting_and_admission_errors() {
    let _serial = serial();
    let mut manager = manager(2 * 1024 * 1024, 64 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_device("large", 1024 * 1024, tick.tick(), false)
        .expect("large allocation");
    manager
        .allocate_device("small", 512 * 1024, tick.tick(), false)
        .expect("small allocation");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 1536 * 1024);

    assert!(matches!(
        manager.allocate_device("large", 1024 * 1024, tick.tick(), false),
        Err(CudaMemoryError::Policy(MemoryError::DuplicateAllocation(id))) if id == "large"
    ));
    assert!(matches!(
        manager.allocate_device("oversized", 3 * 1024 * 1024, tick.tick(), false),
        Err(CudaMemoryError::Policy(MemoryError::RequestTooLarge { .. }))
    ));
    assert!(matches!(
        manager.allocate_pinned("zero", 0, tick.tick()),
        Err(CudaMemoryError::ZeroSizedAllocation)
    ));

    manager
        .release("large", tick.tick())
        .expect("large release");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 512 * 1024);
    assert!(matches!(
        manager.release("large", tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "large"
    ));
    manager
        .release("small", tick.tick())
        .expect("small release");
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 0);
    assert!(matches!(
        manager.release("missing", tick.tick()),
        Err(CudaMemoryError::UnknownAllocation(id)) if id == "missing"
    ));
    println!("HW-C6-03 accounting stayed consistent across release and admission errors");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c7_01_repeated_allocate_transfer_free_cycles() {
    let _serial = serial();
    let len = 64 * 1024;
    let mut manager = manager(8 * 1024 * 1024, 8 * 1024 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", len as u64, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("device", len as u64, tick.tick(), false)
        .expect("device allocation");

    let started = Instant::now();
    for cycle in 0..128_u32 {
        let expected = pattern(len, cycle as u8);
        manager
            .write_pinned("host", 0, &expected, tick.tick())
            .expect("pinned write");
        let h2d = manager
            .copy_h2d("host", "device", len, tick.tick())
            .expect("H2D enqueue");
        manager
            .wait_for_transfer(&h2d, tick.tick())
            .expect("H2D completion");
        manager
            .write_pinned("host", 0, &vec![0_u8; len], tick.tick())
            .expect("pinned clear");
        let d2h = manager
            .copy_d2h("device", "host", len, tick.tick())
            .expect("D2H enqueue");
        manager
            .wait_for_transfer(&d2h, tick.tick())
            .expect("D2H completion");
        assert_eq!(
            manager
                .read_pinned("host", 0, len, tick.tick())
                .expect("pinned read"),
            expected,
            "cycle {cycle}"
        );
    }
    let elapsed = started.elapsed();
    manager
        .release("device", tick.tick())
        .expect("device release");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    println!("HW-C7-01 128 H2D/D2H cycles verified in {elapsed:?}");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c7_02_tight_polling_is_idempotent() {
    let _serial = serial();
    let len = 64 * 1024;
    let mut manager = manager(1024 * 1024, 1024 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", len as u64, tick.tick())
        .expect("pinned allocation");
    manager
        .allocate_device("device", len as u64, tick.tick(), false)
        .expect("device allocation");
    let expected = pattern(len, 23);
    manager
        .write_pinned("host", 0, &expected, tick.tick())
        .expect("pinned write");
    let mut tokens = Vec::new();
    for _ in 0..8 {
        tokens.push(
            manager
                .copy_h2d("host", "device", len, tick.tick())
                .expect("H2D enqueue"),
        );
    }

    let mut retired_total = 0_usize;
    for _ in 0..1000 {
        let retired = manager
            .poll_completed_transfers(tick.tick())
            .expect("non-blocking poll");
        retired_total += retired;
        assert!(retired_total <= tokens.len());
        for token in &tokens {
            let _ = manager
                .is_transfer_complete(token)
                .expect("token query during polling");
        }
    }
    for token in &tokens {
        manager
            .wait_for_transfer(token, tick.tick())
            .expect("final wait");
        assert!(manager
            .is_transfer_complete(token)
            .expect("final token query"));
    }
    assert_eq!(
        manager
            .poll_completed_transfers(tick.tick())
            .expect("poll after wait"),
        0
    );
    manager
        .write_pinned("host", 0, &vec![0_u8; len], tick.tick())
        .expect("pinned clear");
    let d2h = manager
        .copy_d2h("device", "host", len, tick.tick())
        .expect("D2H enqueue");
    manager
        .wait_for_transfer(&d2h, tick.tick())
        .expect("D2H completion");
    assert_eq!(
        manager
            .read_pinned("host", 0, len, tick.tick())
            .expect("pinned read"),
        expected
    );
    manager
        .release("device", tick.tick())
        .expect("device release");
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    println!("HW-C7-02 1000 polls, retired_total={retired_total}/8, state stayed coherent");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c7_03_transfer_and_eviction_alternation() {
    let _serial = serial();
    let mut manager = manager(1024 * 1024, 64 * 1024);
    let mut tick = Ticker::new();
    manager
        .allocate_pinned("host", 4096, tick.tick())
        .expect("pinned allocation");
    manager
        .write_pinned("host", 0, &pattern(4096, 31), tick.tick())
        .expect("pinned write");

    for round in 0..16_u32 {
        let temp_id = format!("temp-{round}");
        let next_id = format!("next-{round}");
        manager
            .allocate_device(&temp_id, 768 * 1024, tick.tick(), false)
            .expect("temp allocation");
        let token = manager
            .copy_h2d("host", &temp_id, 4096, tick.tick())
            .expect("H2D enqueue");
        manager
            .wait_for_transfer(&token, tick.tick())
            .expect("H2D completion");
        manager
            .allocate_device(&next_id, 512 * 1024, tick.tick(), false)
            .expect("admission with eviction");
        assert_eq!(manager.usage_bytes(MemoryTier::Vram), 512 * 1024);
        assert!(matches!(
            manager.release(&temp_id, tick.tick()),
            Err(CudaMemoryError::UnknownAllocation(id)) if id == temp_id
        ));
        manager
            .release(&next_id, tick.tick())
            .expect("next release");
    }
    assert_eq!(manager.usage_bytes(MemoryTier::Vram), 0);
    manager
        .release("host", tick.tick())
        .expect("pinned release");
    println!("HW-C7-03 16 transfer/eviction alternations verified");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m7_cuda_transfer_executor_runs_scheduler_tasks_after_event_completion() {
    let _serial = serial();
    let expected = pattern(1024 * 1024, 73);
    let budgets = budgets(64 * 1024 * 1024, 64 * 1024 * 1024);
    let executor = CudaTransferExecutor::new(0, budgets, EvictionPolicy::Lru)
        .expect("CUDA transfer executor");
    let mut executor = executor;
    let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();

    let allocate_host = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
    let allocate_device = scheduler.submit(TaskSpec::new(TaskKind::Transfer).depends_on(allocate_host)).unwrap();
    let write_host = scheduler.submit(TaskSpec::new(TaskKind::Transfer).depends_on(allocate_device)).unwrap();
    let h2d = scheduler.submit(TaskSpec::new(TaskKind::Transfer).depends_on(write_host)).unwrap();
    let d2h = scheduler.submit(TaskSpec::new(TaskKind::Transfer).depends_on(h2d)).unwrap();
    let read_host = scheduler.submit(TaskSpec::new(TaskKind::Transfer).depends_on(d2h)).unwrap();
    let release_device = scheduler.submit(TaskSpec::new(TaskKind::Transfer).depends_on(read_host)).unwrap();
    let release_host = scheduler.submit(TaskSpec::new(TaskKind::Transfer).depends_on(release_device)).unwrap();

    executor.register(allocate_host, CudaMemoryTask::AllocatePinned {
        id: "host".into(), bytes: expected.len() as u64,
    }).unwrap();
    executor.register(allocate_device, CudaMemoryTask::AllocateDevice {
        id: "device".into(), bytes: expected.len() as u64, expert: false,
    }).unwrap();
    executor.register(write_host, CudaMemoryTask::WritePinned {
        id: "host".into(), offset: 0, data: expected.clone(),
    }).unwrap();
    executor.register(h2d, CudaMemoryTask::CopyH2d {
        pinned_id: "host".into(), device_id: "device".into(), bytes: expected.len(),
    }).unwrap();
    executor.register(d2h, CudaMemoryTask::CopyD2h {
        device_id: "device".into(), pinned_id: "host".into(), bytes: expected.len(),
    }).unwrap();
    executor.register(read_host, CudaMemoryTask::ReadPinned {
        id: "host".into(), offset: 0, len: expected.len(),
    }).unwrap();
    executor.register(release_device, CudaMemoryTask::Release { id: "device".into() }).unwrap();
    executor.register(release_host, CudaMemoryTask::Release { id: "host".into() }).unwrap();

    let mut runner = QueueRunner::new(TaskKind::Transfer, executor);
    for task_id in [
        allocate_host, allocate_device, write_host, h2d, d2h, read_host,
        release_device, release_host,
    ] {
        assert_eq!(
            runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Completed(task_id)
        );
    }
    assert_eq!(runner.executor_mut().take_read_result(read_host), Some(expected));
    let metrics = runner.executor().metrics();
    assert_eq!(metrics.h2d_transfers, 1);
    assert_eq!(metrics.d2h_transfers, 1);
    assert_eq!(metrics.h2d_bytes, 1024 * 1024);
    assert_eq!(metrics.d2h_bytes, 1024 * 1024);
    assert_eq!(metrics.failed, 0);
    assert_eq!(runner.executor().pending_operations(), 0);
    println!(
        "M7 CUDA executor: H2D={} bytes D2H={} bytes event_wait={} ns",
        metrics.h2d_bytes, metrics.d2h_bytes, metrics.event_wait_ns
    );
}

