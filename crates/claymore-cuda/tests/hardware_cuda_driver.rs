//! Hardware validation of the public `claymore-cuda` wrapper contracts.
//!
//! These tests require a real NVIDIA driver and device. They are ignored by
//! default so the standard CI remains hardware-independent. Run them with:
//!
//! ```text
//! RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib \
//! cargo test -p claymore-cuda --features cuda-driver -- --ignored --test-threads=1
//! ```

use std::sync::{Mutex, MutexGuard};

use claymore_cuda::{CudaDriver, CudaError, DeviceInventory};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c1_01_driver_initializes_primary_context() {
    let _serial = serial();
    let driver = CudaDriver::new(0).expect("CUDA driver and device 0");
    let info = driver.device_info();
    assert_eq!(info.ordinal, 0);
    assert!(!info.name.is_empty(), "device name must be reported");
    assert!(
        info.total_memory_bytes > 0,
        "device memory must be reported"
    );
    let (major, minor) = info
        .compute_capability
        .expect("compute capability must be reported");

    let (free, total) = driver.memory_info().expect("cuMemGetInfo");
    assert!(free <= total);
    assert!(total > 0);
    assert_eq!(driver.devices().expect("inventory").len(), 1);

    let peer = CudaDriver::new(0).expect("second driver handle");
    assert!(
        driver.shares_context_with(&peer),
        "driver handles must retain the same primary context"
    );
    assert!(matches!(
        CudaDriver::new(99),
        Err(CudaError::InvalidDeviceOrdinal(99))
    ));
    println!(
        "HW-C1-01 device={} cc={}.{} total={} free={}",
        info.name, major, minor, total, free
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c1_02_stream_and_event_lifecycle() {
    let _serial = serial();
    let driver = CudaDriver::new(0).expect("CUDA driver and device 0");
    for _ in 0..64 {
        let stream = driver.create_stream().expect("stream creation");
        stream.synchronize().expect("empty stream synchronization");
        let event = driver.create_event().expect("event creation");
        event.record(&stream).expect("event record");
        stream.synchronize().expect("stream synchronization");
        assert!(event.is_complete().expect("event query"));
        event.synchronize().expect("event synchronization");
    }
    println!("HW-C1-02 created/destroyed 64 streams and 64 events");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c1_03_context_interop_and_async_round_trip() {
    let _serial = serial();
    let owner = CudaDriver::new(0).expect("owner driver");
    let peer = CudaDriver::new(0).expect("peer driver");
    assert!(owner.shares_context_with(&peer));

    let len = 1024 * 1024;
    let expected = pattern(len, 7);
    let mut host = owner
        .allocate_pinned(len as u64)
        .expect("pinned allocation");
    let mut device = owner.allocate(len as u64).expect("device allocation");
    let stream = peer.create_stream().expect("stream from peer context");

    // SAFETY: the pinned range is exclusively owned by this test; the source
    // buffer stays alive and immutable until the stream is synchronized.
    unsafe {
        std::ptr::copy_nonoverlapping(expected.as_ptr(), host.as_mut_ptr(), len);
        device
            .copy_from_pinned_async(&host, len, &stream)
            .expect("H2D enqueue");
    }
    let h2d = owner.create_event().expect("H2D event");
    h2d.record(&stream).expect("H2D event record");
    h2d.synchronize().expect("H2D event wait");

    // SAFETY: the destination is exclusively owned and the stream is idle.
    unsafe {
        std::ptr::write_bytes(host.as_mut_ptr(), 0, len);
        device
            .copy_to_pinned_async(&mut host, len, &stream)
            .expect("D2H enqueue");
    }
    let d2h = owner.create_event().expect("D2H event");
    d2h.record(&stream).expect("D2H event record");
    d2h.synchronize().expect("D2H event wait");
    stream.synchronize().expect("stream synchronization");

    // SAFETY: no work is in flight and the pinned range is initialized.
    let copied = unsafe { std::slice::from_raw_parts(host.as_mut_ptr(), len) };
    assert_eq!(copied, expected.as_slice());

    // SAFETY: each allocation is freed exactly once.
    unsafe {
        device.free().expect("device free");
        host.free().expect("pinned free");
    }
    println!("HW-C1-03 verified {len} bytes across two driver handles");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c2_01_allocation_sizes_and_explicit_free() {
    let _serial = serial();
    let driver = CudaDriver::new(0).expect("CUDA driver and device 0");
    for len in [1_usize, 37, 4096, 1024 * 1024, 64 * 1024 * 1024] {
        let mut buffer = driver.allocate(len as u64).expect("device allocation");
        assert_eq!(buffer.len(), len);
        assert!(!buffer.is_empty());
        // SAFETY: the test owns the allocation and no work is queued.
        unsafe {
            buffer.free().expect("explicit free");
            buffer.free().expect("double free is a no-op");
        }
    }

    let mut pinned = driver.allocate_pinned(4096).expect("pinned allocation");
    assert_eq!(pinned.len(), 4096);
    // SAFETY: the test owns the allocation and no work is queued.
    unsafe {
        pinned.free().expect("explicit pinned free");
        pinned.free().expect("double pinned free is a no-op");
    }
    println!("HW-C2-01 freed 1 B .. 64 MiB device buffers and 4 KiB pinned");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_c2_02_zero_size_and_bounds_are_rejected() {
    let _serial = serial();
    let driver = CudaDriver::new(0).expect("CUDA driver and device 0");
    assert!(matches!(
        driver.allocate(0),
        Err(CudaError::ZeroSizedAllocation)
    ));
    assert!(matches!(
        driver.allocate_pinned(0),
        Err(CudaError::ZeroSizedAllocation)
    ));
    // On a 64-bit target u64 -> usize cannot overflow, so an absurd request is
    // rejected by the CUDA driver rather than by the SizeOverflow guard.
    let huge = match driver.allocate(u64::MAX) {
        Ok(_) => panic!("u64::MAX allocation unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(
        huge,
        CudaError::Driver {
            operation: "cuMemAlloc_v2",
            code
        } if code != 0
    ));

    let stream = driver.create_stream().expect("stream creation");
    let mut host = driver.allocate_pinned(4096).expect("pinned allocation");
    let mut device = driver.allocate(8192).expect("device allocation");
    // SAFETY: no work is queued; the range check must reject before any copy.
    let h2d = unsafe { device.copy_from_pinned_async(&host, 8193, &stream) };
    assert!(matches!(
        h2d,
        Err(CudaError::BufferTooSmall {
            requested: 8193,
            capacity: 8192
        })
    ));
    // SAFETY: no work is queued; the range check must reject before any copy.
    let d2h = unsafe { device.copy_to_pinned_async(&mut host, 4097, &stream) };
    assert!(matches!(
        d2h,
        Err(CudaError::BufferTooSmall {
            requested: 4097,
            capacity: 4096
        })
    ));
    // SAFETY: each allocation is freed exactly once.
    unsafe {
        device.free().expect("device free");
        host.free().expect("pinned free");
    }
    println!("HW-C2-02 rejected zero, overflowing and out-of-bounds requests");
}
