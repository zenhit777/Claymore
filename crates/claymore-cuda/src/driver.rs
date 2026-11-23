//! Thin, ownership-aware wrappers over the CUDA Driver API.
//!
//! The backend retains the device primary context so Claymore components can
//! share CUDA resources without crossing incompatible context boundaries.
//! Async copy methods remain unsafe: callers must keep buffers alive and avoid
//! conflicting access until the associated stream has completed.
use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr},
    marker::PhantomData,
    ptr,
    rc::Rc,
};

use crate::{CudaDeviceInfo, CudaError, DeviceInventory};

type CuResult = c_int;
type CuDevice = c_int;
type CuDevicePtr = u64;
type CuContext = *mut c_void;
type CuStreamRaw = *mut c_void;
type CuEventRaw = *mut c_void;
pub(crate) type CuModuleRaw = *mut c_void;
pub(crate) type CuFunctionRaw = *mut c_void;

pub(crate) const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NOT_READY: CuResult = 600;
// Do not implicitly synchronize with CUDA's legacy default stream.
const CU_STREAM_NON_BLOCKING: c_uint = 1;
const CU_EVENT_DEFAULT: c_uint = 0;
const CU_MEMHOSTALLOC_PORTABLE: c_uint = 1;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CuResult;
    fn cuDeviceGetCount(count: *mut c_int) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> CuResult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, device: CuDevice) -> CuResult;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, device: CuDevice) -> CuResult;
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRetain(context: *mut CuContext, device: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRelease_v2(device: CuDevice) -> CuResult;
    fn cuCtxSetCurrent(context: CuContext) -> CuResult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CuResult;
    fn cuMemAlloc_v2(pointer: *mut CuDevicePtr, bytes: usize) -> CuResult;
    fn cuMemFree_v2(pointer: CuDevicePtr) -> CuResult;
    fn cuMemHostAlloc(pointer: *mut *mut c_void, bytes: usize, flags: c_uint) -> CuResult;
    fn cuMemFreeHost(pointer: *mut c_void) -> CuResult;
    fn cuMemcpyHtoDAsync_v2(
        dst: CuDevicePtr,
        src: *const c_void,
        bytes: usize,
        stream: CuStreamRaw,
    ) -> CuResult;
    fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: CuDevicePtr,
        bytes: usize,
        stream: CuStreamRaw,
    ) -> CuResult;
    fn cuStreamCreate(stream: *mut CuStreamRaw, flags: c_uint) -> CuResult;
    #[cfg(test)]
    fn cuStreamGetFlags(stream: CuStreamRaw, flags: *mut c_uint) -> CuResult;
    fn cuStreamSynchronize(stream: CuStreamRaw) -> CuResult;
    fn cuStreamDestroy_v2(stream: CuStreamRaw) -> CuResult;
    fn cuEventCreate(event: *mut CuEventRaw, flags: c_uint) -> CuResult;
    fn cuEventRecord(event: CuEventRaw, stream: CuStreamRaw) -> CuResult;
    fn cuEventQuery(event: CuEventRaw) -> CuResult;
    fn cuEventSynchronize(event: CuEventRaw) -> CuResult;
    fn cuEventDestroy_v2(event: CuEventRaw) -> CuResult;
    pub(crate) fn cuModuleLoadData(module: *mut CuModuleRaw, image: *const c_void) -> CuResult;
    pub(crate) fn cuModuleGetFunction(
        function: *mut CuFunctionRaw,
        module: CuModuleRaw,
        name: *const c_char,
    ) -> CuResult;
    pub(crate) fn cuModuleUnload(module: CuModuleRaw) -> CuResult;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuLaunchKernel(
        function: CuFunctionRaw,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_bytes: c_uint,
        stream: CuStreamRaw,
        kernel_params: *const *const c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;
}

pub(crate) fn check(code: CuResult, operation: &'static str) -> Result<(), CudaError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Driver { operation, code })
    }
}

fn as_size(bytes: u64) -> Result<usize, CudaError> {
    if bytes == 0 {
        return Err(CudaError::ZeroSizedAllocation);
    }
    usize::try_from(bytes).map_err(|_| CudaError::SizeOverflow)
}

fn event_query_result(result: CuResult) -> Result<bool, CudaError> {
    if result == CUDA_SUCCESS {
        Ok(true)
    } else if result == CUDA_ERROR_NOT_READY {
        Ok(false)
    } else {
        Err(CudaError::Driver {
            operation: "cuEventQuery",
            code: result,
        })
    }
}

pub(crate) fn ensure_same_context(left: CuContext, right: CuContext) -> Result<(), CudaError> {
    if left == right {
        Ok(())
    } else {
        Err(CudaError::ContextMismatch)
    }
}

pub(crate) struct ContextInner {
    pub(crate) raw: CuContext,
    device: CuDevice,
}

impl ContextInner {
    pub(crate) fn activate(&self) -> Result<(), CudaError> {
        // SAFETY: raw is retained for the lifetime of this shared owner.
        check(unsafe { cuCtxSetCurrent(self.raw) }, "cuCtxSetCurrent")
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // SAFETY: this owner balances exactly one successful primary-context retain.
        unsafe {
            let _ = cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}

pub struct CudaDriver {
    context: Rc<ContextInner>,
    info: CudaDeviceInfo,
}
pub type CudaRuntime = CudaDriver;

impl CudaDriver {
    pub fn new(ordinal: u32) -> Result<Self, CudaError> {
        // SAFETY: CUDA driver initialization has no pointer preconditions.
        check(unsafe { cuInit(0) }, "cuInit")?;
        let mut count = 0;
        // SAFETY: count points to valid writable storage.
        check(unsafe { cuDeviceGetCount(&mut count) }, "cuDeviceGetCount")?;
        let ordinal_i32 =
            c_int::try_from(ordinal).map_err(|_| CudaError::InvalidDeviceOrdinal(ordinal))?;
        if ordinal_i32 < 0 || ordinal_i32 >= count {
            return Err(CudaError::InvalidDeviceOrdinal(ordinal));
        }

        let mut device = 0;
        // SAFETY: device points to valid writable storage and ordinal is in range.
        check(
            unsafe { cuDeviceGet(&mut device, ordinal_i32) },
            "cuDeviceGet",
        )?;

        let mut name = [0_i8; 256];
        // SAFETY: name is a writable buffer and its length is provided.
        check(
            unsafe { cuDeviceGetName(name.as_mut_ptr(), name.len() as c_int, device) },
            "cuDeviceGetName",
        )?;
        // SAFETY: CUDA returns a NUL-terminated device name in this buffer.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_str()
            .map_err(|_| CudaError::InvalidDeviceName)?
            .to_owned();

        let mut total = 0_usize;
        // SAFETY: total points to valid writable storage.
        check(
            unsafe { cuDeviceTotalMem_v2(&mut total, device) },
            "cuDeviceTotalMem_v2",
        )?;

        let (mut major, mut minor) = (0, 0);
        // SAFETY: outputs are valid and device was obtained from the driver.
        check(
            unsafe {
                cuDeviceGetAttribute(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                )
            },
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)",
        )?;
        check(
            unsafe {
                cuDeviceGetAttribute(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                )
            },
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)",
        )?;

        let mut raw = ptr::null_mut();
        // Retaining the primary context makes this backend interoperable with
        // CUDA consumers that also use the device primary context.
        // SAFETY: raw is valid output storage and device is valid.
        check(
            unsafe { cuDevicePrimaryCtxRetain(&mut raw, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        let context = Rc::new(ContextInner { raw, device });
        context.activate()?;

        Ok(Self {
            context,
            info: CudaDeviceInfo {
                ordinal,
                name,
                total_memory_bytes: total as u64,
                compute_capability: Some((major as u32, minor as u32)),
            },
        })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.info
    }

    /// Returns whether two driver handles retain the same CUDA context.
    /// Drivers created for the same device primary context should return true.
    pub fn shares_context_with(&self, other: &Self) -> bool {
        self.context.raw == other.context.raw
    }

    pub fn memory_info(&self) -> Result<(u64, u64), CudaError> {
        self.context.activate()?;
        let (mut free, mut total) = (0_usize, 0_usize);
        // SAFETY: output pointers are valid and the context is current.
        check(
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) },
            "cuMemGetInfo_v2",
        )?;
        Ok((free as u64, total as u64))
    }

    pub fn allocate(&self, bytes: u64) -> Result<CudaDeviceBuffer, CudaError> {
        let len = as_size(bytes)?;
        self.context.activate()?;
        let mut pointer = 0;
        // SAFETY: pointer is valid output storage; CUDA owns the resulting allocation.
        check(unsafe { cuMemAlloc_v2(&mut pointer, len) }, "cuMemAlloc_v2")?;
        Ok(CudaDeviceBuffer {
            context: Rc::clone(&self.context),
            pointer,
            len,
        })
    }

    pub fn allocate_pinned(&self, bytes: u64) -> Result<PinnedHostBuffer, CudaError> {
        let len = as_size(bytes)?;
        self.context.activate()?;
        let mut pointer = ptr::null_mut();
        // SAFETY: pointer is valid output storage; CUDA allocates page-locked host memory.
        check(
            unsafe { cuMemHostAlloc(&mut pointer, len, CU_MEMHOSTALLOC_PORTABLE) },
            "cuMemHostAlloc",
        )?;
        Ok(PinnedHostBuffer {
            context: Rc::clone(&self.context),
            pointer,
            len,
            _not_sync: PhantomData,
        })
    }

    pub fn create_stream(&self) -> Result<CudaStream, CudaError> {
        self.context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: raw is valid output storage.
        check(
            unsafe { cuStreamCreate(&mut raw, CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(CudaStream {
            context: Rc::clone(&self.context),
            raw,
        })
    }

    /// Create a compute engine bound to this driver's primary context. The
    /// embedded PTX is loaded eagerly so compilation/loading failures are
    /// reported before any tensor work is scheduled.
    pub fn create_compute_engine(&self) -> Result<crate::compute::CudaComputeEngine, CudaError> {
        crate::compute::CudaComputeEngine::new(Rc::clone(&self.context))
    }

    pub fn create_event(&self) -> Result<CudaEvent, CudaError> {
        self.context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: raw is valid output storage.
        check(
            unsafe { cuEventCreate(&mut raw, CU_EVENT_DEFAULT) },
            "cuEventCreate",
        )?;
        Ok(CudaEvent {
            context: Rc::clone(&self.context),
            raw,
        })
    }
}

impl DeviceInventory for CudaDriver {
    fn devices(&self) -> Result<Vec<CudaDeviceInfo>, CudaError> {
        Ok(vec![self.info.clone()])
    }
}

pub struct CudaDeviceBuffer {
    pub(crate) context: Rc<ContextInner>,
    pub(crate) pointer: CuDevicePtr,
    len: usize,
}
impl CudaDeviceBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    /// Opaque CUDA device address. It is valid only while this buffer is live
    /// and must never be dereferenced from host code. Kernel launch helpers use
    /// it together with an explicit context/stream check.
    pub fn device_ptr(&self) -> u64 {
        self.pointer
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Explicitly release the allocation and report driver failures to the caller.
    ///
    /// # Safety
    /// No in-flight CUDA operation may access this allocation.
    pub unsafe fn free(&mut self) -> Result<(), CudaError> {
        if self.pointer == 0 {
            return Ok(());
        }
        self.context.activate()?;
        // SAFETY: this object uniquely owns the live device allocation.
        check(unsafe { cuMemFree_v2(self.pointer) }, "cuMemFree_v2")?;
        self.pointer = 0;
        Ok(())
    }

    /// Enqueue pinned-host-to-device transfer.
    ///
    /// # Safety
    /// Keep both buffers alive and prevent conflicting access until stream completion.
    pub unsafe fn copy_from_pinned_async(
        &mut self,
        source: &PinnedHostBuffer,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        ensure_same_context(self.context.raw, source.context.raw)?;
        ensure_same_context(self.context.raw, stream.context.raw)?;
        if bytes > self.len {
            return Err(CudaError::BufferTooSmall {
                requested: bytes,
                capacity: self.len,
            });
        }
        if bytes > source.len {
            return Err(CudaError::BufferTooSmall {
                requested: bytes,
                capacity: source.len,
            });
        }
        self.context.activate()?;
        // SAFETY: caller upholds asynchronous buffer lifetime and access requirements.
        check(
            unsafe { cuMemcpyHtoDAsync_v2(self.pointer, source.pointer, bytes, stream.raw) },
            "cuMemcpyHtoDAsync_v2",
        )
    }

    /// Enqueue device-to-pinned-host transfer.
    ///
    /// # Safety
    /// Do not read/free the host destination or mutate/free the device source until completion.
    pub unsafe fn copy_to_pinned_async(
        &self,
        destination: &mut PinnedHostBuffer,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        ensure_same_context(self.context.raw, destination.context.raw)?;
        ensure_same_context(self.context.raw, stream.context.raw)?;
        if bytes > self.len {
            return Err(CudaError::BufferTooSmall {
                requested: bytes,
                capacity: self.len,
            });
        }
        if bytes > destination.len {
            return Err(CudaError::BufferTooSmall {
                requested: bytes,
                capacity: destination.len,
            });
        }
        self.context.activate()?;
        // SAFETY: caller upholds asynchronous buffer lifetime and access requirements.
        check(
            unsafe { cuMemcpyDtoHAsync_v2(destination.pointer, self.pointer, bytes, stream.raw) },
            "cuMemcpyDtoHAsync_v2",
        )
    }
}
impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.pointer != 0 && self.context.activate().is_ok() {
            // SAFETY: pointer is a live allocation owned by this object.
            unsafe {
                let _ = cuMemFree_v2(self.pointer);
            }
        }
    }
}

pub struct PinnedHostBuffer {
    context: Rc<ContextInner>,
    pointer: *mut c_void,
    len: usize,
    _not_sync: PhantomData<*mut ()>,
}
impl PinnedHostBuffer {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Access pinned storage.
    ///
    /// # Safety
    /// No in-flight CUDA operation may access this buffer.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.cast()
    }

    /// Explicitly release pinned memory and report driver failures.
    ///
    /// # Safety
    /// No in-flight CUDA operation may access this allocation.
    pub unsafe fn free(&mut self) -> Result<(), CudaError> {
        if self.pointer.is_null() {
            return Ok(());
        }
        self.context.activate()?;
        // SAFETY: this object uniquely owns the live pinned allocation.
        check(unsafe { cuMemFreeHost(self.pointer) }, "cuMemFreeHost")?;
        self.pointer = ptr::null_mut();
        Ok(())
    }
}
impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.pointer.is_null() && self.context.activate().is_ok() {
            // SAFETY: pointer is a live pinned allocation owned by this object.
            unsafe {
                let _ = cuMemFreeHost(self.pointer);
            }
        }
    }
}

pub struct CudaStream {
    pub(crate) context: Rc<ContextInner>,
    pub(crate) raw: CuStreamRaw,
}
impl CudaStream {
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.activate()?;
        // SAFETY: raw is a live stream owned by this object.
        check(
            unsafe { cuStreamSynchronize(self.raw) },
            "cuStreamSynchronize",
        )
    }
}
impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.context.activate().is_ok() {
            // SAFETY: raw is a live stream owned by this object.
            unsafe {
                let _ = cuStreamDestroy_v2(self.raw);
            }
        }
    }
}

pub struct CudaEvent {
    context: Rc<ContextInner>,
    raw: CuEventRaw,
}
impl CudaEvent {
    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        ensure_same_context(self.context.raw, stream.context.raw)?;
        self.context.activate()?;
        // SAFETY: both handles are live and belong to the same CUDA context.
        check(
            unsafe { cuEventRecord(self.raw, stream.raw) },
            "cuEventRecord",
        )
    }
    /// Query the event without blocking. CUDA_ERROR_NOT_READY is a normal
    /// incomplete state; every other driver error is propagated.
    pub fn is_complete(&self) -> Result<bool, CudaError> {
        self.context.activate()?;
        // SAFETY: raw is a live event owned by this object.
        let result = unsafe { cuEventQuery(self.raw) };
        event_query_result(result)
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.activate()?;
        // SAFETY: raw is a live event owned by this object.
        check(
            unsafe { cuEventSynchronize(self.raw) },
            "cuEventSynchronize",
        )
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        if self.context.activate().is_ok() {
            // SAFETY: raw is a live event owned by this object.
            unsafe {
                let _ = cuEventDestroy_v2(self.raw);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resources_must_share_the_same_context() {
        let context = 1_usize as CuContext;
        assert!(ensure_same_context(context, context).is_ok());
        assert!(matches!(
            ensure_same_context(context, 2_usize as CuContext),
            Err(CudaError::ContextMismatch)
        ));
    }

    #[test]
    fn event_query_status_conversion_distinguishes_complete_pending_and_driver_failure() {
        assert!(matches!(event_query_result(CUDA_SUCCESS), Ok(true)));
        assert!(matches!(
            event_query_result(CUDA_ERROR_NOT_READY),
            Ok(false)
        ));
        assert!(matches!(
            event_query_result(999),
            Err(CudaError::Driver {
                operation: "cuEventQuery",
                code: 999
            })
        ));
    }

    #[test]
    fn driver_error_conversion_preserves_operation_and_code() {
        assert!(matches!(
            check(999, "cuStreamSynchronize"),
            Err(CudaError::Driver {
                operation: "cuStreamSynchronize",
                code: 999
            })
        ));
    }

    #[test]
    fn zero_sized_allocations_are_rejected_before_driver_calls() {
        assert!(matches!(as_size(0), Err(CudaError::ZeroSizedAllocation)));
        assert_eq!(as_size(1).unwrap(), 1);
    }

    #[test]
    #[ignore = "requires a CUDA driver and a usable NVIDIA device"]
    fn hardware_streams_are_explicitly_non_blocking() {
        let driver = CudaDriver::new(0).expect("CUDA driver and device 0");
        let stream = driver.create_stream().expect("stream creation");
        let mut flags = 0_u32;
        check(
            unsafe { cuStreamGetFlags(stream.raw, &mut flags) },
            "cuStreamGetFlags",
        )
        .expect("driver-reported stream flags");
        assert_eq!(
            flags & CU_STREAM_NON_BLOCKING,
            CU_STREAM_NON_BLOCKING,
            "transfer stream must not synchronize with the legacy default stream"
        );

        // Creation and destruction are balanced; the empty stream is quiescent.
        for _ in 0..32 {
            let extra = driver.create_stream().expect("stream creation");
            let mut extra_flags = 0_u32;
            check(
                unsafe { cuStreamGetFlags(extra.raw, &mut extra_flags) },
                "cuStreamGetFlags",
            )
            .expect("driver-reported stream flags");
            assert_eq!(extra_flags & CU_STREAM_NON_BLOCKING, CU_STREAM_NON_BLOCKING);
            extra.synchronize().expect("empty stream synchronization");
        }
        stream.synchronize().expect("empty stream synchronization");
    }

    #[test]
    #[ignore = "requires a CUDA driver and a usable NVIDIA device"]
    fn hardware_stream_synchronize_completes_async_copy() {
        let driver = CudaDriver::new(0).expect("CUDA driver and device 0");
        let stream = driver.create_stream().expect("stream creation");
        let len = 64 * 1024 * 1024;
        let mut source = driver
            .allocate_pinned(len as u64)
            .expect("pinned allocation");
        let mut destination = driver.allocate(len as u64).expect("device allocation");
        // SAFETY: the pinned range is exclusively owned by this test.
        unsafe { std::ptr::write_bytes(source.as_mut_ptr(), 0x5a, len) };
        // SAFETY: both buffers stay alive and are not accessed until completion.
        unsafe {
            destination
                .copy_from_pinned_async(&source, len, &stream)
                .expect("H2D enqueue");
        }
        let event = driver.create_event().expect("event creation");
        event.record(&stream).expect("event record");
        stream
            .synchronize()
            .expect("explicit stream synchronization");
        assert!(event.is_complete().expect("event query"));
        // SAFETY: the stream is synchronized and each owner frees once.
        unsafe {
            destination.free().expect("device free");
            source.free().expect("pinned free");
        }
    }

    #[test]
    #[ignore = "requires a CUDA driver and a usable NVIDIA device"]
    fn hardware_device_buffer_free_is_explicit_and_idempotent() {
        let driver = CudaDriver::new(0).expect("CUDA driver and device 0");
        let stream = driver.create_stream().expect("stream creation");
        for len in [1_usize, 37, 4096, 1024 * 1024] {
            let mut buffer = driver.allocate(len as u64).expect("device allocation");
            assert_eq!(buffer.len(), len);
            assert!(!buffer.is_empty());
            // SAFETY: this test uniquely owns the allocation and no work is queued.
            unsafe {
                buffer.free().expect("first free succeeds");
                buffer.free().expect("free is idempotent");
            }
        }

        let mut source = driver.allocate_pinned(4096).expect("pinned allocation");
        let mut destination = driver.allocate(4096).expect("device allocation");
        // SAFETY: no work is queued on this allocation.
        unsafe { destination.free().expect("device free") };
        // SAFETY: this is a deliberately invalid CUDA API usage probe, limited
        // to verifying that the wrapper's nulled pointer fails closed after free.
        // Do not generalize this pattern or use a freed allocation in production.
        let error = unsafe {
            destination
                .copy_from_pinned_async(&source, 4096, &stream)
                .expect_err("a freed allocation must not accept an async copy")
        };
        assert!(matches!(
            error,
            CudaError::Driver {
                operation: "cuMemcpyHtoDAsync_v2",
                code
            } if code != 0
        ));
        // SAFETY: the pinned owner is freed exactly once.
        unsafe { source.free().expect("pinned free") };
    }
}
