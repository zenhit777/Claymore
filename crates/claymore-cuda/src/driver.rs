//! Thin, ownership-aware wrappers over the CUDA Driver API.
//!
//! The backend retains the device primary context so resources can interoperate
//! with CUDA libraries (including Candle/cudarc) that use the same context.
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

const CUDA_SUCCESS: CuResult = 0;
const CU_STREAM_DEFAULT: c_uint = 0;
const CU_EVENT_DEFAULT: c_uint = 0;
const CU_MEMHOSTALLOC_PORTABLE: c_uint = 1;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CuResult;
    fn cuDeviceGetCount(count: *mut c_int) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> CuResult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, device: CuDevice) -> CuResult;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, device: CuDevice) -> CuResult;
    fn cuDeviceComputeCapability(
        major: *mut c_int,
        minor: *mut c_int,
        device: CuDevice,
    ) -> CuResult;
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
    fn cuStreamSynchronize(stream: CuStreamRaw) -> CuResult;
    fn cuStreamDestroy_v2(stream: CuStreamRaw) -> CuResult;
    fn cuEventCreate(event: *mut CuEventRaw, flags: c_uint) -> CuResult;
    fn cuEventRecord(event: CuEventRaw, stream: CuStreamRaw) -> CuResult;
    fn cuEventSynchronize(event: CuEventRaw) -> CuResult;
    fn cuEventDestroy_v2(event: CuEventRaw) -> CuResult;
}

fn check(code: CuResult, operation: &'static str) -> Result<(), CudaError> {
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

fn ensure_same_context(left: CuContext, right: CuContext) -> Result<(), CudaError> {
    if left == right {
        Ok(())
    } else {
        Err(CudaError::ContextMismatch)
    }
}

struct ContextInner {
    raw: CuContext,
    device: CuDevice,
}

impl ContextInner {
    fn activate(&self) -> Result<(), CudaError> {
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
        // SAFETY: both outputs are valid and device was obtained from the driver.
        check(
            unsafe { cuDeviceComputeCapability(&mut major, &mut minor, device) },
            "cuDeviceComputeCapability",
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

    pub fn allocate(&self, bytes: u64) -> Result<DeviceBuffer, CudaError> {
        let len = as_size(bytes)?;
        self.context.activate()?;
        let mut pointer = 0;
        // SAFETY: pointer is valid output storage; CUDA owns the resulting allocation.
        check(unsafe { cuMemAlloc_v2(&mut pointer, len) }, "cuMemAlloc_v2")?;
        Ok(DeviceBuffer {
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
            unsafe { cuStreamCreate(&mut raw, CU_STREAM_DEFAULT) },
            "cuStreamCreate",
        )?;
        Ok(CudaStream {
            context: Rc::clone(&self.context),
            raw,
        })
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

pub struct DeviceBuffer {
    context: Rc<ContextInner>,
    pointer: CuDevicePtr,
    len: usize,
}
impl DeviceBuffer {
    pub fn len(&self) -> usize {
        self.len
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
impl Drop for DeviceBuffer {
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
    context: Rc<ContextInner>,
    raw: CuStreamRaw,
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
    fn zero_sized_allocations_are_rejected_before_driver_calls() {
        assert!(matches!(as_size(0), Err(CudaError::ZeroSizedAllocation)));
        assert_eq!(as_size(1).unwrap(), 1);
    }
}
