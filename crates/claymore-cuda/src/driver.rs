//! Thin, ownership-aware wrappers over the CUDA Driver API.
//!
//! Async copy methods are unsafe because the caller must keep host/device
//! buffers alive and avoid conflicting access until the associated stream has
//! completed. Every operation activates the owning CUDA context first.
use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr},
    marker::PhantomData,
    ptr,
    sync::Arc,
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
    fn cuCtxCreate_v2(context: *mut CuContext, flags: c_uint, device: CuDevice) -> CuResult;
    fn cuCtxSetCurrent(context: CuContext) -> CuResult;
    fn cuCtxDestroy_v2(context: CuContext) -> CuResult;
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
    usize::try_from(bytes).map_err(|_| CudaError::SizeOverflow)
}

struct ContextInner {
    raw: CuContext,
}

impl ContextInner {
    fn activate(&self) -> Result<(), CudaError> {
        // SAFETY: raw remains live while this context is held by an Arc.
        check(unsafe { cuCtxSetCurrent(self.raw) }, "cuCtxSetCurrent")
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // SAFETY: this context was created by cuCtxCreate_v2 and is released once.
        unsafe {
            let _ = cuCtxDestroy_v2(self.raw);
        }
    }
}

pub struct CudaDriver {
    context: Arc<ContextInner>,
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
        if ordinal >= count as u32 {
            return Err(CudaError::Driver {
                operation: "device ordinal out of range",
                code: 101,
            });
        }
        let mut device = 0;
        // SAFETY: device points to valid writable storage and ordinal is in range.
        check(
            unsafe { cuDeviceGet(&mut device, ordinal as c_int) },
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
        let mut raw = ptr::null_mut();
        // SAFETY: raw is valid output storage and device is valid.
        check(
            unsafe { cuCtxCreate_v2(&mut raw, 0, device) },
            "cuCtxCreate_v2",
        )?;
        let context = Arc::new(ContextInner { raw });
        context.activate()?;
        Ok(Self {
            context,
            info: CudaDeviceInfo {
                ordinal,
                name,
                total_memory_bytes: total as u64,
                compute_capability: None,
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
        self.context.activate()?;
        let len = as_size(bytes)?;
        let mut pointer = 0;
        // SAFETY: pointer is valid output storage; CUDA owns the resulting allocation.
        check(unsafe { cuMemAlloc_v2(&mut pointer, len) }, "cuMemAlloc_v2")?;
        Ok(DeviceBuffer {
            context: Arc::clone(&self.context),
            pointer,
            len,
        })
    }

    pub fn allocate_pinned(&self, bytes: u64) -> Result<PinnedHostBuffer, CudaError> {
        self.context.activate()?;
        let len = as_size(bytes)?;
        let mut pointer = ptr::null_mut();
        // SAFETY: pointer is valid output storage; CUDA allocates page-locked host memory.
        check(
            unsafe { cuMemHostAlloc(&mut pointer, len, CU_MEMHOSTALLOC_PORTABLE) },
            "cuMemHostAlloc",
        )?;
        Ok(PinnedHostBuffer {
            context: Arc::clone(&self.context),
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
            context: Arc::clone(&self.context),
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
            context: Arc::clone(&self.context),
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
    context: Arc<ContextInner>,
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
    pub fn free(mut self) -> Result<(), CudaError> {
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
    context: Arc<ContextInner>,
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
    pub fn free(mut self) -> Result<(), CudaError> {
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
    context: Arc<ContextInner>,
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
    context: Arc<ContextInner>,
    raw: CuEventRaw,
}
impl CudaEvent {
    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        self.context.activate()?;
        // SAFETY: both handles are live and belong to the same driver context.
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
