/*
 * Link-only CUDA Driver stub for CPU fault-injection tests.
 *
 * Every entry point returns CUDA_ERROR_NOT_INITIALIZED. The test suite uses
 * injected event/stream/buffer doubles and must never execute real CUDA calls.
 * Hardware tests remain ignored unless run on a machine with the real driver.
 */
#include <stddef.h>
#include <stdint.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef uint64_t CUdeviceptr;
typedef void *CUstream;
typedef void *CUevent;

#define CUDA_ERROR_NOT_INITIALIZED 3

CUresult cuInit(unsigned int flags) { (void)flags; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuDeviceGetCount(int *count) { (void)count; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuDeviceGet(CUdevice *device, int ordinal) { (void)device; (void)ordinal; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuDeviceGetName(char *name, int len, CUdevice device) { (void)name; (void)len; (void)device; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice device) { (void)bytes; (void)device; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuDeviceGetAttribute(int *value, int attribute, CUdevice device) { (void)value; (void)attribute; (void)device; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device) { (void)context; (void)device; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice device) { (void)device; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuCtxSetCurrent(CUcontext context) { (void)context; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes) { (void)free_bytes; (void)total_bytes; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemAlloc_v2(CUdeviceptr *pointer, size_t bytes) { (void)pointer; (void)bytes; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemFree_v2(CUdeviceptr pointer) { (void)pointer; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemHostAlloc(void **pointer, size_t bytes, unsigned int flags) { (void)pointer; (void)bytes; (void)flags; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemFreeHost(void *pointer) { (void)pointer; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemsetD8Async(CUdeviceptr destination, unsigned char value, size_t bytes, CUstream stream) { (void)destination; (void)value; (void)bytes; (void)stream; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr destination, const void *source, size_t bytes, CUstream stream) { (void)destination; (void)source; (void)bytes; (void)stream; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuMemcpyDtoHAsync_v2(void *destination, CUdeviceptr source, size_t bytes, CUstream stream) { (void)destination; (void)source; (void)bytes; (void)stream; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuStreamCreate(CUstream *stream, unsigned int flags) { (void)stream; (void)flags; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuStreamGetFlags(CUstream stream, unsigned int *flags) { (void)stream; (void)flags; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuStreamSynchronize(CUstream stream) { (void)stream; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuStreamDestroy_v2(CUstream stream) { (void)stream; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuEventCreate(CUevent *event, unsigned int flags) { (void)event; (void)flags; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuEventRecord(CUevent event, CUstream stream) { (void)event; (void)stream; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuEventQuery(CUevent event) { (void)event; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuEventSynchronize(CUevent event) { (void)event; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuEventDestroy_v2(CUevent event) { (void)event; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuModuleLoadData(void **module, const void *image) { (void)module; (void)image; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuModuleGetFunction(void **function, void *module, const char *name) { (void)function; (void)module; (void)name; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuModuleUnload(void *module) { (void)module; return CUDA_ERROR_NOT_INITIALIZED; }
CUresult cuLaunchKernel(void *function, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared, void *stream, void **params, void **extra) { (void)function; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shared; (void)stream; (void)params; (void)extra; return CUDA_ERROR_NOT_INITIALIZED; }
