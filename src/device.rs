//! Functions and types for enumerating CUDA devices and retrieving information about them.

use crate::error::{CudaResult, ToResult};
use cuda_driver_sys::*;
use std::ffi::CStr;
use std::ops::Range;

/// All supported device attributes for [Device::get_attribute](struct.Device.html#method.get_attribute)
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceAttribute {
    /// Maximum number of threads per block
    MaxThreadsPerBlock = 1,
    /// Maximum x-dimension of a block
    MaxBlockDimX = 2,
    /// Maximum y-dimension of a block
    MaxBlockDimY = 3,
    /// Maximum z-dimension of a block
    MaxBlockDimZ = 4,
    /// Maximum x-dimension of a grid
    MaxGridDimX = 5,
    /// Maximum y-dimension of a grid
    MaxGridDimY = 6,
    /// Maximum z-dimension of a grid
    MaxGridDimZ = 7,
    /// Maximum amount of shared memory available to a thread block in bytes
    MaxSharedMemoryPerBlock = 8,
    /// Memory available on device for constant variables in a kernel in bytes
    TotalConstantMemory = 9,
    /// Warp size in threads
    WarpSize = 10,
    /// Maximum pitch in bytes allowed by the memory copy functions that involve memory regions
    /// allocated through cuMemAllocPitch()
    MaxPitch = 11,
    /// Maximum number of 32-bit registers available to a thread block
    MaxRegistersPerBlock = 12,
    /// Typical clock frequency in kilohertz
    ClockRate = 13,
    /// Alignment requirement for textures
    TextureAlignment = 14,
    //GpuOverlap = 15, - Deprecated.
    /// Number of multiprocessors on device.
    MultiprocessorCount = 16,
    /// Specifies whether there is a run time limit on kernels
    KernelExecTimeout = 17,
    /// Device is integrated with host memory
    Integrated = 18,
    /// Device can map host memory into CUDA address space
    CanMapHostMemory = 19,
    /// Compute Mode
    ComputeMode = 20,
    /// Maximum 1D texture width
    MaximumTexture1DWidth = 21,
    /// Maximum 2D texture width
    MaximumTexture2DWidth = 22,
    /// Maximum 2D texture height
    MaximumTexture2DHeight = 23,
    /// Maximum 3D texture width
    MaximumTexture3DWidth = 24,
    /// Maximum 3D texture height
    MaximumTexture3DHeight = 25,
    /// Maximum 3D texture depth
    MaximumTexture3DDepth = 26,
    /// Maximum 2D layered texture width
    MaximumTexture2DLayeredWidth = 27,
    /// Maximum 2D layered texture height
    MaximumTexture2DLayeredHeight = 28,
    /// Maximum layers in a 2D layered texture
    MaximumTexture2DLayeredLayers = 29,
    /// Alignment requirement for surfaces
    SurfaceAlignment = 30,
    /// Device can possibly execute multiple kernels concurrently
    ConcurrentKernels = 31,
    /// Device has ECC support enabled
    EccEnabled = 32,
    /// PCI bus ID of the device
    PciBusId = 33,
    /// PCI device ID of the device
    PciDeviceId = 34,
    /// Device is using TCC driver model
    TccDriver = 35,
    /// Peak memory clock frequency in kilohertz
    MemoryClockRate = 36,
    /// Global memory bus width in bits
    GlobalMemoryBusWidth = 37,
    /// Size of L2 cache in bytes.
    L2CacheSize = 38,
    /// Maximum resident threads per multiprocessor
    MaxThreadsPerMultiprocessor = 39,
    /// Number of asynchronous engines
    AsyncEngineCount = 40,
    /// Device shares a unified address space with the host
    UnifiedAddressing = 41,
    /// Maximum 1D layered texture width
    MaximumTexture1DLayeredWidth = 42,
    /// Maximum layers in a 1D layered texture
    MaximumTexture1DLayeredLayers = 43,
    //CanTex2DGather = 44, deprecated
    /// Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    MaximumTexture2DGatherWidth = 45,
    /// Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    MaximumTexture2DGatherHeight = 46,
    /// Alternate maximum 3D texture width
    MaximumTexture3DWidthAlternate = 47,
    /// Alternate maximum 3D texture height
    MaximumTexture3DHeightAlternate = 48,
    /// Alternate maximum 3D texture depth
    MaximumTexture3DDepthAlternate = 49,
    /// PCI domain ID of the device
    PciDomainId = 50,
    /// Pitch alignment requirement for textures
    TexturePitchAlignment = 51,
    /// Maximum cubemap texture width/height
    MaximumTextureCubemapWidth = 52,
    /// Maximum cubemap layered texture width/height
    MaximumTextureCubemapLayeredWidth = 53,
    /// Maximum layers in a cubemap layered texture
    MaximumTextureCubemapLayeredLayers = 54,
    /// Maximum 1D surface width
    MaximumSurface1DWidth = 55,
    /// Maximum 2D surface width
    MaximumSurface2DWidth = 56,
    /// Maximum 2D surface height
    MaximumSurface2DHeight = 57,
    /// Maximum 3D surface width
    MaximumSurface3DWidth = 58,
    /// Maximum 3D surface height
    MaximumSurface3DHeight = 59,
    /// Maximum 3D surface depth
    MaximumSurface3DDepth = 60,
    /// Maximum 1D layered surface width
    MaximumSurface1DLayeredWidth = 61,
    /// Maximum layers in a 1D layered surface
    MaximumSurface1DLayeredLayers = 62,
    /// Maximum 2D layered surface width
    MaximumSurface2DLayeredWidth = 63,
    /// Maximum 2D layered surface height
    MaximumSurface2DLayeredHeight = 64,
    /// Maximum layers in a 2D layered surface
    MaximumSurface2DLayeredLayers = 65,
    /// Maximum cubemap surface width
    MaximumSurfacecubemapWidth = 66,
    /// Maximum cubemap layered surface width
    MaximumSurfacecubemapLayeredWidth = 67,
    /// Maximum layers in a cubemap layered surface
    MaximumSurfacecubemapLayeredLayers = 68,
    /// Maximum 1D linear texture width
    MaximumTexture1DLinearWidth = 69,
    /// Maximum 2D linear texture width
    MaximumTexture2DLinearWidth = 70,
    /// Maximum 2D linear texture height
    MaximumTexture2DLinearHeight = 71,
    /// Maximum 2D linear texture pitch in bytes
    MaximumTexture2DLinearPitch = 72,
    /// Maximum mipmapped 2D texture height
    MaximumTexture2DMipmappedWidth = 73,
    /// Maximum mipmapped 2D texture width
    MaximumTexture2DMipmappedHeight = 74,
    /// Major compute capability version number
    ComputeCapabilityMajor = 75,
    /// Minor compute capability version number
    ComputeCapabilityMinor = 76,
    /// Maximum mipammed 1D texture width
    MaximumTexture1DMipmappedWidth = 77,
    /// Device supports stream priorities
    StreamPrioritiesSupported = 78,
    /// Device supports caching globals in L1
    GlobalL1CacheSupported = 79,
    /// Device supports caching locals in L1
    LocalL1CacheSupported = 80,
    /// Maximum shared memory available per multiprocessor in bytes
    MaxSharedMemoryPerMultiprocessor = 81,
    /// Maximum number of 32-bit registers available per multiprocessor
    MaxRegistersPerMultiprocessor = 82,
    /// Device can allocate managed memory on this system
    ManagedMemory = 83,
    /// Device is on a multi-GPU board
    MultiGpuBoard = 84,
    /// Unique ID for a group of devices on the same multi-GPU board
    MultiGpuBoardGroupId = 85,
    /// Link between the device and the host supports native atomic operations (this is a
    /// placeholder attribute and is not supported on any current hardware)
    HostNativeAtomicSupported = 86,
    /// Ratio of single precision performance (in floating-point operations per second) to double
    /// precision performance
    SingleToDoublePrecisionPerfRatio = 87,
    /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it.
    PageableMemoryAccess = 88,
    /// Device can coherently access managed memory concurrently with the CPU
    ConcurrentManagedAccess = 89,
    /// Device supports compute preemption
    ComputePreemptionSupported = 90,
    /// Device can access host registered memory at the same virtual address as the CPU
    CanUseHostPointerForRegisteredMem = 91,
    /// `cuStreamBatchMemOp` and related APIs are supported.
    CanUseStreamMemOps = 92,
    /// 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs.
    CanUse64BitStreamMemOps = 93,
    /// CU_STREAM_WAIT_VALUE_NOR is supported.
    CanUseStreamWaitValueNor = 94,
    /// Device supports launching cooperative kernels via `cuLaunchCooperativeKernel`
    CooperativeLaunch = 95,
    /// Device can participate in cooperative kernels launched via `cuLaunchCooperativeKernelMultiDevice`
    CooperativeMultiDeviceLaunch = 96,
    /// Maximum optin shared memory per block
    MaxSharedMemoryPerBlockOptin = 97,
    /// Both the `CU_STREAM_WAIT_VALUE_FLUSH` flag and the `CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` MemOp are supported on the device.
    CanFlushRemoteWrites = 98,
    /// Device supports using the `cuMemHostRegister` flag `CU_MEMHOSTREGISTER_READ_ONLY` to register memory that must be mapped as read-only to the GPU
    HostRegisterSupported = 99,
    /// Device accesses pageable memory via the host's page tables.
    PageableMemoryAccessUsesHostPageTables = 100,
    /// The host can directly access managed memory on the device without migration.
    DirectManagedMemAccessFromHost = 101,
    /// Device supports virtual memory management APIs
    VirtualMemoryManagementSupported = 102,
    /// Device supports exporting memory to a posix file descriptor with `cuMemExportToShareableHandle`, if requested via `cuMemCreate`
    HandleTypePosixFileDescriptorSupported = 103,
    /// Device supports exporting memory to a Win32 NT handle with `cuMemExportToShareableHandle`, if requested via `cuMemCreate`
    HandleTypeWin32HandleSupported = 104,
    /// Device supports exporting memory to a Win32 KMT handle with `cuMemExportToShareableHandle`, if requested `cuMemCreate`
    HandleTypeWin32KmtHandleSupported = 105,
    /// Maximum number of blocks per multiprocessor
    MaxBlocksPerMultiprocessor = 106,
    /// Device supports compression of memory
    GenericCompressionSupported = 107,
    /// Device's maximum L2 persisting lines capacity setting in bytes
    MaxPersistingL2CacheSize = 108,
    /// The maximum value of `CUaccessPolicyWindow::num_bytes`.
    MaxAccessPolicyWindowSize = 109,
    /// Device supports specifying the GPUDirect RDMA flag with `cuMemCreate`
    GpuDirectRdmaWithCudaVmnSupported = 110,
    /// Shared memory reserved by CUDA driver per block in bytes
    ReservedSharedMemoryPerBlock = 111,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
    SparseCudaArraySupported = 112,
    /// Device supports using the `cuMemHostRegister` flag `CU_MEMHOSTREGISTER_READ_ONLY` to register memory that must be mapped as read-only to the GPU
    ReadOnlyHostRegisterSupported = 113,
    /// External timeline semaphore interop is supported on the device
    TimelineSemaphoreInteropSupported = 114,
    /// Device supports using the ::cuMemAllocAsync and `cuMemPool` family of APIs
    MemoryPoolsSupported = 115,
}

/// P2P Attributes
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceP2PAttribute {
    /// A relative value indicating the performance of the link between two devices
    PerformanceRank = 1,
    /// P2P Access is enabled
    AccessSupported = 2,
    /// Atomic operation over the link supported
    NativeAtomicSupported = 3,
    /// Accessing CUDA arrays over the link supported
    CudaArrayAccessSupported = 4,
}

/// Opaque handle to a CUDA device.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct Device {
    pub(crate) device: CUdevice,
}

impl Device {
    /// Get the number of CUDA-capable devices.
    ///
    /// Returns the number of devices with compute-capability 2.0 or greater which are available
    /// for execution.
    ///
    /// # Example
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let num_devices = Device::num_devices()?;
    /// println!("Number of devices: {}", num_devices);
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_devices() -> CudaResult<u32> {
        unsafe {
            let mut num_devices = 0i32;
            cuDeviceGetCount(&mut num_devices as *mut i32).to_result()?;
            Ok(num_devices as u32)
        }
    }

    /// Get a handle to the `ordinal`'th CUDA device.
    ///
    /// Ordinal must be in the range `0..num_devices()`. If not, an error will be returned.
    ///
    /// # Example
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let device = Device::get_device(0)?;
    /// println!("Device Name: {}", device.name()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_device(ordinal: u32) -> CudaResult<Device> {
        unsafe {
            let mut device = Device { device: 0 };
            cuDeviceGet(&mut device.device as *mut CUdevice, ordinal as i32).to_result()?;
            Ok(device)
        }
    }

    /// Return an iterator over all CUDA devices.
    ///
    /// # Example
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// for device in Device::devices()? {
    ///     let device = device?;
    ///     println!("Device Name: {}", device.name()?);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn devices() -> CudaResult<Devices> {
        Device::num_devices().map(|num_devices| Devices {
            range: 0..num_devices,
        })
    }

    /// Returns the total amount of memory available on the device in bytes.
    ///
    /// # Example
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let device = Device::get_device(0)?;
    /// println!("Device Memory: {}", device.total_memory()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn total_memory(self) -> CudaResult<usize> {
        unsafe {
            let mut memory = 0;
            cuDeviceTotalMem_v2(&mut memory as *mut usize, self.device).to_result()?;
            Ok(memory)
        }
    }

    /// Returns the name of this device.
    ///
    /// # Example
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::Device;
    /// let device = Device::get_device(0)?;
    /// println!("Device Name: {}", device.name()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn name(self) -> CudaResult<String> {
        unsafe {
            let mut name = [0u8; 128]; // Hopefully this is big enough...
            cuDeviceGetName(
                &mut name[0] as *mut u8 as *mut ::std::os::raw::c_char,
                128,
                self.device,
            )
            .to_result()?;
            let nul_index = name
                .iter()
                .cloned()
                .position(|byte| byte == 0)
                .expect("Expected device name to fit in 128 bytes and be nul-terminated.");
            let cstr = CStr::from_bytes_with_nul_unchecked(&name[0..=nul_index]);
            Ok(cstr.to_string_lossy().into_owned())
        }
    }

    /// Returns information about this device.
    ///
    /// # Example
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # init(CudaFlags::empty())?;
    /// use rustacuda::device::{Device, DeviceAttribute};
    /// let device = Device::get_device(0)?;
    /// println!("Max Threads Per Block: {}",
    ///     device.get_attribute(DeviceAttribute::MaxThreadsPerBlock).unwrap());
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_attribute(self, attr: DeviceAttribute) -> CudaResult<i32> {
        unsafe {
            let mut val = 0i32;
            cuDeviceGetAttribute(
                &mut val as *mut i32,
                // This should be safe, as the repr and values of DeviceAttribute should match.
                ::std::mem::transmute(attr),
                self.device,
            )
            .to_result()?;
            Ok(val)
        }
    }

    pub(crate) fn into_inner(self) -> CUdevice {
        self.device
    }

    /// Queries if a device may directly access a peer device's memory.
    pub fn can_access_peer(self, peer: &Device) -> CudaResult<bool> {
        unsafe {
            let mut val = 0i32;
            cuDeviceCanAccessPeer(
                &mut val as *mut i32,
                self.device,
                peer.device
            )
            .to_result()?;
            Ok(val == 1)
        }
    }

    /// Queries attributes of the link between two devices.
    pub fn get_p2p_attribute(self, peer: &Device, attr: DeviceP2PAttribute) -> CudaResult<i32> {
        unsafe {
            let mut val = 0i32;
            cuDeviceGetP2PAttribute(
                &mut val as *mut i32,
                // This should be safe, as the repr and values of DeviceAttribute should match.
                ::std::mem::transmute(attr),
                self.device,
                peer.device
            )
            .to_result()?;
            Ok(val)
        }
    }
}

/// Iterator over all available CUDA devices. See
/// [the Device::devices function](./struct.Device.html#method.devices) for more information.
#[derive(Debug, Clone)]
pub struct Devices {
    range: Range<u32>,
}

impl Iterator for Devices {
    type Item = CudaResult<Device>;

    fn next(&mut self) -> Option<CudaResult<Device>> {
        self.range.next().map(Device::get_device)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::error::Error;

    fn test_init() -> Result<(), Box<dyn Error>> {
        crate::init(crate::CudaFlags::empty())?;
        Ok(())
    }

    #[test]
    fn test_num_devices() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let num_devices = Device::num_devices()?;
        assert!(num_devices > 0);
        Ok(())
    }

    #[test]
    fn test_devices() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let num_devices = Device::num_devices()?;
        let all_devices: CudaResult<Vec<_>> = Device::devices()?.collect();
        let all_devices = all_devices?;
        assert_eq!(num_devices as usize, all_devices.len());
        Ok(())
    }

    #[test]
    fn test_get_name() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let device_name = Device::get_device(0)?.name()?;
        println!("{}", device_name);
        assert!(device_name.len() < 127);
        Ok(())
    }

    #[test]
    fn test_get_memory() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let memory = Device::get_device(0)?.total_memory()?;
        println!("{}", memory);
        Ok(())
    }

    // Simple test that some enums are aligned. Not a comprehensive test for the enums.
    #[test]
    fn test_enums_align() {
        assert_eq!(
            DeviceAttribute::HandleTypeWin32KmtHandleSupported as u32,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED
                as u32
        );
    }
}
