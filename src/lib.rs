//! This crate provides a safe, user-friendly wrapper around the CUDA Driver API.
//!
//! # CUDA Terminology
//!
//! ## Devices and Hosts
//!
//! This crate and its documentation uses the terms "device" and "host" frequently, so it's worth
//! explaining them in more detail.
//!
//! **Device** refers to a CUDA-capable GPU or similar device and its associated external memory space.
//!
//! **Host** is the CPU and its associated memory space. Data must be transferred from host memory
//! to device memory before the device can use the data for computations, and the results must then
//! be transferred back to host memory in order to use the results on the CPU, such as in your Rust
//! code.
//!
//! ## Contexts, Modules, Streams and Functions
//!
//! **Context** is akin to a process on the host — it contains all of the state for working
//! with a device, all memory allocations, etc. Each context is associated with a single device.
//!
//! **Module** is similar to a shared-object library — it is a piece of compiled code which exports
//! functions and global values. Functions can be loaded from modules and launched on a device as
//! one might load a function from a shared-object file and call it. Functions are also known as
//! kernels, both of which are short terms for global kernel function, and these terms will be used
//! interchangeably.
//!
//! **Stream** is akin to a thread — asynchronous work such as kernel execution can be queued into a
//! stream. Work within a single stream will execute sequentially in the order that it was
//! submitted, and may interleave with work from other streams.
//!
//! ## Grids, Blocks, Threads and Shared Memory
//!
//! CUDA devices contain multiple separate processors, known as multiprocessors or MPs. Each
//! multiprocessor is capable of executing large number of **threads** simultaneously, grouped
//! into thread blocks, or **blocks** in short. Each thread block executes on a single
//! multiprocessor and shares an area of fast on-chip memory known as **shared memory**.
//! Accesses to shared memory on the physical multiprocessor are typically more than 10 times
//! faster than accesses to device global memory, which is accessible across all thread blocks.
//! Thread blocks can be one-, two-, or three-dimensional, which is helpful when working with
//! multi-dimensional data such as images.
//!
//! Thread blocks are launched in **grids**, which also can be one-, two-, or three-dimensional.
//! The total number of threads launched by a kernel launch will be the product of grid size
//! multiplied by thread block size.
//!
//! For achieving high computation throughput, it is important to ensure that the grid size
//! is large enough, so that optimal number of multiprocessors are utilized to complete the
//! work. On the other hand, if the number of threads in each thread block gets too small,
//! each multiprocessor may be under-utilized, or kernels using shared memory may not be
//! able to make effective use of the fast shared memory.
//!
//! # Usage
//!
//! Before using RustaCUDA, you must install the CUDA development libraries for your system. Version
//! 8.0 or newer is required. You must also have a CUDA-capable GPU installed with the appropriate
//! drivers.
//!
//! Add the following to your `Cargo.toml`:
//!
//! ```text
//! [dependencies]
//! rustacuda = "0.1"
//! rustacuda_core = "0.1"
//! ```
//!
//! And this to your crate root:
//!
//! ```text
//! #[macro_use]
//! extern crate rustacuda;
//! extern crate rustacuda_core;
//! ```
//!
//! Finally, set the CUDA library path environment variable to the location of your CUDA libraries.
//! 
//! #### Ubuntu
//!
//! ```shell
//! export LIBRARY_PATH=/usr/local/cuda/lib64
//! ```
//!
//! #### Windows
//!
//! ```shell
//! export CUDA_LIBRARY_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\lib\x64"
//! ```
//!
//! # Examples
//!
//! ## Adding two numbers on the device
//!
//! First, download the `resources/add.ptx` file from the RustaCUDA repository and place it in
//! the resources directory for your application.
//!
//! ```
//! #[macro_use]
//! extern crate rustacuda;
//! extern crate rustacuda_core;
//!
//! use rustacuda::prelude::*;
//! use rustacuda::memory::DeviceBox;
//! use std::error::Error;
//! use std::ffi::CString;
//!
//! fn main() -> Result<(), Box<dyn Error>> {
//!     // Initialize the CUDA API
//!     rustacuda::init(CudaFlags::empty())?;
//!
//!     // Get the first device
//!     let device = Device::get_device(0)?;
//!
//!     // Create a context associated to this device
//!     let context = Context::create_and_push(
//!         ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
//!
//!     // Load the module containing the global kernel function we want to call
//!     let module_data = CString::new(include_str!("../resources/add.ptx"))?;
//!     let module = Module::load_from_string(&module_data)?;
//!
//!     // Create a stream to submit work to
//!     let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
//!
//!     // Allocate space on the device and copy numbers to it.
//!     let mut x = DeviceBox::new(&10.0f32)?;
//!     let mut y = DeviceBox::new(&20.0f32)?;
//!     let mut result = DeviceBox::new(&0.0f32)?;
//!
//!     // Launching kernels is unsafe since Rust can't enforce safety.
//!     // Think of kernel launches as a foreign-function call. In this case,
//!     // it is — this kernel is written in CUDA C, compiled to PTX, and then
//!     // loaded and executed on the CUDA device.
//!     unsafe {
//!         // Launch the `add` global kernel function with one block containing one thread
//!         // without shared memory on the given stream.
//!         launch!(module.add<<<1, 1, 0, stream>>>(
//!             x.as_device_ptr(),
//!             y.as_device_ptr(),
//!             result.as_device_ptr(),
//!             1u32 // Number of values in device buffers
//!         ))?;
//!     }
//!
//!     // Kernel launches are always asynchronous, so wait for the stream
//!     // to complete queued kernel launches, as well as any other tasks
//!     // submitted to the stream.
//!     stream.synchronize()?;
//!
//!     // Copy the result back to the host
//!     let mut result_host = 0.0f32;
//!     result.copy_to(&mut result_host)?;
//!
//!     println!("Sum is {}", result_host);
//!     assert_eq!(30.0f32, result_host);
//!
//!     Ok(())
//! }
//! ```

#![warn(
    missing_docs,
    missing_debug_implementations,
    unused_import_braces,
    unused_results,
    unused_qualifications
)]
// TODO: Add the missing_doc_code_examples warning, switch these to Deny later.

// Allow clippy lints
#![allow(unknown_lints, clippy::new_ret_no_self)]

#[macro_use]
extern crate bitflags;
extern crate cuda_driver_sys;
extern crate cuda_runtime_sys;
extern crate rustacuda_core;

pub mod context;
pub mod device;
pub mod error;
pub mod event;
pub mod function;
pub mod memory;
pub mod module;
pub mod prelude;
pub mod stream;

use crate::context::{Context, ContextFlags};
use crate::device::Device;
use crate::error::{CudaResult, ToResult};
use cuda_driver_sys::{cuDriverGetVersion, cuInit};

bitflags! {
    /// Bit flags for initializing the CUDA driver. Currently, no flags are defined,
    /// so `CudaFlags::empty()` is the only valid value.
    pub struct CudaFlags: u32 {
        // We need to give bitflags at least one constant.
        #[doc(hidden)]
        const _ZERO = 0;
    }
}

/// Initialize the CUDA Driver API.
///
/// This must be called before any other RustaCUDA (or CUDA) function is called. Typically, this
/// should be at the start of your program. All other functions will fail unless the API is
/// initialized first.
///
/// The `flags` parameter is used to configure the CUDA API. Currently no flags are defined, so
/// it must be `CudaFlags::empty()`.
pub fn init(flags: CudaFlags) -> CudaResult<()> {
    unsafe { cuInit(flags.bits()).to_result() }
}

/// Shortcut for initializing the CUDA Driver API and creating a CUDA context with default settings
/// for the first device.
///
/// This is useful for testing or just setting up a basic CUDA context quickly. Users with more
/// complex needs (multiple devices, custom flags, etc.) should use `init` and create their own
/// context.
pub fn quick_init() -> CudaResult<Context> {
    init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
}

/// Struct representing the CUDA API version number.
#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub struct CudaApiVersion {
    version: i32,
}
impl CudaApiVersion {
    /// Returns the latest CUDA version supported by the CUDA driver.
    pub fn get() -> CudaResult<CudaApiVersion> {
        unsafe {
            let mut version: i32 = 0;
            cuDriverGetVersion(&mut version as *mut i32).to_result()?;
            Ok(CudaApiVersion { version })
        }
    }

    /// Return the major version number — eg. the 9 in version 9.2
    #[inline]
    pub fn major(self) -> i32 {
        self.version / 1000
    }

    /// Return the minor version number — eg. the 2 in version 9.2
    #[inline]
    pub fn minor(self) -> i32 {
        (self.version % 1000) / 10
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_api_version() {
        let version = CudaApiVersion { version: 9020 };
        assert_eq!(version.major(), 9);
        assert_eq!(version.minor(), 2);
    }

    #[test]
    fn test_init_twice() {
        init(CudaFlags::empty()).unwrap();
        init(CudaFlags::empty()).unwrap();
    }

    
}

// Fake module with a private trait used to prevent outside code from implementing certain traits.
pub(crate) mod private {
    pub trait Sealed {}
}
