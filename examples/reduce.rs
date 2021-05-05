#[macro_use]
extern crate rustacuda;

use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use num::Integer;

fn main() -> Result<(), Box<dyn Error>> {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let ptx = CString::new(include_str!("../resources/block_reduce.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const BLOCK: usize = 128;
    const N: usize = 2 * BLOCK + 1;
    const X: f32 = 0.5;

    // Create buffers for data
    let mut in_x = DeviceBuffer::from_slice(&[X; N])?;
    let mut out = DeviceBuffer::from_slice(&[0f32; 3])?;

    // This kernel adds values in each block segment defined by the block size.
    unsafe {
        // Launch the kernel with defined block sizes
        let function_name = CString::new("block_sum")?;
        let mut kernel = module.get_function(&function_name)?;
        let block = BLOCK as u32;
        let grid = (N as u32).div_ceil(&block);
        let shared_mem = block * std::mem::size_of::<f32>() as u32;
        kernel.set_max_dynamic_shared_mem(shared_mem)?;
        let result = launch!(kernel<<<grid, block, shared_mem, stream>>>(
            in_x.as_device_ptr(),
            out.as_device_ptr(),
            in_x.len() as u32
        ));
        result?;
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0f32; 3];
    out.copy_to(&mut out_host[0..3])?;

    let full_block_sum = BLOCK as f32 * X;
    assert_eq!(out_host, [full_block_sum, full_block_sum, X]);
    println!("Launched kernel successfully.");
    Ok(())
}