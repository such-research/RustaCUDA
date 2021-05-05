/// Block size should be a power of 2
extern "C" __global__ void block_sum(
    const float* x,
    float* out,
    unsigned int n
) {
    extern __shared__ float smem[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

    if (i < n) {
        // Move values to shared memory
        smem[tid] = x[i];
        __syncthreads();

        // Reduce sum in shared memory
        if (blockDim.x == 1024) {
            if (tid < 512 && i + 512 < n)
                smem[tid] += smem[tid + 512];
            __syncthreads();
        }
        if (blockDim.x >= 512) {
            if (tid < 256 && i + 256 < n)
                smem[tid] += smem[tid + 256];
            __syncthreads();
        }
        if (blockDim.x >= 256) {
            if (tid < 128 && i + 128 < n)
                smem[tid] += smem[tid + 128];
            __syncthreads();
        }
        if (blockDim.x >= 128) {
            if (tid < 64 && i + 64 < n)
                smem[tid] += smem[tid + 64];
            __syncthreads();
        }
        if (tid < 32) {
            smem[tid] += smem[tid + 32];
            __syncwarp();
            smem[tid] += smem[tid + 16];
            __syncwarp();
            smem[tid] += smem[tid + 8];
            __syncwarp();
            smem[tid] += smem[tid + 4];
            __syncwarp();
            smem[tid] += smem[tid + 2];
            __syncwarp();
            smem[tid] += smem[tid + 1];
        }

        // Write result of this block to out in global memory
        if (tid == 0) out[blockIdx.x] = smem[0];
    }
}
