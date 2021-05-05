extern "C" __constant__ int my_constant = 314;

extern "C" __global__ void add(
    const float* x,
    const float* y,
    float* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}