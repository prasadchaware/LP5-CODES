// reduction.cu

#include <stdio.h>

__global__ void reduce(int *input, int n, int *min, int *max, int *sum, float *avg) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int local_min = input[idx];
    int local_max = input[idx];
    int local_sum = 0;

    while (idx < n) {
        local_min = min(local_min, input[idx]);
        local_max = max(local_max, input[idx]);
        local_sum += input[idx];
        idx += stride;
    }
    
    atomicMin(min, local_min);
    atomicMax(max, local_max);
    atomicAdd(sum, local_sum);

    if (tid == 0) {
        atomicAdd(avg, (float)local_sum / n);
    }
}
