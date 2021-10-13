
#include <cassert>
#include <cuda_runtime.h>

#include "CUDASleep.h"

#define MAXGPU 256

static double cycles_per_ns[MAXGPU];

typedef long long int clock64_t;

#if 0
__global__ void nanosleepX(unsigned nanos) {
  __nanosleep(nanos);
}
#endif

__global__ void clock_block(clock64_t clock_count) {
  clock64_t end_clock = clock64() + clock_count;
  while (clock64() < end_clock)
    ;
}

void gpu_nsleep(uint64_t nanos, cudaStream_t stream) {
  int device;
  assert(cudaGetDevice(&device) == cudaSuccess);
  assert(device < MAXGPU);

  if (!cycles_per_ns[device]) {
    cudaDeviceProp p;
    assert(cudaGetDeviceProperties(&p, device) == cudaSuccess);
    cycles_per_ns[device] = (double)p.clockRate / (1000.0 * 1000.0);
  }

  clock_block<<<1, 1, 0, stream>>>(
      (clock64_t)((double)nanos * cycles_per_ns[device]));
}
