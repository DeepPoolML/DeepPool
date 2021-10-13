#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void gpu_nsleep(uint64_t nanos, cudaStream_t stream);
