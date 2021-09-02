// Copyright (c) 2021 MIT
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#ifndef TRACER_H
#define TRACER_H

// #include <memory>
// #include <mutex>
#include <vector>
// #include <string>
#include "cuda_runtime.h"
#include "logger.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

class CudaTimer {
 public:
  CudaTimer(CudaTimer* from = nullptr, size_t reservedEntries = 1000)
    : fromTimer(from), evt(), recorded(false), elapsedTimes()
  {
    elapsedTimes.reserve(reservedEntries);
    CUDACHECK(cudaEventCreate(&evt));
  }

  void record() {
    CUDACHECK(cudaEventRecord(evt));
    recorded = true;
  }

  // Be wary for the order of invocation. Should not invoke this method before
  // other CudaTimer measuring time from this timer invokes.
  void saveAndReset() {
    if (fromTimer != nullptr && recorded) {
      CUDACHECK(cudaEventSynchronize(evt));
      assert(fromTimer->recorded);
      float ms;
      CUDACHECK(cudaEventElapsedTime(&ms, fromTimer->evt, evt));
      elapsedTimes.push_back(ms);
    }
    CUDACHECK(cudaEventDestroy(evt));
    CUDACHECK(cudaEventCreate(&evt));
    recorded = false;
  }
  int count() { return static_cast<int>(elapsedTimes.size()); }
  float getAvg(size_t skipIterCount = 0) {
    float sum = 0;
    for (size_t i = skipIterCount; i < elapsedTimes.size(); ++i) {
      sum += elapsedTimes[i];
    }
    if (elapsedTimes.size() == 0) {
      return 0;
    }
    return sum / elapsedTimes.size();
  }

  float getPercentile(float percentile, size_t skipIterCount) {
    std::vector<float> sortedTimes(elapsedTimes.begin() + skipIterCount,
                                   elapsedTimes.end());
    std::sort(sortedTimes.begin(), sortedTimes.end());
    size_t idx = sortedTimes.size() * percentile / 100.0;
    return sortedTimes[idx];
  }

  float getP50(size_t skipIterCount = 0) { return getPercentile(50, skipIterCount); }
  float getP99(size_t skipIterCount = 0) { return getPercentile(99, skipIterCount); }

  cudaEvent_t* getCudaEvent() { return &evt; }
  bool isRecorded() { return recorded; }
  
 private:
  CudaTimer* fromTimer;
  cudaEvent_t evt;
  bool recorded;
  std::vector<float> elapsedTimes;
};

#endif
