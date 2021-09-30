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

#include <vector>
#include "cuda_runtime.h"
#include "logger.h"
#include "utils.h"
#include "runtime.h"
#include "Cycles.h"

#define ENABLE_TIMERS 1

class CpuTimer {
 public:
  CpuTimer(const char* name)
    : name(name) {}

  inline void start() {
    lastStartTick = RAMCloud::Cycles::rdtsc();
  }

  inline void stop() {
    totalCycles += RAMCloud::Cycles::rdtsc() - lastStartTick;
    count++;
  }

  uint64_t avgMicros() {
    return RAMCloud::Cycles::toMicroseconds(
        totalCycles / count);
  }

  const char* name;
  uint64_t lastStartTick;
  uint64_t totalCycles;
  uint64_t count;
};

class CudaTimer {
 public:
  CudaTimer(CudaTimer* from = nullptr, size_t reservedEntries = 2500)
    : fromTimer(from)
  {
#if ENABLE_TIMERS
    elapsedTimes.reserve(reservedEntries);
    CUDACHECK(cudaEventCreateWithFlags(&evt, cudaEventBlockingSync));
#endif
  }

  void record() {
#if ENABLE_TIMERS
    CUDACHECK(cudaEventRecord(evt, rtctx->torch_stream));
#endif
    recorded = true;
  }

  // Be wary for the order of invocation. Should not invoke this method before
  // other CudaTimer measuring time from this timer invokes.
  void saveAndReset() {
    if (fromTimer != nullptr && recorded) {
#if ENABLE_TIMERS
      CUDACHECK(cudaEventSynchronize(evt));
      assert(fromTimer->recorded);
      float ms;
      CUDACHECK(cudaEventElapsedTime(&ms, fromTimer->evt, evt));
      elapsedTimes.push_back(ms);
#endif
      counter++;
    }
    recorded = false;
  }
  size_t count() { return counter; }
  float getAvg(size_t skipIterCount = 0) {
#if ENABLE_TIMERS
    if (skipIterCount >= elapsedTimes.size()) {
      skipIterCount = 0;
    }
    float sum = 0;
    for (size_t i = skipIterCount; i < elapsedTimes.size(); ++i) {
      sum += elapsedTimes[i];
    }
    if (elapsedTimes.size() == 0) {
      return 0;
    }
    return sum / (elapsedTimes.size() - skipIterCount);
#else
    return 0;
#endif
  }

  float getPercentile(float percentile, size_t skipIterCount) {
#if ENABLE_TIMERS
    if (skipIterCount >= elapsedTimes.size()) {
      skipIterCount = 0;
    }
    if (elapsedTimes.size() == 0) {
      return 0;
    }
    std::vector<float> sortedTimes(elapsedTimes.begin() + skipIterCount,
                                   elapsedTimes.end());
    std::sort(sortedTimes.begin(), sortedTimes.end());
    size_t idx = sortedTimes.size() * percentile / 100.0;
    return sortedTimes[idx];
#else
    return 0;
#endif
  }

  float getP50(size_t skipIterCount = 0) { return getPercentile(50, skipIterCount); }
  float getP99(size_t skipIterCount = 0) { return getPercentile(99, skipIterCount); }

  cudaEvent_t* getCudaEvent() { return &evt; }
  bool isRecorded() { return recorded; }
  
 private:
  CudaTimer* fromTimer;
  cudaEvent_t evt;
  bool recorded{false};
  size_t counter{0};
  std::vector<float> elapsedTimes;
};

#endif
