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

#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>

#include <vector>

#include "Cycles.h"

class CpuTimer {
 public:
  CpuTimer(const char* name) : name(name) {}

  inline void start() { lastStartTick = RAMCloud::Cycles::rdtsc(); }

  inline void stop() {
    totalCycles += RAMCloud::Cycles::rdtsc() - lastStartTick;
    count++;
  }

  uint64_t avgMicros() {
    return RAMCloud::Cycles::toMicroseconds(totalCycles / count);
  }

  const char* name;
  uint64_t lastStartTick{0};
  uint64_t totalCycles{0};
  uint64_t count{0};
};

class CudaTimerChain {
 public:
  void Record(std::string name) {
    if (current_index == events.size()) {
      names.push_back(name);
      events.emplace_back(cudaEventDefault);
      elapsedTimes[names.at(current_index)] = {};
    }
    assert(names.at(current_index) == name);
    events.at(current_index++).record();
  };

  /* Warning - synchronizes */
  void SaveAndReset() {
    c10::cuda::device_synchronize();
    for (size_t i = 1; i < current_index; i++) {
      float ms = events.at(i - 1).elapsed_time(events.at(i));
      elapsedTimes.at(names.at(i)).push_back(ms);
    }
    current_index = 0;
  }

  float GetAvg(std::string name, size_t skipIterCount = 0) const {
    if (elapsedTimes.count(name) == 0) return 0;
    auto& times = elapsedTimes.at(name);

    float sum = 0;
    size_t nr = 0;
    for (size_t i = skipIterCount; i < times.size(); ++i, ++nr) sum += times[i];

    return nr > 0 ? sum / static_cast<float>(nr) : 0;
  }

  float GetPercentile(std::string name, float percentile,
                      size_t skipIterCount = 0) const {
    if (elapsedTimes.count(name) == 0) return 0;
    auto times = elapsedTimes.at(name);

    if (skipIterCount >= times.size()) return 0;
    std::vector<float> sortedTimes(times.begin() + skipIterCount, times.end());
    std::sort(times.begin(), times.end());
    size_t idx = sortedTimes.size() * percentile / 100.0;
    return sortedTimes.at(idx);
  }

  float GetP50(std::string name, size_t skipIterCount = 0) const {
    return GetPercentile(name, 50, skipIterCount);
  }
  float GetP99(std::string name, size_t skipIterCount = 0) const {
    return GetPercentile(name, 99, skipIterCount);
  }

 private:
  size_t current_index{0};
  std::map<std::string, std::vector<float>> elapsedTimes;
  std::vector<at::cuda::CUDAEvent> events;
  std::vector<std::string> names;
};
