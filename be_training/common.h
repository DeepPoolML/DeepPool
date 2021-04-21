#pragma once

static inline uint64_t get_now_ns() {
  using namespace std::chrono;
  using nsec = duration<uint64_t, std::nano>;
  return duration_cast<nsec>(steady_clock::now().time_since_epoch()).count();
}
