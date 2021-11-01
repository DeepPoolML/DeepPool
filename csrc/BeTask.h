#pragma once

#include <cstddef>
#include <string>

struct BeTaskConfig {
  long be_batch_size;
  double be_graph_split_ms;
  size_t sample_per_kernel;
  bool use_be_graph;
  std::string be_jit_file;
};

uint64_t GetBeCounter();
void BePause();
void BeResume();
bool IsBeEnabled();
void InitBeTask(BeTaskConfig cfg);