#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "utils.h"

class GraphPieces {
 public:
  static std::shared_ptr<GraphPieces> GraphToExecs(cudaGraph_t graph,
                                                   float ms_piece_split = -1.0);
  static std::shared_ptr<GraphPieces> GraphsToSingleExec(
      std::vector<cudaGraph_t> parts, size_t duplicatorcount = 1);
  static std::shared_ptr<GraphPieces> MergePieces(
      std::vector<std::shared_ptr<GraphPieces>> pieces);

  ~GraphPieces();
  void Launch(cudaStream_t stream) {
    for (auto& p : parts_) CUDACHECK(cudaGraphLaunch(p, stream));
  }
  void LaunchInterleavedCallback(cudaStream_t stream, std::function<void()> pre,
                                 std::function<void()> post) {
    for (auto& p : parts_) {
      if (pre) pre();
      CUDACHECK(cudaGraphLaunch(p, stream));
      if (post) post();
    }
  }

  GraphPieces(const GraphPieces&) = delete;
  GraphPieces& operator=(const GraphPieces&) = delete;
  GraphPieces(GraphPieces&&) = delete;
  GraphPieces& operator=(GraphPieces&&) = delete;

 private:
  GraphPieces(std::vector<cudaGraphExec_t> parts) : parts_(parts) {}
  std::vector<cudaGraphExec_t> parts_;
};