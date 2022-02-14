#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <vector>

class CommunicationHandler;

class GradientSyncManager {
 public:
  GradientSyncManager(std::shared_ptr<CommunicationHandler> commHandler,
                      size_t flush_threshold_bytes)
      : commHandler_(commHandler),
        flush_threshold_bytes_(flush_threshold_bytes) {}

  void Flush(c10::optional<c10::cuda::CUDAStream> stream = {});
  void AddGradient(torch::Tensor grad, size_t comm_group_key);
  void Join(c10::optional<c10::cuda::CUDAStream> stream = {});

  bool HasPendingJoinOrFlush() {
    return has_unjoined_work_ || total_pending_bytes_ > 0;
  }

  void Reset() {
    has_unjoined_work_ = false;
    freeze_ = false;
    total_pending_bytes_ = 0;
    pending_bytes_by_key_.clear();
    grads_by_key_.clear();
  }

  void Freeze() { freeze_ = true; }

 private:
  const std::shared_ptr<CommunicationHandler> commHandler_;
  const size_t flush_threshold_bytes_;
  bool has_unjoined_work_{false};
  bool freeze_{false};
  ssize_t total_pending_bytes_{0};

  void FlushKey(size_t key, c10::optional<c10::cuda::CUDAStream> stream);

  std::map<size_t, size_t> pending_bytes_by_key_;
  std::map<size_t, std::vector<torch::Tensor>> grads_by_key_;
};