#pragma once

#include <torch/torch.h>

#include <vector>

class CommunicationHandler;

class GradientSyncManager {
 public:
  GradientSyncManager(std::shared_ptr<CommunicationHandler> commHandler,
                      size_t flush_threshold_bytes)
      : commHandler_(commHandler),
        flush_threshold_bytes_(flush_threshold_bytes) {}

  void Flush();
  void AddGradient(torch::Tensor grad, size_t comm_group_key);
  void Join();

 private:
  const std::shared_ptr<CommunicationHandler> commHandler_;
  const size_t flush_threshold_bytes_;
  bool has_unjoined_work_{false};

  void FlushKey(size_t key);

  std::map<size_t, size_t> pending_bytes_by_key_;
  std::map<size_t, std::vector<torch::Tensor>> grads_by_key_;
};