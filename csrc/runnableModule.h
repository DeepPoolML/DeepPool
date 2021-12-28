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

#ifndef RUNNABLE_MODULE_H
#define RUNNABLE_MODULE_H

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <deque>
#include <vector>

#include "GraphPieces.h"
#include "communication.h"
#include "json.hpp"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

using json = nlohmann::json;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class Dataset;
class DatasetPipelineWrapper;

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class CommunicationHandler;

typedef int Tag;

struct Xfer {
  std::pair<size_t, size_t> src; /* rank and offset */
  std::pair<size_t, size_t> dst; /* rank and offset */
  size_t nr_samples;
  size_t src_lid;
  Tag tag;
};

/**
 * Flipping status flag. This variable tracks the execution of the layer.
 * Specifically, it is used to (1) prevent duplicate execution and (2) ensure
 * the previous layer is executed so that its output tensor is valid.
 */
enum class LayerStatus {
  PENDING_FP = 0,  // pending forward pass (last done job was backward).
  PENDING_BP       // pending backward pass (last done job was forward).
};

enum class SpecialModuleTypes { NOTSPECIAL = 0, CONCAT };

/**
 * Description / context of a layer for training.
 */
struct Layer {
  Layer(torch::jit::Module module, SpecialModuleTypes specialModule, int id,
        bool active, bool doLocalGradSync)
      : module(module),
        specialModule(specialModule),
        id(id),
        active(active),
        doLocalGradSync(doLocalGradSync) {}

  void DoForward(bool captureLayer);
  void DoBackward(bool captureLayer);

  std::map<size_t, torch::Tensor> tensors_in;
  std::vector<Xfer> xfers;
  std::set<size_t> tx_lids;
  std::set<size_t> rx_lids;

  torch::jit::Module module;
  int64_t fwUsec{0};
  int64_t bwUsec{0};
  const SpecialModuleTypes
      specialModule;  // 0: not special, use module. 1: concat.
  const int id;
  const bool active;  // Inactive means no samples assigned for this runtime.
  const bool doLocalGradSync;  // Perform gradient all-reduce within a host.
                               // (all layers do all-reduce over NIC)
  size_t commGroupKey;
  std::vector<std::shared_ptr<Layer>> prevLayers;  // sorted by id
  std::vector<std::shared_ptr<Layer>> nextLayers;
  torch::Tensor output;  // Used during forward pass.
  LayerStatus status{LayerStatus::PENDING_FP};
  size_t nr_current_depedencies{0};
  long layerLocalBatch;
  std::vector<int64_t> emptyOutSizes;
  std::string moduleName;  // Used to output profiled runtimes.
};

class GradientSyncManager {
 public:
  GradientSyncManager(std::shared_ptr<CommunicationHandler> commHandler,
                      size_t flush_threshold_bytes)
      : commHandler_(commHandler),
        flush_threshold_bytes_(flush_threshold_bytes) {}

  void Flush() {
    if (cur_pending_bytes_) {
      DP_LOG(DEBUG, "Flushing gradients");
      size_t curGroup = 0;
      size_t nbytesCurGroup = 0;
      for (auto& gradp : pending_grads_) {
        /* start a new all reduce group comm */
        if (gradp.first != curGroup) {
          if (curGroup) {
            commHandler_->comm_end();
            DP_LOG(DEBUG, "end sync group %lu (%lu bytes)", curGroup,
                   nbytesCurGroup);
            has_unjoined_work_ = true;
          }

          curGroup = gradp.first;
          nbytesCurGroup = 0;

          if (curGroup) {
            DP_LOG(DEBUG, "starting sync with GPUs on key %lu", gradp.first);
            commHandler_->comm_start(rtctx->grad_sync_stream, gradp.first);
          }
        }

        if (curGroup) {
          DP_LOG(DEBUG, "including grad with %lu nbytes",
                 gradp.second.nbytes());
          nbytesCurGroup += gradp.second.nbytes();
          commHandler_->all_reduce(gradp.second, c10d::ReduceOp::SUM);
        }
      }

      if (curGroup) {
        commHandler_->comm_end();
        DP_LOG(DEBUG, "end sync group %lu (%lu bytes)", curGroup,
               nbytesCurGroup);
        has_unjoined_work_ = true;
      }
    }
    cur_pending_bytes_ = 0;
    last_comm_group_key_ = 0;
    pending_grads_.clear();
  }

  void MaybeFlush(size_t commGroupKey) {
    if (commGroupKey != last_comm_group_key_)
      pending_grads_.emplace_back(0, torch::Tensor());
  }

  void AddGradient(torch::Tensor grad, size_t comm_group_key) {
    assert(comm_group_key > 0);
    if (flush_threshold_bytes_ && comm_group_key != last_comm_group_key_)
      Flush();
    last_comm_group_key_ = comm_group_key;
    cur_pending_bytes_ += grad.nbytes();
    pending_grads_.emplace_back(comm_group_key, grad);

    if (flush_threshold_bytes_ && cur_pending_bytes_ >= flush_threshold_bytes_)
      Flush();
  }

  void Join() {
    if (has_unjoined_work_) {
      commHandler_->sync(rtctx->grad_sync_stream);
      has_unjoined_work_ = false;
    }
  }

 private:
  const std::shared_ptr<CommunicationHandler> commHandler_;
  const size_t flush_threshold_bytes_;
  size_t last_comm_group_key_{0};
  size_t cur_pending_bytes_;
  bool has_unjoined_work_{false};
  std::vector<std::pair<size_t, torch::Tensor>> pending_grads_;
};

enum JobStatus { IN_PROGRESS = 0, COMPLETED, YIELD };

enum class JobState {
  INIT = 0,
  FORWARD,
  BACKWARD,
  SYNC,
  STEP,
  FINISH,
  NUM_JOB_STATES  // must be the last element in the enum
};

/**
 * A module that holds parameters (or submodules) and
 * provides functionalities to run training iteration.
 */
class RunnableModule {
 public:
  RunnableModule(json specInJson,
                 std::shared_ptr<CommunicationHandler> commHandler);

  int AdvanceTraining(bool doGraphCapture, bool layerProfile);

  CudaTimerChain timers;
  void printLayerInGraphTimes();

  void Evaluate();
  void ExecuteXfers(Layer* layer, bool reverse = false);

 protected:
  friend struct Layer;

  long globalBatchSize;
  std::vector<long> sampleIndices;
  std::vector<long> initialBatchSizes;

  std::shared_ptr<Dataset> train_dataset_;
  std::shared_ptr<Dataset> eval_dataset_;
  std::shared_ptr<DatasetPipelineWrapper> dataset_pipeline_;

  inline void TimerRecord(std::string name) {
    if (rtctx->profile && !has_graph && !graph_recording) timers.Record(name);
  }

  void iterInit();
  JobStatus forwardAStep(bool captureLayer);
  JobStatus backwardAStep(bool captureLayer);
  void loss();
  void resetTimers();
  void SetupOptimizer();

  ////////////////////////////////////////////
  // Internal data structure.
  ////////////////////////////////////////////
  std::shared_ptr<CommunicationHandler> commHandler;
  GradientSyncManager sync_manager_;
  // Topologically sorted list of layers.
  std::vector<std::shared_ptr<Layer>> layers;
  std::unique_ptr<torch::optim::SGD> optimizer;
  ////////////////////////////////////////////
  // Context for tracking partial progress.
  ////////////////////////////////////////////
  std::deque<Layer*> layerQ;
  torch::Tensor fpTargets;
  torch::Tensor fpOutput;

  JobState state{JobState::INIT};

  bool backwards_did_sync{false};
  bool has_graph{false};
  bool graph_recording{false};

  std::shared_ptr<GraphPieces> fullgraph;
  at::cuda::CUDAGraph maingraph, syncgraph, stepgraph;
  at::cuda::MempoolId_t graph_mempool;
};

#endif
