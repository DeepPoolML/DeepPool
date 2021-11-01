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
#include "json.hpp"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

using json = nlohmann::json;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class CommunicationHandler;
struct IdleTimeCtx;

typedef int Tag;
typedef int Rank;

struct TsrXfer {
  TsrXfer(std::shared_ptr<CommunicationHandler> comm) : commHandler(comm) {}

  torch::Tensor DoSend(Variable& x);
  torch::Tensor DoRecv(Variable& x);

  std::shared_ptr<CommunicationHandler> commHandler;
  enum Type { None, Send, Recv };
  Type type{None};
  // used only for send's forward or recv's backward.
  std::vector<int64_t> splitSizes;
  int splitCatDim{0};
  int nextLayerId;
  std::vector<std::pair<Tag, Rank>> xferTagAndRank;
};

class TsrXferFunc : public torch::autograd::Function<TsrXferFunc> {
 public:
  static Variable forward(AutogradContext* ctx, Variable x, TsrXfer* xfer);
  static variable_list backward(AutogradContext* ctx,
                                variable_list grad_output);
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
        bool active, bool detachInput, bool doLocalGradSync)
      : module(module),
        specialModule(specialModule),
        id(id),
        active(active),
        doLocalGradSync(doLocalGradSync),
        detachInput(detachInput) {}

  void DoForward(RunnableModule* model, bool captureLayer);
  void DoForwardXferIn();
  void DoBackward(bool captureLayer);

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
  const bool detachInput;  // Detach input before running this layer.
  std::vector<std::shared_ptr<Layer>> prevLayers;  // sorted by id
  std::vector<std::shared_ptr<Layer>> nextLayers;
  torch::Tensor output;              // Used during forward pass.
  torch::Tensor outputBeforeDetach;  // Used during backward when output is used
                                     // multple times.
  std::map<int, torch::Tensor>
      outputsToLayer;  // Output specific to the nextLayerId (key).
  std::map<int, torch::Tensor> detachedInputs;  // Used during backward pass.
  LayerStatus status{LayerStatus::PENDING_FP};
  size_t nr_current_depedencies{0};
  std::vector<TsrXfer> xferIns;
  std::vector<TsrXfer> xferOuts;
  // primarily used for creating empty tensors for recv.
  std::vector<int64_t> emptyOutSizes;
  std::string moduleName;  // Used to output profiled runtimes.
};

class TensorGeneratorPipeline {
 public:
  TensorGeneratorPipeline(){};

  TensorGeneratorPipeline(std::function<torch::Tensor()> gen) {
    for (size_t i = 0; i < 64; i++) cached_.push_back(gen());

    CUDACHECK(cudaEventCreateWithFlags(&hToD_ev_, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&dToD_ev_, cudaEventDisableTiming));

    next_t_ = cached_[iter_idx_++ % 64];
    auto origstream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::setCurrentCUDAStream(rtctx->xfer_stream);
    next_t_ = next_t_.to(rtctx->c10dev, /*non_blocking*/ false, /*copy*/ false);
    CUDACHECK(cudaEventRecord(hToD_ev_, rtctx->xfer_stream));
    c10::cuda::setCurrentCUDAStream(origstream);

    tensorbytes = next_t_.nbytes();
    tensor_buf = c10::cuda::CUDACachingAllocator::raw_alloc(tensorbytes);
    assert(!tensorbytes || tensor_buf);
  }

  torch::Tensor GetNext() {
    return next_t_;

    if (!tensorbytes)
      return cached_[iter_idx_++ % 64].to(rtctx->c10dev, true, false);

    /* send last transmitted tensor into final buf */
    auto origstream = c10::cuda::getCurrentCUDAStream();
    CUDACHECK(cudaStreamWaitEvent(origstream, hToD_ev_));
    CUDACHECK(cudaMemcpyAsync(tensor_buf, next_t_.data_ptr(), next_t_.nbytes(),
                              cudaMemcpyDeviceToDevice, origstream));
    CUDACHECK(cudaEventRecord(dToD_ev_, origstream));

    auto tensor_out =
        torch::from_blob(tensor_buf, next_t_.sizes(), next_t_.options());

    /* wait for DtoD to finish before starting next HtoD transfer */
    CUDACHECK(cudaStreamWaitEvent(rtctx->xfer_stream, dToD_ev_));

    /* run next HtoD transfer */
    next_t_ = cached_[iter_idx_++ % 64];
    c10::cuda::setCurrentCUDAStream(rtctx->xfer_stream);
    next_t_ = next_t_.to(rtctx->c10dev, /*non_blocking*/ true, /*copy*/ false);
    CUDACHECK(cudaEventRecord(hToD_ev_, rtctx->xfer_stream));
    c10::cuda::setCurrentCUDAStream(origstream);

    return tensor_out;
  }

 private:
  size_t tensorbytes;
  void* tensor_buf;
  torch::Tensor next_t_;
  cudaEvent_t hToD_ev_{nullptr};
  cudaEvent_t dToD_ev_{nullptr};
  std::vector<torch::Tensor> cached_;
  uint64_t iter_idx_{0};
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

 protected:
  friend struct Layer;

  inline void TimerRecord(std::string name) {
    if (rtctx->profile && !has_graph && !graph_recording) timers.Record(name);
  }

  void iterInit();
  // void resetForNewIter();
  JobStatus forwardAStep(bool captureLayer);
  JobStatus backwardAStep(bool captureLayer);
  void loss();
  void gradientSyncSync();
  void gradientSync();
  void resetTimers();
  std::vector<torch::Tensor> getActiveParameters();

  ////////////////////////////////////////////
  // Internal data structure.
  ////////////////////////////////////////////
  std::shared_ptr<CommunicationHandler> commHandler;
  // Topologically sorted list of layers.
  std::vector<std::shared_ptr<Layer>> layers;
  std::unique_ptr<torch::optim::SGD> optimizer;
  ////////////////////////////////////////////
  // Context for tracking partial progress.
  ////////////////////////////////////////////
  std::deque<Layer*> layerQ;
  torch::Tensor fpInput;
  torch::Tensor fpTargets;
  torch::Tensor fpOutput;

  JobState state{JobState::INIT};

  TensorGeneratorPipeline input_pipeline, target_pipeline;

  bool backwards_did_sync{false};
  bool has_graph{false};
  bool graph_recording{false};

  std::shared_ptr<GraphPieces> fullgraph;
  at::cuda::CUDAGraph maingraph, syncgraph, stepgraph;
  at::cuda::MempoolId_t graph_mempool;
  // Performance Stat
  // CpuTimer detachTimer;

  // std::vector<ReduceBucket> reduceBuckets;
  // IdleTimeCtx* idleCtxPtr;
  // bool hasInactiveLayer {false};
};

#endif
