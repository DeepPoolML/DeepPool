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

#include <memory>
#include <set>
#include <vector>
#include <deque>
#include <torch/torch.h>
#include <torch/script.h>
#include "json.hpp"

#include "runtime.h"

#include <c10/cuda/CUDAStream.h>

using json = nlohmann::json;
using torch::autograd::Variable;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class CommunicationHandler;
class CudaTimer;
struct Layer;

typedef int Tag;
typedef int Rank;

struct TsrXfer {
  TsrXfer(CommunicationHandler* comm) : commHandler(comm), type(None),
      splitSizes(), splitCatDim(0), xferTagAndRank(), xferTagAndRankBack() {}
  
  CommunicationHandler* commHandler;
  enum Type {
    None, Send, Recv
  };
  Type type;
  std::vector<int64_t> splitSizes; // used only for send's forward or recv's backward.
  int splitCatDim;
  int prevLayerId;
  int nextLayerId;
  std::vector<std::pair<Tag, Rank> > xferTagAndRank;
  std::vector<std::pair<Tag, Rank> > xferTagAndRankBack;
  Layer* recevingLayerForSend; // Used for delayed send.
};

class TsrXferFunc : public torch::autograd::Function<TsrXferFunc> {
 public:
  static Variable forward(AutogradContext* ctx, Variable x, TsrXfer* xfer);
  static variable_list backward(AutogradContext* ctx, variable_list grad_output);
};

/**
 * Flipping status flag. This variable tracks the execution of the layer.
 * Specifically, it is used to (1) prevent duplicate execution and (2) ensure
 * the previous layer is executed so that its output tensor is valid.
 */
enum class LayerStatus {
  PENDING_FP = 0, // pending forward pass (last done job was backward).
  PENDING_BP      // pending backward pass (last done job was forward).
};

enum class SpecialModuleTypes {
  NOTSPECIAL = 0,
  CONCAT
};

/**
 * Description / context of a layer for training.
 */
struct Layer {
  Layer(torch::jit::Module module, SpecialModuleTypes specialModule, int id, bool active,
      bool detachInput, bool detachOutput, std::vector<Layer*>& prevLayerVec)
    : module(module)
    , specialModule(specialModule)
    , id(id)
    , active(active)
    , detachInput(detachInput)
    , detachOutput(detachOutput)
    , prevLayers()
    , nextLayers()
    , output()
    , outputBeforeDetach()
    , detachedInputs()
    , status(LayerStatus::PENDING_FP)
    , xferIns()
    , xferOuts()
  {
    for (auto prevLayerPtr : prevLayerVec) {
      prevLayers.push_back(prevLayerPtr);
      prevLayerPtr->nextLayers.push_back(this);
    }
  }
  
  torch::jit::Module module;
  const SpecialModuleTypes specialModule; // 0: not special, use module. 1: concat.
  const int id;
  const bool active; // Inactive means no samples assigned for this runtime.
  const bool detachInput; // Detach input before running this layer.
  const bool detachOutput; // Detach output when output is used multiple times.
  std::vector<Layer*> prevLayers;
  std::vector<Layer*> nextLayers;
  torch::Tensor output;  // Used during forward pass.
  torch::Tensor outputBeforeDetach; // Used during backward when output is used multple times. 
  std::map<int, torch::Tensor> outputsAfterXfer;  // Output specific to the nextLayerId (key).
  std::map<int, torch::Tensor> detachedInputs; // Used during backward pass.
  LayerStatus status;
  std::vector<TsrXfer> xferIns;
  std::vector<TsrXfer> xferOuts;
  std::vector<int64_t> emptyInSizes;  // primarily used for creating empty tensors for recv.
  std::vector<int64_t> emptyOutSizes; // primarily used for creating empty tensors for recv.
  std::unique_ptr<CudaTimer> fpTimer, bpTimer; // Used during profile mode only.
  std::string moduleName; // Used to output profiled runtimes.
};


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

class TensorGeneratorPipeline {
public:
  TensorGeneratorPipeline() {};
  TensorGeneratorPipeline(std::function<torch::Tensor()> gen, RuntimeContext *rtctx) : rtctx(rtctx) {
    for (size_t i = 0; i < 64; i++) cached_.push_back(gen());

    next_t_ = cached_[iter_idx_++ % 64];
    auto origstream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::setCurrentCUDAStream(rtctx->xfer_stream);
    next_t_ = next_t_.to(device, /*non_blocking*/ true, /*copy*/ false);
    CUDACHECK(cudaEventCreateWithFlags(&next_t_ev_, cudaEventDisableTiming));
    CUDACHECK(cudaEventRecord(next_t_ev_, rtctx->xfer_stream.stream()));
    c10::cuda::setCurrentCUDAStream(origstream);
  }

  torch::Tensor GetNext() {
    /* make sure current stream waits for last one to be ready */
    auto origstream = c10::cuda::getCurrentCUDAStream();
    CUDACHECK(cudaStreamWaitEvent(origstream.stream(), next_t_ev_));
    auto out = next_t_;

    /* generate next */
    next_t_ = cached_[iter_idx_++ % 64];
    c10::cuda::setCurrentCUDAStream(rtctx->xfer_stream);
    next_t_ = next_t_.to(device, /*non_blocking*/ true, /*copy*/ false);
    CUDACHECK(cudaEventRecord(next_t_ev_, rtctx->xfer_stream.stream()));
    c10::cuda::setCurrentCUDAStream(origstream);
    return next_t_;
  }

private:
  RuntimeContext *rtctx;
  torch::Device device{"cuda:0"};
  torch::Tensor next_t_;
  cudaEvent_t next_t_ev_;
  std::vector<torch::Tensor> cached_;
  uint64_t iter_idx_{0};
};


/**
 * A module that holds parameters (or submodules) and
 * provides functionalities to run training iteration.
 */
class RunnableModule : public torch::nn::Module {
 public:
  // RunnableModule();
  RunnableModule(RuntimeContext* rtctx, json specInJson,
      CommunicationHandler* commHandler, c10::Device device);

  void getParameters(std::vector<torch::Tensor>* parameters);
  void iterInit();
  bool forwardAStep();
  // bool forwardAStepOld();
  bool backwardAStep();
  void loss();
  void initProfileTimers(CudaTimer* ct_load, CudaTimer* ct_loss);
  void resetProfileTimers();
  void printProfileTimers(int warmupIters);

  ////////////////////////////////////////////
  // Internal data structure.
  ////////////////////////////////////////////
  RuntimeContext* rtctx;
  int rank;
  int globalBatchSize;
  std::vector<torch::jit::Module> moduleList;
  json layersInJson;
  int initialBatchSize;
  CommunicationHandler* commHandler;
  c10::Device device;
  std::vector<Layer> layers; // Topologically sorted list of layers.

  ////////////////////////////////////////////
  // Context for tracking particial progress.
  ////////////////////////////////////////////
  std::deque<Layer*> layerQ;
  torch::Tensor fpInput;
  torch::Tensor fpTargets;
  torch::Tensor fpOutput;
  torch::Tensor fpLoss;
  // bool runCriterionAndLoss;

  TensorGeneratorPipeline input_pipeline, target_pipeline;
};

#endif