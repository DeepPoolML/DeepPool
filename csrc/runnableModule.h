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
#include <deque>
#include <torch/torch.h>
#include <torch/script.h>
#include "json.hpp"

using json = nlohmann::json;
using torch::autograd::Variable;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class CommunicationHandler;
class RuntimeContext;


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
  std::vector<std::pair<Tag, Rank> > xferTagAndRank;
  std::vector<std::pair<Tag, Rank> > xferTagAndRankBack;
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

/**
 * Description / context of a layer for training.
 */
struct Layer {
  Layer(torch::jit::Module module, int id, bool active, bool detachInput,
      std::vector<Layer*>& prevLayerVec)
    : module(module)
    , id(id)
    , active(active)
    , detachInput(detachInput)
    , prevLayers()
    , nextLayers()
    , output()
    , detachedInput()
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
  const int id;
  const bool active; // Inactive means no samples assigned for this runtime.
  const bool detachInput; // Detach input before running this layer.
  std::vector<Layer*> prevLayers;
  std::vector<Layer*> nextLayers;
  torch::Tensor output;  // Used during forward pass.
  torch::Tensor detachedInput; // Used during backward pass.
  LayerStatus status;
  std::vector<TsrXfer> xferIns;
  std::vector<TsrXfer> xferOuts;
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
  void iterInit(torch::Tensor x, torch::Tensor targets);
  bool forwardAStep();
  // bool forwardAStepOld();
  bool backwardAStep();
  void loss();

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
};

#endif