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

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class CommunicationHandler;
class RuntimeContext;

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
    // , gradOut()
    , detachedInput()
    , status(LayerStatus::PENDING_FP)
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
  // torch::Tensor gradOut; // Used during backward pass.
  torch::Tensor detachedInput; // Used during backward pass.
  LayerStatus status;
};

#if 0
/**
 * A context that tracks the progress of a forward pass of a signle iteration.
 */
struct ForwardPassContext {
  ForwardPassContext(json layersInJson)
    : fpInput()
    , fpTensorToReturn()
    , fpOutputs()
    , layerIsActive()
    , leafIds()
    , layersToProcess()
    , layersProcessed()
    , runCriterionAndLoss()
  {
    size_t numLayers = layersInJson.size();
    fpOutputs.resize(numLayers + 2);
    layerIsActive.reserve(numLayers + 2);
    for (auto ldsc : layersInJson) {
      int layerLocalBatch = ldsc["config"][0].get<int>();
      layerIsActive.push_back(layerLocalBatch > 0);
    }
    layersToProcess.push_back(0);
  }

  void clear() {
    fpOutputs.clear();
    leafIds.clear(); // Maybe not necessary?
    layersToProcess.clear();
    layersToProcess.push_back(0);
    layersProcessed.clear();
  }

  torch::Tensor fpInput;
  torch::Tensor fpTensorToReturn;
  std::vector<torch::Tensor> fpOutputs;
  std::vector<bool> layerIsActive;
  std::set<int> leafIds;
  std::deque<int> layersToProcess;
  std::set<int> layersProcessed;
  bool runCriterionAndLoss;
};
#endif


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
  void iterInit(torch::Tensor x);
  bool forwardAStep();
  // bool forwardAStepOld();
  bool backwardAStep();
  void loss(torch::Tensor targets);

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
  // std::vector< std::pair<int, torch::Tensor> > leavesForBackward;
  // ForwardPassContext fpCtx;
  std::vector<Layer> layers; // Topologically sorted list of layers.

  ////////////////////////////////////////////
  // Context for tracking particial progress.
  ////////////////////////////////////////////
  std::deque<Layer*> layerQ;
  torch::Tensor fpInput;
  torch::Tensor fpOutput;
  torch::Tensor fpLoss;
  // bool runCriterionAndLoss;
};

#endif