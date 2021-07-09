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

class CommunicationHandler;

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

/**
 * A module that holds parameters (or submodules) and
 * provides functionalities to run training iteration.
 */
class RunnableModule : public torch::nn::Module {
 public:
  // RunnableModule();
  RunnableModule(json specInJson, CommunicationHandler* commHandler, c10::Device device);

  bool iterInit(torch::Tensor x);
  bool forwardAStep();
  // torch::Tensor forward(torch::Tensor x);

  int rank;
  int globalBatchSize;
  std::vector<torch::jit::Module> moduleList;
  json layersInJson;
  int initialBatchSize;
  CommunicationHandler* commHandler;
  c10::Device device;
  std::vector< std::pair<int, torch::Tensor> > leavesForBackward;
  ForwardPassContext fpCtx;
};

#endif