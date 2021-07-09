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

#include <torch/script.h>
#include <torch/torch.h>
#include "json.hpp"
#include "runnableModule.h"
#include "logger.h"

/**
 * Constructs RunnableModule
 */
RunnableModule::RunnableModule(json spec,
                               CommunicationHandler* commHandler,
                               c10::Device device)
  : rank(spec["rank"].get<int>())
  , globalBatchSize(spec["globalBatchSize"].get<int>())
  , moduleList()
  , layersInJson(spec["layers"])
  , initialBatchSize(layersInJson[0]["config"][0])
  , commHandler(commHandler)
  , device(device)
  , leavesForBackward()
  , fpCtx(layersInJson)
{
  DP_LOG(DEBUG, "Constructing runnable module.. rank:%d", rank);
  DP_LOG(DEBUG, "             initialBatchSize:%d", initialBatchSize);
  DP_LOG(DEBUG, "             layersInJson's size:%d (from spec)", static_cast<int>(spec["layers"].size()));
  DP_LOG(DEBUG, "             layersInJson's size:%d", static_cast<int>(layersInJson.size()));
  for (auto& ldsc : layersInJson) {
    DP_LOG(DEBUG, " entering a new layer.. %s", ldsc.dump().c_str());
    std::string name = ldsc["name"].get<std::string>();
    DP_LOG(DEBUG, " layer name: %s", name.c_str());
    std::string moduleLoc = ldsc["moduleSavedLocation"].get<std::string>();
    DP_LOG(DEBUG, " layer's name: %s, moduleLoc: %s", name.c_str(), moduleLoc.c_str());
    torch::jit::Module module = torch::jit::load("/home/ubuntu/DeepPoolRuntime/" + moduleLoc);
    DP_LOG(DEBUG, " layer's module is loaded.");
    if (name == "concat") {
      DP_LOG(DEBUG, " layer is concat.");
    } else {
      DP_LOG(DEBUG, " layer is not concat.");
    }
    // std::shared_ptr<torch::nn::Module> modulePtr(std::move(module));
    moduleList.push_back(module);
    DP_LOG(DEBUG, " layer's module is pushed back.");
  }
}

/**
 * Initiate an iteration.
 */
bool
RunnableModule::iterInit(torch::Tensor x)
{
  fpCtx.clear();
  fpCtx.fpInput = x;
  return true;
}

/**
 * Execute a forward pass of this model.
 * 
 * \return Returns true if forward pass is completed.
 */
bool
RunnableModule::forwardAStep()
{
  int lid = fpCtx.layersToProcess.front();
  fpCtx.layersToProcess.pop_front();
  torch::jit::Module& module = moduleList[lid];
  json& ldsc = layersInJson[lid];
  bool skipSinceNotReady = false;
  if (fpCtx.layersProcessed.find(lid) != fpCtx.layersProcessed.end()) {
    DIE("%d-th layer is processed again.", lid);
  }
  DP_LOG(DEBUG, "lid:%d.", lid);

  torch::Tensor input;
  if (ldsc["prevLayers"].size() == 0) {
    input = fpCtx.fpInput;
  } else if (ldsc["prevLayers"].size() == 1) {
    int prevLayerId = ldsc["prevLayers"][0].get<int>();
    input = fpCtx.fpOutputs[prevLayerId];
  } else if (ldsc["name"].get<std::string>() == std::string("concat")) {
    DIE("%d-th layer is concat, which is not implemented.", lid);
    //TODO: implement.
    // skipSinceNotReady should be updated here.
  } else {
    DIE("%d-th layer has more than 2 previous layers (except concat), which is not supported.", lid);
  }

  DP_LOG(DEBUG, "input:%s.", input.toString().c_str());

  if (skipSinceNotReady) {
    return false;
  }

  torch::Tensor output;
  if (fpCtx.layerIsActive[lid]) {
    DP_LOG(DEBUG, "Layer %d is active.", lid);
    if (!input.defined()) {
      DP_LOG(DEBUG, "input is not defined. Using empty tensor.");
      input = torch::empty(0);
      input.to(device, /*non_blocking*/ true, /*copy*/ false);
    }

    std::vector<torch::jit::IValue> inputVec;
    inputVec.push_back(input);
    DP_LOG(DEBUG, "inputVec is prepared.");
    output = module.forward(inputVec).toTensor();
    DP_LOG(DEBUG, "module.forward called.");

    fpCtx.fpTensorToReturn = output;
    fpCtx.runCriterionAndLoss = true;
    DP_LOG(DEBUG, "return values are set.");
    
    bool isOutputLeaf = ldsc["nextLayers"].size() > 0;
    for (auto& nlidjson : ldsc["nextLayers"]) {
      int nlid = nlidjson.get<int>();
      if (fpCtx.layerIsActive[nlid]) {
        isOutputLeaf = false;
      }
    }
    DP_LOG(DEBUG, "isOutputLeaf: %d", (int)isOutputLeaf);
    if (isOutputLeaf) {
      // TODO: maybe no need for this check? Preserve this set structure across iterations?
      if (fpCtx.leafIds.find(lid) != fpCtx.leafIds.end()) {
        DIE("visited a node that was tagged as a leaf."); 
      }
      leavesForBackward.emplace_back(lid, output);
      fpCtx.leafIds.insert(lid);
    }
  } else { // This rank doesn't participate for this layer.
    DP_LOG(DEBUG, "Layer %d is not active.", lid);
    output = torch::empty(0);
    output.to(device, /*non_blocking*/ true, /*copy*/ false);
    fpCtx.runCriterionAndLoss = false;
    fpCtx.fpTensorToReturn.reset();
  }
  DP_LOG(DEBUG, "Preparing output.");
  fpCtx.fpOutputs[lid] = output;
  DP_LOG(DEBUG, "Output is set.");
  fpCtx.layersProcessed.insert(lid);
  DP_LOG(DEBUG, "layersProcessed is inserted.");
  for (auto& nlidjson : ldsc["nextLayers"]) {
    int nlid = nlidjson.get<int>();
    if (fpCtx.layersProcessed.find(nlid) == fpCtx.layersProcessed.end()) {
      fpCtx.layersToProcess.push_back(nlid);
    }
  }
  DP_LOG(DEBUG, "layersToProcess is inserted.");

  // Forward pass is completed.
  if (fpCtx.layersToProcess.empty()) {
    return true;
  }
  return false;
}

// /**
//  * Execute a forward pass of this model.
//  */
// torch::Tensor
// RunnableModule::forward(torch::Tensor x)
// {

// }