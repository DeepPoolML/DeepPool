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
#include "runtime.h"
#include "runnableModule.h"
#include "logger.h"
#include "utils.h"
#include "communication.h"

using torch::autograd::Variable;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

////////////////////////////////////////////////
// TsrXferFunc
////////////////////////////////////////////////
Variable
TsrXferFunc::forward(AutogradContext* ctx, Variable x, TsrXfer* xfer)
{
  ctx->saved_data["xfer"] = reinterpret_cast<int64_t>(xfer);
  DP_LOG(DEBUG, "TsrXferFunc::forward entered.. type: %d", xfer->type);

  if (xfer->type == TsrXfer::Send) {
    std::vector<torch::Tensor> splittedTsrs =
        x.split_with_sizes(xfer->splitSizes, xfer->splitCatDim);
    assert(splittedTsrs.size() == xfer->xferTagAndRank.size() + 1);
    size_t i;
    for (i = 0; i < xfer->xferTagAndRank.size(); ++i) {
      Tag tag = xfer->xferTagAndRank[i].first;
      Rank dest = xfer->xferTagAndRank[i].second;
      torch::Tensor tsr = splittedTsrs[i];
      DP_LOG(DEBUG, "Sending tag:%d to R:%d with %s", tag, dest,
          tsrSizeToStr(tsr).c_str());

      // xfer->recevingLayerForSend->sendOnLayerVisit.emplace_back({tsr, tag, dest});
      xfer->commHandler->send(tsr, tag, dest, /*async*/ true);
    }
    return splittedTsrs[i];
  }
  else if (xfer->type == TsrXfer::Recv) {
    std::vector<int64_t> inputSizes = x.sizes().vec();
    std::vector<torch::Tensor> tsrList;
    size_t i;
    for (i = 0; i < xfer->xferTagAndRank.size(); ++i) {
      Tag tag = xfer->xferTagAndRank[i].first;
      Rank src = xfer->xferTagAndRank[i].second;
      inputSizes[xfer->splitCatDim] = xfer->splitSizes[i];
      torch::Tensor tsr = torch::empty(inputSizes);
      tsr = tsr.to(xfer->commHandler->getDev(), /*non_blocking*/ true, /*copy*/ false);
      // DP_LOG(DEBUG, "Receiving tag:%d from R:%d with tensor: %s, %s, %s", tag, src,
      //     tsr.toString().c_str(), tsrSizeToStr(tsr).c_str(), tsrToStr(tsr).c_str());
      DP_LOG(DEBUG, "Receiving tag:%d from R:%d with tensor: %s", tag, src,
          tsrSizeToStr(tsr).c_str());
      xfer->commHandler->recv(tsr, tag, src, /*async*/ true);
      tsrList.push_back(tsr);
    }
    tsrList.push_back(x);
    DP_LOG(DEBUG, "Concating %d tensors", static_cast<int>(tsrList.size()));
    auto concated = torch::cat(tsrList, xfer->splitCatDim);
    DP_LOG(DEBUG, "Concated tensor: %s", tsrSizeToStr(concated).c_str());
    return concated;
  } else {
    DP_LOG(ERROR, "xfer type is %d, which is not supported.", xfer->type);
    return x;
  }
}

variable_list
TsrXferFunc::backward(AutogradContext* ctx, variable_list grad_output)
{
  TsrXfer* xfer = reinterpret_cast<TsrXfer*>(ctx->saved_data["xfer"].toInt());
  Variable x = grad_output[0];
  DP_LOG(DEBUG, "grad_output size: %d", (int)grad_output.size());

  if (xfer->type == TsrXfer::Recv) {
    std::vector<torch::Tensor> splittedTsrs =
        x.split_with_sizes(xfer->splitSizes, xfer->splitCatDim);
    assert(splittedTsrs.size() == xfer->xferTagAndRank.size() + 1);
    size_t i;
    for (i = 0; i < xfer->xferTagAndRankBack.size(); ++i) {
      Tag tag = xfer->xferTagAndRankBack[i].first;
      Rank dest = xfer->xferTagAndRankBack[i].second;
      torch::Tensor tsr = splittedTsrs[i];
      DP_LOG(DEBUG, "Sending tag:%d to R:%d with %s", tag, dest,
          tsrSizeToStr(tsr).c_str());
      xfer->commHandler->send(tsr, tag, dest, /*async*/ true);
    }
    
    variable_list grad_inputs(2);
    grad_inputs[0] = splittedTsrs[i];

    DP_LOG(DEBUG, "Remainder tensor after sending grads out. %s, %s",
        splittedTsrs[i].toString().c_str(), tsrSizeToStr(splittedTsrs[i]).c_str());
    
    return grad_inputs;
    // return { splittedTsrs[i] };
  }
  else if (xfer->type == TsrXfer::Send) {
    std::vector<int64_t> inputSizes = x.sizes().vec();
    std::vector<torch::Tensor> tsrList;
    size_t i;
    for (i = 0; i < xfer->xferTagAndRankBack.size(); ++i) {
      Tag tag = xfer->xferTagAndRankBack[i].first;
      Rank src = xfer->xferTagAndRankBack[i].second;
      inputSizes[xfer->splitCatDim] = xfer->splitSizes[i];
      torch::Tensor tsr = torch::empty(inputSizes);
      tsr = tsr.to(xfer->commHandler->getDev(), /*non_blocking*/ true, /*copy*/ false);
      DP_LOG(DEBUG, "Receiving tag:%d from R:%d with tensor: %s", tag, src,
          tsr.toString().c_str());
      xfer->commHandler->recv(tsr, tag, src, /*async*/ true);
      tsrList.push_back(tsr);
    }
    tsrList.push_back(x);
    // return { torch::cat(tsrList, xfer->splitCatDim) };

    variable_list grad_inputs(2);
    grad_inputs[0] = torch::cat(tsrList, xfer->splitCatDim);
    return grad_inputs;
  }
  else {
    DP_LOG(ERROR, "xfer type is %d, which is not supported.", xfer->type);
    return grad_output;
  }
}


/**
 * Constructs RunnableModule
 */
RunnableModule::RunnableModule(RuntimeContext* rtctx,
                               json spec,
                               CommunicationHandler* commHandler,
                               c10::Device device)
  : rtctx(rtctx)
  , rank(spec["rank"].get<int>())
  , globalBatchSize(spec["globalBatchSize"].get<int>())
  , moduleList()
  , layersInJson(spec["layers"])
  , initialBatchSize(layersInJson[0]["config"][0])
  , commHandler(commHandler)
  , device(device)
  // , leavesForBackward()
  // , fpCtx(layersInJson)
  , layers()
  , layerQ()
  , fpInput()
  , fpTargets()
  , fpOutput()
  , fpLoss()
{
  DP_LOG(DEBUG, "Constructing runnable module.. rank:%d", rank);
  DP_LOG(DEBUG, "             initialBatchSize:%d", initialBatchSize);
  DP_LOG(DEBUG, "             layersInJson's size:%d (from spec)", static_cast<int>(spec["layers"].size()));
  DP_LOG(DEBUG, "             layersInJson's size:%d", static_cast<int>(layersInJson.size()));
  
  // It's important to reserve the same, so that layers won't get copied over
  // to another address.. (layer's are pointing each other with raw pointer.)
  layers.reserve(layersInJson.size());

  for (auto& ldsc : layersInJson) {
    int id = ldsc["id"].get<int>();
    std::string name = ldsc["name"].get<std::string>();
    std::string moduleLoc = ldsc["moduleSavedLocation"].get<std::string>();
    DP_LOG(DEBUG, " %d-th layer's name: %s, moduleLoc: %s", id, name.c_str(),
        moduleLoc.c_str());
    
    SpecialModuleTypes specialModule = SpecialModuleTypes::NOTSPECIAL;
    if (name == "concat") {
      specialModule = SpecialModuleTypes::CONCAT;
    }
    torch::jit::Module module = torch::jit::load(std::string(rtctx->homedir) +
        "/DeepPoolRuntime/" + moduleLoc);
    DP_LOG(DEBUG, " layer's module is loaded.");
    if (name == "concat") {
      DP_LOG(DEBUG, " layer is concat.");
    } else {
      DP_LOG(DEBUG, " layer is not concat.");
    }

    module.to(device);
    module.train();
    DP_LOG(DEBUG, " layer's module is moved to device and set for train mode.");

    int layerLocalBatch = ldsc["config"][0].get<int>();
    bool layerIsActive = layerLocalBatch > 0;
    bool detachInput = true;
    if (name == "ReLU2d" || name == "ReLU1d") {
      detachInput = false;
      DP_LOG(DEBUG, "ReLU. detachInput: %d", detachInput);
    }
    
    std::vector<Layer*> prevLayers;
    for (auto& plidjson : ldsc["prevLayers"]) {
      int plid = plidjson.get<int>();
      prevLayers.push_back(&layers[plid]);
    }
    bool detachOutput = ldsc["nextLayers"].size() > 1;
    layers.emplace_back(module, specialModule, id, layerIsActive, detachInput,
                        detachOutput, prevLayers);

    // EmptyTensorSizes.
    layers.back().emptyInSizes.push_back(0);
    for (int size : ldsc["inputDim"]) {
      layers.back().emptyInSizes.push_back(size);
    }
    layers.back().emptyOutSizes.push_back(0);
    for (int size : ldsc["outputDim"]) {
      layers.back().emptyOutSizes.push_back(size);
    }
    
    // Communications.
    if (layerIsActive && ldsc.contains("tensorTx")) {
      std::map<int, std::vector<json> > sendListDict;
      for (auto& item : ldsc["tensorTx"]) {
        int nextLayerId = item["prop"]["nextLayerId"].get<int>();
        if (sendListDict.find(nextLayerId) == sendListDict.end()) {
          sendListDict[nextLayerId] = std::vector<json>();
        }
        sendListDict[nextLayerId].push_back(item);
      }
      for (const auto& kv : sendListDict) {
        const int nextLayerId = kv.first;
        const std::vector<json>& sendList = kv.second;

        TsrXfer xfer(commHandler);
        xfer.type = TsrXfer::Send;
        xfer.splitCatDim = 0; // Sample dimension.
        xfer.prevLayerId = id;
        xfer.nextLayerId = nextLayerId;
        xfer.recevingLayerForSend = &layers.back();
        int xferSampleSum = 0;
        for (const json& item : sendList) {
          int xferSamples = item["prop"]["xferSamples"].get<int>();
          xfer.splitSizes.push_back(xferSamples);
          xferSampleSum += xferSamples;

          auto xferName = item["name"].get<std::string>();
          Tag tag = commHandler->getTag(xferName);
          Tag tagB = commHandler->getTag(xferName + "_back");
          Rank dest = item["dest"].get<Rank>();
          xfer.xferTagAndRank.push_back(std::make_pair(tag, dest));
          xfer.xferTagAndRankBack.push_back(std::make_pair(tagB, dest));
        }

        int remainder;
        if (xfer.splitCatDim == 0) {
          DP_LOG(DEBUG, "total samples for layer: %d", ldsc["config"][0].get<int>());
          remainder = ldsc["config"][0].get<int>() - xferSampleSum;
        } else { // Other than sample dimension, use outputDim as its dimension is ordered correctly.
          remainder = ldsc["outputDim"][xfer.splitCatDim - 1].get<int>() - xferSampleSum;
        }
        
        DP_LOG(DEBUG, "remainder: %d, sum: %d", remainder, xferSampleSum);
        xfer.splitSizes.push_back(remainder);
        layers.back().xferOuts.push_back(std::move(xfer));
        DP_LOG(DEBUG, "xferOut registered. len(layer->xferOuts): %d",
            static_cast<int>(layers.back().xferOuts.size()));
      }
    }

    if (ldsc.contains("tensorRxJit")) {
      std::map<int, std::vector<json> > recvListDict;
      for (auto& item : ldsc["tensorRxJit"]) {
        int nextLayerId = item["prop"]["nextLayerId"].get<int>();
        if (recvListDict.find(nextLayerId) == recvListDict.end()) {
          recvListDict[nextLayerId] = std::vector<json>();
        }
        recvListDict[nextLayerId].push_back(item);
      }

      for (const auto& kv : recvListDict) {
        const int nextLayerId = kv.first;
        const std::vector<json>& recvList = kv.second;

        TsrXfer xfer(commHandler);
        xfer.type = TsrXfer::Recv;
        xfer.splitCatDim = 0;
        xfer.prevLayerId = id;
        xfer.nextLayerId = nextLayerId;
        xfer.recevingLayerForSend = &layers.back();
        int xferSampleSum = 0;
        for (const json& item : recvList) {
          int xferSamples = item["prop"]["xferSamples"].get<int>();
          xfer.splitSizes.push_back(xferSamples);
          xferSampleSum += xferSamples;

          auto xferName = item["name"].get<std::string>();
          Tag tag = commHandler->getTag(xferName);
          Tag tagB = commHandler->getTag(xferName + "_back");
          Rank src = item["src"].get<Rank>();
          xfer.xferTagAndRank.push_back(std::make_pair(tag, src));
          xfer.xferTagAndRankBack.push_back(std::make_pair(tagB, src));
        }

        int remainder;
        if (xfer.splitCatDim == 0) {
          remainder = layersInJson[nextLayerId]["config"][0].get<int>() - xferSampleSum;
          // remainder = ldsc["config"][0].get<int>() - xferSampleSum;
        } else { // Other than sample dimension, use inputDim as its dimension is ordered correctly.
          remainder = ldsc["inputDim"][xfer.splitCatDim - 1].get<int>() - xferSampleSum;
        }

        DP_LOG(DEBUG, "remainder: %d, sum: %d", remainder, xferSampleSum);
        xfer.splitSizes.push_back(remainder);
        layers.back().xferIns.push_back(std::move(xfer));
        DP_LOG(DEBUG, "xferIn registered. len(layer->xferIns): %d", static_cast<int>(layers.back().xferIns.size()));
      }
    }

    // if (layerIsActive && ldsc.contains("tensorRx")) {
    //   std::map<int, std::vector<json> > recvListDict;
    //   for (auto& item : ldsc["tensorRx"]) {
    //     int prevLayerId = item["prop"]["prevLayerId"].get<int>();
    //     if (recvListDict.find(prevLayerId) == recvListDict.end()) {
    //       recvListDict[prevLayerId] = std::vector<json>();
    //     }
    //     recvListDict[prevLayerId].push_back(item);
    //   }
    //   for (const auto& kv : recvListDict) {
    //     const int prevLayerId = kv.first;
    //     const std::vector<json>& recvList = kv.second;

    //     TsrXfer xfer(commHandler);
    //     xfer.type = TsrXfer::Recv;
    //     xfer.splitCatDim = 0;
    //     xfer.prevLayerId = prevLayerId;
    //     xfer.nextLayerId = id;
    //     xfer.recevingLayerForSend = &layers.back();
    //     int xferSampleSum = 0;
    //     for (const json& item : recvList) {
    //       int xferSamples = item["prop"]["xferSamples"].get<int>();
    //       xfer.splitSizes.push_back(xferSamples);
    //       xferSampleSum += xferSamples;

    //       auto xferName = item["name"].get<std::string>();
    //       Tag tag = commHandler->getTag(xferName);
    //       Tag tagB = commHandler->getTag(xferName + "_back");
    //       Rank src = item["src"].get<Rank>();
    //       xfer.xferTagAndRank.push_back(std::make_pair(tag, src));
    //       xfer.xferTagAndRankBack.push_back(std::make_pair(tagB, src));
    //     }

    //     int remainder;
    //     if (xfer.splitCatDim == 0) {
    //       remainder = ldsc["config"][0].get<int>() - xferSampleSum;
    //     } else { // Other than sample dimension, use inputDim as its dimension is ordered correctly.
    //       remainder = ldsc["inputDim"][xfer.splitCatDim - 1].get<int>() - xferSampleSum;
    //     }

    //     DP_LOG(DEBUG, "remainder: %d, sum: %d", remainder, xferSampleSum);
    //     xfer.splitSizes.push_back(remainder);
    //     layers.back().xferIns.push_back(std::move(xfer));
    //     DP_LOG(DEBUG, "xferIn registered. len(layer->xferIns): %d", static_cast<int>(layers.back().xferIns.size()));
    //   }
    // }

    moduleList.push_back(module);
    DP_LOG(DEBUG, " layer's module is pushed back.");
    DP_LOG(DEBUG, " id: %d and moduleListsize: %d", id, (int)moduleList.size());
    assert(id + 1 == (int)moduleList.size());
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, xferOuts: %d, xferIns: %d", layer.id,
        static_cast<int>(layer.xferOuts.size()), static_cast<int>(layer.xferIns.size()));
  }


  /* set up fake data pipelines for input + target */
  std::vector<int64_t> inputSizes;
  inputSizes.push_back(initialBatchSize);
  for (int size : layersInJson[0]["inputDim"]) inputSizes.push_back(size);
  auto inputFn = [=] { return torch::randn(inputSizes); };
  input_pipeline = TensorGeneratorPipeline(inputFn, rtctx);
  int targetCount = layersInJson.back()["config"][0];
  auto targetOpts = torch::TensorOptions().dtype(torch::kInt64);
  auto targetFn = [=] { return torch::randint(/*low=*/0, /*high=*/1000, {targetCount}, targetOpts); };
  target_pipeline = TensorGeneratorPipeline(targetFn, rtctx);
}

/**
 * Dumps the entire model parameters into the given vector.
 */
void
RunnableModule::getParameters(std::vector<torch::Tensor>* parameters)
{
  for (const auto& module : moduleList) {
    for (const auto& params : module.parameters()) {
      parameters->push_back(params);
    }
  }
}

/**
 * Initiate an iteration.
 */
void
RunnableModule::iterInit()
{
  layerQ.clear();
  layerQ.push_back(&layers[0]);
  fpInput = input_pipeline.GetNext();
  fpTargets = target_pipeline.GetNext();
  fpOutput.reset();
  fpLoss.reset();
}

/**
 * Execute a forward pass of this model.
 * 
 * \return Returns true if forward pass is completed.
 */
bool
RunnableModule::forwardAStep()
{
  DP_LOG(DEBUG, "layerQ size: %d", (int)layerQ.size());
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  // TODO: potentially we can make track if the cuda kernel is finished
  // or probably finished.
  // bool skipSinceNotReady = false;
  if (layer->status == LayerStatus::PENDING_BP) {
    DP_LOG(DEBUG, "%d-th layer is processed again.", layer->id);
    return false;
  }
  DP_LOG(DEBUG, "lid:%d.", layer->id);
  for (auto& prevLayer : layer->prevLayers) {
    if (prevLayer->status == LayerStatus::PENDING_FP) {
      DP_LOG(DEBUG, "Layer %d is skipped for now, must do %d first.",
          layer->id, prevLayer->id);
      return false;
    }
  }
  
  if (layer->active) {
    DP_LOG(DEBUG, "Layer %d is active.", layer->id);

    std::vector<torch::Tensor> inputVec;
    if (layer->prevLayers.size() == 0) {
      DP_LOG(DEBUG, "Adding to inputVec: %s.", tsrSizeToStr(fpInput).c_str());
      inputVec.push_back(fpInput);
    } else if (layer->prevLayers.size() >= 1) {
      std::map<int, torch::Tensor> inputsByPid;
      for (auto& prevLayer : layer->prevLayers) {
        torch::Tensor prevOut;
        if (prevLayer->outputsAfterXfer.count(layer->id) > 0) {
          prevOut = prevLayer->outputsAfterXfer[layer->id];
        } else {
          prevOut = prevLayer->output;
        }
        if (!prevOut.defined()) {
          DIE("prevOut is not defined.");
          // DP_LOG(DEBUG, "prevOut is not defined. Using empty tensor.");
          // // prevOut = torch::empty(layer->emptyInSizes);
          // prevOut = torch::empty(prevLayer->emptyOutSizes);
          // prevOut = prevOut.to(device, /*non_blocking*/ true, /*copy*/ false);
          // DP_LOG(DEBUG, "Empty input tensor: %s", prevOut.toString().c_str());
        }
        if (layer->detachInput) {
          DP_LOG(DEBUG, "Detaching input");
          layer->detachedInputs[prevLayer->id] = prevOut.detach();
          layer->detachedInputs[prevLayer->id].requires_grad_();
          prevOut = layer->detachedInputs[prevLayer->id];
          DP_LOG(DEBUG, "Detached input tensor: %s", prevOut.toString().c_str());
        }
        inputsByPid[prevLayer->id] = prevOut;
      }

      // // Recv samples before running this layer.
      // for (TsrXfer& xfer : layer->xferIns) {
      //   inputsByPid[xfer.prevLayerId] =
      //       TsrXferFunc::apply(inputsByPid[xfer.prevLayerId], &xfer);
      //   DP_LOG(DEBUG, "Received & concatenated samples.");
      // }
      
      for (auto& plidInputPair : inputsByPid) {
        DP_LOG(DEBUG, "Adding to inputVec: %s.", tsrSizeToStr(plidInputPair.second).c_str());
        inputVec.push_back(plidInputPair.second);
      }
    } else {
      DIE("%d-th layer negative number of previous layers??", layer->id);
    }

    torch::Tensor output;
    if (layer->specialModule == SpecialModuleTypes::CONCAT) {
      // temporary hack to solve the problem of passing list of tensors as input.
      output = torch::cat(inputVec, 1);
    } else {
      std::vector<torch::jit::IValue> ivalVec;
      ivalVec.push_back(inputVec[0]);
      output = layer->module.forward(ivalVec).toTensor();
      DP_LOG(DEBUG, "module.forward called.");
    }

    if (layer->detachOutput) {
      layer->outputBeforeDetach = output;
      output = output.detach();
      output.requires_grad_();
    }

    // Send samples after running this layer.
    DP_LOG(DEBUG, "len(layer->xferOuts): %d", static_cast<int>(layer->xferOuts.size()));
    layer->outputsAfterXfer.clear();
    for (TsrXfer& xfer : layer->xferOuts) {
      torch::Tensor remainingOutput = TsrXferFunc::apply(output, &xfer);
      layer->outputsAfterXfer[xfer.nextLayerId] = remainingOutput;
      DP_LOG(DEBUG, "Split & sent samples.");
    }
    layer->output = output;

    // auto h = layer->output.register_hook([layer](torch::Tensor grad){
    //   DP_LOG(DEBUG, "lid:%d grad: %s", layer->id, tsrToStr(grad).c_str());
    // });

    // fpCtx.fpTensorToReturn = output;
    // fpCtx.runCriterionAndLoss = true;
    // DP_LOG(DEBUG, "return values are set.");

  } else { // This rank doesn't participate for this layer.
    DP_LOG(DEBUG, "Layer %d is not active.", layer->id);
    // output = torch::empty(0);
    // output.to(device, /*non_blocking*/ true, /*copy*/ false);
    // fpCtx.runCriterionAndLoss = false;
    // fpCtx.fpTensorToReturn.reset();
  }

  // Recv parts of output processed by another GPU.
  for (TsrXfer& xfer : layer->xferIns) {
    //TODO: assert that next layer is active.
    torch::Tensor localOut;
    if (layer->active) { // Don't use outputsAfterXfer if not active.
      if (layer->outputsAfterXfer.count(xfer.nextLayerId) > 0) {
        localOut = layer->outputsAfterXfer[xfer.nextLayerId];
      } else {
        localOut = layer->output;
      }
    }

    if (!localOut.defined()) {
      assert(!layer->active);
      DP_LOG(DEBUG, "localOut is not defined. Must be inactive? Using an empty tensor.");
      localOut = torch::empty(layer->emptyOutSizes);
      localOut = localOut.to(device, /*non_blocking*/ true, /*copy*/ false);
      localOut.requires_grad_();
      DP_LOG(DEBUG, "Empty localOut tensor: %s", localOut.toString().c_str());
    }
    torch::Tensor remainingOutput = TsrXferFunc::apply(localOut, &xfer);
    layer->outputsAfterXfer[xfer.nextLayerId] = remainingOutput;
    DP_LOG(DEBUG, "Received (nextLayer: %d) & concatenated samples. %s",
        xfer.nextLayerId, tsrSizeToStr(remainingOutput).c_str());
  }
  
  layer->status = LayerStatus::PENDING_BP;
  DP_LOG(DEBUG, " ** Layer %d is processed.", layer->id);
  
  for (auto& nextLayerPtr : layer->nextLayers) {
    if (nextLayerPtr->status == LayerStatus::PENDING_FP) {
      layerQ.push_back(nextLayerPtr);
      DP_LOG(DEBUG, "nextLayer %d is queued for processing.", nextLayerPtr->id);
    } else {
      DP_LOG(DEBUG, "nextLayer %d is already processed.", nextLayerPtr->id);
    }
  }

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    if (layer->output.defined()) {
      fpOutput = layer->output; // TODO: clean up.
      // runCriterionAndLoss = true;
    } else {
      fpOutput.reset();
      // runCriterionAndLoss = false;
    }
    return true;
  }
  return false;
}

/**
 * Compute the loss from forward pass.
 */
void
RunnableModule::loss()
{
  if (fpOutput.defined()) {    
    fpLoss = torch::nll_loss(fpOutput, fpTargets);
    // DP_LOG(DEBUG, "fpLoss: %s", tsrToStr(fpLoss).c_str());
    fpLoss.backward();
    DP_LOG(DEBUG, "fpLoss.backward() done. ");
  }
}

/**
 * Execute a backward pass of this model.
 * 
 * \return Returns true if backward pass is completed.
 */
bool
RunnableModule::backwardAStep()
{
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  // TODO: potentially we can make track if the cuda kernel is finished
  // or probably finished.
  if (layer->status == LayerStatus::PENDING_FP) {
    DP_LOG(DEBUG, "%d-th layer is processed again.", layer->id);
    return false;
  }
  DP_LOG(DEBUG, "lid:%d.", layer->id);
  for (auto& nextLayer : layer->nextLayers) {
    if (nextLayer->status == LayerStatus::PENDING_BP) {
      DP_LOG(DEBUG, "Layer %d is skipped for now, must do %d first.",
          layer->id, nextLayer->id);
      return false;
    }
  }

  bool mustRunBackward = layer->active;
  if (!layer->active && layer->outputsAfterXfer.size()) {
    mustRunBackward = true;
    DP_LOG(DEBUG, "Layer %d is inactive, but backward is called for sending "
        "out gradients.", layer->id);
#if VERBOSE
    bool someNextLayerIsActive = false;
    for (const auto nextLayer : layer->nextLayers) {
      if (nextLayer->active)
        someNextLayerIsActive = true;
    }
    assert(someNextLayerIsActive);
#endif
  }

  if (mustRunBackward) {
    if (layer->nextLayers.size() == 0) {
      DP_LOG(DEBUG, "No nextLayers.");
    } else if (layer->nextLayers.size() >= 1) {
      // for (auto nextLayerPtr : layer->nextLayers) {
      for (size_t nli = 0; nli < layer->nextLayers.size(); nli++) {
        auto nextLayerPtr = layer->nextLayers[nli];
        if (nextLayerPtr->detachInput) {
          bool retainGraph = nli < layer->nextLayers.size() - 1;
          torch::Tensor grad;
          if (nextLayerPtr->detachedInputs[layer->id].defined()) {
            grad = nextLayerPtr->detachedInputs[layer->id].grad();
          } else {
            DP_LOG(DEBUG, "nextLayerPtr->detachInput is not defined. Using empty tensor.");
            grad = torch::empty(layer->emptyOutSizes);
            grad = grad.to(device, /*non_blocking*/ true, /*copy*/ false);
          }
          DP_LOG(DEBUG, "nextLayerPtr(%d)->detachedInputs[%d]: %s, grad: %s",
              nextLayerPtr->id, layer->id,
              nextLayerPtr->detachedInputs[layer->id].toString().c_str(),
              tsrSizeToStr(grad).c_str());
          
          if (layer->outputsAfterXfer.count(nextLayerPtr->id)) {
            DP_LOG(DEBUG, "Backward on outputsAfterXfer:%s gradIn:%s", 
                tsrSizeToStr(layer->outputsAfterXfer[nextLayerPtr->id]).c_str(),
                tsrSizeToStr(grad).c_str());
            layer->outputsAfterXfer[nextLayerPtr->id].backward(grad, retainGraph);
          } else if (layer->active) {
            DP_LOG(DEBUG, "Backward on output:%s gradIn:%s", 
                tsrSizeToStr(layer->output).c_str(), tsrSizeToStr(grad).c_str());
            layer->output.backward(grad, retainGraph);
          } else {
            DP_LOG(DEBUG, "Backward is not called since inactive layer & "
                "no xferIn for layer %d", nextLayerPtr->id);
          }
        } else {
          DP_LOG(DEBUG, "  nextLayerPtr(%d)->detachInput is false!", nextLayerPtr->id);
        }
      }

      if (layer->active && layer->detachOutput) {
        DP_LOG(DEBUG, "  output was detached previously. Invoking backward on outputBeforeDetach.");
        layer->outputBeforeDetach.backward(layer->output.grad());
      }
    }
    DP_LOG(DEBUG, "Layer %d is active.", layer->id);
  } else { // This rank doesn't participate for this layer.
    DP_LOG(DEBUG, "Layer %d is not active.", layer->id);
  }
  
  layer->status = LayerStatus::PENDING_FP;
  
  for (auto& prevLayerPtr : layer->prevLayers) {
    if (prevLayerPtr->status == LayerStatus::PENDING_BP) {
      layerQ.push_back(prevLayerPtr);
    } else {
      DP_LOG(DEBUG, "prevLayer %d is already processed.", prevLayerPtr->id);
    }
  }

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    return true;
  }
  return false;
}
