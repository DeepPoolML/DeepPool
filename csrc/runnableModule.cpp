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

#include "runnableModule.h"

#include <torch/script.h>
#include <torch/torch.h>

#include "JobContext.h"
#include "communication.h"
#include "json.hpp"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class GraphTimer {
 public:
  static constexpr int kTrials = 200;
  GraphTimer(){};
  void StartCapture() {
    c10::cuda::device_synchronize();
    graph.capture_begin();
  }
  int64_t EndCaptureAndTime() {
    graph.capture_end();
    c10::cuda::device_synchronize();
    CpuTimer timer("fwTimer");
    timer.start();
    for (int i = 0; i < kTrials; ++i) graph.replay();
    c10::cuda::device_synchronize();
    timer.stop();
    return timer.avgMicros() / kTrials;
  }

 private:
  at::cuda::CUDAGraph graph;
};

torch::Tensor TsrXfer::DoSend(Variable& x) {
  std::vector<torch::Tensor> splittedTsrs =
      x.split_with_sizes(splitSizes, splitCatDim);
  assert(splittedTsrs.size() == xferTagAndRank.size() + 1);
  size_t i;
  commHandler->comm_start();
  for (i = 0; i < xferTagAndRank.size(); ++i) {
    Tag tag = xferTagAndRank[i].first;
    Rank dest = xferTagAndRank[i].second;
    torch::Tensor tsr = splittedTsrs[i];
    DP_LOG(DEBUG, "Sending tag:%d to R:%d with %s", tag, dest,
           tsrSizeToStr(tsr).c_str());

    commHandler->send(tsr, tag, dest);
  }
  commHandler->comm_end();
  return splittedTsrs[i];
}

torch::Tensor TsrXfer::DoRecv(Variable& x) {
  std::vector<int64_t> inputSizes = x.sizes().vec();

  int64_t totalInputDim = 0;
  for (auto& dim : splitSizes) totalInputDim += dim;

  assert(inputSizes[splitCatDim] != totalInputDim);
  assert(xferTagAndRank.size() > 0);

  /* allocate a single GPU buffer for the output tensor */
  torch::TensorOptions topts(rtctx->c10dev);
  inputSizes[splitCatDim] = totalInputDim;
  torch::Tensor tensor_out = torch::empty(inputSizes, topts);

  /* get pointers for each piece of the buffer to receive into */
  std::vector<torch::Tensor> tsrList =
      tensor_out.split_with_sizes(splitSizes, splitCatDim);

  size_t i;
  commHandler->comm_start();
  for (i = 0; i < xferTagAndRank.size(); ++i) {
    Tag tag = xferTagAndRank[i].first;
    Rank src = xferTagAndRank[i].second;
    auto& tsr = tsrList[i];
    DP_LOG(DEBUG, "Receiving tag:%d from R:%d with tensor: %s", tag, src,
           tsr.toString().c_str());
    commHandler->recv(tsr, tag, src);
  }
  commHandler->comm_end();

  auto& localTsr = tsrList[i];
  if (localTsr.nbytes()) {
    assert(x.nbytes() == localTsr.nbytes());
    CUDACHECK(cudaMemcpyAsync(localTsr.data_ptr(), x.data_ptr(), x.nbytes(),
                              cudaMemcpyDeviceToDevice, rtctx->torch_stream));
  }

  /* join communication stream after local copy */
  commHandler->sync();

  return tensor_out;
}

////////////////////////////////////////////////
// TsrXferFunc
////////////////////////////////////////////////
Variable TsrXferFunc::forward(AutogradContext* ctx, Variable x, TsrXfer* xfer) {
  ctx->saved_data["xfer"] = reinterpret_cast<int64_t>(xfer);
  DP_LOG(DEBUG, "TsrXferFunc::forward entered.. type: %d", xfer->type);

  if (xfer->type == TsrXfer::Send) {
    return xfer->DoSend(x);
  } else if (xfer->type == TsrXfer::Recv) {
    return xfer->DoRecv(x);
  } else {
    DP_LOG(ERROR, "xfer type is %d, which is not supported.", xfer->type);
    return x;
  }
}

variable_list TsrXferFunc::backward(AutogradContext* ctx,
                                    variable_list grad_output) {
  TsrXfer* xfer = reinterpret_cast<TsrXfer*>(ctx->saved_data["xfer"].toInt());
  Variable x = grad_output[0];
  DP_LOG(DEBUG, "grad_output size: %d", (int)grad_output.size());

  if (xfer->type == TsrXfer::Recv) {
    auto rem = xfer->DoSend(x);
    variable_list grad_inputs(2);
    grad_inputs[0] = rem;
    DP_LOG(DEBUG, "Remainder tensor after sending grads out. %s, %s",
           rem.toString().c_str(), tsrSizeToStr(rem).c_str());
    return grad_inputs;
  } else if (xfer->type == TsrXfer::Send) {
    variable_list grad_inputs(2);
    grad_inputs[0] = xfer->DoRecv(x);
    return grad_inputs;
  } else {
    DP_LOG(ERROR, "xfer type is %d, which is not supported.", xfer->type);
    return grad_output;
  }
}

/**
 * Constructs RunnableModule
 */
RunnableModule::RunnableModule(
    json spec, std::shared_ptr<CommunicationHandler> commHandler)
    : commHandler(commHandler),
      sync_manager_(commHandler, rtctx->sync_bucket_size) {
  auto& layersInJson = spec["layers"];
  long initialBatchSize = layersInJson[0]["config"][0];

  assert(spec["rank"].get<int>() == rtctx->rank);
  DP_LOG(DEBUG, "Constructing runnable module.. rank:%d", rtctx->rank);
  DP_LOG(DEBUG, "             initialBatchSize:%ld", initialBatchSize);
  DP_LOG(DEBUG, "             layersInJson's size:%lu (from spec)",
         spec["layers"].size());
  DP_LOG(DEBUG, "             layersInJson's size:%lu", layersInJson.size());

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
    if (name == "concat") specialModule = SpecialModuleTypes::CONCAT;

    torch::jit::Module module = torch::jit::load(
        std::string(rtctx->homedir) + "/DeepPoolRuntime/" + moduleLoc);

    DP_LOG(DEBUG, " layer's module is loaded.");
    DP_LOG(DEBUG, " layer is%s concat.", name == "concat" ? "" : " not");

    module.to(rtctx->c10dev);
    module.train();
    DP_LOG(DEBUG, " layer's module is moved to device and set for train mode.");

    int layerLocalBatch = ldsc["config"][0].get<int>();
    bool layerIsActive = layerLocalBatch > 0;
    bool detachInput = true;
    if (name == "ReLU2d" || name == "ReLU1d") {
      detachInput = false;
      DP_LOG(DEBUG, "ReLU. detachInput: %d", detachInput);
    }

    bool doLocalGradSync = layerIsActive && ldsc["gpuAssignment"].size() > 1;
    doLocalGradSync &= rtctx->min_layer_sync <= rtctx->worldSize;

    auto layer = std::make_shared<Layer>(
        module, specialModule, id, layerIsActive, detachInput, doLocalGradSync);
    layers.push_back(layer);

    if (doLocalGradSync) {
      layer->commGroupKey = RankVecToKey(ldsc["gpuAssignment"]);
      if (rtctx->nccl_groups.count(layer->commGroupKey) == 0) {
        DP_LOG(ERROR,
               "Runtime does not have a nccl communicator group for %lu\n",
               layer->commGroupKey);
        assert(false);
      }
    }

    DP_LOG(DEBUG, " %d-th layer's name: %s, doLocalGradSync: %d, key: %lu", id,
           name.c_str(), doLocalGradSync, layer->commGroupKey);

    int last_lid = -1;
    for (auto& plidjson : ldsc["prevLayers"]) {
      int plid = plidjson.get<int>();
      // ensure that prevLayers is sorted
      assert(plid > last_lid);
      last_lid = plid;
      layer->prevLayers.push_back(layers.at(plid));
      layers.at(plid)->nextLayers.push_back(layer);
      layer->nr_current_depedencies++;
    }

    layer->emptyOutSizes.push_back(0);
    for (int size : ldsc["outputDim"]) {
      layer->emptyOutSizes.push_back(size);
    }

    // Communications.
    if (layerIsActive && ldsc.contains("tensorTx")) {
      std::map<int, std::vector<json> > sendListDict;
      for (auto& item : ldsc["tensorTx"]) {
        int nextLayerId = item["prop"]["nextLayerId"].get<int>();
        sendListDict[nextLayerId].push_back(item);
      }
      for (const auto& kv : sendListDict) {
        const int nextLayerId = kv.first;
        const std::vector<json>& sendList = kv.second;

        TsrXfer xfer(commHandler);
        xfer.type = TsrXfer::Send;
        xfer.splitCatDim = 0;  // Sample dimension.
        xfer.nextLayerId = nextLayerId;
        int xferSampleSum = 0;
        for (const json& item : sendList) {
          int xferSamples = item["prop"]["xferSamples"].get<int>();
          xfer.splitSizes.push_back(xferSamples);
          xferSampleSum += xferSamples;

          auto xferName = item["name"].get<std::string>();
          Tag tag = commHandler->getTag(xferName);
          Rank dest = item["dest"].get<Rank>();
          xfer.xferTagAndRank.push_back(std::make_pair(tag, dest));
        }

        int remainder;
        if (xfer.splitCatDim == 0) {
          DP_LOG(DEBUG, "total samples for layer: %d",
                 ldsc["config"][0].get<int>());
          remainder = ldsc["config"][0].get<int>() - xferSampleSum;
        } else {  // Other than sample dimension, use outputDim as its dimension
                  // is ordered correctly.
          remainder = ldsc["outputDim"][xfer.splitCatDim - 1].get<int>() -
                      xferSampleSum;
        }

        DP_LOG(DEBUG, "remainder: %d, sum: %d", remainder, xferSampleSum);
        xfer.splitSizes.push_back(remainder);
        layer->xferOuts.push_back(std::move(xfer));
        DP_LOG(DEBUG, "xferOut registered. len(layer->xferOuts): %lu",
               layer->xferOuts.size());
      }
    }

    if (ldsc.contains("tensorRxJit")) {
      std::map<int, std::vector<json> > recvListDict;
      for (auto& item : ldsc["tensorRxJit"]) {
        int nextLayerId = item["prop"]["nextLayerId"].get<int>();
        recvListDict[nextLayerId].push_back(item);
      }

      for (const auto& kv : recvListDict) {
        const int nextLayerId = kv.first;
        const std::vector<json>& recvList = kv.second;

        TsrXfer xfer(commHandler);
        xfer.type = TsrXfer::Recv;
        xfer.splitCatDim = 0;
        xfer.nextLayerId = nextLayerId;
        int xferSampleSum = 0;
        for (const json& item : recvList) {
          int xferSamples = item["prop"]["xferSamples"].get<int>();
          xfer.splitSizes.push_back(xferSamples);
          xferSampleSum += xferSamples;

          auto xferName = item["name"].get<std::string>();
          Tag tag = commHandler->getTag(xferName);
          Rank src = item["src"].get<Rank>();
          xfer.xferTagAndRank.push_back(std::make_pair(tag, src));
        }

        int remainder;
        if (xfer.splitCatDim == 0) {
          remainder =
              layersInJson[nextLayerId]["config"][0].get<int>() - xferSampleSum;
          // remainder = ldsc["config"][0].get<int>() - xferSampleSum;
        } else {  // Other than sample dimension, use inputDim as its dimension
                  // is ordered correctly.
          remainder =
              ldsc["inputDim"][xfer.splitCatDim - 1].get<int>() - xferSampleSum;
        }

        DP_LOG(DEBUG, "remainder: %d, sum: %d", remainder, xferSampleSum);
        xfer.splitSizes.push_back(remainder);
        layer->xferIns.push_back(std::move(xfer));
        DP_LOG(DEBUG, "xferIn registered. len(layer->xferIns): %lu",
               layer->xferIns.size());
      }
    }

    if (rtctx->profile) {
      // TODO: if it's not that accurate, maybe add layer id?
      layer->moduleName = name + ldsc["params"].dump() + "[" +
                          std::to_string(layerLocalBatch) + "]" +
                          ldsc["inputDim"].dump();
      DP_LOG(DEBUG, "moduleName: %s", layer->moduleName.c_str());
    } else {
      layer->fwUsec = ldsc["gpuTime"][0].get<int>();
      layer->bwUsec = ldsc["gpuTime"][1].get<int>();
    }
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, xferOuts: %lu, xferIns: %lu", layer->id,
           layer->xferOuts.size(), layer->xferIns.size());
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, fwUsec: %" PRId64 ", bwUsec: %" PRId64 "",
           layer->id, layer->fwUsec, layer->bwUsec);
  }

  /* set up fake data pipelines for input + target */
  std::vector<int64_t> inputSizes;
  inputSizes.push_back(initialBatchSize);
  for (int size : layersInJson[0]["inputDim"]) inputSizes.push_back(size);
  auto inputFn = [=] { return torch::randn(inputSizes); };
  input_pipeline = TensorGeneratorPipeline(inputFn);
  int targetCount = layersInJson.back()["config"][0];
  auto targetOpts = torch::TensorOptions().dtype(torch::kInt64);
  auto targetFn = [=] {
    return torch::randint(/*low=*/0, /*high=*/1000, {targetCount}, targetOpts);
  };
  target_pipeline = TensorGeneratorPipeline(targetFn);

  SetupOptimizer();


  for (auto &l : layers) {
    for (auto &p : l->prevLayers)
      assert(std::find(p->nextLayers.begin(), p->nextLayers.end(), l) != p->nextLayers.end());
    for (auto &p : l->nextLayers)
      assert(std::find(p->prevLayers.begin(), p->prevLayers.end(), l) != p->prevLayers.end());
  }
}

/**
 * Dumps the entire model parameters into the given vector.
 */
void RunnableModule::SetupOptimizer() {
  std::vector<torch::Tensor> parameters;
  for (auto& layer : layers) {
    if (!layer->active) continue;
    for (const auto& params : layer->module.parameters()) {
      parameters.push_back(params);
    }
  }
  optimizer = std::make_unique<torch::optim::SGD>(parameters, /*lr=*/0.01);
}

/**
 * Initiate an iteration.
 */
void RunnableModule::iterInit() {
  layerQ.clear();
  layerQ.push_back(layers[0].get());
  fpInput = input_pipeline.GetNext();
  fpTargets = target_pipeline.GetNext();
  fpOutput.reset();
}

void Layer::DoForward(RunnableModule* model, bool captureLayer) {
  if (!active) {
    DP_LOG(DEBUG, "Layer %d is not active.", id);
    return;
  }

  DP_LOG(DEBUG, "Layer %d is active.", id);

  std::vector<torch::Tensor> vec;

  for (auto& prevLayer : prevLayers) {
    torch::Tensor prevOut;
    assert(prevLayer->outputsToLayer.count(id) > 0);
    prevOut = prevLayer->outputsToLayer[id];
    assert(prevOut.defined());
    if (detachInput) {
      detachedInputs[prevLayer->id] = prevOut.detach();
      detachedInputs[prevLayer->id].requires_grad_();
      prevOut = detachedInputs[prevLayer->id];
      DP_LOG(DEBUG, "Detached input tensor");
    }
    vec.push_back(prevOut);
  }

  if (vec.size() == 0) vec.push_back(model->fpInput);

  GraphTimer fwdtimer;
  if (captureLayer) fwdtimer.StartCapture();

  std::vector<c10::IValue> iVec;
  if (specialModule == SpecialModuleTypes::CONCAT) {
    iVec.emplace_back(vec);
  } else {
    assert(vec.size() == 1);
    iVec.emplace_back(vec[0]);
  }

  output = module.forward(iVec).toTensor();

  if (captureLayer) fwUsec = fwdtimer.EndCaptureAndTime();

  if (nextLayers.size() > 1) {
    outputBeforeDetach = output;
    output = output.detach();
    output.requires_grad_();
  }

  outputsToLayer.clear();

  for (auto& nl : nextLayers) outputsToLayer[nl->id] = output;

  // Send samples after running this layer.
  DP_LOG(DEBUG, "len(layer->xferOuts): %lu", xferOuts.size());
  for (TsrXfer& xfer : xferOuts) {
    torch::Tensor remainingOutput = TsrXferFunc::apply(output, &xfer);
    outputsToLayer[xfer.nextLayerId] = remainingOutput;
    DP_LOG(DEBUG, "Split & sent samples.");
  }
}

void Layer::DoForwardXferIn() {
  // Recv parts of output processed by another GPU.
  for (TsrXfer& xfer : xferIns) {
    torch::Tensor localOut;
    if (active) {
      assert(outputsToLayer.count(xfer.nextLayerId) > 0);
      localOut = outputsToLayer[xfer.nextLayerId];
      assert(localOut.defined());
    } else {
      torch::TensorOptions topts(rtctx->c10dev);
      topts = topts.requires_grad(true);
      localOut = torch::empty(emptyOutSizes, topts);
      DP_LOG(DEBUG, "Empty localOut tensor: %s", localOut.toString().c_str());
    }

    torch::Tensor remainingOutput = TsrXferFunc::apply(localOut, &xfer);
    outputsToLayer[xfer.nextLayerId] = remainingOutput;
    DP_LOG(DEBUG, "Received (nextLayer: %d) & concatenated samples. %s",
           xfer.nextLayerId, tsrSizeToStr(remainingOutput).c_str());
  }
}

void Layer::DoBackward(bool captureLayer) {
#if VERBOSE
  if (!active && outputsToLayer.size()) {
    DP_LOG(DEBUG,
           "Layer %d is inactive, but backward is called for sending "
           "out gradients.",
           layer->id);
    bool someNextLayerIsActive = false;
    for (const auto nextLayer : nextLayers) {
      if (nextLayer->active) someNextLayerIsActive = true;
    }
    assert(someNextLayerIsActive);
  }
#endif

  if (!active && !outputsToLayer.size()) {
    DP_LOG(DEBUG, "Layer %d is not active.", id);
    return;
  }

  DP_LOG(DEBUG, "Layer %d is active.", id);
  if (nextLayers.size() == 0) {
    DP_LOG(DEBUG, "No nextLayers.");
    return;
  }

  for (size_t nli = 0; nli < nextLayers.size(); nli++) {
    auto nextLayerPtr = nextLayers[nli];
    if (!nextLayerPtr->detachInput) {
      DP_LOG(DEBUG, "  nextLayerPtr(%d)->detachInput is false!",
             nextLayerPtr->id);
      continue;
    }

    bool retainGraph = nli < nextLayers.size() - 1;
    torch::Tensor grad = nextLayerPtr->detachedInputs[id];
    if (!grad.defined()) {
      DP_LOG(DEBUG,
             "nextLayerPtr->detachInput is not defined. Using empty "
             "tensor.");
      torch::TensorOptions topts(rtctx->c10dev);
      grad = torch::empty(emptyOutSizes, topts);
    }
    DP_LOG(DEBUG, "nextLayerPtr(%d)->detachedInputs[%d]: %s, grad: %s",
           nextLayerPtr->id, id,
           nextLayerPtr->detachedInputs[id].toString().c_str(),
           tsrSizeToStr(grad).c_str());

    GraphTimer bwdtimer;
    if (captureLayer) bwdtimer.StartCapture();

    if (outputsToLayer.count(nextLayerPtr->id)) {
      DP_LOG(DEBUG, "Backward on outputsToLayer:%s gradIn:%s",
             tsrSizeToStr(outputsToLayer[nextLayerPtr->id]).c_str(),
             tsrSizeToStr(grad).c_str());
      outputsToLayer[nextLayerPtr->id].backward(grad, retainGraph);
    } else {
      assert(!active);
    }

    if (captureLayer) bwUsec += bwdtimer.EndCaptureAndTime();
  }

  if (active && nextLayers.size() > 1) {
    DP_LOG(DEBUG,
           "  output was detached previously. Invoking backward on "
           "outputBeforeDetach.");
    GraphTimer bwdtimer;
    if (captureLayer) bwdtimer.StartCapture();
    outputBeforeDetach.backward(output.grad());
    if (captureLayer) bwUsec += bwdtimer.EndCaptureAndTime();
  }
}

/**
 * Execute a forward pass of this model.
 *
 * \return Returns true if forward pass is completed.
 */
JobStatus RunnableModule::forwardAStep(bool captureLayer) {
  assert(layerQ.size() > 0);
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  assert(layer->nr_current_depedencies == 0);
  assert(layer->status == LayerStatus::PENDING_FP);

  // TODO: potentially we can make track if the cuda kernel is finished
  // or probably finished.

  layer->DoForward(this, captureLayer);
  layer->DoForwardXferIn();
  layer->status = LayerStatus::PENDING_BP;
  layer->nr_current_depedencies = layer->nextLayers.size();

  DP_LOG(DEBUG, " ** Layer %d is processed.", layer->id);

  for (auto& nl : layer->nextLayers) {
    assert(nl->nr_current_depedencies > 0);
    if (--nl->nr_current_depedencies == 0) layerQ.push_back(nl.get());
  }

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    if (layer->output.defined()) {
      fpOutput = layer->output;
    } else {
      fpOutput.reset();
    }
    return COMPLETED;
  }
  return IN_PROGRESS;
}

/**
 * Compute the loss from forward pass.
 */
void RunnableModule::loss() {
  if (fpOutput.defined()) {
    auto fpLoss = torch::nll_loss(fpOutput, fpTargets);
    fpLoss.backward();
    DP_LOG(DEBUG, "fpLoss.backward() done. ");
    // idleCtxPtr->processLayerTime(1000, true);
  } else {
    // if (idleCtxPtr->jobType == IdleTimeCtx::FG) { // Don't deduct time for
    // BG. idleCtxPtr->processLayerTime(3000, false);  // For WRN.
    // idleCtxPtr->processLayerTime(2000, false);  // For VGG16.
    // }
  }
}

/**
 * Execute a backward pass of this model.
 *
 * \return Returns true if backward pass is completed.
 */
JobStatus RunnableModule::backwardAStep(bool captureLayer) {
  assert(layerQ.size() > 0);
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  assert(layer->nr_current_depedencies == 0);
  assert(layer->status == LayerStatus::PENDING_BP);

  DP_LOG(DEBUG, "lid:%d.", layer->id);

  layer->DoBackward(captureLayer);
  layer->status = LayerStatus::PENDING_FP;
  layer->nr_current_depedencies = layer->prevLayers.size();

  if (layer->doLocalGradSync) {
    for (const auto& param : layer->module.parameters()) {
      auto grad = param.mutable_grad();
      sync_manager_.AddGradient(grad, layer->commGroupKey);
    }
  }

  for (auto& pl : layer->prevLayers) {
    assert(pl->nr_current_depedencies > 0);
    if (--pl->nr_current_depedencies == 0) layerQ.push_back(pl.get());
  }

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    return COMPLETED;
  }
  return IN_PROGRESS;
}

/**
 * A helper to run a job.
 *
 * \param job   a context for the job to train.
 * \param[out] jobCompleted
 *    will be filled with true if the job is completely finished.
 *
 * \return    returns non-zero if iteration is finished.
 */
int RunnableModule::AdvanceTraining(bool doGraphCapture, bool layerProfile) {
  static CUDAPipeline p(2);  // Todo fix....

  if (state == JobState::INIT) {
    DP_LOG(DEBUG, "JobState::INIT.");

    p.Lap();
    TimerRecord("start");

    iterInit();
    TimerRecord("load");

    /* start graph capture */
    if (doGraphCapture) {
      DP_LOG(NOTICE, "Starting capture.");
      assert(!has_graph);
      graph_recording = true;
      c10::cuda::device_synchronize();
      graph_mempool = at::cuda::graph_pool_handle();
      maingraph.capture_begin(graph_mempool);
      commHandler->precapture();
    } else if (has_graph) {
      /* skip to forward phase */
      state = JobState::FORWARD;
      return 0;
    }

    optimizer->zero_grad();
    TimerRecord("zero");
    state = JobState::FORWARD;
    DP_LOG(DEBUG, "Foward pass is starting soon.");
  } else if (state == JobState::FORWARD) {
    DP_LOG(DEBUG, "JobState::FORWARD.");

    if (has_graph) {
      DP_LOG(DEBUG, "Replay iter.");
      fullgraph->Launch(rtctx->torch_stream);
      state = JobState::FINISH;
      return 0;
    }

    JobStatus status = forwardAStep(layerProfile);

    if (status == COMPLETED) {
      TimerRecord("forward");
      // TODO: add a loss calculation here? or as another state?
      DP_LOG(DEBUG, "Foward pass is completed. Calculating loss.");

      loss();
      TimerRecord("loss");
      assert(layerQ.empty());
      layerQ.push_back(layers.back().get());
      DP_LOG(DEBUG, "Moving to backward pass.");
      state = JobState::BACKWARD;
    }
  } else if (state == JobState::BACKWARD) {
    DP_LOG(DEBUG, "JobState::BACKWARD.");

    JobStatus status = backwardAStep(layerProfile);
    // TODO: get idle time for backward separately.
    if (status == COMPLETED) {
      TimerRecord("backward");
      state = JobState::SYNC;
      DP_LOG(DEBUG,
             "Backward pass is completed. Moving to gradient all-reduce.");
    }
  } else if (state == JobState::SYNC) {
    DP_LOG(DEBUG, "JobState::SYNC.");

    if (doGraphCapture) {
      sync_manager_.Join();
      commHandler->postcapture();
      maingraph.capture_end();
      syncgraph.capture_begin(graph_mempool);
    }

    sync_manager_.Flush();
    sync_manager_.Join();

    if (doGraphCapture) {
      syncgraph.capture_end();
      stepgraph.capture_begin(graph_mempool);
    }

    TimerRecord("sync");
    state = JobState::STEP;
  } else if (state == JobState::STEP) {
    DP_LOG(DEBUG, "JobState::STEP");
    optimizer->step();
    TimerRecord("step");
    state = JobState::FINISH;
  } else if (state == JobState::FINISH) {
    DP_LOG(DEBUG, "JobState::FINISH");

    if (doGraphCapture) {
      stepgraph.capture_end();

      float maingraphsplit = -1.0;  // 10.0;
      float stepgraphsplit = -1.0;  // 2.0;

      auto maingraph_e =
          GraphPieces::GraphToExecs(maingraph.getGRAPH(), maingraphsplit);
      auto syncgraph_e = GraphPieces::GraphToExecs(syncgraph.getGRAPH(), -1.0);
      auto stepgraph_e =
          GraphPieces::GraphToExecs(stepgraph.getGRAPH(), stepgraphsplit);
      fullgraph =
          GraphPieces::MergePieces({maingraph_e, syncgraph_e, stepgraph_e});
      has_graph = true;
      graph_recording = false;
      DP_LOG(NOTICE, "Ending capture.");
    }

    state = JobState::INIT;
    TimerRecord("stop");
    resetTimers();
    return 1;
  }
  return 0;
}

/**
 * Reset timers for profiling each layer. Happens every iteration.
 */
void RunnableModule::resetTimers() {
  if (rtctx->profile && !has_graph && !graph_recording) timers.SaveAndReset();
}

void RunnableModule::printLayerInGraphTimes() {
  if (!rtctx->profile) {
    return;
  }

  double sum = 0;
  for (auto& layer : layers) {
    double layerms =
        static_cast<double>(layer->fwUsec + layer->bwUsec) / 1000.0;
    printf(" %110s  %6.3f  %8" PRId64 "  %8" PRId64 "\n",
           layer->moduleName.c_str(), layerms, layer->fwUsec, layer->bwUsec);
    sum += layerms;
  }
  printf("%100s  %.3f\n", "SUM(avg)", sum);
}

// int64_t
// JobContext::getNextStepTime() {
//   if (state == JobState::INIT) {
//     return 1;
//   } else if (state == JobState::FORWARD) {
//     Layer* layer = model->layerQ.front();
//     return layer->fwUsec;
//   } else if (state == JobState::BACKWARD) {
//     Layer* layer = model->layerQ.front();
//     return layer->bwUsec;
//   } else if (state == JobState::SYNC) {
//     return 1;
//   } else if (state == JobState::STEP) {
//     return 1000;
//   } else if (state == JobState::FINISH) {
//     return 1;
//   }
//   return 0;
// }

// void
// RunnableModule::resetForNewIter()
// {
//   for (auto& layer : layers) {
//     layer->status = LayerStatus::PENDING_FP;
//   }
// }
