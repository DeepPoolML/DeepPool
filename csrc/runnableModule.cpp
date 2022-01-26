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

#include <ATen/autocast_mode.h>
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
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::device_synchronize();
    graph.capture_begin();
  }
  int64_t EndCaptureAndTime() {
    graph.capture_end();
    c10::cuda::device_synchronize();
    CpuTimer timer("timer");
    timer.start();
    for (int i = 0; i < kTrials; ++i) graph.replay();
    c10::cuda::device_synchronize();
    timer.stop();
    return timer.avgMicros() / kTrials;
  }

 private:
  at::cuda::CUDAGraph graph;
};

/**
 * Constructs RunnableModule
 */
RunnableModule::RunnableModule(
    json spec, std::shared_ptr<CommunicationHandler> commHandler,
    LossFunctions lf)
    : commHandler(commHandler),
      sync_manager_(commHandler, rtctx->sync_bucket_size),
      lossfn_(lf) {
  auto& layersInJson = spec["layers"];
  long initialBatchSize = layersInJson[0]["config"][0];

  globalBatchSize = spec["globalBatchSize"].get<long>();
  for (long sample : spec["sampleIndices"]) sampleIndices.push_back(sample);
  for (long batch : spec["initialBatchSizes"])
    initialBatchSizes.push_back(batch);

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

    DP_LOG(DEBUG, " layer's module is loaded.");
    DP_LOG(DEBUG, " layer is%s concat.", name == "concat" ? "" : " not");

    DP_LOG(DEBUG, " layer's module is moved to device and set for train mode.");

    long layerLocalBatch = ldsc["config"][0].get<long>();
    bool layerIsActive = layerLocalBatch > 0;

    torch::jit::Module module;

    if (layerIsActive) {
      module = torch::jit::load(moduleLoc);
      module.to(rtctx->c10dev);
      module.train();
    }

    bool doLocalGradSync = layerIsActive && ldsc["gpuAssignment"].size() > 1;
    doLocalGradSync &=
        rtctx->min_layer_sync <= static_cast<size_t>(rtctx->worldSize);

    auto layer = std::make_shared<Layer>(module, specialModule, id,
                                         layerIsActive, doLocalGradSync);
    layers.push_back(layer);

    layer->commGroupKey = RankVecToKey(ldsc["gpuAssignment"]);
    if (doLocalGradSync) {
      if (rtctx->nccl_groups.count(layer->commGroupKey) == 0) {
        DP_LOG(ERROR,
               "Runtime does not have a nccl communicator group for %lu\n",
               layer->commGroupKey);
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

    layer->layerLocalBatch = layerLocalBatch;

    layer->emptyOutSizes.push_back(0);
    for (int size : ldsc["outputDim"]) layer->emptyOutSizes.push_back(size);

    if (ldsc.contains("xfers")) {
      for (auto& item : ldsc["xfers"]) {
        size_t src_lid = item["prop"]["prevLayerId"];

        Xfer n;
        n.src = std::make_pair(item["src"], item["prop"]["txSampleOffset"]);
        n.dst = std::make_pair(item["dest"], item["prop"]["rxSampleOffset"]);
        n.nr_samples = item["prop"]["xferSamples"];
        n.src_lid = src_lid;
        n.tag = commHandler->getTag(item["name"].get<std::string>());

        if (n.src.first == n.dst.first)
          layer->xfers_local.push_back(n);
        else
          layer->xfers.push_back(n);

        if (n.dst.first == static_cast<size_t>(rtctx->rank))
          layer->rx_lids.insert(src_lid);

        if (n.src.first == static_cast<size_t>(rtctx->rank))
          layer->tx_lids.insert(src_lid);
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
           layer->tx_lids.size(), layer->rx_lids.size());
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, fwUsec: %" PRId64 ", bwUsec: %" PRId64 "",
           layer->id, layer->fwUsec, layer->bwUsec);
  }

  SetupOptimizer();

  /* Debug check */
  for (auto& l : layers) {
    for (auto& p : l->prevLayers)
      assert(std::find(p->nextLayers.begin(), p->nextLayers.end(), l) !=
             p->nextLayers.end());
    for (auto& p : l->nextLayers)
      assert(std::find(p->prevLayers.begin(), p->prevLayers.end(), l) !=
             p->prevLayers.end());
  }
}

void RunnableModule::SetMode(bool train) {
  if (train == isTrain_) return;
  isTrain_ = train;
  ResetGraphs();
  for (auto& l : layers) {
    if (!l->active) continue;
    if (train)
      l->module.train();
    else
      l->module.eval();
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

void RunnableModule::SetInputsTargets(torch::Tensor input,
                                      torch::Tensor target) {
  assert(state == JobState::INIT);

  size_t input_nb = input.defined() ? input.nbytes() : 0;

  if (input_nb) {
    /* should already be on device */
    assert(input.is_cuda());

    if (!input_buf.defined() || input_nb != input_buf.nbytes()) {
      /* reallocate input buffer */
      assert(!has_graph);
      input_buf = torch::empty(input.sizes(), input.options());
      assert(input_buf.is_cuda());
    }

    input_buf.copy_(input, /*non_blocking=*/true);
    input.record_stream(rtctx->torch_stream);
  }

  size_t target_nb = target.defined() ? target.nbytes() : 0;

  if (target_nb) {
    /* should already be on device */
    assert(target.is_cuda());

    if (!target_buf.defined() || target_nb != target_buf.nbytes()) {
      /* reallocate target buffer */
      assert(!has_graph);
      target_buf = torch::empty(target.sizes(), target.options());
      assert(target_buf.is_cuda());
    }

    target_buf.copy_(target, /*non_blocking=*/true);
    target.record_stream(rtctx->torch_stream);
  }

  /* enqueue input to first layer */
  layers[0]->tensors_in[0] = input_buf;
  fpTargets = target_buf;
}

torch::Tensor Layer::DoForward(bool captureLayer) {
  if (!active) {
    DP_LOG(DEBUG, "Layer %d is not active.", id);
    return {};
  }

  DP_LOG(DEBUG, "Layer %d is active.", id);
  assert(tensors_in.size());

  std::vector<torch::Tensor> vec;
  for (auto& p : tensors_in) vec.push_back(p.second);


  std::vector<c10::IValue> iVec;
  if (specialModule == SpecialModuleTypes::CONCAT &&
      module.get_method("forward").function().getSchema().arguments().size() <=
          2) {
    /* old list-argument concat */
    iVec.emplace_back(vec);
  } else {
    for (auto& v : vec) iVec.emplace_back(v);
  }

  GraphTimer fwdtimer;
  if (captureLayer) fwdtimer.StartCapture();

  output = module.forward(iVec).toTensor();

  if (captureLayer) fwUsec = fwdtimer.EndCaptureAndTime();

  for (auto& nl : nextLayers)
    nl->tensors_in[id] =
        output.detach().requires_grad_(output.is_floating_point());

  return output;
}

void Layer::DoBackward(bool captureLayer) {
  if (!active) {
    DP_LOG(DEBUG, "Layer %d is not active.", id);
    return;
  }

  if (!output.requires_grad()) {
    DP_LOG(DEBUG, "Layer %d does not require grad", id);
    return;
  }

  DP_LOG(DEBUG, "Layer %d is active.", id);

  GraphTimer bwdtimer;
  if (captureLayer) bwdtimer.StartCapture();

  for (size_t nli = 0; nli < nextLayers.size(); nli++) {
    auto& nl = nextLayers[nli];
    auto& grad = nl->tensors_in[id];
    DP_LOG(DEBUG, "Backward on output:%s grad(%d):%s",
           tsrSizeToStr(output).c_str(), nl->id, tsrSizeToStr(grad).c_str());
    output.backward(grad, nli < nextLayers.size() - 1);
    nl->tensors_in[id].reset();
  }

  if (captureLayer) bwUsec = bwdtimer.EndCaptureAndTime();

  for (auto& pLid : prevLayers) {
    tensors_in[pLid->id] = tensors_in[pLid->id].grad();
    DP_LOG(DEBUG, "Sending backward grad from %d to %d (size %s)", id, pLid->id,
           tsrSizeToStr(tensors_in[pLid->id]).c_str());
  }

  output.reset();
}

// TODO maybe some cleaner way to do in code, but this shouldn't impact
// performance
static torch::Tensor getSampleSlice(torch::Tensor& in, ssize_t offset,
                                    ssize_t nr_samples) {
  ssize_t cur_batch = in.sizes()[0];
  assert(offset + nr_samples <= cur_batch);
  std::vector<long> splits = {offset, nr_samples,
                              cur_batch - offset - nr_samples};
  std::vector<torch::Tensor> subs = in.split_with_sizes(splits);
  return subs[1];
}

/* Prepapre samples needed to process this layer */
void RunnableModule::ExecuteXfers(Layer* layer, bool backward) {
  if (!layer->xfers.size() && !layer->xfers_local.size()) return;

  std::map<size_t, torch::Tensor> inbound_tensors;
  std::map<size_t, torch::Tensor> outbound_tensors;

  torch::TensorOptions topts(rtctx->c10dev);
  topts = topts.requires_grad(!backward);
  if (at::autocast::is_enabled())
    topts = topts.dtype(at::autocast::get_autocast_gpu_dtype());

  auto& outbound_lids = backward ? layer->rx_lids : layer->tx_lids;
  auto& inbound_lids = backward ? layer->tx_lids : layer->rx_lids;

  /* prepare pointers to outbound samples */
  for (size_t lid : outbound_lids)
    outbound_tensors[lid] = layer->tensors_in[lid];

  /* prepare pointers for inbound samples */
  for (size_t lid : inbound_lids) {
    std::vector<long> inDim = layers.at(lid)->emptyOutSizes;
    inDim[0] =
        backward ? layers.at(lid)->layerLocalBatch : layer->layerLocalBatch;

    torch::Tensor in = torch::empty(inDim, topts);
    layer->tensors_in[lid] = in;
    inbound_tensors[lid] = in;
  }

  if (layer->xfers.size()) commHandler->comm_start();

  bool did_recv = false;
  for (const auto& ixfer : layer->xfers) {
    const size_t lid = ixfer.src_lid;
    const std::pair<size_t, size_t>& src = backward ? ixfer.dst : ixfer.src;
    const std::pair<size_t, size_t>& dst = backward ? ixfer.src : ixfer.dst;

    DP_LOG(DEBUG,
           "Transferring samples from rank %lu layer %lu pos %lu to rank %lu "
           "layer %d pos %lu",
           src.first, lid, src.second, dst.first, layer->id, dst.second);

    if (src.first == static_cast<size_t>(rtctx->rank)) {
      auto tsr = getSampleSlice(outbound_tensors.at(lid), src.second,
                                ixfer.nr_samples);
      commHandler->send(tsr, ixfer.tag, dst.first);
    } else {
      assert(dst.first == static_cast<size_t>(rtctx->rank));
      auto tsr =
          getSampleSlice(inbound_tensors.at(lid), dst.second, ixfer.nr_samples);
      commHandler->recv(tsr, ixfer.tag, src.first);
      did_recv = true;
    }
  }

  if (layer->xfers.size()) commHandler->comm_end();

  for (const auto& ixfer : layer->xfers_local) {
    const size_t lid = ixfer.src_lid;
    const std::pair<size_t, size_t>& src = backward ? ixfer.dst : ixfer.src;
    const std::pair<size_t, size_t>& dst = backward ? ixfer.src : ixfer.dst;

    DP_LOG(DEBUG,
           "Copying samples from layer %lu pos %lu to "
           "layer %d pos %lu",
           lid, src.second, layer->id, dst.second);

    auto srcTsr =
        getSampleSlice(outbound_tensors.at(lid), src.second, ixfer.nr_samples);
    auto dstTsr =
        getSampleSlice(inbound_tensors.at(lid), dst.second, ixfer.nr_samples);

    CUDACHECK(cudaMemcpyAsync(dstTsr.data_ptr(), srcTsr.data_ptr(),
                              srcTsr.nbytes(), cudaMemcpyDeviceToDevice,
                              rtctx->torch_stream));
  }

  if (did_recv) commHandler->sync();
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

  ExecuteXfers(layer);
  torch::Tensor output = layer->DoForward(captureLayer);
  layer->status = LayerStatus::PENDING_BP;
  layer->nr_current_depedencies = layer->nextLayers.size();

  DP_LOG(DEBUG, " ** Layer %d is processed.", layer->id);

  for (auto& nl : layer->nextLayers) {
    assert(nl->nr_current_depedencies > 0);
    if (--nl->nr_current_depedencies == 0) layerQ.push_back(nl.get());
  }

  if (!isTrain_) {
    layer->tensors_in.clear();
    layer->output.reset();
  }

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    fpOutput = output;
    return COMPLETED;
  }

  return IN_PROGRESS;
}

/**
 * Compute the loss from forward pass.
 */
void RunnableModule::loss() {
  if (!fpOutput.defined()) return;

  torch::Tensor fpLoss;

  if (lossfn_ == LossFunctions::CrossEntropyLoss) {
    auto shift_logits =
        fpOutput
            .index({torch::indexing::Ellipsis,
                    torch::indexing::Slice(torch::indexing::None, -1),
                    torch::indexing::Slice()})
            .contiguous();
    auto shift_labels =
        fpTargets
            .index({torch::indexing::Ellipsis,
                    torch::indexing::Slice(1, torch::indexing::None)})
            .contiguous();
    auto loss_fct = torch::nn::CrossEntropyLoss();
    fpLoss = loss_fct(shift_logits.view({-1, shift_logits.size(-1)}),
                      shift_labels.view({-1}).to(torch::kLong));
  } else {
    assert(lossfn_ == LossFunctions::NLLLoss);
    fpLoss = torch::nll_loss(fpOutput.log_softmax(1), fpTargets);
  }

  fpLoss.backward();
  fpOutput.reset();
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
  ExecuteXfers(layer, true);
  layer->status = LayerStatus::PENDING_FP;
  layer->nr_current_depedencies = layer->prevLayers.size();

  if (layer->doLocalGradSync) {
    for (const auto& param : layer->module.parameters()) {
      if (param.mutable_grad().defined())
        sync_manager_.AddGradient(param.mutable_grad(), layer->commGroupKey);
    }
  }

  for (auto& pl : layer->prevLayers) {
    assert(pl->nr_current_depedencies > 0);
    if (--pl->nr_current_depedencies == 0) layerQ.push_back(pl.get());
  }

  // Backward pass is completed.
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

  doGraphCapture &= !has_graph;

  if (state == JobState::INIT) {
    DP_LOG(DEBUG, "JobState::INIT.");

    p.Lap();
    TimerRecord("start");

    layerQ.clear();
    layerQ.push_back(layers[0].get());

    if (layers[0]->active && !layers[0]->tensors_in[0].defined())
      assert("MISSING INPUT TO FIRST LAYER!" && false);

    TimerRecord("load");

    /* start graph capture */
    if (doGraphCapture) {
      DP_LOG(NOTICE, "Starting capture.");
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

    fpOutput.reset();
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
      if (!isTrain_) {
        for (auto& layer : layers) {
          layer->status = LayerStatus::PENDING_FP;
          layer->nr_current_depedencies = layer->prevLayers.size();
        }
        DP_LOG(DEBUG, "Foward pass is completed.");
        state = JobState::FINISH;
      } else {
        DP_LOG(DEBUG, "Foward pass is completed. Calculating loss soon.");
        state = JobState::LOSS;
      }
    }
  } else if (state == JobState::LOSS) {
    DP_LOG(DEBUG, "JobState::LOSS.");
    loss();
    TimerRecord("loss");
    assert(layerQ.empty());
    layerQ.push_back(layers.back().get());
    DP_LOG(DEBUG, "Moving to backward pass.");
    state = JobState::BACKWARD;
  } else if (state == JobState::BACKWARD) {
    DP_LOG(DEBUG, "JobState::BACKWARD.");

    JobStatus status = backwardAStep(layerProfile);
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
      float maingraphsplit = -1.0;  // 10.0;
      float stepgraphsplit = -1.0;  // 2.0;

      if (!isTrain_) {
        commHandler->postcapture();
        maingraph.capture_end();
        auto maingraph_e =
            GraphPieces::GraphToExecs(maingraph.getGRAPH(), maingraphsplit);
        fullgraph = GraphPieces::MergePieces({maingraph_e});
      } else {
        stepgraph.capture_end();

        auto maingraph_e =
            GraphPieces::GraphToExecs(maingraph.getGRAPH(), maingraphsplit);
        auto syncgraph_e =
            GraphPieces::GraphToExecs(syncgraph.getGRAPH(), -1.0);
        auto stepgraph_e =
            GraphPieces::GraphToExecs(stepgraph.getGRAPH(), stepgraphsplit);
        fullgraph =
            GraphPieces::MergePieces({maingraph_e, syncgraph_e, stepgraph_e});
      }
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