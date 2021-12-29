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

#include "JobContext.h"

#include <cuda_profiler_api.h>
#include <torch/torch.h>

#include <memory>
#include <string>

#include "BeTask.h"
#include "communication.h"
#include "dataset.h"
#include "logger.h"
#include "runnableModule.h"
#include "runtime.h"
#include "utils.h"

/**
 * Contructs context for a training job.
 */
JobContext::JobContext(std::unique_ptr<RunnableModule> modelIn,
                       std::string name,
                       std::shared_ptr<CommunicationHandler> commHandler,
                       json job_params)
    : model(std::move(modelIn)),
      name(name),
      commHandler(std::move(commHandler)) {
  run_with_be_ = job_params["run_with_be"].get<bool>();
  nr_gpus_ = job_params["nr_gpus"].get<size_t>();

  std::string dset = "random";
  if (job_params.contains("cifar_training")) {
    if (job_params["cifar_training"].get<bool>()) {
      dset = "cifar";
      /* cifar default includes 10 epochs with test routine */
      runTestRoutine_ = true;
      epochsToTrain = 10;
    }
  }

  if (job_params.contains("run_test_routine"))
    runTestRoutine_ = job_params["run_test_routine"].get<bool>();

  if (job_params.contains("epochs_to_train"))
    epochsToTrain = job_params["epochs_to_train"].get<size_t>();

  train_dataset_.reset(
      Dataset::fromName(dset, rtctx->rank, model->globalBatchSize,
                        model->initialBatchSizes, model->sampleIndices, 2000));
  eval_dataset_.reset(
      Dataset::fromName(dset + "_eval", rtctx->rank, model->globalBatchSize,
                        model->initialBatchSizes, model->sampleIndices, 10));
  dataset_pipeline_.reset(new DatasetPipelineWrapper(train_dataset_));

  if (!rtctx->use_fg_graph)
    iters_before_graph_capture = itersToTrain * epochsToTrain;
}

/**
 * Destructs context for a training job.
 * Defined here to avoid incomplete type destruction in 'JobContext.h' of
 * some unique_ptr members.
 */
JobContext::~JobContext() {}

void JobContext::printJobStatistics() {
  model->printLayerInGraphTimes();
  size_t iters = totiters - warmupIters;
  using msec = std::chrono::duration<double, std::milli>;
  double elapsed_ms = std::chrono::duration_cast<msec>(end - start).count();
  double total_iter_ms = elapsed_ms / (double)iters;
  double total_iter_ps = 1e3 / total_iter_ms;
  double be_img_ps = be_img_end - be_img_start;
  be_img_ps = 1e3 * be_img_ps / elapsed_ms;
  const auto &timers = model->GetTimers();
  DP_LOG(
      NOTICE,
      "A training job %s is completed (%lu iters, %.2f ms/iter, %.2f iter/s, "
      "%.2f be img/s, %lu globalBatchSize)."
      " AverageTiming (ms) => zero: %.1f, load:%.1f, fp:%.1f, loss:%.1f, "
      "bp:%.1f, opt: %.1f, iter:%.1f"
      " P50 (ms) => fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f",
      name.c_str(), iters, total_iter_ms, total_iter_ps, be_img_ps,
      model->globalBatchSize, timers.GetAvg("zero", warmupIters),
      timers.GetAvg("load", warmupIters), timers.GetAvg("forward", warmupIters),
      timers.GetAvg("loss", warmupIters),
      timers.GetAvg("backward", warmupIters),
      timers.GetAvg("step", warmupIters), timers.GetAvg("stop", warmupIters),
      timers.GetP50("forward", warmupIters), timers.GetP50("loss", warmupIters),
      timers.GetP50("backward", warmupIters),
      timers.GetP50("stop", warmupIters));
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
void JobContext::StepOne(bool *iter_done, bool *job_done) {
  bool graphCapture = totiters == iters_before_graph_capture;
  bool profile = rtctx->profile && totiters == iters_before_graph_capture - 5;

  if (!iter_in_progress) {
    if (totiters == profile_iter_start) CUDA_API_CALL(cudaProfilerStart());
    if ((graphCapture || profile) && IsBeEnabled()) BePause();
    if (totiters == warmupIters) {
      rtctx->torch_stream.synchronize();
      start = std::chrono::steady_clock::now();
      be_img_start = GetBeCounter();
    }
  }

  iter_in_progress = !model->AdvanceTraining(graphCapture, profile);

  if (iter_done) *iter_done = !iter_in_progress;

  if (!iter_in_progress) {
    if ((graphCapture || profile) && IsBeEnabled() && run_with_be_) BeResume();
    if (totiters == profile_iter_start + niter_to_profile - 1)
      CUDA_API_CALL(cudaProfilerStop());
    rtctx->fgcounter++;
    if (profile) {
      if (job_done) *job_done = true;
      return;
    }
    ++totiters;
  }
}

void JobContext::Test() {
  double total = 0.0;
  torch::Tensor correct = torch::zeros({1}).to(at::kLong).to(rtctx->c10dev);

  eval_dataset_->Reset();

  if (iters_before_graph_capture < totiters && rtctx->use_fg_graph)
    iters_before_graph_capture = totiters + 5;

  size_t i = 0;
  while (!eval_dataset_->IsDone()) {
    total += model->GetGlobalBatchSize();

    auto batch = eval_dataset_->getNextThisRank();
    torch::Tensor input = batch.data;
    if (input.defined()) input = input.to(rtctx->c10dev);
    auto output = Infer(input);
    if (output.defined() && output.nbytes() > 0) {
      auto pred = output.argmax(1);
      correct += pred.eq(batch.target.to(rtctx->c10dev)).sum();
    }
    DP_LOG(DEBUG, "Evaluate iteration %lu/%lu\n", ++i,
           eval_dataset_->GetItersPerEpoch());
  }

  iters_before_graph_capture = 0;

  if (nr_gpus_ > 1) {
    rtctx->torch_stream.synchronize();  // sync before calling into NCCL
    commHandler->comm_start();
    commHandler->all_reduce(correct, c10d::ReduceOp::SUM);
    commHandler->comm_end();
    commHandler->sync();
  }

  double corr = correct.item().toDouble();

  DP_LOG(NOTICE, "Evaluate: Total: %.1f Correct: %.1f | Accuracy: %.3f", total,
         corr, static_cast<double>(corr) / total);
}

torch::Tensor JobContext::Infer(torch::Tensor input) {
  torch::NoGradGuard guard;
  model->SetEval();
  model->SetInputsTargets(input, {});
  FinishIteration();
  return model->getOutput();
}

void JobContext::Train(torch::Tensor input, torch::Tensor target) {
  model->SetTrain();
  model->SetInputsTargets(input, target);
  FinishIteration();
}

void JobContext::TrainOneEpoch() {
  dataset_pipeline_->Reset();
  size_t i = 0;
  if (iters_before_graph_capture < totiters && rtctx->use_fg_graph)
    iters_before_graph_capture = totiters + 5;
  while (!dataset_pipeline_->IsDone()) {
    auto batch = dataset_pipeline_->getNextThisRank();
    Train(batch.data, batch.target);
    DP_LOG(DEBUG, "Training iteration %lu/%lu\n", ++i,
           dataset_pipeline_->GetItersPerEpoch());
  }
  iters_before_graph_capture = 0;
  rtctx->torch_stream.synchronize();
  end = std::chrono::steady_clock::now();
  be_img_end = GetBeCounter();
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
void JobContext::FinishIteration() {
  bool iter_done = false, job_done = false;
  do {
    StepOne(&iter_done, &job_done);
  } while (!iter_done && !job_done);
}
