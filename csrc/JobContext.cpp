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
#include "logger.h"
#include "runnableModule.h"
#include "runtime.h"
#include "utils.h"

/**
 * Contructs context for a training job.
 */
JobContext::JobContext(std::unique_ptr<RunnableModule> modelIn,
                       std::string name,
                       std::shared_ptr<CommunicationHandler> commHandler)
    : model(std::move(modelIn)),
      name(name),
      commHandler(std::move(commHandler)) {
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
  auto &timers = model->timers;
  DP_LOG(
      NOTICE,
      "A training job %s is completed (%lu iters, %.2f ms/iter, %.2f iter/s, "
      "%.2f be img/s)."
      " AverageTiming (ms) => zero: %.1f, load:%.1f, fp:%.1f, loss:%.1f, "
      "bp:%.1f, opt: %.1f, iter:%.1f"
      " P50 (ms) => fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f",
      name.c_str(), iters, total_iter_ms, total_iter_ps, be_img_ps,
      timers.GetAvg("zero", warmupIters), timers.GetAvg("load", warmupIters),
      timers.GetAvg("forward", warmupIters), timers.GetAvg("loss", warmupIters),
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
bool JobContext::TrainSingleStep() {
  size_t end_iter = itersToTrain * epochsToTrain;
  assert(totiters < end_iter);

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

  if (!iter_in_progress) {
    if ((graphCapture || profile) && IsBeEnabled() && run_with_be) BeResume();
    if (totiters == profile_iter_start + niter_to_profile - 1)
      CUDA_API_CALL(cudaProfilerStop());
    if (++iter >= itersToTrain) {
      DP_LOG(DEBUG, "epoch is completed.");
      iter = 0;
      epoch++;
    }
    rtctx->fgcounter++;
    if (profile) return true;
    if (++totiters == end_iter) {
      rtctx->torch_stream.synchronize();
      end = std::chrono::steady_clock::now();
      be_img_end = GetBeCounter();
      return true;
    }
  }

  return false;
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
void JobContext::TrainToCompletion() {
  assert(totiters == 0);
  while (!TrainSingleStep())
    ;
  return;
}
