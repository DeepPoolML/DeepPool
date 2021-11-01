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

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <cuda_profiler_api.h>
#include <torch/torch.h>

#include <memory>
#include <string>

#include "BeTask.h"
#include "Cycles.h"
#include "communication.h"
#include "logger.h"
#include "rpcService.h"
#include "runnableModule.h"
#include "runtime.h"
#include "utils.h"

using Cycles = RAMCloud::Cycles;

#define EXPLICIT_INTERLEAVING 0

/**
 * Contructs context for a training job.
 */
JobContext::JobContext(std::unique_ptr<RunnableModule> modelIn,
                       std::string name, std::unique_ptr<DataLoader> dataLoader,
                       std::shared_ptr<CommunicationHandler> commHandler,
                       std::unique_ptr<TargetShuffler> targetShuffler,
                       int epochsToTrain)
    : model(std::move(modelIn)),
      name(name),
      dataLoader(std::move(dataLoader)),
      commHandler(std::move(commHandler)),
      targetShuffler(std::move(targetShuffler)),
      epochsToTrain(epochsToTrain) {
  if (rtctx->use_fg_graph) {
    iters_before_graph_capture = 50;
  } else {
    iters_before_graph_capture = 5000;
  }
}

/**
 * Destructs context for a training job.
 * Defined here to avoid incomplete type destruction in 'JobContext.h' of
 * some unique_ptr members.
 */
JobContext::~JobContext() {}

#if 0
/**
 * Set the BG job after parsing it from the json file.
 */
int
TaskManager::addBgJob()
{
  if (!rtctx->bg_json_file.empty()) {
    ScheduleTrainingRequest req;
    // std::string path = format("%s/DeepPoolRuntime/lastReq%d.txt", ctx->homedir, ctx->device);
    std::ifstream ifs(rtctx->bg_json_file.c_str());
    req.ParseFromIstream(&ifs);
    ifs.close();
    DP_LOG(NOTICE, "Parsed BG job request from %s.", rtctx->bg_json_file.c_str());
    
    bgJob = rtctx->grpcService->parseAndCreateTrainingTask(&req);
    DP_LOG(NOTICE, "Parsing BG json completed.");
    bgJob->idleCtx.jobType = IdleTimeCtx::BG;
    bgJob->itersToTrain = 500000; // Let it to run as long as we have any idle gaps.
    bgJob->epochsToTrain = 5000; // Let it to run as long as we have any idle gaps.
    bgJob->iters_before_graph_capture = 500000; // Capture will happen by the main job.
    return 1;
  }
  return 0;
}
#endif

#if 0
bool
TaskManager::trainAllTheWayWithBg(JobContext* mainJob)
{
    if (be_bsize > 0 && mainJob->totiters == 0) {
    if (!mainJob->run_with_be) {
      be_controller.Pause();
    } else {
      be_controller.Resume();
    }
  }

  // const int bgItersPerCapture = 2;
  // int fgItersPerCapture = 0;
  const int fgItersPerCapture = rtctx->iters_per_capture; //10;
  int bgItersPerCapture = 0;
  int bgCaptureStartIter = 9999999;
  bool runBgJob = (bool)bgJob && mainJob->model->hasInactiveLayer;
  DP_LOG(NOTICE, "Run bg job during the gap: %d", runBgJob);
  bool stopCapture = false;
  bool jobCompleted = false;
  bool iterFinished = false;
  at::cuda::CUDAGraph mixedGraph;
  while (!jobCompleted) {
    if (iterFinished) {
      // DP_LOG(NOTICE, "Finished %" PRId64 "-th iter before capture.", mainJob->totiters);
      if (mainJob->totiters == mainJob->iters_before_graph_capture) {
        if (mainJob->run_with_be && be_bsize > 0) {
          DP_LOG(NOTICE, "Pausing BE.");
          be_controller.Pause();
        }

        // Prepare BG job.
        if (bgJob) {
          bgJob->model->resetProfileTimers();
          bgJob->model->resetForNewIter();
          bgJob->model->iterInit();
          bgJob->state = JobState::FORWARD;
          bgCaptureStartIter = bgJob->totiters;
        }

        // Start capture.
        c10::cuda::device_synchronize();
        DP_LOG(NOTICE, "Starting capture.");
        mixedGraph.capture_begin();
        mainJob->commHandler->precapture();
        // captureStarted = true;
      }
      // Use different condition... like after 10 BG iterations, finish FG.. 
      if (stopCapture) {
        DP_LOG(NOTICE, "Ending capture.");
        mainJob->commHandler->postcapture();
        mixedGraph.capture_end();
        if (mainJob->run_with_be && be_bsize > 0) {
          DP_LOG(NOTICE, "Resuming BE.");
          be_controller.Resume();
        }
        // fgItersPerCapture = mainJob->totiters - mainJob->iters_before_graph_capture;
        bgItersPerCapture = bgJob->totiters - bgCaptureStartIter;
        jobCompleted = true;
      }
    }

    // if (!runBgJob && mainJob->totiters == rtctx->iters_per_capture + mainJob->iters_before_graph_capture - 1) { //captureStarted) {
    if (mainJob->totiters == rtctx->iters_per_capture + mainJob->iters_before_graph_capture - 1) { //captureStarted) {
      // DP_LOG(NOTICE, "Asking to end capture. (no bg job)");
      stopCapture = true;
    }

    // DP_LOG(NOTICE, "FG job run. remainingIdle: %" PRId64 " us, next bgJobStep: %" PRId64 "",
    //     mainJob->idleCtx.remainingIdleUsec, getNextStepTime(bgJob.get()));
    iterFinished = mainJob->trainSingleStep(&jobCompleted);
#if 0
    while (runBgJob && mainJob->idleCtx.remainingIdleUsec > getNextStepTime(bgJob.get())) {
      bool bgJobCompleted = false;
      bgJob->idleCtx.idleUsecOfMainPtr = &mainJob->idleCtx.remainingIdleUsec;
      DP_LOG(NOTICE, "  BG job run. remainingIdle: %" PRId64 " us, nextStepTime: %" PRId64 " us",
          mainJob->idleCtx.remainingIdleUsec, getNextStepTime(bgJob.get()));
      bool bgIterFinished = bgJob->trainSingleStep(&bgJobCompleted);
      DP_LOG(NOTICE, "    finished a step in %" PRId64 "-th iter (fin:%d). remainingIdle: %" PRId64 " us",
          bgJob->totiters, bgIterFinished, mainJob->idleCtx.remainingIdleUsec);
      // if (bgIterFinished && bgJob->totiters == bgCaptureStartIter + bgItersPerCapture) {
      //   DP_LOG(NOTICE, "Asking to end capture. (with bg job)");
      //   runBgJob = false;
      //   stopCapture = true;
      // }
    }
#endif
  }

  DP_LOG(NOTICE, "mixedGraph => fgItersPerCapture: %d, bgItersPerCapture: %d",
      fgItersPerCapture, bgItersPerCapture);

  // Replay the mixed graph.
  bool measurementStarted = false;
  assert (mainJob->totiters < 200);
  assert (mainJob->itersToTrain > 600);
  while (mainJob->totiters < mainJob->itersToTrain) {
    // DP_LOG(NOTICE, "totiters: %" PRId64 "", mainJob->totiters);
    if (!measurementStarted && mainJob->totiters >= 200) {
      DP_LOG(NOTICE, "measurementStarted. totiters: %" PRId64 "", mainJob->totiters);
      measurementStarted = true;
      rtctx->torch_stream.synchronize();
      mainJob->be_img_start = becounter.load();
      mainJob->start = std::chrono::steady_clock::now();
    }

    static CUDAPipeline p(8);
    p.Lap();
    mixedGraph.replay();
    mainJob->totiters += fgItersPerCapture;
    mainJob->iter += fgItersPerCapture;
    fgcounter += fgItersPerCapture;
    if (bgJob) {
      bgJob->totiters += bgItersPerCapture;
      bgJob->iter += bgItersPerCapture;
    }
  }
  if (bgJob) {
    rtctx->torch_stream.synchronize();
    int bgImg = bgJob->model->globalBatchSize * (bgJob->totiters - bgCaptureStartIter);
    mainJob->be_img_end += bgImg; // Temporarily adding bg tput to be tput.
  }
  return jobCompleted;
}
#endif

void JobContext::printJobStatistics() {
  // mainJob->model->printProfileTimers(warmupIters);
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
