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

#include "taskManager.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include "Cycles.h"
#include "utils.h"
#include "runnableModule.h"
#include "runtime.h"
#include "communication.h"
#include "logger.h"

using Cycles = RAMCloud::Cycles;

/**
 * Contructs context for a training job.
 */
JobContext::JobContext(std::unique_ptr<RunnableModule> model, std::string name,
    std::unique_ptr<DataLoader> dataLoader,
    std::unique_ptr<CommunicationHandler> commHandler,
    std::unique_ptr<TargetShuffler> targetShuffler,
    c10::Device device,
    int epochsToTrain,
    std::unique_ptr<torch::optim::Optimizer> optimizer)
  : model(std::move(model))
  , name(name)
  , dataLoader(std::move(dataLoader))
  , commHandler(std::move(commHandler))
  , targetShuffler(std::move(targetShuffler))
  , epochsToTrain(epochsToTrain)
  , optimizer(std::move(optimizer))
  , device(device)
  , epoch(0)
  , iter(0)
  , itersToTrain(1000) // = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
  , state(JobState::INIT)
  , timers()
  , modelToVerify()
{
  // self.dataLoaderIt = iter(self.dataLoader) if dataLoader != None else None
  // self.criterion = nn.CrossEntropyLoss().cuda(device) if criterion == None else criterion
  
  timers.reserve(CT_NUM_OF_EVENTS);
  timers.emplace_back();
  CudaTimer* startTimer = &timers.back();
  CudaTimer* lastTimer = startTimer;
  for (int i = 1; i < CT_NUM_OF_EVENTS - 1; ++i) {
    timers.emplace_back(lastTimer);
    lastTimer = &timers.back();
  }
  timers.emplace_back(startTimer); // CT_STOP measures from CT_START to CT_STOP;
}

/**
 * Destructs context for a training job.
 * Defined here to avoid incomplete type destruction in 'taskManager.h' of
 * some unique_ptr members.
 */
JobContext::~JobContext() {}

/**
 * Constructs a TaskManager.
 */
TaskManager::TaskManager(RuntimeContext* rtctx)
  : rtctx(rtctx)
  , _mutex()
  , jobList()
{
  rtctx->taskManager = this;
}

/**
 * Adds a new training job submitted by coordinator.
 * 
 * \return  The number of jobs currently scheduled.
 */
int
TaskManager::addTrainingJob(std::unique_ptr<JobContext> job)
{
  std::lock_guard<std::mutex> lock(_mutex);
  jobList.push_back(std::move(job));
  DP_LOG(LogLevel::NOTICE, "Added a new job. %s", jobList.back()->name.c_str());
  return jobList.size();
}

/**
 * A poller to make a progress on training tasks.
 *
 * \return The number of jobs that are executed (or scheduled to CUDA).
 */
int
TaskManager::poll()
{
  std::lock_guard<std::mutex> lock(_mutex);
  // Cycles::sleep(1000000);
  if (jobList.empty()) {
    return 0;
  }

  int jobsScheduled = 0;
  JobContext* mainJob = jobList[0].get();
  bool jobCompleted = false;
  trainSingleStep(mainJob, &jobCompleted);
  if (jobCompleted) {
    size_t warmupIters = 100;
    DP_LOG(NOTICE, "A training job %s is completed (%d iters)."
        " AverageTiming (ms) => load:%.1f, fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f"
        " P50 (ms) => fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f",
        mainJob->name.c_str(), mainJob->timers[CT_FP].count(),
        mainJob->timers[CT_LOAD].getAvg(warmupIters),
        mainJob->timers[CT_FP].getAvg(warmupIters),
        mainJob->timers[CT_LOSS].getAvg(warmupIters),
        mainJob->timers[CT_BP].getAvg(warmupIters),
        mainJob->timers[CT_STOP].getAvg(warmupIters),
        mainJob->timers[CT_FP].getP50(warmupIters),
        mainJob->timers[CT_LOSS].getP50(warmupIters),
        mainJob->timers[CT_BP].getP50(warmupIters),
        mainJob->timers[CT_STOP].getP50(warmupIters));

    jobList.erase(jobList.begin());
    DP_LOG(NOTICE, "Removed the completed job. Remaining: %d",
        static_cast<int>(jobList.size()));
  }
  jobsScheduled++;
  return jobsScheduled;
}

/**
 * A helper to run a job.
 * 
 * \param job   a context for the job to train.
 * \param[out] jobCompleted 
 *    will be filled with true if the job is completely finished.
 * 
 * \return    returns non-zero if it actively worked.
 */
int
TaskManager::trainSingleStep(JobContext* job, bool* jobCompleted)
{
  if (job->iter > job->itersToTrain) {
    DP_LOG(DEBUG, "epoch is completed.");
    job->iter = 0;
    job->epoch++;
  }
  if (job->epoch >= job->epochsToTrain) {
    DP_LOG(DEBUG, "training is completed.");
    *jobCompleted = true;
    return 0;
  }
  if (job->state == JobState::INIT) {
    for (int tpIdx = CT_NUM_OF_EVENTS - 1; tpIdx >= CT_START; --tpIdx) {
      DP_LOG(DEBUG, "timer.saveAndReset() for %d. recorded:%d", tpIdx, job->timers[tpIdx].isRecorded());
      job->timers[tpIdx].saveAndReset();
    }
    job->timers[CT_START].record();

    DP_LOG(DEBUG, "JobState::INIT.");
    // TODO: load data from real data loader.
    auto x = torch::randn({16, 3, 224, 224});
    // x = x.to(job->device);
    // TODO: replace this fake targets with real ones.
    auto targetOpts = torch::TensorOptions().dtype(torch::kInt64);
    auto targets = torch::randint(/*low=*/0, /*high=*/1000, {16}, targetOpts);
    // targets = targets.to(job->device);
    // DP_LOG(DEBUG, "targets: %s dim: %d sizes: %d", tsrToStr(targets).c_str(),
    //     (int)targets.dim(), (int)targets.sizes().size());

    job->model->iterInit(x, targets);
    job->state = JobState::FORWARD;

    if (rtctx->verify && job->iter == 0) {
      auto x2 = x.clone();
      x2 = x2.to(job->device);
      DP_LOG(NOTICE, "Verify two inputs.. fpInput: %s x2: %s",
            tsrSizeToStr(job->model->fpInput).c_str(),
            tsrSizeToStr(x2).c_str());
      // DP_LOG(NOTICE, "fpInput: %s", tsrToStr(job->model->fpInput).c_str());
      // DP_LOG(NOTICE, "x2:      %s", tsrToStr(x2).c_str());
      std::vector<torch::jit::IValue> inputVec;
      inputVec.push_back(x2);
      job->outputToVerify = job->modelToVerify.forward(inputVec).toTensor();
    }
    // cudaDeviceSynchronize();

    job->timers[CT_LOAD].record();
    DP_LOG(DEBUG, "Foward pass is starting soon.");
  } else if (job->state == JobState::FORWARD) {
    DP_LOG(DEBUG, "JobState::FORWARD.");
    bool completed = job->model->forwardAStep();

    if (completed) {
      job->timers[CT_FP].record();
      // TODO: add a loss calculation here? or as another state?
      DP_LOG(DEBUG, "Foward pass is completed. Calculating loss.");

      if (rtctx->verify && job->iter == 0) {
        DP_LOG(NOTICE, "Verify two outputs.. fpOutput: %s outputToVerify: %s",
            tsrSizeToStr(job->model->fpOutput).c_str(),
            tsrSizeToStr(job->outputToVerify).c_str());
        // DP_LOG(NOTICE, "fpOutput:       %s", tsrToStr(job->model->fpOutput).c_str());
        // DP_LOG(NOTICE, "outputToVerify: %s", tsrToStr(job->outputToVerify).c_str());
      }
      
      job->model->loss();
      job->timers[CT_LOSS].record();
      assert(job->model->layerQ.empty());
      job->model->layerQ.push_back(&job->model->layers.back());
      DP_LOG(DEBUG, "Moving to backward pass.");
      job->state = JobState::BACKWARD;
    }
  } else if (job->state == JobState::BACKWARD) {
    DP_LOG(DEBUG, "JobState::BACKWARD.");
    // DP_LOG(WARNING, "Backward pass is not implemented yet.");
    bool completed = job->model->backwardAStep();
    if (completed) {
      job->timers[CT_BP].record();
      job->state = JobState::SYNC;
      DP_LOG(DEBUG, "Backward pass is completed. Moving to gradient all-reduce.");
    }
  } else if (job->state == JobState::SYNC) {
    DP_LOG(DEBUG, "JobState::SYNC.");
    job->iter++;
    DP_LOG(DEBUG, "All-reduce parameter sync is not implemented yet.");
    job->timers[CT_SYNC].record();
    job->state = JobState::INIT;
    job->timers[CT_STOP].record();
  }
  return 1;
}