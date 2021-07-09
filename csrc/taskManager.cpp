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
  , itersToTrain(1) // = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
  , state(JobState::INIT)
{
  // self.dataLoaderIt = iter(self.dataLoader) if dataLoader != None else None
  // self.criterion = nn.CrossEntropyLoss().cuda(device) if criterion == None else criterion
}

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
  Cycles::sleep(1000000);
  if (jobList.empty()) {
    return 0;
  }

  int jobsScheduled = 0;
  JobContext* mainJob = jobList[0].get();
  bool jobCompleted = false;
  trainSingleStep(mainJob, &jobCompleted);
  if (jobCompleted) {
    jobList.erase(jobList.begin());
    DP_LOG(NOTICE, "A training job %s is completed. Remaining: %d",
        mainJob->name.c_str(), static_cast<int>(jobList.size()));
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
  if (job->iter >= job->itersToTrain) {
    DP_LOG(DEBUG, "iteration is completed.");
    job->epoch++;
  }
  if (job->epoch >= job->epochsToTrain) {
    DP_LOG(DEBUG, "training is completed.");
    *jobCompleted = true;
  }
  if (job->state == JobState::INIT) {
    DP_LOG(DEBUG, "JobState::INIT.");
    // TODO: load data from real data loader.
    auto x = torch::randn({16, 3, 224, 224});
    x.to(job->device);
    job->model->iterInit(x);
    job->state = JobState::FORWARD;
  } else if (job->state == JobState::FORWARD) {
    DP_LOG(DEBUG, "JobState::FORWARD.");
    bool completed = job->model->forwardAStep();
    if (completed) {
      // TODO: add a loss calculation here? or as another state?
      job->state = JobState::BACKWARD;
    }
  } else if (job->state == JobState::BACKWARD) {
    DP_LOG(DEBUG, "JobState::BACKWARD.");
    DP_LOG(WARNING, "Backward pass is not implemented yet.");
    job->state = JobState::SYNC;
  } else if (job->state == JobState::SYNC) {
    DP_LOG(DEBUG, "JobState::SYNC.");
    job->iter++;
    DP_LOG(WARNING, "All-reduce parameter sync is not implemented yet.");
    job->state = JobState::INIT;
  }
}