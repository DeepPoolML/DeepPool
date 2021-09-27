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

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>


using Cycles = RAMCloud::Cycles;

/**
 * Contructs context for a training job.
 */
JobContext::JobContext(std::unique_ptr<RunnableModule> modelIn, std::string name,
    std::unique_ptr<DataLoader> dataLoader,
    std::unique_ptr<CommunicationHandler> commHandler,
    std::unique_ptr<TargetShuffler> targetShuffler,
    c10::Device device,
    int epochsToTrain,
    std::unique_ptr<torch::optim::Optimizer> optimizer)
  : model(std::move(modelIn))
  , name(name)
  , dataLoader(std::move(dataLoader))
  , commHandler(std::move(commHandler))
  , targetShuffler(std::move(targetShuffler))
  , epochsToTrain(epochsToTrain)
  , optimizer(std::move(optimizer))
  , device(device)
  , epoch(0)
  , iter(0)
  , itersToTrain(200) // = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
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

  // Initialize timers.
  model->initProfileTimers(&timers[CT_LOAD], &timers[CT_LOSS]);
}

/**
 * Destructs context for a training job.
 * Defined here to avoid incomplete type destruction in 'taskManager.h' of
 * some unique_ptr members.
 */
JobContext::~JobContext() {}

#include <condition_variable>
#include <mutex>

std::mutex mtx;
std::condition_variable cv;
bool beinited = false;

std::atomic<uint64_t> becounter{0};

/* tremendous WIP */
void BeRunner(long bsize) {
  assert(bsize % 32 == 0);
  long splitways = bsize / 32;

  std::string filename("/home/friedj/mlsf/multimodel/resnet_dropped.jit");
  torch::jit::script::Module m = torch::jit::load(filename);
  m.train();
  m.to(torch::Device("cuda:0"));

  long px = filename.find("inception") == std::string::npos ? 224 : 299;
  auto tensor = torch::rand({bsize, 3, px, px});
  tensor = tensor.to(torch::Device("cuda:0"));

  // assert(bsize % splitways == 0);
  std::vector<int64_t> splitSizes(splitways, bsize / splitways);
  std::cerr << "split: " << splitSizes << std::endl;
  auto tenss = tensor.split_with_sizes(splitSizes);
  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < tenss.size(); i++) streams.push_back(c10::cuda::getStreamFromPool(false));
  auto target =
        torch::empty(bsize).uniform_(0, 1000).to(at::kLong).to(torch::Device("cuda:0"));
  auto targs = target.split_with_sizes(splitSizes);

  at::autocast::set_enabled(true);

  auto fn = [&] {
    auto orig_stream = c10::cuda::getCurrentCUDAStream();
    at::cuda::CUDAEvent ev;
    ev.record(orig_stream);
    for (size_t i = 0; i < tenss.size(); i++) {
      auto &st = streams.at(i);
      c10::cuda::setCurrentCUDAStream(st);
      ev.block(st);
      auto ret = m.operator()({tenss.at(i)});
      auto loss = torch::nll_loss(ret.toTensor().log_softmax(1), targs.at(i));
      loss.backward();
      at::cuda::CUDAEvent ev2;
      ev2.record(st);
      ev2.block(orig_stream);
    }

    c10::cuda::setCurrentCUDAStream(orig_stream);

    at::autocast::clear_cache();
  };

  auto cstream = c10::cuda::getStreamFromPool(false);
  c10::cuda::setCurrentCUDAStream(cstream);

  for (size_t i = 0; i < 50; i++) fn();
  at::cuda::CUDAGraph graph;
  c10::cuda::device_synchronize();
  graph.capture_begin();
  fn();
  graph.capture_end();
  c10::cuda::device_synchronize();
  // auto cgr_exec_ = graph.getGEXEC();
  {
    std::lock_guard<std::mutex> lk(mtx);
    beinited = true;
  }
  cv.notify_one();

  cudaEvent_t waiter;
  CUDACHECK(cudaEventCreateWithFlags(&waiter, cudaEventDisableTiming));

  while (true) {
    graph.replay();
    // CUDACHECK(cudaGraphLaunch(cgr_exec_, cstream.stream()));
    CUDACHECK(cudaEventRecord(waiter, cstream.stream()));
    while (cudaEventQuery(waiter) != cudaSuccess) { usleep(100); }
    becounter.store(becounter.load() + bsize);
  }
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
  if (rtctx->be_batch_size > 0) {
    long bsize = rtctx->be_batch_size;
    std::thread([=] { BeRunner(bsize); }).detach();
    std::thread([&] {
      using namespace std::chrono;
      size_t lastc = becounter.load();
      auto lastt = steady_clock::now();
      while (true) {
        sleep(5);
        size_t newtr = becounter.load();
        auto now = steady_clock::now();
        auto s = duration_cast<seconds>(now - lastt).count();
        std::cerr << "Trained " << (newtr - lastc) / s << " img/s" << std::endl;
        lastt = now;
        lastc = newtr;
      }
    }).detach();
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, []{return beinited;});
  }
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
    mainJob->model->printProfileTimers(warmupIters);
    size_t totiters = mainJob->totiters - warmupIters;
    using msec = std::chrono::duration<double, std::milli>;
    double elapsed_ms = std::chrono::duration_cast<msec>(mainJob->end - mainJob->start).count();
    double total_iter_ms = elapsed_ms / (double)totiters;
    double total_iter_ps = 1e3 / total_iter_ms;
    double be_img_ps = mainJob->be_img_end - mainJob->be_img_start;
    be_img_ps = 1e3 * be_img_ps / elapsed_ms;
    DP_LOG(NOTICE, "A training job %s is completed (%lu iters, %.2f ms/iter, %.2f iter/s, %.2f be img/s)."
        " AverageTiming (ms) => load:%.1f, fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f"
        " P50 (ms) => fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f",
        mainJob->name.c_str(), totiters, total_iter_ms, total_iter_ps, be_img_ps,
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
  /* record starting point for BE training */
  if (job->totiters == 100) {
    rtctx->torch_stream.synchronize();
    job->be_img_start = becounter.load();
    job->start = std::chrono::steady_clock::now();
  }

  if (job->iter >= job->itersToTrain) {
    DP_LOG(DEBUG, "epoch is completed.");
    job->iter = 0;
    job->epoch++;
  }
  if (job->epoch >= job->epochsToTrain) {
    DP_LOG(DEBUG, "training is completed.");
    rtctx->torch_stream.synchronize();
    job->end = std::chrono::steady_clock::now();
    job->be_img_end = becounter.load();
    *jobCompleted = true;
    return 0;
  }
  if (job->state == JobState::INIT) {
    job->model->resetProfileTimers();
    for (int tpIdx = CT_NUM_OF_EVENTS - 1; tpIdx >= CT_START; --tpIdx) {
      DP_LOG(DEBUG, "timer.saveAndReset() for %d. recorded:%d", tpIdx, job->timers[tpIdx].isRecorded());
      job->timers[tpIdx].saveAndReset();
    }
    job->timers[CT_START].record();

    DP_LOG(DEBUG, "JobState::INIT.");

    job->model->iterInit();
    job->state = JobState::FORWARD;

    if (job->totiters == job->iters_before_graph_capture) {
      c10::cuda::device_synchronize();
      DP_LOG(NOTICE, "Starting capture.");
      job->model->graph.capture_begin();
      job->commHandler->precapture();
    } else if (job->iter > job->totiters) {
      DP_LOG(DEBUG, "Replay iter.");
      job->model->graph.replay();
      c10::cuda::device_synchronize(); // TODO remove me
      job->state = JobState::SYNC;
      return 1;
    }


    if (rtctx->verify && job->iter == 0) {
      auto x2 = job->model->fpInput.clone();
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
    if (job->iter == job->iters_before_graph_capture) {
      job->commHandler->postcapture();
      job->model->graph.capture_end();
      DP_LOG(NOTICE, "Ending capture.");
    }

    job->totiters++;
    job->iter++;
    DP_LOG(DEBUG, "All-reduce parameter sync is not implemented yet.");
    job->timers[CT_SYNC].record();
    job->state = JobState::INIT;
    job->timers[CT_STOP].record();
  }
  return 1;
}