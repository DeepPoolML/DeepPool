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
#include "rpcService.h"

#include <cuda_profiler_api.h>

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>

using Cycles = RAMCloud::Cycles;

class CUDAPipeline {
 public:
  CUDAPipeline(size_t depth) : depth_(depth) {}
  void Lap() {
    if (cur_idx_++ % depth_ != 0) return;
    while (!ev_.query()) usleep(100);
    ev_ = at::cuda::CUDAEvent();
    ev_.record();
  }

 private:
  size_t depth_;
  size_t cur_idx_{0};
  at::cuda::CUDAEvent ev_;
};

class BeRunner {
public:
  void Lap() {
    while (status.load() != 0) {
      int s = 1;
      if (status.load() != 2) status.compare_exchange_strong(s, 2);
      usleep(100);
    }
  }
  void Pause() {
    auto stat = status.load();
    if (stat == 2) return;
    assert(stat == 0);
    status.store(1);
    while (status.load() != 2) usleep(100);
  }
  void Resume() {
    status.store(0);
  }
private:
  std::atomic<int> status{2};
};


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
  , itersToTrain(1900) // = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
  , state(JobState::INIT)
  , timers()
{
  if (rtctx->use_fg_graph) {
    iters_before_graph_capture = 50;
  } else {
    iters_before_graph_capture = 5000;
  }

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

static std::atomic<uint64_t> fgcounter{0};
static std::atomic<uint64_t> becounter{0};
static BeRunner be_controller;
static long be_bsize = 0;

/* tremendous WIP */
void BeRunner(long bsize) {
  be_bsize = bsize;
  int samplePerKernel = rtctx->samplePerKernel;
  assert(bsize % samplePerKernel == 0);
  long splitways = bsize / samplePerKernel;

  torch::jit::script::Module m = torch::jit::load(rtctx->be_jit_file);
  m.train();
  m.to(rtctx->c10dev);

  std::vector<torch::Tensor> params;
  for (const auto &p : m.parameters()) params.push_back(p);

  torch::optim::SGD optim(params, torch::optim::SGDOptions(0.1).momentum(0.9));

  long px = rtctx->be_jit_file.find("inception") == std::string::npos ? 224 : 299;
  auto tensor = torch::rand({bsize, 3, px, px}).to(rtctx->c10dev);

  std::vector<int64_t> splitSizes(splitways, bsize / splitways);
  std::cerr << "split: " << splitSizes << std::endl;
  auto tenss = tensor.split_with_sizes(splitSizes);
  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < tenss.size(); i++) streams.push_back(c10::cuda::getStreamFromPool(false));
  auto target =
        torch::empty(bsize).uniform_(0, 1000).to(at::kLong).to(rtctx->c10dev);
  auto targs = target.split_with_sizes(splitSizes);

  at::autocast::set_enabled(true);

  auto fn = [&] {
    auto orig_stream = c10::cuda::getCurrentCUDAStream();
    optim.zero_grad();
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
    optim.step();

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
  {
    std::lock_guard<std::mutex> lk(mtx);
    beinited = true;
  }

  be_controller.Resume();
  cv.notify_one();

  CUDAPipeline p(1);

  while (true) {
    be_controller.Lap();
    p.Lap();
    if (rtctx->use_be_graph)
      graph.replay();
    else
      fn();
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
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, []{return beinited;});
  }

  std::thread([&] {
    using namespace std::chrono;
    size_t lastc = becounter.load();
    size_t lastfg = fgcounter.load();
    auto lastt = steady_clock::now();
    while (true) {
      sleep(1);
      size_t newtr = becounter.load();
      size_t newfg = fgcounter.load();
      auto now = steady_clock::now();
      auto s = duration_cast<seconds>(now - lastt).count();
      std::cerr << "BE im/s: " << (newtr - lastc) / s << " FG iter/s: " << (newfg - lastfg) / s << std::endl;
      lastt = now;
      lastc = newtr;
      lastfg = newfg;
    }
  }).detach();
}

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

  JobContext* mainJob = jobList[0].get();
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
    iterFinished = trainSingleStep(mainJob, &jobCompleted);
    while (runBgJob && mainJob->idleCtx.remainingIdleUsec > getNextStepTime(bgJob.get())) {
      bool bgJobCompleted = false;
      bgJob->idleCtx.idleUsecOfMainPtr = &mainJob->idleCtx.remainingIdleUsec;
      DP_LOG(NOTICE, "  BG job run. remainingIdle: %" PRId64 " us, nextStepTime: %" PRId64 " us",
          mainJob->idleCtx.remainingIdleUsec, getNextStepTime(bgJob.get()));
      bool bgIterFinished = trainSingleStep(bgJob.get(), &bgJobCompleted);
      DP_LOG(NOTICE, "    finished a step in %" PRId64 "-th iter (fin:%d). remainingIdle: %" PRId64 " us",
          bgJob->totiters, bgIterFinished, mainJob->idleCtx.remainingIdleUsec);
      // if (bgIterFinished && bgJob->totiters == bgCaptureStartIter + bgItersPerCapture) {
      //   DP_LOG(NOTICE, "Asking to end capture. (with bg job)");
      //   runBgJob = false;
      //   stopCapture = true;
      // }
    }
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
  rtctx->torch_stream.synchronize();
  mainJob->end = std::chrono::steady_clock::now();
  mainJob->be_img_end = becounter.load();
  if (bgJob) {
    int bgImg = bgJob->model->globalBatchSize * (bgJob->totiters - bgCaptureStartIter);
    mainJob->be_img_end += bgImg; // Temporarily adding bg tput to be tput.
  }

  // if (jobCompleted) {
  printJobStatistics(mainJob);
  jobList.erase(jobList.begin());
  DP_LOG(NOTICE, "Removed the completed job. Remaining: %lu", jobList.size());
  // }
  return 1;
}

void
TaskManager::printJobStatistics(JobContext* job)
{
  size_t warmupIters = 200;
  // mainJob->model->printProfileTimers(warmupIters);
  job->model->printLayerInGraphTimes();
  size_t totiters = job->totiters - warmupIters;
  using msec = std::chrono::duration<double, std::milli>;
  double elapsed_ms = std::chrono::duration_cast<msec>(job->end - job->start).count();
  double total_iter_ms = elapsed_ms / (double)totiters;
  double total_iter_ps = 1e3 / total_iter_ms;
  double be_img_ps = job->be_img_end - job->be_img_start;
  be_img_ps = 1e3 * be_img_ps / elapsed_ms;
  DP_LOG(NOTICE, "A training job %s is completed (%lu iters, %.2f ms/iter, %.2f iter/s, %.2f be img/s)."
      " AverageTiming (ms) => zero: %.1f, load:%.1f, fp:%.1f, loss:%.1f, bp:%.1f, opt: %.1f, iter:%.1f"
      " P50 (ms) => fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f",
      job->name.c_str(), totiters, total_iter_ms, total_iter_ps, be_img_ps,
      job->timers[CT_ZERO].getAvg(warmupIters),
      job->timers[CT_LOAD].getAvg(warmupIters),
      job->timers[CT_FP].getAvg(warmupIters),
      job->timers[CT_LOSS].getAvg(warmupIters),
      job->timers[CT_BP].getAvg(warmupIters),
      job->timers[CT_OPT].getAvg(warmupIters),
      job->timers[CT_STOP].getAvg(warmupIters),
      job->timers[CT_FP].getP50(warmupIters),
      job->timers[CT_LOSS].getP50(warmupIters),
      job->timers[CT_BP].getP50(warmupIters),
      job->timers[CT_STOP].getP50(warmupIters));
  // DP_LOG(NOTICE, " -- detachTime: %" PRIu64" us", job->model->detachTimer.avgMicros());
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
int
TaskManager::trainSingleStep(JobContext* job, bool* jobCompleted)
{
  if (job->state == JobState::INIT) {
    if (job->totiters == job->profile_iter_start)
      CUDA_API_CALL(cudaProfilerStart());

    if (job->iter >= job->itersToTrain) {
      DP_LOG(DEBUG, "epoch is completed.");
      job->iter = 0;
      job->epoch++;
    }
    
    if (rtctx->profile && job->totiters == job->iters_before_graph_capture) {
      *jobCompleted = true;
      return 0;
    }    
    // if (job->epoch >= job->epochsToTrain || 
    //     (rtctx->profile && job->totiters == job->iters_before_graph_capture)) {
    //   DP_LOG(DEBUG, "training is completed.");
    //   rtctx->torch_stream.synchronize();
    //   job->end = std::chrono::steady_clock::now();
    //   job->be_img_end = becounter.load();
    //   *jobCompleted = true;
    //   return 0;
    // }

    job->model->resetProfileTimers();
    for (int tpIdx = CT_NUM_OF_EVENTS - 1; tpIdx >= CT_START; --tpIdx) {
      DP_LOG(DEBUG, "timer.saveAndReset() for %d. recorded:%d", tpIdx, job->timers[tpIdx].isRecorded());
      job->timers[tpIdx].saveAndReset();
    }
    job->timers[CT_START].record();
    DP_LOG(DEBUG, "JobState::INIT.");

    job->model->iterInit();

    /* start graph capture */
    // if (job->totiters == job->iters_before_graph_capture) {
    //   if (job->run_with_be && be_bsize > 0) be_controller.Pause();
    //   c10::cuda::device_synchronize();
    //   DP_LOG(NOTICE, "Starting capture.");
    //   job->model->graph.capture_begin();
    //   job->commHandler->precapture();
    // } else if (job->totiters >= rtctx->iters_per_capture + job->iters_before_graph_capture) {
    //   /* skip to forward phase */
    //   job->state = JobState::FORWARD;
    //   return 0;
    // }

    job->optimizer->zero_grad();
    job->timers[CT_ZERO].record();

    job->timers[CT_LOAD].record();
    job->state = JobState::FORWARD;
    DP_LOG(DEBUG, "Foward pass is starting soon.");
  } else if (job->state == JobState::FORWARD) {
    DP_LOG(DEBUG, "JobState::FORWARD.");
    // if (job->totiters >= rtctx->iters_per_capture + job->iters_before_graph_capture) {
    //   DP_LOG(DEBUG, "Replay iter.");

    //   if ((job->totiters - job->iters_before_graph_capture) % rtctx->iters_per_capture) {
    //     /* advance state machine, no replay */
    //     job->state = JobState::FINISH;
    //     return 0;
    //   }

    //   static CUDAPipeline p(8);
    //   p.Lap();
    //   job->model->graph.replay();
    //   job->state = JobState::FINISH;
    //   return 0;
    // }
    
    bool capture = rtctx->profile && job->totiters == job->iters_before_graph_capture - 1;
    JobStatus status = job->model->forwardAStep(capture);

    if (status == COMPLETED) {
      job->timers[CT_FP].record();
      // TODO: add a loss calculation here? or as another state?
      DP_LOG(DEBUG, "Foward pass is completed. Calculating loss.");
      
      job->model->loss();
      job->timers[CT_LOSS].record();
      assert(job->model->layerQ.empty());
      job->model->layerQ.push_back(&job->model->layers.back());
      DP_LOG(DEBUG, "Moving to backward pass.");
      job->state = JobState::BACKWARD;
    }
  } else if (job->state == JobState::BACKWARD) {
    DP_LOG(DEBUG, "JobState::BACKWARD.");
    
    bool capture = rtctx->profile && job->totiters == job->iters_before_graph_capture - 1;
    JobStatus status = job->model->backwardAStep(capture);
    // TODO: get idle time for backward separately.
    
    if (status == COMPLETED) {
      job->timers[CT_BP].record();
      job->state = JobState::SYNC;
      DP_LOG(DEBUG, "Backward pass is completed. Moving to gradient all-reduce.");
    }
  } else if (job->state == JobState::SYNC) {
    DP_LOG(DEBUG, "JobState::SYNC.");
    // DP_LOG(DEBUG, "All-reduce parameter sync is not implemented yet.");
    job->model->gradientSync();
    job->timers[CT_SYNC].record();
    job->state = JobState::STEP;
  } else if (job->state == JobState::STEP) {
    DP_LOG(DEBUG, "JobState::STEP");
    job->optimizer->step();
    job->timers[CT_OPT].record();
    job->state = JobState::FINISH;
  } else if (job->state == JobState::FINISH) {
    DP_LOG(DEBUG, "JobState::FINISH");

    if (job->totiters == job->profile_iter_start + job->niter_to_profile)
      CUDA_API_CALL(cudaProfilerStop());

    // if (job->totiters == rtctx->iters_per_capture - 1 + job->iters_before_graph_capture) {
    //   job->commHandler->postcapture();
    //   job->model->graph.capture_end();
    //   if (job->run_with_be && be_bsize > 0) be_controller.Resume();
    //   DP_LOG(NOTICE, "Ending capture.");
    // }
    job->totiters++;
    job->iter++;
    fgcounter++;

    job->state = JobState::INIT;
    job->timers[CT_STOP].record();
    return 1;
  }
  return 0;
}

int64_t
TaskManager::getNextStepTime(JobContext* job) {
  if (job->state == JobState::INIT) {
    return 1;
  } else if (job->state == JobState::FORWARD) {
    Layer* layer = job->model->layerQ.front();
    return layer->fwUsec;
  } else if (job->state == JobState::BACKWARD) {
    Layer* layer = job->model->layerQ.front();
    return layer->bwUsec;
  } else if (job->state == JobState::SYNC) {
    return 1;
  } else if (job->state == JobState::STEP) {
    return 1000;
  } else if (job->state == JobState::FINISH) {
    return 1;
  }
  return 0;
}