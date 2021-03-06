
#include "BeTask.h"

// clang-format off
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
// clang-format on
#include <ATen/cuda/CUDAEvent.h>
#include <absl/flags/flag.h>
#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "CUDAGraph.h"
#include "GraphPieces.h"
#include "Manager.h"
#include "runtime.h"

ABSL_FLAG(long, be_image_px, 0,
          "number of pixels in image used for be training");
ABSL_FLAG(bool, run_be_as_hipri, false,
          "run be task with high priority stream");
ABSL_FLAG(double, be_graph_split_ms, 0.5, "");

static long bsize;

static std::mutex mtx;
static std::condition_variable cv;
static bool beinited = false;

static std::atomic<uint64_t> becounter{0};
static std::atomic<int> status{2};

uint64_t GetBeCounter() { return becounter.load(); }
bool IsBeEnabled() { return bsize > 0; }

void BePause() {
  auto stat = status.load();
  if (stat == 2) return;
  assert(stat == 0);
  status.store(1);
  while (status.load() != 2) usleep(100);
}

void BeResume() { status.store(0); }

static void BeLap() {
  while (status.load() != 0) {
    int s = 1;
    if (status.load() != 2) status.compare_exchange_strong(s, 2);
    usleep(100);
  }
}

static void InitDone() {
  BeResume();
  std::lock_guard<std::mutex> lk(mtx);
  beinited = true;
  cv.notify_one();
}

static void BeRunner(BeTaskConfig cfg) {
  bool use_graph_partitioner = absl::GetFlag(FLAGS_be_graph_split_ms) > 0.0;
  cfg.sample_per_kernel = bsize;  // DISABLE SPLITTING, unsupported
  assert(bsize % cfg.sample_per_kernel == 0);
  long splitways = bsize / cfg.sample_per_kernel;
  assert(!use_graph_partitioner || splitways == 1);
  assert(!use_graph_partitioner || cfg.use_be_graph);

  torch::jit::script::Module m = torch::jit::load(cfg.be_jit_file);
  m.train();
  m.to(rtctx->c10dev);

  std::vector<torch::Tensor> params;
  for (const auto& p : m.parameters()) params.push_back(p);

  torch::optim::SGD optim(params, torch::optim::SGDOptions(0.1).momentum(0.9));

  bool be_is_hi_pri = absl::GetFlag(FLAGS_run_be_as_hipri);
  long px = absl::GetFlag(FLAGS_be_image_px);
  if (!px)
    px = cfg.be_jit_file.find("inception") == std::string::npos ? 224 : 299;
  auto tensor = torch::rand({bsize, 3, px, px}).to(rtctx->c10dev);

  std::vector<int64_t> splitSizes(splitways, bsize / splitways);
  std::cerr << "split: " << splitSizes << std::endl;
  auto tenss = tensor.split_with_sizes(splitSizes);
  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < tenss.size(); i++)
    streams.push_back(c10::cuda::getStreamFromPool(be_is_hi_pri));
  auto target =
      torch::empty(bsize).uniform_(0, 1000).to(at::kLong).to(rtctx->c10dev);
  auto targs = target.split_with_sizes(splitSizes);

  at::autocast::set_enabled(true);

  assert(static_cast<size_t>(splitways) == tenss.size());
  auto fn = [&] {
    auto orig_stream = c10::cuda::getCurrentCUDAStream();

    for (auto& group : optim.param_groups())
      for (auto& param : group.params()) param.mutable_grad() = torch::Tensor();

    DeepPool::CUDAEvent ev;
    ev.record(orig_stream);
    for (size_t i = 0; i < tenss.size(); i++) {
      auto& st = streams.at(i);
      if (splitways > 1) {
        c10::cuda::setCurrentCUDAStream(st);
        ev.block(st);
      }
      auto ret = m.operator()({tenss.at(i)});
      auto loss = torch::nll_loss(ret.toTensor().log_softmax(1), targs.at(i));
      loss.backward();
      if (splitways > 1) {
        at::cuda::CUDAEvent ev2;
        ev2.record(st);
        ev2.block(orig_stream);
      }
    }

    c10::cuda::setCurrentCUDAStream(orig_stream);
    optim.step();

    at::autocast::clear_cache();
  };

  auto cstream = c10::cuda::getStreamFromPool(false);
  c10::cuda::setCurrentCUDAStream(cstream);

  for (size_t i = 0; i < 50; i++) fn();
  DeepPool::CUDAGraph graph;
  c10::cuda::device_synchronize();
  graph.capture_begin();
  fn();
  graph.capture_end();
  c10::cuda::device_synchronize();

  if (use_graph_partitioner) {
    auto gr = GraphPieces::GraphToExecs(graph.getGRAPH(),
                                        absl::GetFlag(FLAGS_be_graph_split_ms));
    auto gtask =
        std::make_shared<GpuTask>(be_is_hi_pri, cstream, gr->ExtractParts());

    InitDone();
    auto fn = [&]() {
      while (status.load() != 0) {
        gtask->FinishCompletion();
        int s = 1;
        if (status.load() != 2) status.compare_exchange_strong(s, 2);
        usleep(100);
      }
    };
    GpuManager::getInstance()->AddTask(gtask);
    while (true) {
      fn();
      gtask->ExecuteTasks();
      becounter.store(becounter.load() + bsize);
    }
  }

  CUDAPipeline p(1, 1000);
  InitDone();
  while (true) {
    BeLap();
    p.Lap();
    if (cfg.use_be_graph)
      graph.replay();
    else
      fn();
    becounter.store(becounter.load() + bsize);
  }
}

void InitBeTask(BeTaskConfig cfg) {
  bsize = cfg.be_batch_size;
  if (!bsize) return;

  std::thread([=] { BeRunner(cfg); }).detach();
  std::unique_lock<std::mutex> lk(mtx);
  cv.wait(lk, [] { return beinited; });
}
