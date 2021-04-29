#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <unistd.h>

#include "be_training.h"
#include "common.h"
#include "model.h"

using Tensor = torch::Tensor;
using namespace std::chrono;

std::vector<double> TimeLayers(std::shared_ptr<TrainableModel> model,
                               long batch_size,
                               std::string cached_timings_file) {
  std::vector<std::vector<double>> timings;
  std::vector<double> layer_time_us;

  auto stream = c10::cuda::getCurrentCUDAStream();

  /* Warmup */
  model->Iterate();
  model->Iterate();
  stream.synchronize();

  if (!cached_timings_file.empty()) {
    std::ifstream infile(cached_timings_file);
    uint64_t idx, micros;
    long batch;
    while (infile >> idx >> batch >> micros) {
      if (batch != batch_size) {
        std::cerr << "Invalid batch size in cached timings file" << std::endl;
        break;
      }
      layer_time_us.push_back(micros);
    }
    infile.close();
    if (layer_time_us.size() == model->GetNumLayers()) return layer_time_us;
    layer_time_us.clear();
    std::cerr << "Rebuilding timing cache" << std::endl;
  }

  std::unique_ptr<at::cuda::CUDAEvent> start;
  auto pre = [&](int layern, Tensor &t) {
    start.reset(new at::cuda::CUDAEvent(0));
    start->record(stream);
  };

  auto post = [&](int layern, Tensor &t) {
    at::cuda::CUDAEvent ev(0);
    ev.record(stream);
    stream.synchronize();
    if (static_cast<size_t>(layern) >= timings.size()) {
      timings.resize(layern + 1);
      timings.at(layern).reserve(1000);
    }
    auto &s = *start.get();
    timings.at(layern).push_back(s.elapsed_time(ev) * 1000.0);
    start.reset();
  };

  model->HookPreLayer(pre);
  model->HookPostLayer(post);

  std::cerr << "Building layer timing set... ";

  for (int i = 0; i < 500; i++) model->Iterate();

  for (auto &ts : timings) {
    auto micros = std::accumulate(ts.begin(), ts.end(), 0);
    layer_time_us.push_back(micros / ts.size());
  }
  std::cerr << "Done building." << std::endl;

  model->HookPreLayer({});
  model->HookPostLayer({});

  if (!cached_timings_file.empty()) {
    std::ofstream outfile(cached_timings_file);
    for (size_t i = 0; i < layer_time_us.size(); i++)
      outfile << i << " " << batch_size << " " << layer_time_us[i] << std::endl;
    outfile.close();
  }

  return layer_time_us;
}

static void ConsumeOrBlock(ExternalController *c, uint64_t micros,
                           uint64_t layern, bool invalidate,
                           TrainingStats *stats) {
  static bool first_round = true;
  static GrantMsg cur_grant;
  uint64_t now;

  if (c == nullptr) return;
  uint64_t nanos = micros * 1000;

  while (invalidate || nanos > cur_grant.granted_nanos) {
    if (c->IsDone()) return;
    invalidate = false;
    if (cur_grant.do_writeback || first_round) {
      first_round = false;
      torch::cuda::synchronize();
#if 0
      now = get_now_ns();
      if (now > cur_grant.clock_ns)
        std::cerr << "Grant done, over by " << now - cur_grant.clock_ns
                  << " ns " << std::endl;
#endif
      c->AckGrant(cur_grant);
    }
    cur_grant = c->BlockNextGrant();
    stats->individual_grants++;
    stats->total_granted_nanos += cur_grant.granted_nanos;
#if 0
    if (cur_grant.granted_nanos < nanos)
      std::cerr << "Got grant for " << last_ns_grant << " neeed grant for "
                << nanos << " layer no " << layern << std::endl;
#endif
  }
  stats->total_used_nanos += nanos;
  cur_grant.granted_nanos -= nanos;
}

void Train(ExternalController *c, std::shared_ptr<TrainableModel> model,
           long bsize, uint64_t iters, std::vector<double> timings,
           TrainingStats *stats) {
  TrainingStats nullstats;
  if (!stats) stats = &nullstats;
  uint64_t last_log_iter = 0, last_total_grants = 0;
  auto last_log = steady_clock::now();
  std::cout << "Train thread created " << getpid() << std::endl << std::flush;
  bool invalidate_grant = false;

  if (c && timings.size() > 0) {
    model->HookPreLayer([&](int layern, torch::Tensor &t) {
      ConsumeOrBlock(c, timings.at(layern), layern, invalidate_grant, stats);
      invalidate_grant = false;
    });
  }

  auto stream = c10::cuda::getCurrentCUDAStream();

  for (uint64_t i = 0; i < iters && (!c || !c->IsDone()); i++) {
    model->Iterate();
    stats->full_iterations++;

    if (steady_clock::now() - last_log > seconds(5)) {
      invalidate_grant = true;
      stream.synchronize();
      auto now = steady_clock::now();
      auto iter_ps = static_cast<double>(i - last_log_iter) /
                     duration_cast<duration<double>>(now - last_log).count();
      last_log_iter = i;
      last_log = now;

      stats->grants_per_sec =
          static_cast<double>(stats->individual_grants - last_total_grants) /
          duration_cast<duration<double>>(now - last_log).count();
      last_total_grants = stats->individual_grants;
      stats->images_per_sec = iter_ps * bsize;

      auto dtime = static_cast<double>(get_now_ns()) / 1000000000.0;
      std::cout << dtime << " Training iter/s: " << iter_ps << " ("
                << iter_ps * bsize << " im/s)" << std::endl
                << std::flush;
    }
  }

  model->HookPreLayer({});
}
