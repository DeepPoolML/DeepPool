
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <torch/torch.h>

#include "be_training.h"
#include "common.h"
#include "model.h"

using Tensor = torch::Tensor;
using namespace std::chrono;
using nsec = std::chrono::duration<uint64_t, std::nano>;
using usec = std::chrono::duration<double, std::micro>;

std::vector<double> TimeLayers(std::shared_ptr<TrainableModel> model,
                               long batch_size,
                               std::string cached_timings_file) {
  time_point<steady_clock> start;
  std::vector<std::vector<double>> timings;
  std::vector<double> layer_time_us;

  /* Warmup */
  model->Iterate();
  model->Iterate();
  torch::cuda::synchronize();

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

  auto pre = [&](int layern, Tensor &t) { start = steady_clock::now(); };

  auto post = [&](int layern, Tensor &t) {
    torch::cuda::synchronize();
    auto end = steady_clock::now();
    if (static_cast<size_t>(layern) >= timings.size()) {
      timings.resize(layern + 1);
      timings.at(layern).reserve(1000);
    }
    timings.at(layern).push_back(duration_cast<usec>(end - start).count());
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

void ConsumeOrBlock(ExternalController *c, uint64_t micros, uint64_t layern) {
  static bool first_round = true;
  static GrantMsg cur_grant;
  static uint64_t last_ns_grant;
  uint64_t now;

  if (c == nullptr) return;
  uint64_t nanos = micros * 1000;

  while (nanos > last_ns_grant) {
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

    now = get_now_ns();

    if (cur_grant.clock_ns > now) {
      last_ns_grant = cur_grant.clock_ns - now;
#if 0
      if (last_ns_grant < nanos)
        std::cerr << "Got grant for " << last_ns_grant << " neeed grant for "
                  << nanos << " layer no " << layern << std::endl;
#endif
    }
  }
  last_ns_grant -= nanos;
}

void Train(ExternalController *c, std::shared_ptr<TrainableModel> model,
           long bsize, uint64_t iters, int device_no) {
  uint64_t last_log_iter = 0;
  // , last_wasted = 0;
  auto last_log = steady_clock::now();
  std::cout << "Train thread created " << getpid() << std::endl << std::flush;
  auto stream = c10::cuda::getStreamFromPool(false, device_no);
  c10::cuda::setCurrentCUDAStream(stream);

  for (uint64_t i = 0; i < iters && (!c || !c->IsDone()); i++) {
    model->Iterate();

    if (steady_clock::now() - last_log > seconds(5)) {
      stream.synchronize();
      auto now = steady_clock::now();
      auto iter_ps = static_cast<double>(i - last_log_iter) /
                     duration_cast<duration<double>>(now - last_log).count();
      last_log_iter = i;
      last_log = now;

      // auto waste = wasted_time_ns - last_wasted;
      // last_wasted = wasted_time_ns;

      auto dtime = static_cast<double>(get_now_ns()) / 1000000000.0;
      std::cout << dtime << " Training iter/s: " << iter_ps << " ("
                << iter_ps * bsize
                << " im/s)"
                // << " -- wasted ns: " << waste
                << std::endl
                << std::flush;
    }
  }
}
