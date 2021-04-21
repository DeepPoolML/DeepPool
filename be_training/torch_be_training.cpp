#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <sys/eventfd.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <chrono>

#include "be_training.h"
#include "common.h"
#include "model.h"

using Tensor = torch::Tensor;

using namespace std::chrono;
using nsec = std::chrono::duration<uint64_t, std::nano>;
using usec = std::chrono::duration<double, std::micro>;

#undef assert
#define assert(x)      \
  do {                 \
    if (!(x)) abort(); \
  } while (0);

std::string cached_timings_file = "/tmp/training_timings.dat";
SameProcessController *c = nullptr;
int device_no;

void TrainFor(uint64_t micros) {
  pybind11::gil_scoped_release no_gil;
  auto stream = c10::cuda::getCurrentCUDAStream(device_no);
  stream.synchronize();
  c->Grant(micros);
}

static std::thread th;

static void Init(long bsize, int device) {
  pybind11::gil_scoped_release no_gil;
  c = new SameProcessController();

  device_no = device;
  auto model = std::make_shared<TrainableModel>(resnet50(), bsize, device);
  auto v = TimeLayers(model, bsize, cached_timings_file);
  model->HookPreLayer([&, v](int layern, Tensor &t) {
    ConsumeOrBlock(c, v.at(layern), layern);
  });

  th = std::thread([=] { Train(c, model, bsize, -1L, device); });
}

static void Terminate() {
  if (c->IsDone()) return;
  c->Close();
  th.join();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &Init, "Initialize training thread");
  m.def("train_for", &TrainFor, "run training model for specified us");
  m.def("terminate", &Terminate, "terminate");
}
