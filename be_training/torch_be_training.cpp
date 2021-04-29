#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "be_training.h"
#include "common.h"
#include "model.h"

std::string cached_timings_file = "/tmp/training_timings.dat";
SameProcessController *c = nullptr;
TrainingStats stats;

static void TrainFor(uint64_t micros) {
  pybind11::gil_scoped_release no_gil;
  c->Grant(micros);
}

static std::thread th;

static void Init(long bsize, int device) {
  pybind11::gil_scoped_release no_gil;
  c = new SameProcessController();

  c10::cuda::CUDAGuard guard_(device);

  auto model = std::make_shared<TrainableModel>(resnet34(), bsize, device);
  auto v = TimeLayers(model, bsize, cached_timings_file);
  th = std::thread([=] {
    c10::cuda::set_device(device);
    auto stream = c10::cuda::getStreamFromPool(false, device);
    c10::cuda::setCurrentCUDAStream(stream);
    Train(c, model, bsize, -1L, v, &stats);
  });
}

static void Terminate() {
  if (!c || c->IsDone()) return;
  c->Close();
  pybind11::gil_scoped_release no_gil;
  th.join();
}

static auto QueryStats() {
  py::dict dict;
  dict["images_per_sec"] = stats.images_per_sec;
  dict["grants_per_sec"] = stats.grants_per_sec;
  dict["full_iterations"] = stats.full_iterations;
  dict["individual_grants"] = stats.individual_grants;
  dict["total_granted_nanos"] = stats.total_granted_nanos;
  dict["total_used_nanos"] = stats.total_used_nanos;
  return dict;
}

auto cleanup_callback = []() { Terminate(); };

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &Init, "Initialize training thread");
  m.def("train_for", &TrainFor, "run training model for specified us");
  m.def("terminate", &Terminate, "terminate");
  m.def("query", &QueryStats, "query");

  m.add_object("_cleanup", py::capsule(cleanup_callback));
}
