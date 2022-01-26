#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>

using namespace std::chrono;
using usec = duration<double, std::micro>;

#define CAPTURE_REPLAYS 200

static double benchFn(std::function<void()> fn) {
  at::cuda::CUDAGraph graph;

  c10::cuda::getCurrentCUDAStream().synchronize();

  /* capture graph  */
  graph.capture_begin();
  fn();
  graph.capture_end();

  /* benchmark function */
  c10::cuda::getCurrentCUDAStream().synchronize();
  auto start = steady_clock::now();
  for (size_t i = 0; i < CAPTURE_REPLAYS; i++) graph.replay();
  c10::cuda::getCurrentCUDAStream().synchronize();
  auto end = steady_clock::now();
  return duration_cast<usec>(end - start).count() / CAPTURE_REPLAYS;
}

static std::pair<double, double> benchModule(
    torch::jit::script::Module module, std::vector<torch::Tensor> inputs) {
  pybind11::gil_scoped_release no_gil;

  c10::cuda::CUDACachingAllocator::emptyCache();

  /* setup module */
  module.train();

  /* use non-default stream for capturing */
  auto stream = c10::cuda::getStreamFromPool();
  auto origstream = c10::cuda::getCurrentCUDAStream();
  c10::cuda::setCurrentCUDAStream(stream);

  /* prepare inputs */
  std::vector<c10::IValue> v;
  for (auto &i : inputs) v.emplace_back(i.cuda().detach().requires_grad_(true));

  /* benchmark forward pass */
  double fwtime = benchFn([&] { module.forward(v); });

  /* prepare for backward pass */
  auto output = module.forward(v).toTensor();
  auto rand = torch::randn(output.sizes()).cuda().requires_grad_(false);

  /* warmup backward pass */
  output.backward(rand, true);

  /* clear gradients */
  for (const auto &param : module.parameters())
    param.mutable_grad() = torch::Tensor();
  for (auto &vv : v) vv.toTensor().mutable_grad() = torch::Tensor();

  /* benchmark backward pass */
  double bwtime = benchFn([&] { output.backward(rand, false); });

  c10::cuda::setCurrentCUDAStream(origstream);
  return std::make_pair(fwtime, bwtime);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("benchmodule", &benchModule, "Benchmark jit module");
}
