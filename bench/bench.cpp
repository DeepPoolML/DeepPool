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

#define WAMRUP_ITERS 10
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

static double benchLossTime(torch::Tensor output, torch::Tensor target,
                            std::string kind) {
  pybind11::gil_scoped_release no_gil;

  c10::cuda::CUDACachingAllocator::emptyCache();

  output = output.cuda().detach().requires_grad_(true);
  /* use non-default stream for capturing */
  auto stream = c10::cuda::getStreamFromPool();
  auto origstream = c10::cuda::getCurrentCUDAStream();
  c10::cuda::setCurrentCUDAStream(stream);

  std::function<void()> fn;

  if (kind == "CrossEntropyLoss") {
    fn = [&] {
      auto loss = torch::nn::CrossEntropyLoss()(output, target.view({-1}));
      loss.backward({}, true);
    };
  } else if (kind == "NLLLoss") {
    fn = [&] {
      auto loss = torch::nll_loss(output.log_softmax(1), target);
      loss.backward({}, true);
    };
  } else {
    assert(false && "Missing/bad loss kind");
  }

  fn();
  output.mutable_grad() = torch::Tensor();
  double tm = benchFn(fn);
  c10::cuda::setCurrentCUDAStream(origstream);
  return tm;
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
  bool hasgradinput = false;
  for (auto &i : inputs) {
    bool usegrad = i.is_floating_point();
    hasgradinput |= usegrad;
    v.emplace_back(i.cuda().detach().requires_grad_(usegrad));
  }

  /* benchmark forward pass */
  for (size_t i = 0; i < WAMRUP_ITERS; i++) module.forward(v);

  double fwtime;
  fwtime = benchFn([&] { module.forward(v); });

  /* prepare for backward pass */
  double bwtime = 0;

  if (hasgradinput || module.forward(v).toTensor().requires_grad()) {
    torch::Tensor output, grad;

    for (size_t i = 0; i < WAMRUP_ITERS; i++) {
      output = module.forward(v).toTensor();
      grad = torch::randn(output.sizes()).cuda().requires_grad_(false);
      output.backward(grad, false);
      for (const auto &param : module.parameters())
        param.mutable_grad() = torch::Tensor();
      for (auto &vv : v) vv.toTensor().mutable_grad() = torch::Tensor();
    }

    output = module.forward(v).toTensor();
    grad = torch::randn(output.sizes()).cuda().requires_grad_(false);
    bwtime = benchFn([&] { output.backward(grad, true); });
  }

  c10::cuda::setCurrentCUDAStream(origstream);
  return std::make_pair(fwtime, bwtime);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("benchmodule", &benchModule, "Benchmark jit module");
  m.def("benchloss", &benchLossTime, "Benchmark loss time");
}
