#pragma once

#include <torch/torch.h>

//#define MODEL_DEBUG 1
#ifdef MODEL_DEBUG
#include <nvToolsExt.h>
#endif

struct WithHooks {
  using FwdHookFn =
      std::function<void(int layern, torch::nn::AnyModule, at::Tensor &)>;
  using BwdHookFn = std::function<void(int layern, at::Tensor &)>;
  void InstallHooks(FwdHookFn forward_pre = FwdHookFn(),
                    FwdHookFn forward = FwdHookFn(),
                    BwdHookFn backward_pre = BwdHookFn()) {
    forward_pre_hook = forward_pre;
    forward_hook = forward;
    backward_pre_hook = backward_pre;
  }
  FwdHookFn forward_pre_hook;
  FwdHookFn forward_hook;
  BwdHookFn backward_pre_hook;
  static thread_local int layercount;
  static void StartForward();
  template <class A>
  at::Tensor record_forward(A &a, at::Tensor &b);
};

struct CustomSequentialImpl : torch::nn::SequentialImpl, WithHooks {
  using SequentialImpl::SequentialImpl;
  at::Tensor forward(at::Tensor x);
};
TORCH_MODULE(CustomSequential);

template <class Block>
struct ResNet : torch::nn::Module, WithHooks {
  std::vector<std::shared_ptr<Block>> allblocks;
  int64_t inplanes = 64;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::MaxPool2d mp;
  CustomSequential layer1;
  CustomSequential layer2;
  CustomSequential layer3;
  CustomSequential layer4;
  torch::nn::Linear fc;
  torch::nn::ReLU relu;
  torch::nn::AvgPool2d avgpool;
  ResNet(torch::ArrayRef<int64_t> layers, int64_t num_classes = 1000);
  at::Tensor forward(at::Tensor x);
  void InstallHooks(FwdHookFn forward_pre = FwdHookFn(),
                    FwdHookFn forward = FwdHookFn(),
                    BwdHookFn backward_pre = BwdHookFn());

 private:
  CustomSequential _make_layer(int64_t planes, int64_t blocks,
                               int64_t stride = 1);
};

struct BasicBlock : torch::nn::Module, WithHooks {
  static const int expansion;
  int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d bn2;
  CustomSequential downsample;
  torch::nn::ReLU relu;
  BasicBlock(int64_t, int64_t, int64_t, CustomSequential);
  at::Tensor forward(at::Tensor);
  void InstallHooks(FwdHookFn forward_pre = FwdHookFn(),
                    FwdHookFn forward = FwdHookFn(),
                    BwdHookFn backward_pre = BwdHookFn());
};

struct BottleNeck : torch::nn::Module, WithHooks {
  static const int expansion;
  int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d bn2;
  torch::nn::Conv2d conv3;
  torch::nn::BatchNorm2d bn3;
  CustomSequential downsample;
  torch::nn::ReLU relu;
  BottleNeck(int64_t inplanes, int64_t planes, int64_t stride_,
             CustomSequential downsample_);
  at::Tensor forward(at::Tensor x1);
  void InstallHooks(FwdHookFn forward_pre = FwdHookFn(),
                    FwdHookFn forward = FwdHookFn(),
                    BwdHookFn backward_pre = BwdHookFn());
};

template class ResNet<BasicBlock>;
template class ResNet<BottleNeck>;

ResNet<BasicBlock> resnet18();
ResNet<BasicBlock> resnet34();
ResNet<BottleNeck> resnet50();
ResNet<BottleNeck> resnet101();
ResNet<BottleNeck> resnet152();

class FakeDataLoader {
 public:
  FakeDataLoader(unsigned int batch_size);
  const std::pair<at::Tensor, at::Tensor> &GetBatch(unsigned int idx) const {
    return dummy_data[idx % dummy_data.size()];
  }

 private:
  std::vector<std::pair<at::Tensor, at::Tensor>> dummy_data;
};

class TrainableModel {
 public:
  using HookFn = std::function<void(int layern, at::Tensor &)>;

  TrainableModel(ResNet<BasicBlock> model, long bsize, int device,
                 bool train = true, bool low_pri = true);
  void Iterate();
  void HookPreLayer(HookFn fn);
  void HookPostLayer(HookFn fn);

  TrainableModel(const TrainableModel &) = delete;
  TrainableModel(TrainableModel &&) = delete;
  TrainableModel &operator=(const TrainableModel &) = delete;
  TrainableModel &operator=(TrainableModel &&) = delete;

  size_t GetNumLayers();

 private:
  ResNet<BasicBlock> model_;
  torch::optim::SGD optimizer_;
  FakeDataLoader dataloader_;
  uint64_t training_iteration_;
  torch::Device device_;

  void LayerStart(at::Tensor t = {}, std::string name = {}) {
    if (pre_hook_) pre_hook_(layer_counter_, t);
#ifdef MODEL_DEBUG
    std::stringstream ss;
    ss << layer_counter_;
    if (!name.empty()) {
      ss << " " << name;
    }
    std::string str = ss.str();
    if (!in_backward_) nvtxRangePush(str.c_str());
#endif
  }

  void LayerEnd(at::Tensor t = {}) {
    if (post_hook_) post_hook_(layer_counter_, t);
    layer_counter_++;
#ifdef MODEL_DEBUG
    if (!in_backward_) nvtxRangePop();
#endif
  }

  struct LayerRunner {
    LayerRunner(TrainableModel &m, at::Tensor t = {}, std::string name = {})
        : m_(m), t_(t) {
      m.LayerStart(t, name);
    }
    ~LayerRunner() { m_.LayerEnd(t_); }
    TrainableModel &m_;
    at::Tensor t_;
  };

  HookFn pre_hook_;
  HookFn post_hook_;

  size_t layer_counter_;

  bool in_backward_;
  bool train_;
  bool low_pri_;

  /* optimizer */
  void zero_grad();
  at::Tensor do_step();
};
