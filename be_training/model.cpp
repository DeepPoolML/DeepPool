#include "model.h"

using Tensor = torch::Tensor;

static torch::nn::Conv2dOptions conv_options(
    int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
  torch::nn::Conv2dOptions conv_options =
      torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
  return conv_options.stride(stride).padding(padding).bias(with_bias);
}

thread_local int WithHooks::layercount;
void WithHooks::StartForward() { layercount = 0; }

template <class A>
Tensor WithHooks::record_forward(A &a, Tensor &b) {
  torch::nn::AnyModule mod(a);

  int thislayer = layercount;
  bool do_hook_layer = a.ptr()->children().size() == 0;
  if (do_hook_layer) {
    layercount++;
    if (forward_pre_hook) forward_pre_hook(thislayer, mod, b);
  }
  Tensor z = mod.forward<Tensor>(b);
  if (do_hook_layer) {
    if (forward_hook) forward_hook(thislayer, mod, z);
    if (backward_pre_hook) {
      // assert(!z.has_hooks());
      z.register_hook(
          [&, thislayer](Tensor grad) { backward_pre_hook(thislayer, grad); });
    }
  }
  return z;
}

Tensor CustomSequentialImpl::forward(Tensor x) {
  for (auto iterator = begin(); iterator != end(); ++iterator)
    x = record_forward(*iterator, x);
  return x;
}

BasicBlock::BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
                       CustomSequential downsample_ = CustomSequential())
    : conv1(conv_options(inplanes, planes, 3, stride_, 1)),
      bn1(planes),
      conv2(conv_options(planes, planes, 3, 1, 1)),
      bn2(planes),
      downsample(downsample_) {
  register_module("conv1", conv1);
  register_module("bn1", bn1);
  register_module("conv2", conv2);
  register_module("bn2", bn2);
  stride = stride_;
  if (!downsample->is_empty()) {
    register_module("downsample", downsample);
  }
  register_module("relu", relu);
}

void BasicBlock::InstallHooks(FwdHookFn forward_pre, FwdHookFn forward,
                              BwdHookFn backward_pre) {
  downsample->InstallHooks(forward_pre, forward, backward_pre);
  WithHooks::InstallHooks(forward_pre, forward, backward_pre);
}

void BottleNeck::InstallHooks(FwdHookFn forward_pre, FwdHookFn forward,
                              BwdHookFn backward_pre) {
  downsample->InstallHooks(forward_pre, forward, backward_pre);
  WithHooks::InstallHooks(forward_pre, forward, backward_pre);
}

Tensor BasicBlock::forward(Tensor x1) {
  auto x = record_forward(conv1, x1);
  x = record_forward(bn1, x);
  x = record_forward(relu, x);

  x = record_forward(conv2, x);
  x = record_forward(bn2, x);

  if (!downsample->is_empty()) {
    x += downsample->forward(x1);
  } else {
    x += x1;
  }

  x = record_forward(relu, x);

  return x;
}

const int BasicBlock::expansion = 1;

BottleNeck::BottleNeck(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
                       CustomSequential downsample_ = CustomSequential())
    : conv1(conv_options(inplanes, planes, 1)),
      bn1(planes),
      conv2(conv_options(planes, planes, 3, stride_, 1)),
      bn2(planes),
      conv3(conv_options(planes, planes * expansion, 1)),
      bn3(planes * expansion),
      downsample(downsample_) {
  register_module("conv1", conv1);
  register_module("bn1", bn1);
  register_module("conv2", conv2);
  register_module("bn2", bn2);
  register_module("conv3", conv3);
  register_module("bn3", bn3);
  stride = stride_;
  if (!downsample->is_empty()) {
    register_module("downsample", downsample);
  }
  register_module("relu", relu);
}

Tensor BottleNeck::forward(Tensor x1) {
  auto x = record_forward(conv1, x1);
  x = record_forward(bn1, x);
  x = record_forward(relu, x);

  x = record_forward(conv2, x);
  x = record_forward(bn2, x);
  x = record_forward(relu, x);

  x = record_forward(conv3, x);
  x = record_forward(bn3, x);

  if (!downsample->is_empty()) {
    x += downsample->forward(x1);
  } else {
    x += x1;
  }

  x = record_forward(relu, x);

  return x;
}

const int BottleNeck::expansion = 4;

template <class Block>
ResNet<Block>::ResNet(torch::ArrayRef<int64_t> layers, int64_t num_classes)
    : allblocks(),
      conv1(conv_options(3, 64, 7, 2, 3)),
      bn1(64),
      mp(torch::nn::MaxPool2dOptions({3, 3}).stride(2).padding(1)),
      layer1(_make_layer(64, layers[0])),
      layer2(_make_layer(128, layers[1], 2)),
      layer3(_make_layer(256, layers[2], 2)),
      layer4(_make_layer(512, layers[3], 2)),
      fc(512 * Block::expansion, num_classes),
      avgpool(torch::nn::AvgPool2dOptions({7, 7}).stride(1)) {
  register_module("conv1", conv1);
  register_module("bn1", bn1);
  register_module("mp", mp);
  register_module("layer1", layer1);
  register_module("layer2", layer2);
  register_module("layer3", layer3);
  register_module("layer4", layer4);
  register_module("fc", fc);
  register_module("relu", relu);
  register_module("avgpool", avgpool);

  // Initializing weights
  // for(auto m: this->modules()){
  //   if (m->name() == "torch::nn::Conv2dImpl"){
  //     for (auto p: m->parameters()){
  //       torch::nn::init::xavier_normal_(p);
  //     }
  //   }
  //   else if (m->name() == "torch::nn::BatchNormImpl"){
  //     for (auto p: m->parameters()){
  //       if (p.key == "weight"){
  //         torch::nn::init::constant_(p, 1);
  //       }
  //       else if (p.key == "bias"){
  //         torch::nn::init::constant_(p, 0);
  //       }
  //     }
  //   }
  // }
}

template <class Block>
void ResNet<Block>::InstallHooks(FwdHookFn forward_pre, FwdHookFn forward,
                                 BwdHookFn backward_pre) {
  layer1->InstallHooks(forward_pre, forward, backward_pre);
  layer2->InstallHooks(forward_pre, forward, backward_pre);
  layer3->InstallHooks(forward_pre, forward, backward_pre);
  layer4->InstallHooks(forward_pre, forward, backward_pre);
  for (auto &b : allblocks) b->InstallHooks(forward_pre, forward, backward_pre);
  WithHooks::InstallHooks(forward_pre, forward, backward_pre);
}

template <class Block>
Tensor ResNet<Block>::forward(Tensor x) {
  WithHooks::StartForward();
  x = record_forward(conv1, x);
  x = record_forward(bn1, x);
  x = record_forward(relu, x);
  x = record_forward(mp, x);

  x = layer1->forward(x);
  x = layer2->forward(x);
  x = layer3->forward(x);
  x = layer4->forward(x);

  x = record_forward(avgpool, x);
  x = x.view({x.sizes()[0], -1});
  x = record_forward(fc, x);

  return x;
}

template <class Block>
CustomSequential ResNet<Block>::_make_layer(int64_t planes, int64_t blocks,
                                            int64_t stride) {
  CustomSequential downsample;
  if (stride != 1 or inplanes != planes * Block::expansion) {
    downsample =
        CustomSequential(torch::nn::Conv2d(conv_options(
                             inplanes, planes * Block::expansion, 1, stride)),
                         torch::nn::BatchNorm2d(planes * Block::expansion));
  }
  CustomSequential layers;
  auto block = std::make_shared<Block>(inplanes, planes, stride, downsample);
  layers->push_back(block);
  allblocks.push_back(block);
  inplanes = planes * Block::expansion;
  for (int64_t i = 0; i < blocks; i++) {
    auto block = std::make_shared<Block>(inplanes, planes);
    layers->push_back(block);
    allblocks.push_back(block);
  }
  return layers;
}

ResNet<BasicBlock> resnet18() {
  ResNet<BasicBlock> model({2, 2, 2, 2});
  return model;
}

ResNet<BasicBlock> resnet34() {
  ResNet<BasicBlock> model({3, 4, 6, 3});
  return model;
}

ResNet<BottleNeck> resnet50() {
  ResNet<BottleNeck> model({3, 4, 6, 3});
  return model;
}

ResNet<BottleNeck> resnet101() {
  ResNet<BottleNeck> model({3, 4, 23, 3});
  return model;
}

ResNet<BottleNeck> resnet152() {
  ResNet<BottleNeck> model({3, 8, 36, 3});
  return model;
}

FakeDataLoader::FakeDataLoader(unsigned int batch_size) {
  for (int i = 0; i < 5; i++) {
    auto input = torch::rand({batch_size, 3, 224, 224});
    auto target = torch::empty(batch_size).uniform_(0, 1000).to(at::kLong);
    dummy_data.emplace_back(input, target);
  }
}

TrainableModel::TrainableModel(ResNet<BottleNeck> model, long bsize, int device,
                               bool train)
    : model_(model),
      optimizer_(model_.parameters(),
                 torch::optim::SGDOptions(0.1).momentum(0.9)),
      dataloader_(bsize),
      training_iteration_(0),
      device_(c10::DeviceType::CUDA, device),
      train_(train) {
  if (train_)
    model_.train();
  else
    model_.eval();
  model_.to(device_);

  auto fwdpre = [&](int layern, torch::nn::AnyModule m, at::Tensor &t) {
    std::stringstream mm;
    m.ptr()->pretty_print(mm);
    LayerStart(t, mm.str());
  };

  auto fwdpost = [&](int layern, torch::nn::AnyModule, at::Tensor &t) {
    LayerEnd(t);
  };

  auto bwdpre = [&](int layern, at::Tensor &t) {
    if (last_backward_++ != -1) {
      LayerEnd(t);
    }
    LayerStart(t, "Bwd");
  };

  if (train_)
    model_.InstallHooks(fwdpre, fwdpost, bwdpre);
  else
    model_.InstallHooks(fwdpre, fwdpost);
}

void TrainableModel::HookPreLayer(HookFn fn) { pre_hook_ = fn; }

void TrainableModel::HookPostLayer(HookFn fn) { post_hook_ = fn; }

size_t TrainableModel::GetNumLayers() {
  if (training_iteration_ < 2)
    throw std::runtime_error("call Iterate() twice before GetNumLayers()");
  return layer_counter_;
}

void TrainableModel::Iterate() {
  auto data = dataloader_.GetBatch(training_iteration_++);

  Tensor empty, input, target, loss, output;

  /* Move data to device */
  layer_counter_ = 0;
  {
    LayerRunner l(*this, empty, "inputs");
    input = data.first.to(device_, data.first.scalar_type(), true);
    target = data.second.to(device_, data.second.scalar_type(), true);
    if (train_) input.set_requires_grad(true);
  }

  /* Zero out existing gradients layer by layer*/
  if (train_) zero_grad();

  /* Forward pass */
  output = model_.forward(input);

  if (!train_) return;

  /* Compute loss */
  {
    LayerRunner l(*this, empty, "loss");
    loss = torch::nll_loss(output.log_softmax(1), target);
  }

  /* Backward Pass */
  last_backward_ = -1;
  loss.backward();
  LayerEnd(empty);

  /* Step optimizer */
  do_step();
}

#define ZERO_GRAD_BATCH 20
#define STEP_GRAD_BATCH 6

void TrainableModel::zero_grad() {
  for (auto &group : optimizer_.param_groups()) {
    const auto &params = group.params();
    for (size_t i = 0, j = params.size(); i < j;) {
      LayerRunner l(*this, {}, "zero");
      for (size_t til = std::min(j, i + ZERO_GRAD_BATCH); i < til; i++) {
        const auto &p = params.at(i);
        if (p.grad().defined()) {
          p.grad().detach_();
          p.grad().zero_();
        }
      }
    }
  }
}

Tensor TrainableModel::do_step() {
  using namespace torch::optim;
  torch::NoGradGuard no_grad;
  Tensor loss = {};
  for (auto &group : optimizer_.param_groups()) {
    auto &options = static_cast<SGDOptions &>(group.options());
    auto weight_decay = options.weight_decay();
    auto momentum = options.momentum();
    auto dampening = options.dampening();
    auto nesterov = options.nesterov();

    const auto &params = group.params();
    for (size_t i = 0, j = params.size(); i < j;) {
      LayerRunner l(*this, {}, "step");
      for (size_t til = std::min(j, i + STEP_GRAD_BATCH); i < til; i++) {
        const auto &p = params.at(i);
        if (!p.grad().defined()) {
          continue;
        }
        auto d_p = p.grad().data();
        if (weight_decay != 0) {
          d_p = d_p.add(p.data(), weight_decay);
        }
        if (momentum != 0) {
          Tensor buf;
          auto param_state = optimizer_.state().find(
              c10::guts::to_string(p.unsafeGetTensorImpl()));
          if (param_state == optimizer_.state().end()) {
            buf = torch::clone(d_p).detach();
            auto state = std::make_unique<SGDParamState>();
            state->momentum_buffer(buf);
            optimizer_.state()[c10::guts::to_string(p.unsafeGetTensorImpl())] =
                std::move(state);
          } else {
            buf = static_cast<SGDParamState &>(*param_state->second)
                      .momentum_buffer();
            buf.mul_(momentum).add_(d_p, 1 - dampening);
          }
          if (nesterov) {
            d_p = d_p.add(buf, momentum);
          } else {
            d_p = buf;
          }
        }
        p.data().add_(d_p, -1 * options.lr());
      }
    }
  }
  return loss;
}
