#include "dataset.h"

#include <absl/flags/flag.h>
#include <torch/torch.h>

#include "cifar10.h"

ABSL_FLAG(std::string, cifar_dataset,
          "/home/friedj/mlsf/multimodel/data/cifar-10-batches-bin/", "");

class FakeDataset : public Dataset {
 public:
  FakeDataset(long globalBatchSize, std::vector<long> inputShape,
              size_t target_classes, size_t images_per_epoch_train,
              size_t images_per_eval);
  torch::data::Example<> getNext() override;
  bool IsDone() override;
  void Reset(bool eval = false) override;

 private:
  size_t batches_per_training_epoch_;
  size_t batches_per_testing_epoch_;
  size_t ctr_;
  std::vector<torch::data::Example<>> cached_;
};

class CifarDataset : public Dataset {
 public:
  CifarDataset(long globalBatchSize);
  torch::data::Example<> getNext() override;
  bool IsDone() override;
  void Reset(bool eval = false) override;

 private:
  long globalBatchSize_;
  c10::optional<torch::data::Iterator<torch::data::Example<>>> cur_iter;

  std::unique_ptr<torch::data::StatelessDataLoader<
      torch::data::datasets::MapDataset<
          torch::data::datasets::MapDataset<
              CIFAR10, torch::data::transforms::Normalize<>>,
          torch::data::transforms::Stack<torch::data::Example<>>>,
      torch::data::samplers::SequentialSampler>>
      train_loader, eval_loader;
};

FakeDataset::FakeDataset(long globalBatchSize, std::vector<long> inputShape,
                         size_t target_classes, size_t images_per_epoch_train,
                         size_t images_per_eval) {
  std::vector<long> fullShape;
  fullShape.push_back(globalBatchSize);
  fullShape.insert(fullShape.end(), inputShape.begin(), inputShape.end());
  auto targetOpts = torch::TensorOptions().dtype(torch::kInt64);
  for (size_t i = 0; i < 64; i++) {
    auto data = torch::randn(inputShape);
    auto target = torch::randint(/*low=*/0, /*high=*/target_classes,
                                 {globalBatchSize}, targetOpts);
    cached_.emplace_back(data, target);
  }
  batches_per_training_epoch_ = images_per_epoch_train / globalBatchSize;
  batches_per_testing_epoch_ = images_per_eval / globalBatchSize;
}

bool FakeDataset::IsDone() {
  return ctr_ >= (is_eval_mode_ ? batches_per_testing_epoch_
                                : batches_per_training_epoch_);
}

torch::data::Example<> FakeDataset::getNext() {
  assert(!IsDone());
  return cached_[ctr_++ % cached_.size()];
}

void FakeDataset::Reset(bool eval) {
  is_eval_mode_ = eval;
  ctr_ = 0;
}

CifarDataset::CifarDataset(long globalBatchSize)
    : globalBatchSize_(globalBatchSize) {
  auto c = CIFAR10(absl::GetFlag(FLAGS_cifar_dataset), CIFAR10::Mode::kTrain)
               .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406},
                                                         {0.229, 0.224, 0.225}))
               .map(torch::data::transforms::Stack<>());
  train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(c), globalBatchSize);
  auto t = CIFAR10(absl::GetFlag(FLAGS_cifar_dataset), CIFAR10::Mode::kTest)
               .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406},
                                                         {0.229, 0.224, 0.225}))
               .map(torch::data::transforms::Stack<>());
  eval_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(t), globalBatchSize);

  cur_iter = train_loader->begin();
}

bool CifarDataset::IsDone() {
  if (is_eval_mode_ && cur_iter == eval_loader->end())
    return true;
  else if (!is_eval_mode_ && cur_iter == train_loader->end())
    return true;
  else if (cur_iter.value()->data.sizes().vec()[0] < globalBatchSize_)
    return true;

  return false;
}

torch::data::Example<> CifarDataset::getNext() {
  assert(!IsDone());
  auto cur_example = *cur_iter.value();
  cur_iter = ++cur_iter.value();
  return cur_example;
}

void CifarDataset::Reset(bool eval) {
  is_eval_mode_ = eval;
  if (eval)
    cur_iter = eval_loader->begin();
  else
    cur_iter = train_loader->begin();
}

Dataset *Dataset::fromName(std::string name, long globalBatchSize) {
	if (name.find("cifar") != std::string::npos)
		return new CifarDataset(globalBatchSize);
	long px = name.find("inception") != std::string::npos ? 299 : 224;
	return new FakeDataset(globalBatchSize, {3, px, px}, 1000, 100000, 1000);
}