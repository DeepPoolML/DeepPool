#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

#include "runtime.h"
#include "utils.h"

class Dataset {
 public:
  virtual torch::data::Example<> getNext() = 0;
  virtual bool IsDone() = 0;
  virtual void Reset() = 0;
  static Dataset *fromName(std::string name, size_t rank, long globalBatchSize,
                           std::vector<long> initialBatchSizes,
                           std::vector<long> sampleIndices);

  torch::data::Example<> getNextThisRank() {
    auto ex = getNext();

    torch::Tensor data, target;
    if (initialBatchSizes_.at(rank_))
      data = ex.data.split_with_sizes(initialBatchSizes_)[rank_];

    if (sampleIndices_.size()) {
      std::vector<long> spl(globalBatchSize_, 1);
      auto splitsamples =
          ex.target.split_with_sizes(spl);  // TODO make this clean....
      std::vector<torch::Tensor> samplesOrdered;
      for (auto &s : sampleIndices_)
        samplesOrdered.push_back(splitsamples.at(s));
      target = torch::cat(samplesOrdered);
    }
    return {data, target};
  }

 protected:
  long globalBatchSize_;
  Dataset(size_t rank, long globalBatchSize,
          std::vector<long> initialBatchSizes, std::vector<long> sampleIndices)
      : globalBatchSize_(globalBatchSize),
        rank_(rank),
        initialBatchSizes_(initialBatchSizes),
        sampleIndices_(sampleIndices){};

 private:
  size_t rank_;
  std::vector<long> initialBatchSizes_;
  std::vector<long> sampleIndices_;
};

class TensorPipeline {
 public:
  TensorPipeline(torch::Tensor next) {
    tensorbytes = next.defined() ? next.nbytes() : 0;
    tensor_buf = c10::cuda::CUDACachingAllocator::raw_alloc(tensorbytes);
    assert(!tensorbytes || tensor_buf);

    SupplyNext(next);
  }

  void SupplyNext(torch::Tensor next) {
    if (!tensorbytes) {
      next_up_ = next;
      return;
    }

    auto origstream = c10::cuda::getCurrentCUDAStream();
    dToD_ev_.block(rtctx->xfer_stream);
    /* run next HtoD transfer */
    c10::cuda::setCurrentCUDAStream(rtctx->xfer_stream);
    next_up_ = next.to(rtctx->c10dev, /*non_blocking*/ true, /*copy*/ false);
    hToD_ev_.record(rtctx->xfer_stream);
    c10::cuda::setCurrentCUDAStream(origstream);
  }

  torch::Tensor GetNext(c10::optional<torch::Tensor> next) {
    assert(next_up_);
    auto &tsr = next_up_.value();

    if (!tensorbytes)
      return torch::Tensor();  //.has_device() ? tsr.to(rtctx->c10dev, true,
                               //false) : tsr; // fixme

    /* send last transmitted tensor into final buf */
    auto origstream = c10::cuda::getCurrentCUDAStream();
    hToD_ev_.block(origstream);
    CUDACHECK(cudaMemcpyAsync(tensor_buf, tsr.data_ptr(), tsr.nbytes(),
                              cudaMemcpyDeviceToDevice, origstream));
    dToD_ev_.record(origstream);

    auto tensor_out = torch::from_blob(tensor_buf, tsr.sizes(), tsr.options());

    next_up_ = {};
    if (next) SupplyNext(next.value());
    return tensor_out;
  }

 private:
  size_t tensorbytes;
  void *tensor_buf;
  c10::optional<torch::Tensor> next_up_;
  at::cuda::CUDAEvent hToD_ev_;
  at::cuda::CUDAEvent dToD_ev_;
};

class DatasetPipelineWrapper {
 public:
  DatasetPipelineWrapper(std::shared_ptr<Dataset> dataset) : dataset_(dataset) {
    auto next_sample = dataset_->getNextThisRank();
    data_pipeline_.reset(new TensorPipeline(next_sample.data));
    target_pipeline_.reset(new TensorPipeline(next_sample.target));
  }

  bool IsDone() { return is_done_; }

  void Reset() {
    dataset_->Reset();
    is_done_ = false;
    auto next_sample = dataset_->getNextThisRank();
    data_pipeline_->SupplyNext(next_sample.data);
    target_pipeline_->SupplyNext(next_sample.target);
  }

  torch::data::Example<> getNextThisRank() {
    assert(!is_done_);
    if (dataset_->IsDone()) {
      auto data = data_pipeline_->GetNext({});
      auto target = target_pipeline_->GetNext({});
      is_done_ = true;
      return {data, target};
    }
    auto next_sample = dataset_->getNextThisRank();
    auto data = data_pipeline_->GetNext(next_sample.data);
    auto target = target_pipeline_->GetNext(next_sample.target);
    return {data, target};
  }

 private:
  std::shared_ptr<Dataset> dataset_;
  std::unique_ptr<TensorPipeline> data_pipeline_;
  std::unique_ptr<TensorPipeline> target_pipeline_;
  bool is_done_{false};
};