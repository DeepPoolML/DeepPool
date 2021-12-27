#pragma once

#include <torch/torch.h>

class Dataset {
 public:
  virtual torch::data::Example<> getNext() = 0;
  virtual bool IsDone() = 0;
  virtual void Reset(bool eval = false) = 0;
  static Dataset *fromName(std::string name, long globalBatchSize);

  torch::data::Example<> getNextForRank(size_t rank, long globalBatchSize,
                                        std::vector<long> initialBatchSizes,
                                        std::vector<long> sampleIndices) {
    auto ex = getNext();

    torch::Tensor data, target;
    if (initialBatchSizes.at(rank))
      data = ex.data.split_with_sizes(initialBatchSizes)[rank];

    if (sampleIndices.size()) {
      std::vector<long> spl(globalBatchSize, 1);
      auto splitsamples = ex.target.split_with_sizes(spl); // TODO make this clean....
      std::vector<torch::Tensor> samplesOrdered;
      for (auto &s : sampleIndices)
        samplesOrdered.push_back(splitsamples.at(s));
      target = torch::cat(samplesOrdered);
    }
    return {data, target};
  }

 protected:
  bool is_eval_mode_{false};
};

// class DatasetPipelineWrapper : public Dataset {

// };