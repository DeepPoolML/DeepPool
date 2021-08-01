// Copyright (c) 2021 MIT
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#ifndef TASK_MANAGER_H
#define TASK_MANAGER_H

#include <torch/torch.h>
#include <memory>
#include <mutex>
#include <vector>
#include <string>

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class RuntimeContext;
class RunnableModule;
namespace torch {
  namespace optim {
    class Optimizer;
  }
}
// namespace c10 {
//   class Device;
// }
class CommunicationHandler;

class DataLoader {
};

class TargetShuffler {
};

enum class JobState {
  INIT = 0,
  FORWARD,
  BACKWARD,
  SYNC,
  NUM_JOB_STATES // must be the last element in the enum
};

/**
 * Context holding data for each training task.
 */
struct JobContext {
 public:
  JobContext(std::unique_ptr<RunnableModule> model, std::string name,
      std::unique_ptr<DataLoader> dataLoader,
      std::unique_ptr<CommunicationHandler> commHandler,
      std::unique_ptr<TargetShuffler> targetShuffler,
      c10::Device device,
      int epochsToTrain = 1,
      std::unique_ptr<torch::optim::Optimizer> optimizer = nullptr
      // std::unique_ptr<torch::optim::SGD> optimizer = nullptr
      );
  ~JobContext();

  std::unique_ptr<RunnableModule> model;
  std::string name;
  std::unique_ptr<DataLoader> dataLoader;
  std::unique_ptr<CommunicationHandler> commHandler;
  std::unique_ptr<TargetShuffler> targetShuffler;
  int epochsToTrain;
  std::unique_ptr<torch::optim::Optimizer> optimizer;
  c10::Device device;
  // self.dataLoaderIt = iter(self.dataLoader) if dataLoader != None else None
  // self.criterion = nn.CrossEntropyLoss().cuda(device) if criterion == None else criterion
  int epoch;
  int iter;
  int itersToTrain; // = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
  // self.itersPerPoll = 50
  // self.training_initialized = False
  // self.itersToCapture = set(range(250, 260))
  JobState state;
};

/**
 * Manages training jobs by scheduling tasks to CUDA devices.
 * 
 * Public methods are thread-safe.
 */
class TaskManager {
 public:
  TaskManager(RuntimeContext* rtctx);
  ~TaskManager() {} // TODO(seojin): implement
  int addTrainingJob(std::unique_ptr<JobContext> job);
  int poll();
 private:
  int trainSingleStep(JobContext* job, bool* jobCompleted);

  RuntimeContext* rtctx;
  std::mutex _mutex;                // Monitor lock for TaskManager.
  std::vector< std::unique_ptr<JobContext> > jobList;  
                                    // Holds unfinished training jobs.
                                    // Assumes jobs are ordered by priority.
  // pointer to comm.
};

#endif // TASK_MANAGER_H