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
#include <string>

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class RunnableModule;
class CommunicationHandler;

/**
 * Context holding data for each training task.
 */
class JobContext {
 public:
  JobContext(std::unique_ptr<RunnableModule> model, std::string name,
             std::shared_ptr<CommunicationHandler> commHandler);
  ~JobContext();

  bool TrainSingleStep();
  void TrainToCompletion();
  int trainSingleStep(bool* jobCompleted);
  void printJobStatistics();
  size_t GetIter() const { return iter; }

  std::unique_ptr<RunnableModule> model;
  std::string name;
  std::shared_ptr<CommunicationHandler> commHandler;
  bool run_with_be{false};

  std::chrono::time_point<std::chrono::steady_clock> start, end;
  uint64_t be_img_start, be_img_end;

 private:
  bool iter_in_progress{false};
  size_t epochsToTrain{50};
  size_t epoch{0};
  size_t iter{0};
  size_t totiters{0};  // total iters executed
  size_t warmupIters{200};
  size_t itersToTrain{1900};
  size_t profile_iter_start{ULLONG_MAX};
  size_t niter_to_profile{5};
  size_t iters_before_graph_capture{50};  // set high to disable graph capture
};

#endif  // TASK_MANAGER_H