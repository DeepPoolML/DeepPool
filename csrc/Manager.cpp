
#include "Manager.h"

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "CUDAGraph.h"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

Tasklet::Tasklet(std::shared_ptr<DeepPool::CUDAGraph> graph, unsigned int flags,
                 std::string debug_name)
    : type_flags_(flags), debug_name_(debug_name) {
  CUDA_API_CALL(cudaGraphClone(&graph_holder_.graph, graph->getGRAPH()));
  refs.push_back(graph);
}

Tasklet::Tasklet(cudaGraphExec_t exec, unsigned int flags,
                 std::string debug_name)
    : type_flags_(flags), debug_name_(debug_name) {
  graph_holder_.graph_exec = exec;
}

Tasklet::Tasklet(std::function<void(c10::cuda::CUDAStream stream)> cb,
                 unsigned int flags, std::string debug_name)
    : type_flags_(flags), debug_name_(debug_name), cb_(std::move(cb)) {}

Tasklet::Tasklet(std::vector<Tasklet> subtasks)
    : subtasks_(std::move(subtasks)) {
  assert(subtasks_.size());

  cudaGraphNode_t curDep = 0;
  graph_holder_.Init();

  for (size_t i = 0; i < subtasks_.size(); i++) {
    auto& t = subtasks_.at(i);
    type_flags_ |= t.type_flags_;
    assert(t.graph_holder_.getGraph());
    t.graph_holder_.DestroyExec();
    CUDA_API_CALL(cudaGraphAddChildGraphNode(&curDep, graph_holder_.getGraph(),
                                             i ? &curDep : nullptr, i ? 1 : 0,
                                             t.graph_holder_.getGraph()));
  }
}

double Tasklet::BenchMicros(size_t trials) {
  Run(c10::cuda::getCurrentCUDAStream());
  c10::cuda::device_synchronize();
  CpuTimer timer("timer");
  timer.start();
  for (size_t i = 0; i < trials; ++i) Run(c10::cuda::getCurrentCUDAStream());
  c10::cuda::device_synchronize();
  timer.stop();
  return timer.avgMicros() / trials;
}

void Tasklet::Run(c10::cuda::CUDAStream stream) {
  if (cb_)
    cb_(stream);
  else
    graph_holder_.Exec(stream);
  ev_.record(stream);
}

std::ostream& operator<<(std::ostream& out, const Tasklet& k) {
  if (k.subtasks_.size())
    out << "Supertasklet";
  else
    out << "Tasklet";
  if (!k.debug_name_.empty()) out << " [" << k.debug_name_ << "]";
  if (k.subtasks_.size()) {
    out << " [";
    for (auto& t : k.subtasks_) out << t.debug_name_ << ",";
    out << "]";
  }
  out << " (" << k.timings_.recorded_us << " us)";
  return out;
}

GpuTask::GpuTask(bool hipri, c10::cuda::CUDAStream execution_stream,
                 std::vector<cudaGraphExec_t> pieces)
    : is_high_priority(hipri),
      execution_stream(execution_stream),
      nccl_stream(c10::cuda::getStreamFromPool(hipri)) {
  for (auto& p : pieces)
    tasks_.emplace_back(p, TASK_FLAGS_COMPUTE | TASK_FLAGS_EXTERNAL,
                        "BE task piece");
  TimeRun();
}

unsigned int GpuTask::GetFlagsQueuedTasks() {
  unsigned int flags = 0;

  for (uint32_t curptr = tail_ptr; curptr != head_ptr; curptr++) {
    auto& t = tasks_[curptr % tasks_.size()];
    flags |= t.type_flags_;
  }

  return flags;
}

bool GpuTask::RunNext(c10::cuda::CUDAStream stream) {
  assert(next_run_idx < tasks_.size());
  auto& t = tasks_[next_run_idx++];

  head_ptr++;
  outstanding_micros_ += t.timings_.recorded_us;

  if (t.IsNccl()) {
    ev.record(stream);
    ev.block(nccl_stream);
    t.Run(nccl_stream);
    if (t.type_flags_ & (TASK_FLAGS_P2PCOMM_RECV | TASK_FLAGS_ALLREDUCE))
      waiting_recv = true;
    return next_run_idx >= tasks_.size();
  }

  if ((t.type_flags_ & TASK_FLAGS_COMPUTE) > 0 && waiting_recv) {
    ev.record(nccl_stream);
    ev.block(stream);
    waiting_recv = false;
  }

  t.Run(stream);
  return next_run_idx >= tasks_.size();
}

void GpuTask::ExecuteTasks() {
  MakeManagerOwned();
  while (IsManagerOwned())
    ;
}

void GpuTask::Reset() {
  FinishCompletion();
  tasks_.clear();
  waiting_recv = false;
}

void GpuTask::TimeRun() {
  std::vector<at::cuda::CUDAEvent> start_events;
  std::vector<at::cuda::CUDAEvent> end_events;

  /* initalize events */
  for (size_t i = 0; i < tasks_.size(); i++) {
    start_events.emplace_back(cudaEventDefault);
    end_events.emplace_back(cudaEventDefault);
    start_events.at(i).record();
    end_events.at(i).record();
  }

  c10::cuda::device_synchronize();

  for (size_t i = 0; i < tasks_.size(); i++) {
    start_events[i].record();
    tasks_[i].Run(c10::cuda::getCurrentCUDAStream());
    end_events[i].record();
  }

  c10::cuda::device_synchronize();
  for (size_t i = 0; i < tasks_.size(); i++)
    tasks_.at(i).timings_.recorded_us =
        1000.0 * start_events.at(i).elapsed_time(end_events.at(i));
}

void GpuTask::CombineGraphs() {
  TimeRun();
  for (auto& t : tasks_) {
    if (t.type_flags_ & TASK_FLAGS_DO_NOT_BENCH) continue;
    t.timings_.recorded_us = t.BenchMicros();
  }

  std::vector<Tasklet> newTasks;
  std::vector<Tasklet> curMergers;
  double total_us = 0;
  unsigned int prevtype = 0;

  auto finMerge = [&] {
    if (!curMergers.size()) return;
    total_us = 0;
    if (curMergers.size() == 1) {
      newTasks.push_back(std::move(curMergers.front()));
      curMergers.clear();
    } else {
      newTasks.emplace_back(std::move(curMergers));
      assert(curMergers.size() == 0);
    }
  };

  while (tasks_.size()) {
    auto& tf = tasks_.front();

    if (tf.cb_) {
      finMerge();
      newTasks.push_back(std::move(tf));
      tasks_.erase(tasks_.begin());
      continue;
    }

    if (prevtype != tf.type_flags_ ||
        total_us + tf.timings_.recorded_us > 1000) {
      finMerge();
      prevtype = tf.type_flags_;
    }

    total_us += tf.timings_.recorded_us;
    curMergers.push_back(std::move(tf));
    tasks_.erase(tasks_.begin());

    if (total_us > 500 || curMergers.size() > 10) finMerge();
  }

  finMerge();

  tasks_ = std::move(newTasks);
  TimeRun();
}

GpuManager* GpuManager::instance;

void GpuManager::DoWork() {
  std::lock_guard<std::mutex> lock(jobListMutex);
  bool waitforhp = false;
  if (hptask) {
    hptask->PollCompletions();
    while (hptask->IsManagerOwned() && (hptask->GetEnqueuedNr() < 4 || hptask->GetEnqueuedMicros() < 1000)) {
      hptask->RunNext();
      if (hptask->next_run_idx == hptask->tasks_.size()) {
        hptask->next_run_idx = 0;
        hptask->ManagerReleaseOwnership();
        break;
      }
    }

    waitforhp = (hptask->GetFlagsQueuedTasks() &
                 (TASK_FLAGS_P2PCOMM_RECV | TASK_FLAGS_P2PCOMM | TASK_FLAGS_ALLREDUCE)) > 0;
  }

  if (!lptask) return;
  lptask->PollCompletions();
  if (waitforhp || !lptask->IsManagerOwned()) return;
  if (lptask->GetEnqueuedNr() >= 1) return;
  lptask->RunNext();
  if (lptask->next_run_idx == lptask->tasks_.size()) {
    lptask->next_run_idx = 0;
    lptask->ManagerReleaseOwnership();
  }
}