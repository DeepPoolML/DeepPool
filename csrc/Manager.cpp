
#include "Manager.h"

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <absl/flags/flag.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "CUDAGraph.h"
#include "Cycles.h"
#include "communication.h"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

ABSL_FLAG(size_t, be_tasklet_depth, 1,
          "Max number of BE tasklets in flight at once");
ABSL_FLAG(size_t, fg_tasklet_depth, 2,
          "Max number of FG tasklets in flight at once");
ABSL_FLAG(double, target_slowdown_factor, 1.1, "");
ABSL_FLAG(size_t, graph_merge_min_us, 250,
          "target lower bound for graph execution latency");
ABSL_FLAG(size_t, graph_merge_max_us, 500,
          "target upper bound for graph execution latency");
ABSL_FLAG(size_t, graph_merge_max_subgraphs, 10,
          "Maximum number of subgraphs in one merged graph");

static size_t be_tasklet_depth;
static size_t fg_tasklet_depth;
static double target_slowdown_factor;
static size_t graph_merge_min_us;
static size_t graph_merge_max_us;
static size_t graph_merge_max_subgraphs;

static void parse_flags() {
  static bool done;
  if (done) return;
  be_tasklet_depth = absl::GetFlag(FLAGS_be_tasklet_depth);
  fg_tasklet_depth = absl::GetFlag(FLAGS_fg_tasklet_depth);
  target_slowdown_factor = absl::GetFlag(FLAGS_target_slowdown_factor);
  graph_merge_min_us = absl::GetFlag(FLAGS_graph_merge_min_us);
  graph_merge_max_us = absl::GetFlag(FLAGS_graph_merge_max_us);
  graph_merge_max_subgraphs = absl::GetFlag(FLAGS_graph_merge_max_subgraphs);
  done = true;
}

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

void Tasklet::BenchMicros(size_t trials) {
  Run(c10::cuda::getCurrentCUDAStream());
  c10::cuda::device_synchronize();
  CpuTimer timer("timer");
  timer.start();
  for (size_t i = 0; i < trials; ++i) Run(c10::cuda::getCurrentCUDAStream());
  c10::cuda::device_synchronize();
  timer.stop();
  timings_.benchmark_us = timer.avgMicros() / trials;
}

void Tasklet::Run(c10::cuda::CUDAStream stream) {
  if (cb_)
    cb_(stream);
  else
    graph_holder_.Exec(stream);
  timings_.ev.record(stream);
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
  out << " (benchmarked " << k.timings_.benchmark_us << " us, observed "
      << k.timings_.recent_us << " us, last " << k.timings_.last_us << "us";
  if (k.timings_.last_us)
    out << " " << k.timings_.last_us / k.timings_.benchmark_us << " slowdown";
#if PROFILING_ENABLE
  if (k.timings_.has_profiledata) {
    out << " Occupancy: " << k.timings_.profiledata.achieved_occupancy
        << " SM Efficiency: " << k.timings_.profiledata.sm_efficiency
        << " DRAM GBps " << k.timings_.dram_gbps;
  }
#endif
  if (k.IsNccl()) {
    out << " [NCCL";
    if (k.IsAsync()) out << " ASYNC";
    out << "]";
  }

  out << " Flags: ";
  for (size_t i = 0; i < 64; i++) {
    if (k.type_flags_ & (1UL << i)) out << i << ",";
  }

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
  TimeRun(true);
}

bool GpuTask::RunNext(c10::cuda::CUDAStream stream) {
  assert(next_run_idx < tasks_.size());
  auto& t = tasks_[next_run_idx++];

  bool first = head_ptr++ == tail_ptr;

  t.timings_.enqueue_time = Cycles::microtime();
  if (first) t.timings_.execute_time_begin = t.timings_.enqueue_time;

  outstanding_micros_ += t.timings_.benchmark_us;

  if (t.IsNccl()) {
    ev.record(stream);
    ev.block(nccl_stream);
    t.Run(nccl_stream);
    if (!t.IsAsync()) waiting_recv = true;
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

void GpuTask::TimeRun(bool no_bench) {
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

  for (size_t i = 0; i < tasks_.size(); i++)
    tasks_[i].Run(c10::cuda::getCurrentCUDAStream());

  for (size_t i = 0; i < tasks_.size(); i++) {
    start_events[i].record();
    tasks_[i].Run(c10::cuda::getCurrentCUDAStream());
    end_events[i].record();
  }

  c10::cuda::device_synchronize();

#if PROFILING_ENABLE
  DP_LOG(NOTICE, "Starting benchmarks...");
  for (int r = 0; r < rtctx->worldSize && !no_bench; r++) {
    if (r == rtctx->rank) {
      for (size_t i = 0; i < tasks_.size(); i++) {
        auto& t = tasks_[i];
        if (t.IsNccl()) continue;
        DP_LOG(NOTICE, "Benchmarking %lu/%lu", i + 1, tasks_.size());
        bool needs_replay = (t.type_flags_ & TASK_FLAGS_EXTERNAL) != 0;
        t.timings_.profiledata = ProfileFN(
            [&]() { t.Run(c10::cuda::getCurrentCUDAStream()); }, needs_replay);
        t.timings_.has_profiledata = true;
      }
    }
    if (rtctx->global_comms) rtctx->global_comms->Barrier();
  }
#else
  // avoid unused variable warning
  (void)(no_bench);
#endif

  double total_us = 0;
  for (size_t i = 0; i < tasks_.size(); i++) {
    auto& t = tasks_.at(i);
    float tm = 1000.0 * start_events.at(i).elapsed_time(end_events.at(i));
    t.timings_.benchmark_us = tm;
    t.timings_.recent_us = tm;
#if PROFILING_ENABLE
    if (t.timings_.has_profiledata) {
      t.timings_.dram_gbps =
          t.timings_.profiledata.dram_gbytes * (1000000.0 / tm);
      if (t.timings_.dram_gbps > 1483) t.timings_.dram_gbps = 1483;
    }
#endif
    total_us += tm;
    std::cerr << t << std::endl;
  }

  std::cerr << "Total " << total_us << " us" << std::endl;
}

void GpuTask::DumpState() {
  std::set<size_t> active_tasks;
  for (uint32_t curptr = tail_ptr; curptr != head_ptr; curptr++)
    active_tasks.insert(curptr % tasks_.size());

  double total_us_bench = 0, total_us_recent = 0, total_us_last = 0;

  for (size_t i = 0; i < tasks_.size(); i++) {
    if (active_tasks.count(i)) std::cerr << "ACTIVE ";
    auto& t = tasks_.at(i);
    std::cerr << t << std::endl;
    total_us_bench += t.timings_.benchmark_us;
    total_us_recent += t.timings_.recent_us;
    total_us_last += t.timings_.last_us;
  }

  std::cerr << "Total benchmark: " << total_us_bench
            << "us, recent: " << total_us_recent
            << "us, last: " << total_us_last << "us" << std::endl;

  std::cerr << "Manager gen " << manager_gen_ << std::endl;
  std::cerr << "Hipri: " << is_high_priority
            << " waiting_recv: " << waiting_recv << std::endl;
  std::cerr << "ManagerOwned: " << IsManagerOwned()
            << " outstanding_micros: " << outstanding_micros_ << std::endl;
  std::cerr << "head: " << head_ptr << " tail: " << tail_ptr
            << " time_between_polls: " << time_between_polls << "us"
            << std::endl;
}

void GpuTask::CombineGraphs() {
  TimeRun(true);

  for (auto& t : tasks_) {
    if (t.type_flags_ & TASK_FLAGS_DO_NOT_BENCH) continue;
    t.BenchMicros();
  }

  std::vector<Tasklet> newTasks;
  std::vector<Tasklet> curMergeSet;
  double total_us = 0;
  unsigned int prevtype = 0;

  auto finMerge = [&] {
    if (!curMergeSet.size()) return;
    total_us = 0;
    if (curMergeSet.size() == 1) {
      newTasks.push_back(std::move(curMergeSet.front()));
      curMergeSet.clear();
    } else {
      newTasks.emplace_back(std::move(curMergeSet));
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
        total_us + tf.timings_.benchmark_us > graph_merge_max_us) {
      finMerge();
      prevtype = tf.type_flags_;
    }

    total_us += tf.timings_.benchmark_us;
    curMergeSet.push_back(std::move(tf));
    tasks_.erase(tasks_.begin());

    if (total_us > graph_merge_min_us ||
        curMergeSet.size() > graph_merge_max_subgraphs)
      finMerge();
  }

  finMerge();

  tasks_ = std::move(newTasks);
  TimeRun(true);
}

GpuManager* GpuManager::instance;

void GpuManager::Pause() {
  std::lock_guard<std::mutex> lock(jobListMutex);
  paused_ = true;
  for (auto& task : hptasks)
    while (task->GetEnqueuedNr()) task->PollCompletions();
  for (auto& task : lptasks)
    while (task->GetEnqueuedNr()) task->PollCompletions();
}

void GpuManager::Resume() {
  std::lock_guard<std::mutex> lock(jobListMutex);
  paused_ = false;
}

void GpuManager::EnableBe() {
  std::lock_guard<std::mutex> lock(jobListMutex);
  be_enabled_ = true;
}

void GpuManager::DisableBe() {
  std::lock_guard<std::mutex> lock(jobListMutex);
  be_enabled_ = false;
}

GpuManager::GpuManager() {
  Cycles::init();
  parse_flags();
  rth_ = std::thread([=] {
    while (!done_) DoWork();
  });
}

void GpuTask::PollCompletions() {
  unsigned int nr_completed = 0;
  uint64_t now = Cycles::microtime();

  while (head_ptr != tail_ptr) {
    auto& t = tasks_[tail_ptr % tasks_.size()];
    if (nr_completed >= 1) t.timings_.execute_time_begin = now;
    t.timings_.elapsed_execution_time = now - t.timings_.execute_time_begin;
    if (!t.timings_.ev.query()) break;
    outstanding_micros_ -= t.timings_.benchmark_us;
    nr_completed++;
    assert(now >= t.timings_.execute_time_begin);
    t.timings_.last_us = now - t.timings_.execute_time_begin;
    double uf = t.timings_.last_us > target_slowdown_factor * t.timings_.benchmark_us ? 1.0 : 0.1;
    ewma_update(t.timings_.recent_us, t.timings_.last_us, uf);
    tail_ptr.store(tail_ptr + 1, std::memory_order_release);
  }

  ewma_update(time_between_polls, now - last_poll);
  last_poll = now;
}

void GpuManager::DoWork() {
  std::lock_guard<std::mutex> lock(jobListMutex);

  if (paused_) return;

  Tasklet* cur_tasklet = nullptr;

  for (auto& task : hptasks) {
    task->PollCompletions();
    cur_tasklet = task->GetRunningTasklet();
    while (task->IsManagerOwned() &&
           (task->GetEnqueuedNr() < fg_tasklet_depth)) {
      task->RunNext();
      if (task->next_run_idx == task->tasks_.size()) {
        task->next_run_idx = 0;
        task->ManagerReleaseOwnership();
        break;
      }
    }
  }

  if (!be_enabled_) return;

  for (auto& task : lptasks) {
    task->PollCompletions();
    if (!task->IsManagerOwned()) continue;
    if (task->GetEnqueuedNr() >= be_tasklet_depth) continue;

    bool can_run_be = true;
    auto curflags = cur_tasklet ? cur_tasklet->type_flags_ : 0;
    double cur_slowdown = 0;

    if (cur_tasklet)
      cur_slowdown =
          cur_tasklet->timings_.recent_us / cur_tasklet->timings_.benchmark_us;

    /* if cur_slowdown is high, don't run */
    if (cur_slowdown > target_slowdown_factor) can_run_be = false;

    /* cannot run if we are at all reduce */
    if (curflags & TASK_FLAGS_ALLREDUCE) can_run_be = false;

    /* can run if we are blocking for a while on NCCL RECV */
    if (curflags & TASK_FLAGS_P2PCOMM_RECV) {
      auto recv_time = cur_tasklet->timings_.benchmark_us;
      /* make sure NCCL recv is going to run for a long time */
      if (recv_time > 100) can_run_be = true;
    }

    /* always run if we have flag MULTIPLEX */
    if (curflags & TASK_FLAGS_MULTIPLEX) can_run_be = true;

    if (!can_run_be) continue;

    task->RunNext();
    if (task->next_run_idx == task->tasks_.size()) {
      task->next_run_idx = 0;
      task->ManagerReleaseOwnership();
    }
  }
}