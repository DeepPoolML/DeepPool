#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "CUDAGraph.h"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

#define TASK_FLAGS_COMPUTE (1U << 0)
#define TASK_FLAGS_MEMCPY (1U << 1)
#define TASK_FLAGS_P2PCOMM (1U << 2)
#define TASK_FLAGS_P2PCOMM_RECV (1U << 3)
#define TASK_FLAGS_ALLREDUCE (1U << 4)
#define TASK_FLAGS_DO_NOT_BENCH (1U << 5)
#define TASK_FLAGS_EXTERNAL (1U << 6)

struct OwnedGraph {
  OwnedGraph() {}
  void Init() {
    assert(!graph);
    CUDA_API_CALL(cudaGraphCreate(&graph, 0));
  }
  void Instantiate(c10::optional<c10::cuda::CUDAStream> stream = {}) {
    assert(graph);
    assert(!graph_exec);
    CUDA_API_CALL(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    if (stream) {
      CUDA_API_CALL(cudaGraphUpload(graph_exec, stream.value()));
    }
  }
  void Exec(c10::cuda::CUDAStream stream) {
    if (!graph_exec) Instantiate(stream);
    CUDA_API_CALL(cudaGraphLaunch(graph_exec, stream));
  }
  void DestroyExec() {
    if (!graph_exec) return;
    CUDA_API_CALL(cudaGraphExecDestroy(graph_exec));
    graph_exec = nullptr;
  }
  void DestroyGraph() {
    if (!graph) return;
    CUDA_API_CALL(cudaGraphDestroy(graph));
    graph = nullptr;
  }
  cudaGraph_t getGraph() { return graph; }
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t graph_exec{nullptr};
  ~OwnedGraph() {
    DestroyGraph();
    DestroyExec();
  }
  OwnedGraph(const OwnedGraph&) = delete;
  OwnedGraph& operator=(const OwnedGraph&) = delete;
  OwnedGraph(OwnedGraph&& other) {
    std::swap(graph, other.graph);
    std::swap(graph_exec, other.graph_exec);
  }
  OwnedGraph& operator=(OwnedGraph&& other) {
    std::swap(graph, other.graph);
    std::swap(graph_exec, other.graph_exec);
    return *this;
  };
};

class Tasklet {
 public:
  Tasklet(std::shared_ptr<DeepPool::CUDAGraph> graph, unsigned int flags,
          std::string debug_name = "");

  Tasklet(std::function<void(c10::cuda::CUDAStream stream)> cb,
          unsigned int flags, std::string debug_name = "");

  Tasklet(cudaGraphExec_t exec, unsigned int flags,
          std::string debug_name = "");

  Tasklet(std::vector<Tasklet> subtasks);

  inline bool IsNccl() {
    return (type_flags_ & (TASK_FLAGS_ALLREDUCE | TASK_FLAGS_P2PCOMM |
                           TASK_FLAGS_P2PCOMM_RECV)) > 0;
  }
  double BenchMicros(size_t trials = 200);

  void Run(c10::cuda::CUDAStream stream);

  Tasklet(const Tasklet&) = delete;
  Tasklet& operator=(const Tasklet&) = delete;
  Tasklet(Tasklet&& other) { moveHelper(std::move(other)); }
  Tasklet& operator=(Tasklet&& other) {
    moveHelper(std::move(other));
    return *this;
  };

  struct Timings {
    double recorded_us{0};
  };

  friend std::ostream& operator<<(std::ostream& out, const Tasklet& k);

 protected:
  friend class GpuTask;
  OwnedGraph graph_holder_;
  unsigned int type_flags_{0};
  std::string debug_name_;
  std::vector<Tasklet> subtasks_;
  Timings timings_;
  std::function<void(c10::cuda::CUDAStream stream)> cb_;
  std::vector<std::shared_ptr<void>> refs;

  /* execution data */
  at::cuda::CUDAEvent ev_;

 private:
  void moveHelper(Tasklet&& other) {
    std::swap(graph_holder_, other.graph_holder_);
    std::swap(type_flags_, other.type_flags_);
    std::swap(debug_name_, other.debug_name_);
    std::swap(subtasks_, other.subtasks_);
    std::swap(timings_, other.timings_);
    std::swap(cb_, other.cb_);
    std::swap(refs, other.refs);
    std::swap(ev_, other.ev_);
  }
};

class GpuTask {
 public:
  GpuTask(bool hipri, c10::cuda::CUDAStream execution_stream)
      : is_high_priority(hipri),
        execution_stream(execution_stream),
        nccl_stream(c10::cuda::getStreamFromPool(hipri)) {}
  GpuTask(bool hipri)
      : is_high_priority(hipri),
        execution_stream(c10::cuda::getStreamFromPool(hipri)),
        nccl_stream(c10::cuda::getStreamFromPool(hipri)) {}
  GpuTask(bool hipri, c10::cuda::CUDAStream execution_stream,
          std::vector<cudaGraphExec_t> pieces);

  void AddTask(Tasklet&& t) { tasks_.push_back(std::move(t)); }
  bool RunNext(c10::cuda::CUDAStream stream);
  bool RunNext() { return RunNext(execution_stream); }
  unsigned int GetFlagsQueuedTasks();

  void ExecuteTasks();
  void Reset();
  void TimeRun();
  void CombineGraphs();

  bool IsManagerOwned() { return owner_.load(std::memory_order_acquire) == 1; }

  void FinishCompletion() {
    assert(!IsManagerOwned());
    auto hp = head_ptr.load(std::memory_order_acquire);
    while (tail_ptr.load() != hp)
      ;
  }

  void MakeManagerOwned() {
    assert(!IsManagerOwned());
    owner_.store(1, std::memory_order_release);
  }

  void ManagerReleaseOwnershipNoCheck() {
    owner_.store(0, std::memory_order_release);
  }

  void ManagerReleaseOwnership() {
    assert(IsManagerOwned());
    ManagerReleaseOwnershipNoCheck();
  }

  unsigned int GetEnqueuedNr() { return head_ptr - tail_ptr; }

  size_t GetEnqueuedMicros() { return outstanding_micros_; }

  void PollCompletions() {
    while (head_ptr != tail_ptr) {
      auto& t = tasks_[tail_ptr % tasks_.size()];
      if (!t.ev_.query()) break;
      outstanding_micros_ -= t.timings_.recorded_us;
      tail_ptr.store(tail_ptr + 1, std::memory_order_release);
    }
  }

 private:
  friend class GpuManager;
  std::vector<Tasklet> tasks_;
  bool is_high_priority{true};
  bool waiting_recv{false};
  std::atomic<size_t> owner_{0};
  size_t outstanding_micros_{0};
  size_t next_run_idx{0};
  at::cuda::CUDAEvent ev;
  CUDAPipeline p{1};
  c10::cuda::CUDAStream execution_stream, nccl_stream;
  std::atomic<uint32_t> head_ptr{0}, tail_ptr{0};
};

class GpuManager {
 public:
  static GpuManager* getInstance() {
    if (!instance) instance = new GpuManager;
    return instance;
  }

  void RemoveTask(std::shared_ptr<GpuTask> task) {
    std::lock_guard<std::mutex> lock(jobListMutex);

    if (task == hptask) {
      hptask->ManagerReleaseOwnershipNoCheck();
      hptask.reset();
      return;
    }

    if (task == lptask) {
      lptask->ManagerReleaseOwnershipNoCheck();
      lptask.reset();
      return;
    }

    assert(false && "Removing task that was never added");
  }

  void AddTask(std::shared_ptr<GpuTask> task) {
    std::lock_guard<std::mutex> lock(jobListMutex);
    if (task->is_high_priority)
      hptask = task;
    else
      lptask = task;
  }

  void DoWork();

 private:
  GpuManager() {
    rth_ = std::thread([=] {
      while (!done_) DoWork();
    });
  }
  std::thread rth_;
  bool done_{false};
  std::mutex jobListMutex;
  std::shared_ptr<GpuTask> hptask;
  std::shared_ptr<GpuTask> lptask;
  static GpuManager* instance;
};
