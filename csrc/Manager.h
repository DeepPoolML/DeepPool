#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <set>
#include <vector>

#include "CUDAGraph.h"
#include "Cycles.h"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

#define PROFILING_ENABLE 0
#if PROFILING_ENABLE
#include "proflib/profiling.h"
#endif

#define TASK_FLAGS_COMPUTE (1U << 0)
#define TASK_FLAGS_MEMCPY (1U << 1)
#define TASK_FLAGS_P2PCOMM (1U << 2)
#define TASK_FLAGS_P2PCOMM_RECV (1U << 3)
#define TASK_FLAGS_ALLREDUCE (1U << 4)
#define TASK_FLAGS_DO_NOT_BENCH (1U << 5)
#define TASK_FLAGS_EXTERNAL (1U << 6)
#define TASK_FLAGS_MULTIPLEX (1U << 7)

class OwnedGraph {
 public:
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

 private:
  friend class Tasklet;
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t graph_exec{nullptr};
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

  inline bool IsNccl() const {
    return (type_flags_ & (TASK_FLAGS_ALLREDUCE | TASK_FLAGS_P2PCOMM |
                           TASK_FLAGS_P2PCOMM_RECV)) > 0;
  }

  inline bool IsAsync() const {
    return IsNccl() && (type_flags_ &
                        (TASK_FLAGS_ALLREDUCE | TASK_FLAGS_P2PCOMM_RECV)) == 0;
  }

  void BenchMicros(size_t trials = 200);
  void Run(c10::cuda::CUDAStream stream);

  Tasklet(const Tasklet&) = delete;
  Tasklet& operator=(const Tasklet&) = delete;
  Tasklet(Tasklet&& other) { moveHelper(std::move(other)); }
  Tasklet& operator=(Tasklet&& other) {
    moveHelper(std::move(other));
    return *this;
  };

  struct Timings {
    double benchmark_us{0};
    double recent_us{0};
    double last_us{0};
    at::cuda::CUDAEvent ev;
    uint64_t enqueue_time;
    uint64_t execute_time_begin;
    uint64_t elapsed_execution_time;
#if PROFILING_ENABLE
    bool has_profiledata{false};
    double dram_gbps;
    ProfileData profiledata;
#endif
  };

  friend std::ostream& operator<<(std::ostream& out, const Tasklet& k);

 protected:
  friend class GpuTask;
  friend class GpuManager;
  unsigned int type_flags_{0};
  std::string debug_name_;

  /**
   three types of tasklets:
     1. Supertasklet, has a graph in graph_holder_ comprised of multiple graphs
     stored in subtasks_
     2. Regular tasklet, graph stored in graph_holder_
     3. Callback tasklet, launches a callback function stored at cb_
  **/

  OwnedGraph graph_holder_;
  std::vector<Tasklet> subtasks_;
  std::function<void(c10::cuda::CUDAStream stream)> cb_;
  std::vector<std::shared_ptr<void>> refs;

  Timings timings_;

 private:
  void moveHelper(Tasklet&& other) {
    std::swap(graph_holder_, other.graph_holder_);
    std::swap(type_flags_, other.type_flags_);
    std::swap(debug_name_, other.debug_name_);
    std::swap(subtasks_, other.subtasks_);
    std::swap(cb_, other.cb_);
    std::swap(refs, other.refs);
    std::swap(timings_, other.timings_);
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

  void ExecuteTasks();
  void Reset();
  void TimeRun(bool);
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
    manager_gen_++;
    owner_.store(1, std::memory_order_release);
  }

  void ManagerReleaseOwnershipNoCheck() {
    owner_.store(0, std::memory_order_release);
  }

  void DumpState();

  void ManagerReleaseOwnership() {
    assert(IsManagerOwned());
    ManagerReleaseOwnershipNoCheck();
  }

  unsigned int GetEnqueuedNr() { return head_ptr - tail_ptr; }

  size_t GetEnqueuedMicros() { return outstanding_micros_; }

  Tasklet* GetRunningTasklet() {
    if (head_ptr == tail_ptr) return nullptr;
    return &tasks_[tail_ptr % tasks_.size()];
  }

  void PollCompletions();

 private:
  static inline void ewma_update(double& variable, uint64_t newval,
                                 double weight = 0.1) {
    // constexpr double weight = 0.1;
    variable = variable * (1 - weight) + static_cast<double>(newval) * weight;
  }

  friend class GpuManager;
  std::vector<Tasklet> tasks_;
  bool is_high_priority{true};
  bool waiting_recv{false};
  std::atomic<size_t> owner_{0};
  size_t outstanding_micros_{0};
  size_t manager_gen_{0};
  size_t next_run_idx{0};
  at::cuda::CUDAEvent ev;
  c10::cuda::CUDAStream execution_stream, nccl_stream;
  std::atomic<uint32_t> head_ptr{0}, tail_ptr{0};

  uint64_t last_poll{0}, last_log{0};
  double time_between_polls{0};
};

class GpuManager {
 public:
  static GpuManager* getInstance() {
    if (!instance) instance = new GpuManager;
    return instance;
  }

  void Pause();
  void Resume();
  void EnableBe();
  void DisableBe();

  void RemoveTask(std::shared_ptr<GpuTask> task) {
    std::lock_guard<std::mutex> lock(jobListMutex);

    auto it = hptasks.find(task);
    if (it != hptasks.end()) {
      task->ManagerReleaseOwnershipNoCheck();
      hptasks.erase(it);
      return;
    }

    it = lptasks.find(task);
    assert(it != lptasks.end() && "Removing task that was never added");
    task->ManagerReleaseOwnershipNoCheck();
    lptasks.erase(it);
  }

  void AddTask(std::shared_ptr<GpuTask> task) {
    std::lock_guard<std::mutex> lock(jobListMutex);
    if (task->is_high_priority)
      hptasks.insert(task);
    else
      lptasks.insert(task);
  }

  void DoWork();

 private:
  GpuManager();
  std::thread rth_;
  volatile bool done_{false};
  bool be_enabled_{true};
  bool paused_{false};
  std::mutex jobListMutex;
  std::set<std::shared_ptr<GpuTask>> hptasks;
  std::set<std::shared_ptr<GpuTask>> lptasks;

  static GpuManager* instance;
};
