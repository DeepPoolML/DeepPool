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

#ifndef RUNTIME_H
#define RUNTIME_H

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "JobContext.h"

#define VERBOSE 0

/**
 * Forward declarations
 */
class RuntimeServiceImpl;
namespace grpc {
class Server;
};

struct NcclGroupConfig {
  torch::cuda::nccl::ncclUniqueId ncclGroupId;
  torch::cuda::nccl::ncclComm_t ncclCommObj;
  std::vector<int> ranks;
  int myRank;
  size_t group_key;
};

/* Get 64-bit bitmap key for a set of ranks */
static inline size_t RankVecToKey(std::vector<int> ranks) {
  size_t key = 0;
  for (auto& i : ranks) {
    assert(i < 64);
    key |= 1UL << i;
  }
  return key;
}

/**
 * Context holding data for Runtime.
 */
struct RuntimeContext {
  RuntimeContext()
      : c10dev(c10::DeviceType::CUDA, 0),
        torch_stream(c10::cuda::getStreamFromPool(true)),
        xfer_stream(c10::cuda::getStreamFromPool(true)),
        grad_sync_stream(c10::cuda::getStreamFromPool(true)) {
    c10::cuda::setCurrentCUDAStream(torch_stream);
  }

  ~RuntimeContext();  // Defined in cpp file because of incomplete unique_ptrs.

  /**
   * Populated by commandline arguments
   */
  int device;
  std::string c10dBackend;
  int rank;
  int worldSize;
  bool profile_stage_time;
  bool profile_layer_times_graph;
  bool profile_layer_times_timers;
  bool cuda_profile;
  bool debug;
  std::string logdir;
  c10::Device c10dev;
  bool use_fg_graph;
  size_t min_layer_sync;
  size_t sync_bucket_size;
  std::atomic<uint64_t> fgcounter{0};

  std::mutex jobListMutex;
  std::vector<std::unique_ptr<JobContext>> jobList;
  int addTrainingJob(std::unique_ptr<JobContext> job);
  int poll();


  std::shared_ptr<CommunicationHandler> global_comms;

  /**
   *  additional variables.
   */
  RuntimeServiceImpl* grpcService;
  grpc::Server* grpcServer;
  std::atomic<bool>
      shutdownRequested;  // Set to true when coordinator shuts down.
  std::map<std::string, CommunicationHandler*> commHandlerMap;
  std::vector<std::string> rankToIpAndPort;
  std::atomic<bool> grpcCommReady;

  /**
   * variables to maintain per NCCL comm group
   * need to be expanded if one node participates in more than one comm group
   */

  std::atomic<bool> ncclCommReady{false};
  std::map<size_t, NcclGroupConfig> nccl_groups;
  NcclGroupConfig maingroup;
  c10::cuda::CUDAStream torch_stream, xfer_stream;
  c10::cuda::CUDAStream grad_sync_stream;
};

extern RuntimeContext* rtctx;

#endif  // RUNTIME_H