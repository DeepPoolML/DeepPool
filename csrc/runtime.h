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

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <map>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>

#define VERBOSE 0

/**
 * Forward declarations
 */
class CommunicationHandler;
class RuntimeServiceImpl;
class RunnableModule;
class TaskManager;
namespace grpc {
  class Server;
};
namespace torch {
  namespace optim {
    class Optimizer;
  }
}

/**
 * Context holding data for Runtime.
 */
struct RuntimeContext {
  RuntimeContext() : coordinatorAddr(0), myAddr(0), device(0), c10dBackend(0),
      c10dMasterPort(0), rank(), worldSize(), logdir(), be_batch_size(0),
      profile(false), debug(false), verify(false), homedir(0), c10dev(c10::DeviceType::CUDA, 0),
      grpcService(), grpcServer(), taskManager(), shutdownRequested(),
      commHandlerMap(), rankToIpAndPort(), grpcCommReady(),
      ncclGroupId(), ncclGroupSize(), ranks(), ncclCommReady(), ncclCommObj(),
      torch_stream(c10::cuda::getStreamFromPool(true)), xfer_stream(c10::cuda::getStreamFromPool(true)), grad_sync_stream(c10::cuda::getStreamFromPool(true)) {
        c10::cuda::setCurrentCUDAStream(torch_stream);
      }

  ~RuntimeContext(); // Defined in cpp file because of incomplete unique_ptrs.

  /**
   * Populated by commandline arguments
   */
  char* coordinatorAddr;  // includes port number.
  char* myAddr;           // includes port number.
  int device;
  char* c10dBackend;
  int c10dMasterPort;
  int rank;
  int worldSize;
  char* logdir;
  int be_batch_size;
  bool profile;
  bool debug;
  bool verify;
  char *homedir;
  c10::Device c10dev;

  /**
   *  additional variables.
   */
  RuntimeServiceImpl* grpcService;
  grpc::Server* grpcServer;
  TaskManager* taskManager;
  std::atomic<bool> shutdownRequested;  // Set to true when coordinator shuts down.
  std::map< std::string, CommunicationHandler* > commHandlerMap;
  std::vector<std::string> rankToIpAndPort;
  std::atomic<bool> grpcCommReady;

  /**
   * variables to maintain per NCCL comm group
   * need to be expanded if one node participates in more than one comm group
   */

  torch::cuda::nccl::ncclUniqueId ncclGroupId;
  int ncclGroupSize;
  std::vector<int> ranks;
  std::atomic<bool> ncclCommReady{false};
  torch::cuda::nccl::ncclComm_t ncclCommObj;
  c10::cuda::CUDAStream torch_stream, xfer_stream;
  c10::cuda::CUDAStream grad_sync_stream;

};

extern RuntimeContext *rtctx;


#endif // RUNTIME_H