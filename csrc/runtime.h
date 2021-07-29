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


/**
 * Forward declarations
 */
class RuntimeServiceImpl;
class TaskManager;
namespace grpc {
  class Server;
};
class RunnableModule;
namespace torch {
  namespace optim {
    class Optimizer;
  }
}

/**
 * Context holding data for Runtime.
 */
struct RuntimeContext {
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
  char *homedir;

  /**
   *  additional variables.
   */
  std::unique_ptr<RuntimeServiceImpl> grpcService;
  std::unique_ptr<grpc::Server> grpcServer;
  TaskManager* taskManager;
  std::atomic<bool> shutdownRequested;           // Set to true when coordinator shuts down.
};




#endif // RUNTIME_H