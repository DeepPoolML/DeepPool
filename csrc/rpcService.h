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

#ifndef RPC_SERVICE_H
#define RPC_SERVICE_H

#include <torch/torch.h>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "runtime.grpc.pb.h"

using grpc::ServerContext;
using grpc::Status;

class RuntimeContext;

/**
 * GRPC service implementation for runtime.
 */
class RuntimeServiceImpl final : public Runtime::Service {
 public:
  RuntimeServiceImpl(RuntimeContext* ctx)
    : Runtime::Service(), rtctx(ctx) {}

//  private:
 public: // temporarily made public for testing.
  Status InitCommGRPC(ServerContext* context,
                      const InitCommGRPCRequest* request,
                      StandardReply* reply) override;
  Status ScheduleTraining(ServerContext* context,
                          const ScheduleTrainingRequest* request,
                          StandardReply* reply) override;
  Status Poke(ServerContext* context, const Empty* request,
              StandardReply* reply) override;
  Status Shutdown(ServerContext* context, const Empty* request,
                  StandardReply* reply) override;
  Status P2PCommunication(ServerContext* context,
                          const P2PCommunicationRequest* request,
                          StandardReply* reply) override;

 private:
  RuntimeContext* rtctx;
};

/**
 * GRPC client for runtime service.
 * It's particially implemented as most services are used by cluster.py.
 */
class RuntimeClient {
 public:
  RuntimeClient(std::shared_ptr<grpc::ChannelInterface> channel)
    : stub_(Runtime::NewStub(channel)) {}
  
  std::string P2PCommunication(const std::string& taskName,
                               const torch::Tensor& tensor, int tag);

 private:
  std::unique_ptr<Runtime::Stub> stub_;
};


#endif // RPC_SERVICE_H