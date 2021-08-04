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

#include <torch/torch.h>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <getopt.h>
#include <unistd.h>    // For homedir
#include <sys/types.h> // For homedir
#include <pwd.h>       // For homedir
#include "runtime.h"
#include "taskManager.h"
#include "utils.h"
#include "logger.h"
#include "rpcService.h"
#include "json.hpp"
#include "communication.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "runtime.grpc.pb.h"

using json = nlohmann::json;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

/**
 * Destructing RuntimeContext.
 */
RuntimeContext::~RuntimeContext()
{
  delete grpcService;
  delete grpcServer;
}

void initGrpcServer(RuntimeContext* ctx) {
  std::string server_address(ctx->myAddr);
  ctx->grpcService = new RuntimeServiceImpl(ctx);
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(ctx->grpcService);
  ctx->grpcServer = builder.BuildAndStart().release();
  std::cout << "Server listening on " << server_address << std::endl;
  // server->Wait();  
}

void debugging(RuntimeContext* ctx) {
  DP_LOG(DEBUG, "runtime debugging function.");

  ServerContext serverCtx;
  ScheduleTrainingRequest req;
  StandardReply reply;
  std::string path = std::string(ctx->homedir) + "/DeepPoolRuntime/lastReq.txt";
  std::ifstream ifs(path.c_str());
  req.ParseFromIstream(&ifs);
  ifs.close();
  DP_LOG(DEBUG, "parsed from saved request.");
  
  ctx->grpcService->ScheduleTraining(&serverCtx, &req, &reply);

  DP_LOG(DEBUG, "runtime debugging function exits.");
}

void debuggingGrpcComm(RuntimeContext* rtctx) {
  rtctx->rankToIpAndPort.resize(2);
  rtctx->rankToIpAndPort[0] = std::string("172.31.112.33:11140");
  rtctx->rankToIpAndPort[1] = std::string("172.31.112.33:11141");
  rtctx->grpcCommReady = true;
}

void grpcCommTest(RuntimeContext* rtctx) {
  json tensorTags;
  json jobRankToGlobalRank;
  c10::Device dev(c10::DeviceType::CUDA, rtctx->device);
  auto commHandler = std::make_unique<CommunicationHandlerGRPC>(
      rtctx, "default", rtctx->worldSize, tensorTags, rtctx->rank, jobRankToGlobalRank, dev);
  DP_LOG(DEBUG, "a default commHandler created for testing.");
  sleep(5);
  commHandler->testRingP2P();
}

void parse_args(RuntimeContext* ctx, int argc, char** argv) {
  // parser.add_argument("--coordinatorAddr", type=str, default="localhost:12340",
  //                     help="IP:port to the cluster coordinator")
  // parser.add_argument("--myAddr", type=str, default="localhost:1234",
  //                     help="IP:port this runtime should listen to."
  //                     "coordinator will talk to this node on this address")
  // parser.add_argument("--device", type=int, default=0,
  //                     help="cuda device for pytorch.")
  // parser.add_argument("--c10dBackend", type=str, default="nccl",
  //                     help="pytorch c10d communication backend. Type either nccl or gloo")
  // parser.add_argument("--c10dMasterPort", type=int, default="55555",
  //                     help="coordinator's port for c10d communication package initialization")
  // parser.add_argument("--rank", type=int, default=0,
  //                     help="global rank for c10d.")
  // parser.add_argument("--worldSize", type=int, default=1,
  //                     help="global world size for c10d.")
  // parser.add_argument("--logdir", default=None, type=str)
  // parser.add_argument("--be_batch_size", default=16, type=int, help="best effort batch size, 0 for disabled")
  // parser.add_argument("--profile", default=False, action='store_true', help="runtime will be profiled")

  static struct option long_options[] = {
      {"coordinatorAddr", required_argument, NULL, 'c'},
      {"myAddr", required_argument, NULL, 'm'},
      {"device", required_argument, NULL, 'd'},
      {"c10dBackend", required_argument, NULL, 'b'},
      {"c10dMasterPort", required_argument, NULL, 'p'},
      {"rank", required_argument, NULL, 'r'},
      {"worldSize", required_argument, NULL, 'w'},
      {"logdir", required_argument, NULL, 'l'},
      {"be_batch_size", required_argument, NULL, 'e'},
      {"profile", no_argument, NULL, 'f'},
      {"debug", no_argument, NULL, 'g'},
      {NULL, 0, NULL, 0}
  };

  // loop over all of the options
  char ch;
  while ((ch = getopt_long(argc, argv, "t:a:", long_options, NULL)) != -1) {
    switch (ch) {
      case 'c':
        ctx->coordinatorAddr = optarg; // or copy it if you want to
        break;
      case 'm':
        ctx->myAddr = optarg;
        break;
      case 'd':
        ctx->device = atoi(optarg);
        break;
      case 'b':
        ctx->c10dBackend = optarg;
        break;
      case 'p':
        ctx->c10dMasterPort = atoi(optarg);
        break;
      case 'r':
        ctx->rank = atoi(optarg);
        break;
      case 'w':
        ctx->worldSize = atoi(optarg);
        break;
      case 'l':
        ctx->logdir = optarg;
        break;
      case 'e':
        ctx->be_batch_size = atoi(optarg);
        break;
      case 'f':
        ctx->profile = true;
        break;
      case 'g':
        ctx->debug = true;
        break;
      default:
        printf("?? getopt returned character code 0%o ??\n", ch);
    }
  }
}

int main(int argc, char** argv) {
  RuntimeContext ctx;
  ctx.shutdownRequested = false;
  parse_args(&ctx, argc, argv);
  
  std::string logFilePath = format("%scpprt%d.out", ctx.logdir, ctx.rank);
  Logger::get().setLogFile(logFilePath.c_str(), true);
  Logger::get().setLogLevel(DEBUG);
  // Logger::get().setLogLevel(NOTICE);

  // Retrieve homedir path.
  if ((ctx.homedir = getenv("HOME")) == NULL) {
    ctx.homedir = getpwuid(getuid())->pw_dir;
  }
  DP_LOG(NOTICE, "The home dir path: %s", ctx.homedir);
  DP_LOG(NOTICE, "Current file path: %s", argv[0]);

  TaskManager taskMngr(&ctx);

  std::cout << "myAddr: " << ctx.myAddr << " rank: " << ctx.rank << std::endl;
  initGrpcServer(&ctx);

  if (ctx.debug) {
    debugging(&ctx);
    // debuggingGrpcComm(&ctx);
  }

  if (strcmp(ctx.c10dBackend, "grpc") == 0) {
    DP_LOG(DEBUG, "GRPC commBackend is used. Waiting for InitCommGRPC.");
    while (!ctx.grpcCommReady.load(std::memory_order_relaxed)) {}
    DP_LOG(DEBUG, "InitCommGRPC done. Running test now.");
    grpcCommTest(&ctx);
  }

  std::cout << "poller is starting." << std::endl;
  while (!ctx.shutdownRequested.load(std::memory_order_relaxed)) {
    taskMngr.poll();
  }
  std::cout << "poller exits." << std::endl;
  ctx.grpcServer->Shutdown();
  std::cout << "grpc shutdown." << std::endl;
  ctx.grpcServer->Wait();
  return 0;
}