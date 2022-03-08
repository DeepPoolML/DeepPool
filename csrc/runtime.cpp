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

#include "runtime.h"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <string>

#include "BeTask.h"
#include "JobContext.h"
#include "Manager.h"
#include "communication.h"
#include "json.hpp"
#include "logger.h"
#include "rpcService.h"
#include "runtime.grpc.pb.h"
#include "utils.h"

using json = nlohmann::json;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

ABSL_FLAG(std::string, myAddr, "localhost:1234", "");
ABSL_FLAG(int, device, 0, "");
ABSL_FLAG(std::string, c10dBackend, "nccl", "");
ABSL_FLAG(int, rank, 0, "");
ABSL_FLAG(int, worldSize, 1, "");
ABSL_FLAG(std::string, logdir, "", "");
ABSL_FLAG(long, be_batch_size, 0, "");
ABSL_FLAG(size_t, sample_per_kernel, 32, "");
ABSL_FLAG(bool, profile_stage_time, false, "");
ABSL_FLAG(bool, profile_layer_times_graph, false, "");
ABSL_FLAG(bool, profile_layer_times_timers, false, "");
ABSL_FLAG(bool, cuda_profile, false,
          "use cuda profiler API to mark an iteration for profiling");

ABSL_FLAG(bool, debug, false, "");
ABSL_FLAG(std::string, be_jit_file,
          "/home/seojin/DeepPoolRuntime/beModules/resnet.jit", "");
ABSL_FLAG(bool, use_fg_graph, true, "");
ABSL_FLAG(bool, use_be_graph, true, "");
ABSL_FLAG(size_t, iters_per_capture, 4, "");
ABSL_FLAG(size_t, min_layer_sync, 8, "");
ABSL_FLAG(size_t, sync_bucket_size, 10 * 1000 * 1000, "");
ABSL_FLAG(std::string, bg_json_file, "", "");
ABSL_FLAG(double, be_graph_split_ms, 0.5, "");

/**
 * Destructing RuntimeContext.
 */
RuntimeContext::~RuntimeContext() {
  delete grpcService;
  delete grpcServer;
}

void initGrpcServer(RuntimeContext* ctx) {
  ctx->grpcService = new RuntimeServiceImpl(ctx);
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  builder.AddListeningPort(absl::GetFlag(FLAGS_myAddr),
                           grpc::InsecureServerCredentials());
  builder.RegisterService(ctx->grpcService);
  ctx->grpcServer = builder.BuildAndStart().release();
  std::cout << "Server listening on " << absl::GetFlag(FLAGS_myAddr)
            << std::endl;
  // server->Wait();
}

void debugging(RuntimeContext* ctx) {
  DP_LOG(DEBUG, "runtime debugging function.");

  ServerContext serverCtx;
  ScheduleTrainingRequest req;
  StandardReply reply;
  std::string path = format("%s/lastReq%d.txt",
                            absl::GetFlag(FLAGS_logdir).c_str(), ctx->rank);
  std::ifstream ifs(path.c_str());
  req.ParseFromIstream(&ifs);
  ifs.close();
  DP_LOG(DEBUG, "parsed from saved request.");

  ctx->grpcService->ScheduleTraining(&serverCtx, &req, &reply);

  DP_LOG(DEBUG, "runtime debugging function exits.");
}

void debuggingGrpcComm() {
  rtctx->rankToIpAndPort.resize(2);
  rtctx->rankToIpAndPort[0] = std::string("172.31.112.33:11140");
  rtctx->rankToIpAndPort[1] = std::string("172.31.112.33:11141");
  rtctx->grpcCommReady = true;
}

void grpcCommTest() {
  json tensorTags;
  json jobRankToGlobalRank;
  auto commHandler = std::make_unique<CommunicationHandlerGRPC>(
      "default", rtctx->worldSize, tensorTags, rtctx->rank,
      jobRankToGlobalRank);
  DP_LOG(DEBUG, "a default commHandler created for testing.");
  sleep(5);
  commHandler->testRingP2P();
}

void ncclCommTest() {
  json tensorTags;
  json jobRankToGlobalRank;
  auto commHandler = std::make_shared<CommunicationHandlerNCCL>(
      "default", rtctx->worldSize, tensorTags, rtctx->rank,
      jobRankToGlobalRank);
  DP_LOG(DEBUG, "a default commHandler created for testing.");
  commHandler->testRingP2P();
  commHandler->testAllReduce();
  rtctx->global_comms = commHandler;
}

/**
 * Adds a new training job submitted by coordinator.
 *
 * \return  The number of jobs currently scheduled.
 */
int RuntimeContext::addTrainingJob(std::unique_ptr<JobContext> job) {
  std::lock_guard<std::mutex> lock(jobListMutex);
  jobList.push_back(std::move(job));
  DP_LOG(LogLevel::NOTICE, "Added a new job. %s", jobList.back()->name.c_str());
  return jobList.size();
}

/**
 * A poller to make a progress on training tasks.
 *
 * \return The number of jobs that are executed (or scheduled to CUDA).
 */
int RuntimeContext::poll() {
  std::lock_guard<std::mutex> lock(jobListMutex);

  if (jobList.empty()) {
    return 0;
  }

  JobContext* mainJob = jobList[0].get();

  if (IsBeEnabled()) {
    if (mainJob->RunWithBe())
      GpuManager::getInstance()->EnableBe();
    else
      GpuManager::getInstance()->DisableBe();
  }

  if (mainJob->ShouldRunTest()) mainJob->Test();
  for (size_t i = 0; i < mainJob->GetEpochsToTrain(); i++) {
    mainJob->TrainOneEpoch();
    if (mainJob->ShouldRunTest()) mainJob->Test();
  }

  torch_stream.synchronize();
  mainJob->printJobStatistics();
  jobList.erase(jobList.begin());
  DP_LOG(NOTICE, "Removed the completed job. Remaining: %lu", jobList.size());
  c10::cuda::CUDACachingAllocator::emptyCache();
  return 1;
}

static BeTaskConfig becfg;

void parse_args(RuntimeContext& ctx, int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

#define PARSEFLAG(x) ctx.x = absl::GetFlag(FLAGS_##x);
  PARSEFLAG(min_layer_sync);
  PARSEFLAG(sync_bucket_size);
  PARSEFLAG(iters_per_capture);
  PARSEFLAG(bg_json_file);
  PARSEFLAG(device);
  PARSEFLAG(c10dBackend);
  PARSEFLAG(use_fg_graph);
  PARSEFLAG(rank);
  PARSEFLAG(worldSize);
  PARSEFLAG(profile_stage_time);
  PARSEFLAG(profile_layer_times_graph);
  PARSEFLAG(profile_layer_times_timers);
  PARSEFLAG(cuda_profile);
  PARSEFLAG(debug);
  PARSEFLAG(logdir);
#undef PARSEFLAG

#define PARSEFLAG(x) becfg.x = absl::GetFlag(FLAGS_##x);
  PARSEFLAG(be_batch_size);
  PARSEFLAG(sample_per_kernel);
  PARSEFLAG(use_be_graph);
  PARSEFLAG(be_jit_file);
  PARSEFLAG(be_graph_split_ms);
#undef PARSEFLAG

  ctx.c10dev = c10::Device(c10::DeviceType::CUDA, ctx.device);
}

RuntimeContext* rtctx; /* global rtctx variable */

int main(int argc, char** argv) {
  // ProfilerInit(0);
  RuntimeContext ctx;
  rtctx = &ctx;
  ctx.shutdownRequested = false;
  parse_args(ctx, argc, argv);

  std::string logFilePath =
      format("%s/cpprt%d.out", rtctx->logdir.c_str(), ctx.rank);
  Logger::get().setLogFile(logFilePath.c_str(), false);
  // Logger::get().setLogLevel(DEBUG);
  Logger::get().setLogLevel(NOTICE);

  DP_LOG(NOTICE, "Current file path: %s", argv[0]);

  InitBeTask(becfg);

  std::cout << "myAddr: " << absl::GetFlag(FLAGS_myAddr)
            << " rank: " << ctx.rank << std::endl;
  std::cout << "myPID: " << getpid() << std::endl;
  initGrpcServer(&ctx);

  if (ctx.debug) {
    debugging(&ctx);
    // debuggingGrpcComm();
  }

  if (ctx.c10dBackend == "grpc") {
    DP_LOG(DEBUG, "GRPC commBackend is used. Waiting for InitCommGRPC.");
    while (!ctx.grpcCommReady.load(std::memory_order_relaxed)) {
    }
    DP_LOG(DEBUG, "InitCommGRPC done. Running test now.");
    grpcCommTest();
    DP_LOG(DEBUG, "GRPC comm test done.");
  }

  if (ctx.c10dBackend == "nccl" && rtctx->worldSize > 1) {
    DP_LOG(DEBUG, "NCCL commBackend is used. Waiting for InitCommNCCL.");
    while (!ctx.ncclCommReady.load(std::memory_order_relaxed)) {
    }
    DP_LOG(DEBUG, "InitCommNCCL done. Running test now.");
    ncclCommTest();
  }

  int version;
  CUDACHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    std::cout
        << "WARNING: old cuda version detected, may have bugs with CUDA graphs"
        << std::endl;
    std::cerr
        << "WARNING: old cuda version detected, may have bugs with CUDA graphs"
        << std::endl;
    DP_LOG(
        ERROR,
        "WARNING: old cuda version detected, may have bugs with CUDA graphs");
  }

  std::thread([&] {
    using namespace std::chrono;
    size_t lastc = GetBeCounter();
    size_t lastfg = ctx.fgcounter.load();
    auto lastt = steady_clock::now();
    while (true) {
      sleep(1);
      size_t newtr = GetBeCounter();
      size_t newfg = ctx.fgcounter.load();
      auto now = steady_clock::now();
      auto s = duration_cast<seconds>(now - lastt).count();
      std::cerr << "BE im/s: " << (newtr - lastc) / s
                << " FG iter/s: " << (newfg - lastfg) / s << std::endl;
      lastt = now;
      lastc = newtr;
      lastfg = newfg;
    }
  }).detach();

  std::cout << "poller is starting." << std::endl << std::flush;
  DP_LOG(DEBUG, "Poller is starting.");
  while (!ctx.shutdownRequested.load(std::memory_order_relaxed)) {
    ctx.poll();
  }
  std::cout << "poller exits." << std::endl;
  ctx.grpcServer->Shutdown();
  std::cout << "grpc shutdown." << std::endl;
  ctx.grpcServer->Wait();
  return 0;
}
