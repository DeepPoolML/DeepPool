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


#include "rpcService.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include "communication.h"
#include "runtime.h"
#include "json.hpp"
#include "utils.h"
#include "logger.h"
#include "runnableModule.h"
#include "taskManager.h"
#include "nccl.h"
#include "cuda_runtime.h"

#include <grpcpp/grpcpp.h>
#include "runtime.grpc.pb.h"

using json = nlohmann::json;
using grpc::Server;
using grpc::ServerContext;
using grpc::Status;

Status
RuntimeServiceImpl::InitCommGRPC(ServerContext* context,
    const InitCommGRPCRequest* request,
    StandardReply* reply)
{
  UNUSED(context);
  DP_LOG(DEBUG, "Received InitCommGRPC().");

  json rankToIpMapJson = json::parse(request->rank_to_ip_map_in_json());
  
  rtctx->rankToIpAndPort.resize(rankToIpMapJson.size());
  for (auto& el : rankToIpMapJson.items()) {
    int rank = atoi(el.key().c_str());
    std::string ipAndPort = el.value().get<std::string>();
    rtctx->rankToIpAndPort[rank] = ipAndPort;
    DP_LOG(DEBUG, "Rank %d 's address: %s", rank, ipAndPort.c_str());
  }
  rtctx->grpcCommReady = true;

  std::string replyMsg("InitCommGRPC invoked.");
  reply->set_message(replyMsg);
  return Status::OK;
}

Status
RuntimeServiceImpl::InitCommNCCL(ServerContext* context,
    const InitCommNCCLMsg* request,
    InitCommNCCLMsg* reply)
{
  UNUSED(context);
  DP_LOG(DEBUG, "Received InitCommNCCL().");

  int msg_type = request->msg_type();
  int group_size = request->group_size();

  if (msg_type == 0) { // Generate comm group ID
    if (rtctx->rank == 0) { // Only rank 0 generates ID
      rtctx->ncclGroupSize = group_size;
      NCCL_API_CALL(ncclGetUniqueId(&rtctx->ncclGroupId));

      std::string replyMsg("Comm group ID generated at rank 0.");
      reply->set_message(replyMsg);
      reply->set_group_id(&rtctx->ncclGroupId, sizeof(rtctx->ncclGroupId));
    }
  }
  else if (msg_type == 1) { // Join comm group specified by ID
    if (rtctx->rank != 0) // Ranks 1+ need to receive ID before joining
      memcpy(&rtctx->ncclGroupId, request->group_id().c_str(), sizeof(rtctx->ncclGroupId));

    NCCL_API_CALL(ncclCommInitRank(&rtctx->ncclCommObj, rtctx->worldSize, rtctx->ncclGroupId, rtctx->rank));
    rtctx->ncclCommReady = true;

    std::string replyMsg("Comm group ID broadcast & joined.");
    reply->set_message(replyMsg);
  }

  return Status::OK;
}

Status
RuntimeServiceImpl::ScheduleTraining(ServerContext* context,
    const ScheduleTrainingRequest* request,
    StandardReply* reply)
{
  UNUSED(context);

  //TODO(seojin): currently ignoring request->data_dir();

  DP_LOG(DEBUG, "Received ScheduleTraining().");

  if (!rtctx->debug) {
    DP_LOG(DEBUG, "Not in DEBUGGING_MODE, so saving request to lastReq.txt.");
    std::ofstream ofs;
    // auto path = std::string(rtctx->homedir) + "/DeepPoolRuntime/lastReq.txt";
    std::string path = format("%s/DeepPoolRuntime/lastReq%d.txt",
                              rtctx->homedir, rtctx->device);
    ofs.open(path.c_str());
    request->SerializeToOstream(&ofs);
    ofs.close();
    DP_LOG(DEBUG, "Saved the serialized ScheduleTrainingRequest.");
  }
  
  std::string name = request->name();
  DP_LOG(DEBUG, "retrieved name. %s", name.c_str());
  json jobSpec = json::parse(request->job_in_json());
  DP_LOG(DEBUG, "parsed jobSpec into json");
  int rank = jobSpec["rank"].get<int>();
  int worldSize = jobSpec["maxGpusUsed"].get<int>();
  DP_LOG(DEBUG, "rank:%d worldSize: %d", rank, worldSize);
  json tensorTags = json::parse(request->tensor_tags_in_json());
  DP_LOG(DEBUG, "parsed tensorTags %s", tensorTags.dump().c_str());
  json jobRankToGlobalRank = json::parse(request->job_rank_to_global_rank_in_json());
  DP_LOG(DEBUG, "parsed jobRankToGlobalRank %s", jobRankToGlobalRank.dump().c_str());
  
  c10::Device dev(c10::DeviceType::CUDA, rtctx->device);
  DP_LOG(DEBUG, "dev constructed.");
  std::unique_ptr<CommunicationHandler> commHandler =
      std::make_unique<CommunicationHandlerGRPC>(
          rtctx, name, worldSize, tensorTags, rank, jobRankToGlobalRank, dev);
  DP_LOG(DEBUG, "commHandler constructed.");
  auto runnableModule = std::make_unique<RunnableModule>(rtctx, jobSpec, commHandler.get(), dev);
  DP_LOG(DEBUG, "runnableModule constructed.");
  std::vector<torch::Tensor> parameters;
  runnableModule->getParameters(&parameters);
  auto optimizer = std::make_unique<torch::optim::SGD>(parameters, /*lr=*/0.01);
  DP_LOG(DEBUG, "optimizer constructed.");

  auto job = std::make_unique<JobContext>(std::move(runnableModule), name,
      nullptr, std::move(commHandler), nullptr, std::move(dev), 1, std::move(optimizer));
  DP_LOG(DEBUG, "job constructed.");

  rtctx->taskManager->addTrainingJob(std::move(job));
  DP_LOG(DEBUG, "added the training job.");

  std::cout << request->name() << " " << request->job_rank_to_global_rank_in_json()
            << " this job's worldSize: " << worldSize << std::endl << std::flush;

  std::string replyMsg("ScheduleTraining invoked.");
  reply->set_message(replyMsg);
  return Status::OK;
}

Status
RuntimeServiceImpl::Poke(ServerContext* context, const Empty* request,
    StandardReply* reply)
{
  UNUSED(context);
  UNUSED(request);

  DP_LOG(NOTICE, "poked.");
  std::string replyMsg("Poke invoked.");
  reply->set_message(replyMsg);
  return Status::OK;
}

Status
RuntimeServiceImpl::Shutdown(ServerContext* context, const Empty* request,
    StandardReply* reply)
{
  UNUSED(context);
  UNUSED(request);

  DP_LOG(NOTICE, "Shutdown requested.");
  rtctx->shutdownRequested = true;
  std::cout << "shutdownRequested " << rtctx->shutdownRequested.load() << std::endl;
  std::string replyMsg("Shutdown invoked.");
  reply->set_message(replyMsg);
  return Status::OK;
}

Status
RuntimeServiceImpl::P2PCommunication(ServerContext* context,
    const P2PCommunicationRequest* request,
    StandardReply* reply)
{
  UNUSED(context);

  DP_LOG(DEBUG, "P2PCommunication requested.");
  std::string taskName = request->task_name();
  std::string tsrData = request->tensor_data();
  DP_LOG(DEBUG, "P2PCommunication requested. TaskName: %s", taskName.c_str());
  int tag = request->tag();

  auto search = rtctx->commHandlerMap.find(taskName);
  if (search == rtctx->commHandlerMap.end()) {
    DP_LOG(ERROR, "No commHandler for taskName: %s", taskName.c_str());
  }

  CommunicationHandlerGRPC* commHandler =
      reinterpret_cast<CommunicationHandlerGRPC*>(
          rtctx->commHandlerMap[taskName]);
  commHandler->saveData(tsrData, tag);
  
  std::string replyMsg("P2PCommunication received.");
  reply->set_message(replyMsg);
  return Status::OK;
}


////////////////////////////////////////////////////////
// GRPC Client code.
////////////////////////////////////////////////////////

std::string
RuntimeClient::Poke() {
  Empty request;
  grpc::ClientContext context;
  StandardReply reply;
  Status status = stub_->Poke(&context, request, &reply);
  if (status.ok()) {
    return reply.message();
  } else {
    DP_LOG(ERROR, "Failed to invoke Poke. code: %d, msg: %s.",
          status.error_code(), status.error_message().c_str());
    return "Failed to invoke Poke.";
  }
}

std::string
RuntimeClient::P2PCommunication(const std::string& taskName,
                                const std::string& tsrData, int tag) {
  P2PCommunicationRequest request;
  request.set_task_name(taskName);
  request.set_tensor_data(tsrData);
  request.set_tag(tag);

  grpc::ClientContext context;
  StandardReply reply;
  Status status = stub_->P2PCommunication(&context, request, &reply);
  if (status.ok()) {
    return reply.message();
  } else {
    DP_LOG(ERROR, "Failed to invoke P2PCommunication. code: %d, msg: %s.",
          status.error_code(), status.error_message().c_str());
    return "Failed to invoke P2PCommunication.";
  }
}
