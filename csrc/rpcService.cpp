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

#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <memory>
#include <string>

#include "JobContext.h"
#include "communication.h"
#include "cuda_runtime.h"
#include "json.hpp"
#include "logger.h"
#include "runnableModule.h"
#include "runtime.grpc.pb.h"
#include "runtime.h"
#include "utils.h"

using json = nlohmann::json;
using grpc::Server;
using grpc::ServerContext;
using grpc::Status;

Status RuntimeServiceImpl::InitCommGRPC(ServerContext* context,
                                        const InitCommGRPCRequest* request,
                                        StandardReply* reply) {
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

Status RuntimeServiceImpl::InitCommNCCL(ServerContext* context,
                                        const InitCommNCCLMsg* request,
                                        InitCommNCCLMsg* reply) {
  UNUSED(context);

  int msg_type = request->msg_type();
  DP_LOG(DEBUG, "Received InitCommNCCL() (%d).", msg_type);

  if (msg_type == 0) {  // Generate comm group ID
    // Coordinator has requested a new unique comm ID
    DP_LOG(NOTICE, "Generate ID NCCL.");

    torch::cuda::nccl::ncclUniqueId id;
    torch::cuda::nccl::get_unique_id(id);

    std::string replyMsg("Comm group ID generated at rank 0.");
    reply->set_message(replyMsg);
    reply->set_group_id(&id, sizeof(id));
  } else if (msg_type == 1) {  // Join comm group specified by ID

    NcclGroupConfig cfg;
    memcpy(&cfg.ncclGroupId, request->group_id().c_str(),
           sizeof(cfg.ncclGroupId));
    cfg.myRank = -1;

    int lastRank = -1;
    for (int i = 0; i < request->members_size(); i++) {
      auto r = request->members(i);
      /* ensure that member list is sorted */
      assert(r > lastRank);
      lastRank = r;
      if (r == rtctx->rank) cfg.myRank = i;
      cfg.ranks.push_back(r);
    }
    assert(cfg.myRank != -1);

    cfg.ncclCommObj = torch::cuda::nccl::comm_init_rank(
        cfg.ranks.size(), cfg.ncclGroupId, cfg.myRank);
    cfg.group_key = RankVecToKey(cfg.ranks);

    if (rtctx->nccl_groups.count(cfg.group_key) > 0)
      DIE("NCCL (sub)group has already been created (%lu)", cfg.group_key);

    rtctx->nccl_groups[cfg.group_key] = cfg;

    if (rtctx->nccl_groups.size() == 1) {
      rtctx->maingroup = cfg;
      rtctx->ncclCommReady = true;
    }

    std::string replyMsg("Comm group ID broadcast & joined.");
    reply->set_message(replyMsg);
  }

  return Status::OK;
}

Status RuntimeServiceImpl::ScheduleTraining(
    ServerContext* context, const ScheduleTrainingRequest* request,
    StandardReply* reply) {
  UNUSED(context);

  c10::cuda::setCurrentCUDAStream(rtctx->torch_stream);

  // TODO(seojin): currently ignoring request->data_dir();
  DP_LOG(DEBUG, "Received ScheduleTraining().");

  if (!rtctx->debug) {
    DP_LOG(DEBUG, "Not in DEBUGGING_MODE, so saving request to lastReq.txt.");
    std::ofstream ofs;
    std::string path = format("%s/lastReq%d.txt", rtctx->logdir, rtctx->rank);
    ofs.open(path.c_str());
    request->SerializeToOstream(&ofs);
    ofs.close();
    DP_LOG(DEBUG, "Saved the serialized ScheduleTrainingRequest.");
  }

  std::unique_ptr<JobContext> job;
  try {
    job = parseAndCreateTrainingTask(request);
  } catch (std::exception& e) {
    std::cerr << "exception: " << e.what() << std::endl;
    throw e;
  }
  rtctx->addTrainingJob(std::move(job));
  DP_LOG(DEBUG, "added the training job.");

  std::string replyMsg("ScheduleTraining invoked.");
  reply->set_message(replyMsg);
  return Status::OK;
}

Status RuntimeServiceImpl::Poke(ServerContext* context, const Empty* request,
                                StandardReply* reply) {
  UNUSED(context);
  UNUSED(request);

  DP_LOG(DEBUG, "poked.");
  std::string replyMsg("Poke invoked.");
  reply->set_message(replyMsg);
  return Status::OK;
}

Status RuntimeServiceImpl::Shutdown(ServerContext* context,
                                    const Empty* request,
                                    StandardReply* reply) {
  UNUSED(context);
  UNUSED(request);

  DP_LOG(NOTICE, "Shutdown requested.");
  rtctx->shutdownRequested = true;
  std::cout << "shutdownRequested " << rtctx->shutdownRequested.load()
            << std::endl;
  std::string replyMsg("Shutdown invoked.");
  reply->set_message(replyMsg);
  return Status::OK;
}

Status RuntimeServiceImpl::P2PCommunication(
    ServerContext* context, const P2PCommunicationRequest* request,
    StandardReply* reply) {
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

std::unique_ptr<JobContext> RuntimeServiceImpl::parseAndCreateTrainingTask(
    const ScheduleTrainingRequest* request) {
  std::string name = request->name();
  DP_LOG(DEBUG, "retrieved name. %s", name.c_str());
  json jobSpec = json::parse(request->job_in_json());
  DP_LOG(DEBUG, "parsed jobSpec into json");
  int rank = jobSpec["rank"].get<int>();
  int worldSize = jobSpec["maxGpusUsed"].get<int>();
  DP_LOG(DEBUG, "rank:%d worldSize: %d", rank, worldSize);
  json tensorTags = json::parse(request->tensor_tags_in_json());
  DP_LOG(DEBUG, "parsed tensorTags %s", tensorTags.dump().c_str());
  json jobRankToGlobalRank =
      json::parse(request->job_rank_to_global_rank_in_json());
  DP_LOG(DEBUG, "parsed jobRankToGlobalRank %s",
         jobRankToGlobalRank.dump().c_str());

  DP_LOG(DEBUG, "dev constructed.");

  std::shared_ptr<CommunicationHandler> commHandler;
  if (rtctx->c10dBackend == "nccl") {
    commHandler = std::make_shared<CommunicationHandlerNCCL>(
        name, worldSize, tensorTags, rank, jobRankToGlobalRank);
  } else if (rtctx->c10dBackend == "grpc") {
    commHandler = std::make_shared<CommunicationHandlerGRPC>(
        name, worldSize, tensorTags, rank, jobRankToGlobalRank);
  }

  DP_LOG(DEBUG, "commHandler constructed.");
  auto runnableModule = std::make_unique<RunnableModule>(jobSpec, commHandler);
  DP_LOG(DEBUG, "runnableModule constructed.");

  json jobParams = json::parse(request->job_meta_params_in_json());
  DP_LOG(DEBUG, "parsed jobParams into json");

  assert(jobParams.contains("run_with_be"));
  assert(jobParams.contains("nr_gpus"));

  auto job = std::make_unique<JobContext>(std::move(runnableModule), name,
                                          std::move(commHandler), jobParams);
  DP_LOG(DEBUG, "job constructed.");
  return job;
}

////////////////////////////////////////////////////////
// GRPC Client code.
////////////////////////////////////////////////////////

std::string RuntimeClient::Poke() {
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

std::string RuntimeClient::P2PCommunication(const std::string& taskName,
                                            const std::string& tsrData,
                                            int tag) {
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
