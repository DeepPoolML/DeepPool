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

#include "communication.h"

#include <torch/torch.h>

#include <memory>
#include <string>

#include "Cycles.h"
#include "json.hpp"
#include "logger.h"
#include "runtime.h"
#include "utils.h"

using Cycles = RAMCloud::Cycles;
using json = nlohmann::json;

/**
 * Constructs communicationHandler base class.
 *
 * \param worldSize   Number of ranks.
 * \param tensorTags  Mapping from xferName to p2p communication tag.
 * \param rank        Rank of the current node.
 * \param jobRankToGlobalRank   Mapping from job's internal rank to cluster
 * rank.
 */
CommunicationHandler::CommunicationHandler(int worldSize, json tensorTags,
                                           int rank, json jobRankToGlobalRank)
    : worldSize(worldSize),
      tensorTags(tensorTags),
      rank(rank),
      jobRankToGlobalRank(jobRankToGlobalRank) {}

int CommunicationHandler::getTag(const std::string& xferName) {
  return tensorTags[xferName].get<int>();
}

///////////////////////////////////////////////////////////////////////
// CommunicationHandlerGRPC
///////////////////////////////////////////////////////////////////////

CommunicationHandlerGRPC::CommunicationHandlerGRPC(std::string taskName,
                                                   int worldSize,
                                                   json tensorTags, int rank,
                                                   json jobRankToGlobalRank)
    : CommunicationHandler(worldSize, tensorTags, rank, jobRankToGlobalRank),
      taskName(taskName) {
  DP_LOG(DEBUG, "Constructing CommunicationHandlerGRPC for %s",
         taskName.c_str());
  rtctx->commHandlerMap[taskName] = this;
  DP_LOG(DEBUG, "set CommunicationHandlerGRPC to commHandlerMap.");
}

/**
 * Save the received data for p2p communication.
 */
void CommunicationHandlerGRPC::saveData(const std::string& tensorData,
                                        int tag) {
  // DP_LOG(DEBUG, "awaiting grpc lock.");
  std::lock_guard<std::mutex> lock(_mutex);
  receivedData[tag] = tensorData;
  _cv.notify_all();
}

/**
 * Implements p2p send (refer to CommunicationHandler::send)
 */
void CommunicationHandlerGRPC::send(const torch::Tensor& tensor, int tag,
                                    int dest) {
  RuntimeClient* destClient;
  {
    DP_LOG(DEBUG, "Grabbing a lock");
    std::lock_guard<std::mutex> lock(_mutex);
    auto search = clientPool.find(dest);
    if (search == clientPool.end()) {
      std::string ipAndPort = rtctx->rankToIpAndPort[dest];
      DP_LOG(DEBUG, "Dest:%d(%s) isn't in clientPool yet.", dest,
             ipAndPort.c_str());

      auto channel =
          grpc::CreateChannel(ipAndPort, grpc::InsecureChannelCredentials());
      auto client = std::make_unique<RuntimeClient>(channel);
      clientPool[dest] = std::move(client);
      clientPool[dest]->Poke();
      DP_LOG(DEBUG, "Poked dest:%d", dest);
    }
    destClient = clientPool[dest].get();
    DP_LOG(DEBUG, "Releasing a lock");
  }

  std::string tsrData;
  tsrData.resize(tensor.nbytes());
  c10::cuda::device_synchronize(); /* ensure data is ready to be copied out */
  CUDACHECK(cudaMemcpy(&tsrData[0], tensor.data_ptr(), tensor.nbytes(),
                       cudaMemcpyDefault));
  DP_LOG(DEBUG, "Copied tensor data (potentially CUDA) to CPU.");
  destClient->P2PCommunication(taskName, tsrData, tag);
}

/**
 * Implements p2p recv (refer to CommunicationHandler::recv)
 */
void CommunicationHandlerGRPC::recv(torch::Tensor& tensor, int tag, int src) {
  UNUSED(src);
  DP_LOG(DEBUG, "Grabbing a lock");
  std::unique_lock<std::mutex> lock(_mutex);
  _cv.wait(lock, [&] { return receivedData.count(tag); });
  auto search = receivedData.find(tag);
  assert(search != receivedData.end());
  std::string tensorData = search->second;
  assert(tensor.nbytes() == tensorData.size());
  CUDACHECK(cudaMemcpyAsync(tensor.data_ptr(), tensorData.data(),
                            tensorData.size(), cudaMemcpyHostToDevice,
                            rtctx->torch_stream));
  receivedData.erase(search);
  DP_LOG(DEBUG, "Releasing a lock");
}

void CommunicationHandlerGRPC::all_reduce(torch::Tensor& tensor,
                                          c10d::ReduceOp op) {
  UNUSED(tensor);
  UNUSED(op);
  fprintf(stderr, "GRPC all reduce not implemented\n");
}

void CommunicationHandlerGRPC::testRingP2P() {
  torch::Tensor tsr2Send = torch::ones({2}, rtctx->c10dev);
  // DP_LOG(DEBUG, "Created tensor [%s] in dev.", tsrToStr(tsr2Send).c_str());
  int dest = (rtctx->rank + 1) % rtctx->worldSize;
  send(tsr2Send, 1, dest);
  DP_LOG(DEBUG, "Sent tensor [%s] to %d.", tsrToStr(tsr2Send).c_str(), dest);

  torch::Tensor tsr2Recv = torch::zeros({2}, rtctx->c10dev);
  // DP_LOG(DEBUG, "Created tensor [%s] in dev.", tsrToStr(tsr2Recv).c_str());
  int src = (rtctx->worldSize + rtctx->rank - 1) % rtctx->worldSize;
  recv(tsr2Recv, 1, src);
  DP_LOG(DEBUG, "Rcvd tensor [%s] from %d.", tsrToStr(tsr2Recv).c_str(), src);
}

///////////////////////////////////////////////////////////////////////
// CommunicationHandlerNCCL
///////////////////////////////////////////////////////////////////////

CommunicationHandlerNCCL::CommunicationHandlerNCCL(std::string taskName,
                                                   int worldSize,
                                                   json tensorTags, int rank,
                                                   json jobRankToGlobalRank)
    : CommunicationHandler(worldSize, tensorTags, rank, jobRankToGlobalRank),
      default_comm_stream(c10::cuda::getStreamFromPool(true)) {
  DP_LOG(DEBUG, "Constructing CommunicationHandlerNCCL for %s",
         taskName.c_str());
  rtctx->commHandlerMap[taskName] = this;
  DP_LOG(DEBUG, "set CommunicationHandlerNCCL to commHandlerMap.");
}

void CommunicationHandlerNCCL::sync(
    c10::optional<c10::cuda::CUDAStream> stream) {
  if (in_group_call) return;
  sync_event.record(stream ? stream.value() : default_comm_stream);
  sync_event.block(rtctx->torch_stream);
}

void CommunicationHandlerNCCL::send(const torch::Tensor& tensor, int tag,
                                    int dest) {
  UNUSED(tag);

  assert(in_group_call);
  tensor.record_stream(group_call_stream.value());
  torch::cuda::nccl::send(tensor, group_call_commObj, group_call_stream.value(),
                          dest);
}

void CommunicationHandlerNCCL::recv(torch::Tensor& tensor, int tag, int src) {
  UNUSED(tag);
  DP_LOG(DEBUG, "NCCL recv.");

  assert(in_group_call);
  tensor.record_stream(group_call_stream.value());
  torch::cuda::nccl::recv(tensor, group_call_commObj, group_call_stream.value(),
                          src);
}

static ncclDataType_t to_nccl_data_type(c10::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return ncclDataType_t::ncclFloat;
    case at::kHalf:
      return ncclDataType_t::ncclHalf;
    case at::kDouble:
      return ncclDataType_t::ncclDouble;
    case at::kLong:
      return ncclDataType_t::ncclInt64;
    case at::kInt:
      return ncclDataType_t::ncclInt;
    case at::kChar:
      return ncclDataType_t::ncclChar;
    case at::kByte:
      return ncclDataType_t::ncclUint8;
    case at::kBool:
      return ncclDataType_t::ncclUint8;
    default:
      TORCH_CHECK(false, "Unconvertible NCCL type ", type);
  }
}

void CommunicationHandlerNCCL::all_reduce(torch::Tensor& tensor,
                                          c10d::ReduceOp op) {
  // NCCL op mapping from c10d
  const std::map<c10d::ReduceOp, ncclRedOp_t> ncclOp = {
      {c10d::ReduceOp::MIN, ncclMin},
      {c10d::ReduceOp::MAX, ncclMax},
      {c10d::ReduceOp::SUM, ncclSum},
      {c10d::ReduceOp::PRODUCT, ncclProd},
  };

  assert(in_group_call);
  NCCL_API_CALL(
      ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
                    to_nccl_data_type(tensor.scalar_type()), ncclOp.at(op),
                    reinterpret_cast<ncclComm_t>(group_call_commObj),
                    group_call_stream.value()));
}

void CommunicationHandlerNCCL::testAllReduce() {
  if (rtctx->worldSize == 1) return;

  /* sum */
  torch::Tensor t = torch::full({3, 3}, rtctx->rank + 1, rtctx->c10dev);
  comm_start();
  all_reduce(t, c10d::ReduceOp::SUM);
  comm_end();
  sync();
  int sum = 0;
  for (int i = 1; i < rtctx->worldSize + 1; i++) sum += i;
  torch::Tensor expected = torch::full({3, 3}, sum, rtctx->c10dev);
  assert(at::equal(t, expected));

  /* prod */
  int prod = 1;
  for (int i = 1; i < rtctx->worldSize + 1; i++) prod *= i;
  t = torch::full({3, 3}, rtctx->rank + 1, rtctx->c10dev);
  expected = torch::full({3, 3}, prod, rtctx->c10dev);
  comm_start();
  all_reduce(t, c10d::ReduceOp::PRODUCT);
  comm_end();
  sync();
  assert(at::equal(t, expected));

  /* max */
  t = torch::full({3, 3}, rtctx->rank + 1, rtctx->c10dev);
  expected = torch::full({3, 3}, rtctx->worldSize, rtctx->c10dev);
  comm_start();
  all_reduce(t, c10d::ReduceOp::MAX);
  comm_end();
  sync();
  assert(at::equal(t, expected));

  DP_LOG(NOTICE, "Completed 3 NCCL all reduce tests.");
}

void CommunicationHandlerNCCL::testRingP2P() {
  if (rtctx->worldSize == 1) return;

  int dest = (rtctx->rank + 1) % rtctx->worldSize;
  int src = (rtctx->rank + rtctx->worldSize - 1) % rtctx->worldSize;

  torch::Tensor send_tensor =
      torch::full({3, 3}, rtctx->rank + 1, rtctx->c10dev);
  torch::Tensor recv_tensor = torch::full({3, 3}, src, rtctx->c10dev);
  torch::Tensor expected = torch::full({3, 3}, src + 1, rtctx->c10dev);

  comm_start();

  if (rtctx->rank == 0) {
    send(send_tensor, 1, dest);
    recv(recv_tensor, 1, src);
  } else {
    recv(recv_tensor, 1, src);
    send(send_tensor, 1, dest);
  }

  comm_end();
  sync();

  DP_LOG(DEBUG, "Sent tensor [%s] to %d.", tsrToStr(send_tensor).c_str(), dest);
  DP_LOG(DEBUG, "Received tensor [%s] from %d.", tsrToStr(recv_tensor).c_str(),
         src);
  DP_LOG(DEBUG, "Expected tensor [%s] from %d.", tsrToStr(expected).c_str(),
         src);

  assert(at::equal(recv_tensor, expected));
}

void CommunicationHandlerNCCL::comm_start(
    c10::optional<c10::cuda::CUDAStream> stream, size_t commKey) {
  if (++in_group_call != 1)
    return;

  if (commKey > 0) {
    auto& config = rtctx->nccl_groups.at(commKey);
    group_call_commObj = config.ncclCommObj;
  } else {
    group_call_commObj = rtctx->maingroup.ncclCommObj;
  }

  group_call_stream = stream ? stream : default_comm_stream;
  sync_event.record(rtctx->torch_stream);
  sync_event.block(group_call_stream.value());
  NCCL_API_CALL(ncclGroupStart());
}

void CommunicationHandlerNCCL::comm_end() {
  if (--in_group_call != 0)
    return;
  group_call_stream = {};
  NCCL_API_CALL(ncclGroupEnd());
}

void CommunicationHandlerNCCL::precapture() {
  sync_event.record(rtctx->torch_stream);
  sync_event.block(default_comm_stream);
}

void CommunicationHandlerNCCL::postcapture() {
  sync_event.record(default_comm_stream);
  sync_event.block(rtctx->torch_stream);
}
