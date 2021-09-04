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
#include <cuda_runtime.h>
#include <nccl.h>
#include "utils.h"
#include "logger.h"
#include "runtime.h"
#include "json.hpp"

using Cycles = RAMCloud::Cycles;
using json = nlohmann::json;

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

/**
 * Constructs communicationHandler base class.
 * 
 * \param worldSize   Number of ranks.
 * \param tensorTags  Mapping from xferName to p2p communication tag.
 * \param rank        Rank of the current node.
 * \param jobRankToGlobalRank   Mapping from job's internal rank to cluster rank.
 * \param tensorInCuda  tensor given to send/recv methods are cuda tensors (false if CPU tensor).
 */
CommunicationHandler::CommunicationHandler(int worldSize, json tensorTags,
    int rank, json jobRankToGlobalRank, c10::Device device, bool tensorInCuda)
  : worldSize(worldSize)
  , tensorTags(tensorTags)
  , rank(rank)
  , jobRankToGlobalRank(jobRankToGlobalRank)
  , device(device)
  , tensorInCuda(tensorInCuda)
{
}

int
CommunicationHandler::getTag(const std::string& xferName)
{
  return tensorTags[xferName].get<int>();
}

///////////////////////////////////////////////////////////////////////
// CommunicationHandlerGRPC
///////////////////////////////////////////////////////////////////////

CommunicationHandlerGRPC::CommunicationHandlerGRPC(RuntimeContext* rtctx,
    std::string taskName, int worldSize, json tensorTags, int rank,
    json jobRankToGlobalRank, c10::Device dev, bool tensorInCuda)
  : CommunicationHandler(worldSize, tensorTags, rank, jobRankToGlobalRank,
                         dev, tensorInCuda)
  , rtctx(rtctx)
  , taskName(taskName)
  , _mutex()
  , receivedData()
  , clientPool()
{
  DP_LOG(DEBUG, "Constructing CommunicationHandlerGRPC for %s", taskName.c_str());
  rtctx->commHandlerMap[taskName] = this;
  DP_LOG(DEBUG, "set CommunicationHandlerGRPC to commHandlerMap.");
}

/**
 * Save the received data for p2p communication.
 */
void
CommunicationHandlerGRPC::saveData(const std::string& tensorData, int tag)
{
  // DP_LOG(DEBUG, "awaiting grpc lock.");
  std::lock_guard<std::mutex> lock(_mutex);
  receivedData[tag] = tensorData;
}

/**
 * Implements p2p send (refer to CommunicationHandler::send)
 */
void
CommunicationHandlerGRPC::send(const torch::Tensor& tensor, int tag, int dest,
    bool async)
{
  UNUSED(async);
  RuntimeClient* destClient;
  {
    DP_LOG(DEBUG, "Grabbing a lock");
    std::lock_guard<std::mutex> lock(_mutex);
    auto search = clientPool.find(dest);
    if (search == clientPool.end()) {
      std::string ipAndPort = rtctx->rankToIpAndPort[dest];
      DP_LOG(DEBUG, "Dest:%d(%s) isn't in clientPool yet.", dest, ipAndPort.c_str());

      auto channel = grpc::CreateChannel(ipAndPort, grpc::InsecureChannelCredentials());
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
  CUDACHECK(cudaMemcpy(&tsrData[0], tensor.data_ptr(),
                       tensor.nbytes(), cudaMemcpyDefault));
  DP_LOG(DEBUG, "Copied tensor data (potentially CUDA) to CPU.");
  destClient->P2PCommunication(taskName, tsrData, tag);
}

/**
 * Implements p2p recv (refer to CommunicationHandler::recv)
 */
void
CommunicationHandlerGRPC::recv(torch::Tensor& tensor, int tag, int src,
    bool async)
{
  UNUSED(async);
  UNUSED(src);
  DP_LOG(DEBUG, "Grabbing a lock");
  std::unique_lock<std::mutex> lock(_mutex);
  bool found = false;
  while (!found) {
    auto search = receivedData.find(tag);
    if (search != receivedData.end()) {
      std::string tensorData = search->second;
      // tensor.data_ptr()
      assert(tensor.nbytes() == tensorData.size());
      CUDACHECK(cudaMemcpy(tensor.data_ptr(), tensorData.data(),
                           tensorData.size(), cudaMemcpyDefault)); //cudaMemcpyHostToDevice));
      receivedData.erase(search);
      found = true;
    } else {
      // This is very hacky... implement thread queue. and async mode.
      // DP_LOG(DEBUG, "Releasing a lock");
      lock.unlock();
      Cycles::sleep(100);
      // DP_LOG(DEBUG, "Grabbing a lock");
      lock.lock();
    }
  }
  DP_LOG(DEBUG, "Releasing a lock");
}

void
CommunicationHandlerGRPC::all_reduce(torch::Tensor &tensor,
                                     c10d::ReduceOp op,
                                     bool async) {
  UNUSED(tensor);
  UNUSED(op);
  UNUSED(async);
  fprintf(stderr, "GRPC all reduce not implemented\n");
}

void
CommunicationHandlerGRPC::testRingP2P()
{
  torch::Tensor tsr2Send = torch::ones({2}, torch::Device(torch::kCUDA, rtctx->device));
  // DP_LOG(DEBUG, "Created tensor [%s] in dev.", tsrToStr(tsr2Send).c_str());
  int dest = (rtctx->rank + 1) % rtctx->worldSize;
  send(tsr2Send, 1, dest, false);
  DP_LOG(DEBUG, "Sent tensor [%s] to %d.", tsrToStr(tsr2Send).c_str(), dest);
  
  torch::Tensor tsr2Recv = torch::zeros({2}, torch::Device(torch::kCUDA, rtctx->device));
  // DP_LOG(DEBUG, "Created tensor [%s] in dev.", tsrToStr(tsr2Recv).c_str());
  int src = (rtctx->worldSize + rtctx->rank - 1) % rtctx->worldSize;
  recv(tsr2Recv, 1, src, false);
  DP_LOG(DEBUG, "Rcvd tensor [%s] from %d.", tsrToStr(tsr2Recv).c_str(), src);
}

///////////////////////////////////////////////////////////////////////
// CommunicationHandlerNCCL
///////////////////////////////////////////////////////////////////////

CommunicationHandlerNCCL::CommunicationHandlerNCCL(RuntimeContext* rtctx,
    std::string taskName, int worldSize, json tensorTags, int rank,
    json jobRankToGlobalRank, c10::Device dev, bool tensorInCuda)
  : CommunicationHandler(worldSize, tensorTags, rank, jobRankToGlobalRank,
                         dev, tensorInCuda)
  , rtctx(rtctx)
  , taskName(taskName)
  , _mutex()
  , receivedData()
  , clientPool()
{
  DP_LOG(DEBUG, "Constructing CommunicationHandlerNCCL for %s", taskName.c_str());
  rtctx->commHandlerMap[taskName] = this;
  DP_LOG(DEBUG, "set CommunicationHandlerNCCL to commHandlerMap.");

  cudaStream_t stream;

  for (int i = 0; i < worldSize; i++) {
    /* skip myself */
    if (i == rank) {
      send_streams.push_back(0);
      recv_streams.push_back(0);
      continue;
    }
    CUDA_API_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0));
    send_streams.push_back(stream);
    CUDA_API_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0));
    recv_streams.push_back(stream);
  }

  CUDA_API_CALL(cudaStreamCreateWithPriority(&comm_sync_stream, cudaStreamNonBlocking, 0));
}

CommunicationHandlerNCCL::~CommunicationHandlerNCCL()
{
  for (auto &stream : send_streams)
    if (stream)
      CUDA_API_CALL(cudaStreamDestroy(stream));
  for (auto &stream : recv_streams)
    if (stream)
      CUDA_API_CALL(cudaStreamDestroy(stream));
  CUDA_API_CALL(cudaStreamDestroy(comm_sync_stream));
}

// borrowed from c10d
static const std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
    {at::kBool, ncclUint8},
};

void
CommunicationHandlerNCCL::send(const torch::Tensor& tensor, int tag, int dest,
    bool async)
{
  UNUSED(tag);
  cudaStream_t send_stream = send_streams[dest];

  /* ensure ncclSend happens after most recent kernel has finished on rtctx->torch_stream */
  cudaEvent_t event;
  CUDA_API_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  CUDA_API_CALL(cudaEventRecord(event, rtctx->torch_stream));
  CUDA_API_CALL(cudaStreamWaitEvent(send_stream, event, 0)); // this is nonblocking for the host
  CUDA_API_CALL(cudaEventDestroy(event));

  /* send the data */
  auto dtype = ncclDataType.at(tensor.scalar_type());
  NCCL_API_CALL(ncclSend(tensor.data_ptr(), tensor.numel(), dtype, dest, rtctx->ncclCommObj, send_stream));

  if (async) {
    /* create a depedency on the send stream for sync stream */
    CUDA_API_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_API_CALL(cudaEventRecord(event, send_stream));
    CUDA_API_CALL(cudaStreamWaitEvent(comm_sync_stream, event, 0)); // this is nonblocking for the host
    CUDA_API_CALL(cudaEventDestroy(event));
  } else {
    cudaStreamSynchronize(send_stream);
  }
}

void
CommunicationHandlerNCCL::recv(torch::Tensor& tensor, int tag, int src,
    bool async)
{
  UNUSED(tag);
  DP_LOG(DEBUG, "NCCL recv.");
  cudaStream_t recv_stream = recv_streams[src];

  /* ensure ncclRecv happens after most recent kernel has finished on rtctx->torch_stream */
  cudaEvent_t event;
  CUDA_API_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  CUDA_API_CALL(cudaEventRecord(event, rtctx->torch_stream));
  CUDA_API_CALL(cudaStreamWaitEvent(recv_stream, event, 0)); // this is nonblocking for the host
  CUDA_API_CALL(cudaEventDestroy(event));

  /* receive the data */
  auto dtype = ncclDataType.at(tensor.scalar_type());
  NCCL_API_CALL(ncclRecv(tensor.data_ptr(), tensor.numel(), dtype, src, rtctx->ncclCommObj, recv_stream));

  if (async) {
    /* create a depedency on the recv stream for sync stream */
    CUDA_API_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_API_CALL(cudaEventRecord(event, recv_stream));
    CUDA_API_CALL(cudaStreamWaitEvent(comm_sync_stream, event, 0)); // this is nonblocking for the host
    CUDA_API_CALL(cudaEventDestroy(event));
  } else {
    cudaStreamSynchronize(recv_stream);
  }
}

void
CommunicationHandlerNCCL::all_reduce(torch::Tensor &tensor,
                                     c10d::ReduceOp op, bool async)
{
  // NCCL op mapping from c10d
  const std::map<c10d::ReduceOp, ncclRedOp_t> ncclOp = {
      {c10d::ReduceOp::MIN, ncclMin},
      {c10d::ReduceOp::MAX, ncclMax},
      {c10d::ReduceOp::SUM, ncclSum},
      {c10d::ReduceOp::PRODUCT, ncclProd},
  };

  auto dtype = ncclDataType.at(tensor.scalar_type());
  NCCL_API_CALL(ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(),
                              tensor.numel(), dtype, ncclOp.at(op),
                              rtctx->ncclCommObj, rtctx->torch_stream));

  if (async) {
    /* create a depedency on the recv stream for sync stream */
    cudaEvent_t event;
    CUDA_API_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_API_CALL(cudaEventRecord(event, rtctx->torch_stream));
    CUDA_API_CALL(cudaStreamWaitEvent(comm_sync_stream, event,
                                      0)); // this is nonblocking for the host
    CUDA_API_CALL(cudaEventDestroy(event));
  } else {
    cudaStreamSynchronize(rtctx->torch_stream);
  }
}

void
CommunicationHandlerNCCL::testAllReduce()
{
  torch::Device dev(torch::kCUDA, rtctx->device);

  /* sum */
  torch::Tensor t = torch::full({3, 3}, rtctx->rank + 1, dev);
  all_reduce(t, c10d::ReduceOp::SUM, false);
  int sum = 0;
  for (int i = 1; i < rtctx->worldSize + 1; i++)
    sum += i;
  torch::Tensor expected = torch::full({3, 3}, sum, dev);
  assert(at::equal(t, expected));

  /* prod */
  int prod = 1;
  for (int i = 1; i < rtctx->worldSize + 1; i++)
    prod *= i;
  t = torch::full({3, 3}, rtctx->rank + 1, dev);
  expected = torch::full({3, 3}, prod, dev);
  all_reduce(t, c10d::ReduceOp::PRODUCT, true);
  sync();
  assert(at::equal(t, expected));

  /* max */
  t = torch::full({3, 3}, rtctx->rank + 1, dev);
  expected = torch::full({3, 3}, rtctx->worldSize, dev);
  all_reduce(t, c10d::ReduceOp::MAX, true);
  sync();
  assert(at::equal(t, expected));

  DP_LOG(DEBUG, "Completed 3 NCCL all reduce tests.");
}

void
CommunicationHandlerNCCL::testRingP2P()
{
  int dest = (rtctx->rank + 1) % rtctx->worldSize;
  int src = (rtctx->rank + rtctx->worldSize - 1) % rtctx->worldSize;

  torch::Tensor send_tensor = torch::full({3,3}, rtctx->rank + 1, torch::Device(torch::kCUDA, rtctx->device));
  torch::Tensor recv_tensor = torch::full({3,3}, src, torch::Device(torch::kCUDA, rtctx->device));
  torch::Tensor expected = torch::full({3,3}, src + 1, torch::Device(torch::kCUDA, rtctx->device));

  if (rtctx->rank == 0) {
    send(send_tensor, 1, dest, true);
    recv(recv_tensor, 1, src, true);
  } else {
    recv(recv_tensor, 1, src, true);
    send(send_tensor, 1, dest, true);
  }

  sync();

  DP_LOG(DEBUG, "Sent tensor [%s] to %d.", tsrToStr(send_tensor).c_str(), dest);
  DP_LOG(DEBUG, "Received tensor [%s] from %d.", tsrToStr(recv_tensor).c_str(), src);
  DP_LOG(DEBUG, "Expected tensor [%s] from %d.", tsrToStr(expected).c_str(), src);

  assert(at::equal(recv_tensor, expected));
}