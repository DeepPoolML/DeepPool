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
#include "cuda_runtime.h"
// #include "nccl.h"
#include "utils.h"
#include "logger.h"
#include "runtime.h"
#include "json.hpp"
#include "nccl.h"

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
  RuntimeClient* destClient;
  {
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
      lock.unlock();
      Cycles::sleep(100);
      lock.lock();
    }
  }
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
}

void
CommunicationHandlerNCCL::send(const torch::Tensor& tensor, int tag, int dest,
    bool async)
{
  ncclSend((void*)tensor.data_ptr(), tensor.nbytes()/tensor.itemsize(), ncclFloat, dest, *rtctx->ncclCommObj, *rtctx->cudaStream);
  cudaSetDevice(rtctx->device);
  cudaStreamSynchronize(*rtctx->cudaStream);
}

void
CommunicationHandlerNCCL::recv(torch::Tensor& tensor, int tag, int src,
    bool async)
{
  ncclRecv((void*)tensor.data_ptr(), tensor.nbytes()/tensor.itemsize(), ncclFloat, src, *rtctx->ncclCommObj, *rtctx->cudaStream);
  cudaSetDevice(rtctx->device);
  cudaStreamSynchronize(*rtctx->cudaStream);
}

void
CommunicationHandlerNCCL::testRingP2P()
{
  torch::Tensor send_tensor = torch::ones({3,3}, torch::Device(torch::kCUDA, rtctx->device));
  torch::Tensor recv_tensor = torch::zeros({1,9}, torch::Device(torch::kCUDA, rtctx->device));

  int dest = (rtctx->rank + 1) % rtctx->worldSize;
  int src = (rtctx->rank + rtctx->worldSize - 1) % rtctx->worldSize;

  send(send_tensor, 1, dest, false);
  DP_LOG(DEBUG, "Sent tensor [%s] to %d.", tsrToStr(send_tensor).c_str(), dest);

  recv(recv_tensor, 1, src, false);
  DP_LOG(DEBUG, "Rcvd tensor [%s] to %d.", tsrToStr(recv_tensor).c_str(), src);
}