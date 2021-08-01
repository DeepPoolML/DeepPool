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
    int rank, json jobRankToGlobalRank, bool tensorInCuda)
  : worldSize(worldSize)
  , tensorTags(tensorTags)
  , rank(rank)
  , jobRankToGlobalRank(jobRankToGlobalRank)
  , tensorInCuda(tensorInCuda)
{
}

///////////////////////////////////////////////////////////////////////
// CommunicationHandlerGRPC
///////////////////////////////////////////////////////////////////////

CommunicationHandlerGRPC::CommunicationHandlerGRPC(RuntimeContext* rtctx,
    std::string taskName, int worldSize, json tensorTags, int rank,
    json jobRankToGlobalRank, bool tensorInCuda)
  : CommunicationHandler(worldSize, tensorTags, rank, jobRankToGlobalRank,
                         tensorInCuda)
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
  std::lock_guard<std::mutex> lock(_mutex);
  auto search = clientPool.find(dest);
  if (search == clientPool.end()) {
    std::string ipAndPort = rtctx->rankToIpAndPort[dest];
    auto channel = grpc::CreateChannel(ipAndPort, grpc::InsecureChannelCredentials());
  }
  clientPool[dest]->P2PCommunication(taskName, tensor, tag);
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
                          tensorData.size(), cudaMemcpyHostToDevice));
                          //TODO: try cudaMemcpyDefault.
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