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

#ifndef COMMUNICATION_H
#define COMMUNICATION_H



class CommunicationHandler {
 public:
  /**
   * Constructs communicationHandler base class.
   * 
   * \param worldSize   Number of ranks.
   * \param tensorTags  Mapping from xferName to p2p communication tag.
   * \param rank        Rank of the current node.
   * \param jobRankToGlobalRank   Mapping from job's internal rank to cluster rank.
   * \param tensorInCuda  tensor given to send/recv methods are cuda tensors (false if CPU tensor).
   */
  CommunicationHandler(int worldSize, json tensorTags, int rank,
      json jobRankToGlobalRank, bool tensorInCuda = true);

  /**
   * Changes from Python runtime.
   * - Compute tensor dimension from json spec.
   * - recv takes empty tensor that is ready to be filled.
   * - No separate async/sync methods.
   * 
   * Undecided: take tensorName or tag? maybe just take tag? It may not be
   * that difficult to save the tag in runnableModule's layer..?
   */
  virtual void send(const torch::Tensor& tensor, int tag, int dest,
                    bool async = false);
  virtual void recv(torch::Tensor& tensor, int tag, int src,
                    bool async = false);
  
  /**
   * Returns the tag for p2p communication send/recv.
   *
   * \param xferName  Transfer name specificed in spec. Sender and receiver
   *                  should use the same xferName.
   */
  int getTag(const std::string& xferName);
};

class CommunicationHandlerNCCL : public CommunicationHandler {
};

#endif