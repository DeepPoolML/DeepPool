# Copyright (c) 2021 MIT
# 
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os
import threading
import torch
import torch.distributed as dist

class CommunicationBackend:
    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int, backend='gloo'):
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port + 1)
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.initialized = False
        return
    
    def init_comm_group_if_not(self):
        if not self.initialized:
            args = {"backend": self.backend, "rank": self.rank, "world_size": self.world_size}
            print("comm group initialization started. args: %s" % str(args), flush=True)
            dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size)
            assert dist.get_world_size() == self.world_size
            print("comm group initialized.", flush=True)
            self.initialized = True

class CommunicationHandler:
    # Features.
    # - mapping from a rank for a training job to global runtime rank.
    # - keeps tensor dimension information for recv operation. (c10d recv needs a tensor with correct size)
    def __init__(self, worldSize, tensor_tags, shouldSendSizes: bool = True):
        self.tensorSizes = {}
        self.tensor_tags = tensor_tags
        self.shouldSendSizes = shouldSendSizes
        self.sendFromCpu = False
        self.jobRankToGlobalRank = list(range(worldSize))
        self.asyncReqs = []
    
    def stopSendingSizes(self):
        """ Should be called after 1st iteration for performance """
        self.shouldSendSizes = False
    
    def send(self, tensor: torch.Tensor, tensorName: str, dest: int):
        self.sendAsync(tensor, tensorName, dest)
        self.waitForAll()

    def sendAsync(self, tensor: torch.Tensor, tensorName: str, dest: int):
        # assert tensor.is_cuda
        if self.sendFromCpu:
            tensor = tensor.cpu()

        dstRank = self.jobRankToGlobalRank[dest]
        tag = self.tensor_tags[tensorName]
        if self.shouldSendSizes:
            tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
            tensor_shape_len = torch.tensor(len(tensor.shape), dtype=torch.int)
            dist.send(tensor=tensor_shape_len, dst=dstRank, tag=tag)
            dist.send(tensor=tensor_shape, dst=dstRank, tag=tag)
        tensorReq = dist.isend(tensor=tensor, dst=dstRank, tag=tag)
        self.asyncReqs.append(tensorReq)
        return tensorReq

    def waitForAll(self):
        for req in self.asyncReqs:
            req.wait()
        self.asyncReqs.clear()

    def recv(self, tensorName: str, src: int, dtype=torch.float32) -> torch.Tensor:
        tensor = self.recvAsync(tensorName, src, dtype)
        self.waitForAll()
        # if self.sendFromCpu:
        #     tensor = tensor.cuda()
        return tensor

    def recvAsync(self, tensorName: str, src: int, dtype=torch.float32) -> torch.Tensor:
        src_rank = self.jobRankToGlobalRank[src]
        tag = self.tensor_tags[tensorName]
        if self.shouldSendSizes:
            tensor_shape_len = torch.zeros(1, dtype=torch.int)
            dist.recv(tensor=tensor_shape_len, src=src_rank, tag=tag)
            tensor_shape_len = list(map(lambda x: int(x), tensor_shape_len))
            
            tensor_shape = torch.zeros(tensor_shape_len, dtype=torch.int)
            dist.recv(tensor=tensor_shape, src=src_rank, tag=tag)
            tensor_shape = list(map(lambda x: int(x), tensor_shape))

            self.tensorSizes[tensorName] = tensor_shape
        else:
            tensor_shape = self.tensorSizes[tensorName]
        # Receive tensor.
        tensor = torch.zeros(tensor_shape, dtype=dtype)
        asyncReq = dist.irecv(tensor=tensor, src=src_rank, tag=tag)
        self.asyncReqs.append(asyncReq)
        return tensor
