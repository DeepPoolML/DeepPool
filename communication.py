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

import torch
import torch.distributed as dist
from logger import Logger

class CommunicationBackend:
    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int, backend: str, device='cuda:0'):
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.device = device
        self.initialized = False
        self.commGrpHandlerDicts = {} # maps from <jobName> to <commGrpHandlerDict>
        return
    
    def init_comm_group_if_not(self):
        if not self.initialized:
            init_method = 'tcp://%s:%d'%(self.master_addr, self.master_port)
            args = {"backend": self.backend, "init_method": init_method, "rank": self.rank, "world_size": self.world_size}
            Logger.log("default comm group initialization started. args: %s" % str(args), flush=True)
            dist.init_process_group(self.backend, init_method=init_method, rank=self.rank, world_size=self.world_size)
            assert dist.get_world_size() == self.world_size
            Logger.log("default comm group initialized.", flush=True)
            self.initialized = True
            Logger.log("Testing c10d backend..", flush=True)
            self.testRing()

    def testRing(self):
        dstRank = (self.rank + 1) % self.world_size
        tag = 0
        tensor2send = torch.tensor(self.rank, dtype=torch.int)
        if self.backend == 'nccl':
            Logger.log("[CommunicationBackend] currently testRing deadlocks with nccl. Skipping test.", level=2, flush=True)
            return
            tensor2send = tensor2send.to(self.device)
        
        argsToPrint = {"tensor": tensor2send, "dst": dstRank, "tag": tag}
        Logger.log("[CommunicationBackend] testRing dist.send(%s)"%str(argsToPrint), level=0, flush=True)
        sendReq = dist.isend(tensor=tensor2send, dst=dstRank, tag=tag)
        
        srcRank = (self.world_size + self.rank - 1) % self.world_size
        tensor2recv = torch.zeros(1, dtype=torch.int)
        if self.backend == 'nccl':
            tensor2recv = tensor2recv.to(self.device)
        
        recvReq = dist.irecv(tensor=tensor2recv, src=srcRank, tag=tag)
        sendReq.wait()
        recvReq.wait()
        Logger.log("[CommunicationBackend] testRing completed. received %s"%str(tensor2recv), level=0, flush=True)
    
    def initCommGroups(self, jobName: str, commGrpDict: dict):
        commGrpHandlerDict = {}
        for grpName in commGrpDict:
            globalGrpRanks = commGrpDict[grpName]
            grpHandler = dist.new_group(globalGrpRanks)
            commGrpHandlerDict[grpName] = grpHandler
        self.commGrpHandlerDicts[jobName] = commGrpHandlerDict
        assert 'all' in self.commGrpHandlerDicts[jobName]
        Logger.log("Testing default collective comm all-reduce for %s among %d ranks." % (jobName, len(commGrpDict['all'])), flush=True)
        self.testAllGroupComm(jobName)

    def testAllGroupComm(self, jobName: str):
        commGrpHandlerDict = self.commGrpHandlerDicts[jobName]
        commGrpHandler = commGrpHandlerDict['all']
        tsr = torch.ones(2, dtype=torch.int)
        if self.backend == 'nccl':
            tsr = tsr.to(self.device)
        Logger.log("[CommunicationBackend] testAllGroupComm started for %s. Exchange %s" % (jobName, str(tsr)), flush=True)
        dist.all_reduce(tsr, dist.ReduceOp.SUM, commGrpHandler)
        Logger.log("[CommunicationBackend] testAllGroupComm completed for %s. Receive %s" % (jobName, str(tsr)), flush=True)

    def makeCommunicationHandler(self, jobName, worldSize, tensor_tags, jobRankToGlobalRank):
        sendFromCpu = (self.backend == 'gloo')
        deviceForComm = 'cpu' if self.backend == 'gloo' else self.device
        Logger.log("[CommunicationBackend] makeCommunicationHandler sendFromCpu(%s)"%str(sendFromCpu), level=0)
        if jobName not in self.commGrpHandlerDicts:
            Logger.log("Error in makeCommunicationHandler. commGroupHandlers are not previously initialized for %s." % jobName, level=2, flush=True)
        commGrpHandlerDict = self.commGrpHandlerDicts.pop(jobName)
        return CommunicationHandler(worldSize, tensor_tags, jobRankToGlobalRank, sendFromCpu, deviceForComm, commGrpHandlerDict, shouldSendSizes=True)

class CommunicationHandler:
    # Features.
    # - mapping from a rank for a training job to global runtime rank.
    # - keeps tensor dimension information for recv operation. (c10d recv needs a tensor with correct size)
    def __init__(self, worldSize, tensor_tags, jobRankToGlobalRank, sendFromCpu, deviceForComm, commGrpHandlerDict, shouldSendSizes: bool = True):
        self.tensorSizes = {}
        self.tensor_tags = tensor_tags
        self.jobRankToGlobalRank = jobRankToGlobalRank #list(range(worldSize))
        self.shouldSendSizes = shouldSendSizes
        self.sendFromCpu = sendFromCpu
        self.deviceForComm = deviceForComm
        self.asyncReqs = []
        # self.commGrpDict = commGrpDict
        self.commGrpHandlerDict = commGrpHandlerDict
        # self.addCommGroups(commGrpDict)

    # def initCommGroups(self, commGrpDict):
    #     for grpName in commGrpDict:
    #         grpRanks = commGrpDict[grpName]
    #         globalGrpRanks = [self.jobRankToGlobalRank[rank] for rank in grpRanks]
    #         grpHandler = dist.new_group(globalGrpRanks)
    #         self.commGrpHandlerDict[grpName] = grpHandler

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
            tensor_shape = torch.tensor(tensor.shape, dtype=torch.int, device=self.deviceForComm)
            tensor_shape_len = torch.tensor(len(tensor.shape), dtype=torch.int, device=self.deviceForComm)
            Logger.log("dist.send(%s)"%str({"tensor": tensor_shape_len.size(), "dst": dstRank, "tag": tag}), level=0, flush=True)
            dist.send(tensor=tensor_shape_len, dst=dstRank, tag=tag)
            Logger.log("dist.send(%s)"%str({"tensor": tensor_shape.size(), "dst": dstRank, "tag": tag}), level=0, flush=True)
            dist.send(tensor=tensor_shape, dst=dstRank, tag=tag)
            Logger.log("dist.isend(%s)"%str({"tensor": tensor.size(), "dst": dstRank, "tag": tag, "bytes": tensor.element_size()*tensor.nelement(), "elems": tensor.nelement(), "elemSize": tensor.element_size()}), level=0, flush=True)
        # Logger.log("dist.isend(%s)"%str({"tensor": tensor.size(), "dst": dstRank, "tag": tag}), level=0, flush=True)
        # dist.send(tensor=tensor, dst=dstRank, tag=tag)
        tensorReq = dist.isend(tensor=tensor, dst=dstRank, tag=tag)
        self.asyncReqs.append(tensorReq)
        # return tensorReq

    def waitForAll(self):
        for req in self.asyncReqs:
            req.wait()
        self.asyncReqs.clear()

    def recv(self, tensorName: str, src: int, dtype=torch.float32) -> torch.Tensor:
        self.waitForAll()
        tensor = self.recvAsync(tensorName, src, dtype)
        self.waitForAll()
        if self.sendFromCpu:
            tensor = tensor.cuda()
        return tensor

    def recvAsync(self, tensorName: str, src: int, dtype=torch.float32) -> torch.Tensor:
        src_rank = self.jobRankToGlobalRank[src]
        tag = self.tensor_tags[tensorName]
        if self.shouldSendSizes:
            tensor_shape_len = torch.zeros(1, dtype=torch.int, device=self.deviceForComm)
            Logger.log("dist.recv(%s)"%str({"tensor": tensor_shape_len.size(), "src": src_rank, "tag": tag}), level=0, flush=True)
            dist.recv(tensor=tensor_shape_len, src=src_rank, tag=tag)
            tensor_shape_len = list(map(lambda x: int(x), tensor_shape_len))
            
            tensor_shape = torch.zeros(tensor_shape_len, dtype=torch.int, device=self.deviceForComm)
            Logger.log("dist.recv(%s)"%str({"tensor": tensor_shape.size(), "src": src_rank, "tag": tag}), level=0, flush=True)
            dist.recv(tensor=tensor_shape, src=src_rank, tag=tag)
            tensor_shape = list(map(lambda x: int(x), tensor_shape))

            self.tensorSizes[tensorName] = tensor_shape
        else:
            tensor_shape = self.tensorSizes[tensorName]
        # Receive tensor.
        tensor = torch.empty(tensor_shape, dtype=dtype, device=self.deviceForComm, requires_grad=True)
        # Logger.log("dist.irecv(%s)"%str({"tensor": tensor.size(), "src": src_rank, "tag": tag, "require_grad": tensor.requires_grad}), level=0, flush=True)
        # dist.recv(tensor=tensor, src=src_rank, tag=tag)
        asyncReq = dist.irecv(tensor=tensor, src=src_rank, tag=tag)
        self.asyncReqs.append(asyncReq)
        # Logger.log("dist.irecv(%s)"%str({"require_grad": tensor.requires_grad}), level=0, flush=True)
        return tensor

    def allGather(self, tensorList, tensor, grpName):
        commGrpHandler = self.commGrpHandlerDict[grpName]
        dist.all_gather(tensorList, tensor, commGrpHandler)

    def allReduce(self, tensor, operation, grpName):
        commGrpHandler = self.commGrpHandlerDict[grpName]
        enum = dist.ReduceOp.SUM # operation argument should specify, now default to SUM
        if self.shouldSendSizes:
            Logger.log("dist.all_reduce(%s)"%str({"tensor": tensor.size(), "enum": enum, "grpName": grpName, "kbytes": (tensor.element_size()*tensor.nelement() / 1024)}), level=0, flush=True)
        dist.all_reduce(tensor, enum, commGrpHandler)
