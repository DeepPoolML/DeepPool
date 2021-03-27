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

import xmlrpc.server
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import re
import threading
import signal
import sys
import time
import json
import sys
from datetime import datetime
from argparse import ArgumentParser, REMAINDER
from runnableModule import RunnableModule
from runnableModule import VisionDataLoaderGenerator
from communication import CommunicationBackend
from logger import Logger

class JobContext:
    def __init__(self, model: nn.Module, name: str, dataLoader, epochsToTrain = 1, optimizer = None, criterion = nn.CrossEntropyLoss(), device="cpu"):
        self.model = model
        self.name = name
        self.dataLoader = dataLoader
        self.dataLoaderIt = iter(self.dataLoader)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochsToTrain = epochsToTrain
        self.epoch = 0
        self.iter = 0
        self.itersToTrain = len(dataLoader) # this may be overwritten for development.
        self.itersPerPoll = 30
        self.training_initialized = False
        

    def train_init(self):
        Logger.log("[JobContext] <%s> train_init at device:%s"%(self.name, self.device), flush=True)
        self.model.to(self.device)
        self.model.train()
        self.training_initialized = True
    
    def limit_iters_to_train(self, iterationCount):
        self.itersToTrain = min(iterationCount, len(self.dataLoader))
        
    def train_single_iter(self):
        Logger.log("[JobContext] <%s> train_single_iter epoch:%d/%d iter:%d/%d" % 
                (self.name, self.epoch, self.epochsToTrain, self.iter, self.itersToTrain), level=0)
        if not self.training_initialized:
            self.train_init()
        
        data, target = next(self.dataLoaderIt) # self.dataLoader[self.iter]
        data, target = data.to(self.device), target.to(self.device)
        # optimizer.zero_grad()
        
        Logger.log("forward pass is starting.. data: %s" % str(data.size()), level=0)
        output, runCriterionAndLoss = self.model(data)
        Logger.log("forward pass is completed.. output: %s runCriterionAndLoss: %s" %
                    (str(output.size()), str(runCriterionAndLoss)), level=0, flush=True)
        # output = torch.flatten(output, 1)
        if runCriterionAndLoss:
            output = F.log_softmax(output, dim=1)
            
            # Hack to match target's sample count with the output at this node.
            if output.size()[0] != target.size()[0]:
                target = torch.repeat_interleave(target, int(1 + output.size()[0] / target.size()[0]), dim=0)
                target = target.narrow(0, 0, output.size()[0])

            loss = self.criterion(output, target)
            Logger.log("backward pass is starting", level=0, flush=True)
            loss.backward()
        else:
            Logger.log("backward pass is starting", level=0, flush=True)
            output.backward(output) # gradient passed is dummy.

        # optimizer.step()
        self.iter += 1
        if self.iter == 1:
            self.model.commHandler.stopSendingSizes()
        if self.iter == self.itersToTrain:
            self.iter = 0
            self.epoch += 1
        # TODO: check iter is over len(dataLoader), bump epoch when down with iter.

    def isCompleted(self):
        if self.epoch == self.epochsToTrain:
            return True
        else:
            return False

class Runtime(xmlrpc.server.SimpleXMLRPCServer):
    """A pool runtime that reside perpetually for each GPU in the cluster.
    
    This class is launched by ClusterCoordinator.
    """

    def __init__(self, coordinatorAddr: str, coordinatorPort: int, myAddr: str,
                myPort: int, device: int, c10dBackend: str, c10dMasterPort: int, rank: int, worldSize: int):
        super(Runtime, self).__init__((myAddr, myPort))
        self.coordinatorAddr = coordinatorAddr
        self.coordinatorPort = coordinatorPort
        self.myAddr = myAddr
        self.myPort = myPort
        self.device = ("cuda:%d" % device) if device is not "cpu" else device
        print("self.device=%s"%str(self.device))
        print("torch.cuda.current_device(): ", torch.cuda.current_device())
        print('torch.cuda availability: ', torch.cuda.is_available())
        print('torch.cuda.nccl version: ', torch.cuda.nccl.version())
        print('torch.distributed availability: ', dist.is_available())
        print('torch.distributed.nccl availability: ', dist.is_nccl_available())
        self.jobs = []
        self.pollInvokeCounter = 0
        self.shutdownRequested = False
        self.commBackend = CommunicationBackend(rank, worldSize, coordinatorAddr, c10dMasterPort, c10dBackend, self.device)
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("[%s] Runtime initialized with coordAddr=%s:%d, myAddr=%s:%d, device=%d" %
            (date_time, coordinatorAddr, coordinatorPort, myAddr, myPort, device) )
        sys.stdout.flush()
    
    def _dispatch(self, method, params):
        """ Custom dispatcher for XML-RPC server. """
        try:
            # We are forcing the 'export_' prefix on methods that are
            # callable through XML-RPC for security.
            func = getattr(self, 'export_' + method)
        except AttributeError:
            raise Exception('method "%s" is not supported' % method)
        else:
            return func(*params)
    
    ######################################################
    ## RPC handlers
    ######################################################
    def export_initCommBackend(self):
        self.commBackend.init_comm_group_if_not()
        return "commBackend initialized. @ %s!"%self.myAddr
    
    def export_scheduleTraining(self, name: str, jobInJson: str, dataDir: str, tensorTagsInJson: str):
        # self.commBackend.init_comm_group_if_not()
        worldSize = json.loads(jobInJson)["maxGpusUsed"]
        tensorTags = json.loads(tensorTagsInJson)

        # all_gather and all_reduce testing code inserted
        test_comm_grp_dict = {'all': [0,1,2,3], 'partial': [1,0]}
        commHandler = self.commBackend.makeCommunicationHandler(worldSize, test_comm_grp_dict, tensorTags)
        commHandler.addCommGroupDict(test_comm_grp_dict)
        tsr = torch.zeros(2, dtype=torch.int, device=self.device) + 10 + self.commBackend.rank
        Logger.log("my tensor: %s" % str(tsr), flush=True)
        for grp_name in test_comm_grp_dict:
            grp_ranks = test_comm_grp_dict[grp_name]
            Logger.log("grp_name: %s grp_ranks: %s" % (grp_name, str(grp_ranks)), flush=True)
            if self.commBackend.rank in grp_ranks:
                tsr_list = [torch.zeros(2, dtype=torch.int, device=self.device) for _ in range(len(grp_ranks))]
                Logger.log("BEFORE all_gather tensor_list: %s" % str(tsr_list), flush=True)
                commHandler.allGather(tsr_list, tsr, grp_name)
                Logger.log(" AFTER all_gather tensor_list: %s" % str(tsr_list), flush=True)
                Logger.log("BEFORE all_reduce tensor: %s" % str(tsr), flush=True)
                commHandler.allReduce(tsr, 0, grp_name)
                Logger.log(" AFTER all_reduce tensor: %s" % str(tsr), flush=True)
        # testing code end

        # commHandler = self.commBackend.makeCommunicationHandler(worldSize, {}, tensorTags)
        module = RunnableModule(jobInJson, commHandler)
        if dataDir == "SYNTHETIC":
            dataDir = None # Use synthetic dataset.
        loader = VisionDataLoaderGenerator.genDataLoader(
            jobInJson, dataDir, syntheticDataLength=1600)
        job = JobContext(module, name, loader, device=self.device)
        
        job.limit_iters_to_train(500)
        self.jobs.append(job)
        Logger.log("Scheduled a training job (%s). Total jobs on queue: %d" % (name, len(self.jobs)))
        return "Scheduled a training job. @ %s!"%self.myAddr

    def export_poke(self):
        return 'Returned from poke at %s' % self.myAddr

    def export_shutdown(self):
        self.shutdownRequested = True
        Logger.log("Shutdown requested.", flush=True)
        # shutdown_thread.join()
        # self.__shutdown_request = True # TODO: testing
        return 'Returned from remote_shutdown at %s:%d' % (self.myAddr, self.myPort)

    ######################################################
    ## Internal processing
    ######################################################
    def getCoordinatorProxy(self):
        return xmlrpc.client.ServerProxy("http://%s:%d/"%(self.coordinatorAddr, self.coordinatorPort))

    def poll(self):
        """ This method manages ongoing training tasks.
        WARNING: this method should never block.
        It is invoked every BaseServer::poll_interval
        """
        hadWork = False
        if len(self.jobs) > 0:
            hadWork = True
            startTime = time.time()
            job = self.jobs[0]
            for itersRan in range(job.itersPerPoll):
                job.train_single_iter()
                if job.isCompleted():
                    self.getCoordinatorProxy().notifyTrainingFinished(self.myAddr, job.name, len(self.jobs) - 1)
                    Logger.log("Training job <%s> is finished." % job.name, flush=True)
                    self.jobs.pop(0)
                    break
            elapsed = time.time() - startTime
            Logger.log("[poll] <%s> epoch:%d/%d iter:%d/%d  %3.1f ms per iter." % 
                (job.name, job.epoch, job.epochsToTrain, job.iter, job.itersToTrain, (1000.0*elapsed)/ job.itersPerPoll))

        # self.pollInvokeCounter += 1
        # if self.pollInvokeCounter % 1 == 0:
        #     print("poll() invoked %d times at %s for device: %s" % (self.pollInvokeCounter, self.myAddr, self.device))
        return hadWork

    def service_actions(self):
        self.poll()
        if self.shutdownRequested:
            def invoke_shutdown():
                Logger.log("Shutting down from shutdown thread.", flush=True)
                self.shutdown()
            shutdown_thread = threading.Thread(name='invoke_shutdown', target=invoke_shutdown)
            Logger.log("Shutting down in 1 sec.", flush=True)
            time.sleep(1)
            shutdown_thread.start()

    def run(self, poll_interval=1):
        # TODO: remove... This method blocks! Switched to overwriting service_actions().
        self.shutdownRequested = False
        while not self.shutdownRequested:
            self.handle_request()
            hadWork = self.poll()
            if not hadWork:
                time.sleep(poll_interval)
        Logger.log("Shutdown is requested.", flush=True)

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Runtime")
    # Optional arguments for the launch helper
    parser.add_argument("--coordinatorAddr", type=str, default="localhost:12340",
                        help="IP:port to the cluster coordinator")
    parser.add_argument("--myAddr", type=str, default="localhost:1234",
                        help="IP:port this runtime should listen to."
                        "coordinator will talk to this node on this address")
    parser.add_argument("--device", type=int, default=0,
                        help="cuda device for pytorch.")
    parser.add_argument("--c10dBackend", type=str, default="nccl",
                        help="pytorch c10d communication backend. Type either nccl or gloo")
    parser.add_argument("--c10dMasterPort", type=int, default="55555",
                        help="coordinator's port for c10d communication package initialization")
    parser.add_argument("--rank", type=int, default=0,
                        help="global rank for c10d.")
    parser.add_argument("--worldSize", type=int, default=1,
                        help="global world size for c10d.")
    parser.add_argument("--logdir", default=None, type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    coordinatorAddrCombined = re.split('[-:]', args.coordinatorAddr)
    coordinatorAddr = coordinatorAddrCombined[0]
    coordinatorPort = int(coordinatorAddrCombined[1])
    myAddrCombined = re.split('[-:]', args.myAddr)
    myAddr = myAddrCombined[0]
    myPort = int(myAddrCombined[1])

    runtime = Runtime(coordinatorAddr, coordinatorPort, myAddr, myPort,
                      args.device, args.c10dBackend, args.c10dMasterPort, args.rank, args.worldSize)
    # runtime.run()
    runtime.serve_forever(poll_interval=0)

if __name__ == "__main__":
    main()