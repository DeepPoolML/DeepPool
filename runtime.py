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
import traceback
from typing import Optional, List, Any
from datetime import datetime
from argparse import ArgumentParser, REMAINDER
from runnableModule import RunnableModule
from runnableModule import VisionDataLoaderGenerator
from runnableModule import TargetShuffler
from communication import CommunicationBackend
from logger import Logger
from timetrace import EventTypes
from timetrace import Timetrace as TT
from contextlib import nullcontext

from collections import defaultdict

try:
    import be_training
except:
    be_training = None


import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
import pyprof
# pyprof.init()

class JobContext:
    def __init__(self, model: nn.Module, name: str, dataLoader, commHandler, targetShuffler,
            epochsToTrain = 1, optimizer = None, criterion = None, device="cpu", runtime=None):
        self.model = model
        self.name = name
        self.dataLoader = dataLoader
        self.dataLoaderIt = iter(self.dataLoader) if dataLoader != None else None
        self.commHandler = commHandler
        self.targetShuffler = targetShuffler
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss().cuda(device) if criterion == None else criterion
        self.device = device
        self.epochsToTrain = epochsToTrain
        self.epoch = 0
        self.iter = 0
        self.itersToTrain = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
        self.itersPerPoll = 50
        self.training_initialized = False
        self.itersToCapture = set(range(250, 260))
        self.iterTimeDuringLastRun = 0
        self.runtime = runtime

        self.idle_tracking_init()

    # Allow a job to track and report its intra-round idle time

    def idle_tracking_init(self):
        self.idle_timings = []
        self.idle_timings_raw = defaultdict(list)
        self.cur_idle_round = 0
        self.idle_start_events = []
        self.idle_end_events = []
        self.idle_iter_track_start = 100
        self.idle_iter_track_end = 200

    def get_be_stats(self):
        if not self.idle_timings:
            return 0, 0
        cur_be_stats = be_training.query()
        be_iters_trained = cur_be_stats["full_iterations"] - self.init_stats["full_iterations"]
        total_fg_iters = (cur_be_stats["individual_grants"] - self.init_stats["individual_grants"])
        total_fg_iters //= len(self.idle_timings)
        images_trained_per_iter = (be_iters_trained / total_fg_iters) * self.runtime.be_task_enabled
        msec_idle_per_iter = sum(self.idle_timings)
        return images_trained_per_iter, msec_idle_per_iter

    def idle_track_should_record(self):
        return (self.iter >= self.idle_iter_track_start and
                self.iter < self.idle_iter_track_end)

    def mark_idle(self):
        if self.idle_track_should_record():
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.idle_start_events.append(ev)
        elif self.idle_timings:
            self.runtime.report_idle(self.idle_timings[self.cur_idle_round])

    def mark_non_idle(self):
        self.cur_idle_round += 1
        if not self.idle_track_should_record():
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.idle_end_events.append(ev)

    def idle_measurement_finalize(self):
        for layerno, times in self.idle_timings_raw.items():
            times = sorted(times)
            med = times[len(times) // 2]
            self.idle_timings.append(med)
            Logger.log("[{}] Idle time discovered: {} ms".format(layerno, med))
        if self.idle_timings:
            self.init_stats = be_training.query()

    def idle_measurement_round_done(self):
        self.cur_idle_round = 0
        if not self.idle_track_should_record():
            if self.iter == self.idle_iter_track_end:
                self.idle_measurement_finalize()
            return
        torch.cuda.synchronize()
        s, e = self.idle_start_events, self.idle_end_events
        assert len(s) == len(e)
        for layer, (start_ev, end_ev) in enumerate(zip(s, e)):
            ms = start_ev.elapsed_time(end_ev)
            assert ms >= 0
            self.idle_timings_raw[layer].append(ms)
        self.idle_start_events.clear()
        self.idle_end_events.clear()


    def train_init(self):
        Logger.log("[JobContext] <%s> train_init at device:%s"%(self.name, self.device), flush=True)
        self.model.to(self.device)
        self.model.train()
        self.training_initialized = True
    
    def limit_iters_to_train(self, iterationCount):
        if self.dataLoader == None:
            self.itersToTrain = iterationCount
        else:
            self.itersToTrain = min(iterationCount, len(self.dataLoader))

    def train_single_iter(self):
        Logger.log("[JobContext] <%s> train_single_iter epoch:%d/%d iter:%d/%d" % 
                (self.name, self.epoch, self.epochsToTrain, self.iter, self.itersToTrain), level=0)
        if not self.training_initialized:
            self.train_init()
        
        TT.cudaInitIter(self.iter)
        nvtx.range_push("Copy to device")
        if self.dataLoaderIt != None:
            data, targetRaw = next(self.dataLoaderIt)
            data.requires_grad_(False)
            targetRaw.requires_grad_(False)
            data = data.to(device=self.device, non_blocking=True)
            data.requires_grad_(True)
            # Logger.log("train_single_iter target's size: %s"%str(target.size()), level=0, flush=True)
        else:
            data = None
            targetRaw = None
        self.optimizer.zero_grad()
        nvtx.range_pop()


        if self.iter in self.itersToCapture and (self.iter - 1) not in self.itersToCapture:
            profiler.start()
        
        nvtx.range_push("Target shuffle")
        target = self.targetShuffler.shuffle(targetRaw)
        nvtx.range_pop()
        TT.cudaRecord(EventTypes.target_shuffle)
        torch.cuda.synchronize(device=self.device)

        Logger.log("forward pass is starting.", level=0)
        nvtx.range_push("Forward Pass")
        if data != None and data.size()[0] > 0:
            TT.cudaRecord(EventTypes.fp_start)
        else:
            TT.cudaRecord(EventTypes.fp_start_idle)
        
        output, runCriterionAndLoss = self.model(data)
        nvtx.range_pop()
        TT.cudaRecord(EventTypes.fp_done)
        # Logger.log("forward pass is completed.. output: %s runCriterionAndLoss: %s" %
        #             (str(output.size() if output != None else None), str(runCriterionAndLoss)), level=0, flush=True)
        # output = torch.flatten(output, 1)
        nvtx.range_push("Backward Pass")
        if runCriterionAndLoss:
            output = F.log_softmax(output, dim=1)
            
            if output.size()[0] != target.size()[0]:
                Logger.log("error! target size doesn't match even after shuffling.", flush=True, level=2)
                # Hack to match target's sample count with the output at this node.
                # target = torch.repeat_interleave(target, int(1 + output.size()[0] / target.size()[0]), dim=0)
                # target = target.narrow(0, 0, output.size()[0])

            loss = self.criterion(output, target)
            TT.cudaRecord(EventTypes.bp_start)
            Logger.log("backward pass is starting", level=0, flush=True)
            loss.backward()
        # else:
        #     Logger.log("backward pass is starting", level=0, flush=True)
        #     output.backward(output) # gradient passed is dummy.
        Logger.log("backward remainder is starting", level=0, flush=True)
        TT.cudaRecord(EventTypes.bp_remainder_start)
        self.model.backwardRemainder()
        nvtx.range_pop()
        TT.cudaRecord(EventTypes.bp_done)

        if self.iter in self.itersToCapture and (self.iter + 1) not in self.itersToCapture:
            profiler.stop()

        # optimizer.step()

        TT.cudaFinishIter(self.iter)
        self.idle_measurement_round_done()

        self.iter += 1
        if self.iter == 1:
            self.model.commHandler.stopSendingSizes()
        if self.iter == self.itersToTrain:
            self.iter = 0
            self.epoch += 1

    def isCompleted(self):
        return self.epoch == self.epochsToTrain

class Runtime(xmlrpc.server.SimpleXMLRPCServer):
    """A pool runtime that reside perpetually for each GPU in the cluster.
    
    This class is launched by ClusterCoordinator.
    """

    def __init__(self, coordinatorAddr: str, coordinatorPort: int, myAddr: str,
                myPort: int, device: int, c10dBackend: str, c10dMasterPort: int, rank: int, worldSize: int, args):
        super(Runtime, self).__init__((myAddr, myPort))
        self.coordinatorAddr = coordinatorAddr
        self.coordinatorPort = coordinatorPort
        self.myAddr = myAddr
        self.myPort = myPort
        self.device = ("cuda:%d" % device) if device != "cpu" else device
        self.be_task_enabled = 0
        self.profile = args.profile
        torch.cuda.set_device(device)
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

    def export_initCommGroups(self, jobName: str, commGroupsInJson: str):
        commGrpDict = json.loads(commGroupsInJson)
        self.commBackend.initCommGroups(jobName, commGrpDict)
        Logger.log("[initCommGroups] Initialized new groups for %s (%d groups)" % (jobName, len(commGrpDict)))
        return "commBackend initialized new groups for %s" % jobName
        
    def export_scheduleTraining(self, name: str, jobInJson: str, dataDir: str, tensorTagsInJson: str, jobRankToGlobalRankInJson: str):
        """ Schedules a training task to this runtime. """
        jobSpec = json.loads(jobInJson)
        worldSize = jobSpec["maxGpusUsed"]
        tensorTags = json.loads(tensorTagsInJson)
        jobRankToGlobalRank = json.loads(jobRankToGlobalRankInJson)

        commHandler = self.commBackend.makeCommunicationHandler(name, worldSize, tensorTags, jobRankToGlobalRank)

        # # all_gather and all_reduce testing code inserted
        # test_comm_grp_dict = {'all': [0,1,2,3], 'partial': [1,0]}
        # tsr = torch.zeros(2, dtype=torch.int, device=self.device) + 10 + self.commBackend.rank
        # Logger.log("my tensor: %s" % str(tsr), flush=True)
        # for grp_name in test_comm_grp_dict:
        #     grp_ranks = test_comm_grp_dict[grp_name]
        #     Logger.log("grp_name: %s grp_ranks: %s" % (grp_name, str(grp_ranks)), flush=True)
        #     if self.commBackend.rank in grp_ranks:
        #         tsr_list = [torch.zeros(2, dtype=torch.int, device=self.device) for _ in range(len(grp_ranks))]
        #         Logger.log("BEFORE all_gather tensor_list: %s" % str(tsr_list), flush=True)
        #         commHandler.allGather(tsr_list, tsr, grp_name)
        #         Logger.log(" AFTER all_gather tensor_list: %s" % str(tsr_list), flush=True)
        #         Logger.log("BEFORE all_reduce tensor: %s" % str(tsr), flush=True)
        #         commHandler.allReduce(tsr, 0, grp_name)
        #         Logger.log(" AFTER all_reduce tensor: %s" % str(tsr), flush=True)
        # # testing code end

        module = RunnableModule(jobInJson, commHandler, self.device, self)
        if dataDir == "SYNTHETIC":
            dataDir = None # Use synthetic dataset.
        loader = VisionDataLoaderGenerator.genDataLoader(jobInJson, dataDir, syntheticDataLength=320000)
        targetShuffler = TargetShuffler(commHandler, jobSpec["rank"], jobSpec["initialBatchSizes"],
                                        jobSpec["sampleIndices"], device=self.device)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        job = JobContext(module, name, loader, commHandler, targetShuffler, optimizer=optimizer, device=self.device, runtime=self)
        
        job.limit_iters_to_train(1000)
        # try:
        #     job.limit_iters_to_train(1)
        # except Exception as e:
        #     print(e)
        #     print(traceback.format_exc())

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
    ## utilizing idle time
    ######################################################

    def enable_be_task(self, batch_size):
        assert "cuda:" in self.device
        device = int(self.device.split(":")[1])
        if be_training is None:
            Logger.log("Failed to initialize BE task... is it installed?")
            return
        be_training.init(batch_size, device)
        self.be_task_enabled = batch_size

    # A job reports it is temporarily idle
    def report_idle(self, idle_time_ms):
        if self.be_task_enabled:
            be_training.train_for(int(idle_time_ms * 1000))


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
            ctx = torch.autograd.profiler.emit_nvtx if self.profile else nullcontext
            with ctx():
                for itersRan in range(job.itersPerPoll):
                    self.cur_job = job
                    job.train_single_iter()
                    self.cur_job = None
                    if job.isCompleted():
                        stats = TT.getStats()
                        if self.be_task_enabled:
                            be_im_iter, idle_ms_iter = job.get_be_stats()
                        else:
                            be_im_iter, idle_ms_iter = 0, 0
                        self.getCoordinatorProxy().notifyTrainingFinished(self.myAddr, job.name, be_im_iter, idle_ms_iter, len(self.jobs) - 1, *stats) # job.iterTimeDuringLastRun
                        Logger.log("Training job <%s> is finished." % job.name, flush=True)
                        self.jobs.pop(0)
                        TT.printAllTraces()
                        TT.reset()
                        break
            elapsed = time.time() - startTime
            job.iterTimeDuringLastRun = (1000.0*elapsed)/ job.itersPerPoll
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
    parser.add_argument("--be_batch_size", default=16, type=int, help="best effort batch size, 0 for disabled")
    parser.add_argument("--profile", default=False, action='store_true', help="runtime will be profiled")

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
                      args.device, args.c10dBackend, args.c10dMasterPort, args.rank, args.worldSize, args)

    if args.be_batch_size > 0:
        runtime.enable_be_task(args.be_batch_size)

    # runtime.run()
    runtime.serve_forever(poll_interval=0)

if __name__ == "__main__":
    main()
