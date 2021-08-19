import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
import collections
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import jsonpickle
import json

import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
import pyprof

class Perf(object):
    def __init__(self, eidToStr = {}):
        super(Perf, self).__init__()
        self.measurements = []
        self.sum = []
        self.count = []
        self.eidToStr = eidToStr
        
    def recordTime(self, eid, elapsedTime):
        if eid >= len(self.measurements):
            self.measurements += [[]] * (eid - len(self.measurements) + 1)
            self.sum += [0.0] * (eid - len(self.sum) + 1)
            self.count += [0] * (eid - len(self.count) + 1)
        self.measurements[eid].append(elapsedTime)
        self.sum[eid] += elapsedTime
        self.count[eid] += 1
        
    def printStats(self):
        # maxEventStrLen = max([len(eventStr) for eventStr in self.eidToStr.values()])
        for eid in range(len(self.measurements)):
            if self.count[eid] == 0:
                continue
            median = sorted(self.measurements[eid])[int(len(self.measurements[eid]) / 2)]
            if eid in self.eidToStr:
                print("Event %15s ==> avg: %8.1f us,  median: %8.1f us" % (self.eidToStr[eid], self.sum[eid] / self.count[eid], median))
            else:
                print("Event %5d ==> avg: %8.1f us,  median: %8.1f us" % (eid, self.sum[eid] / self.count[eid], median))

    def getStat(self, eid):
        return sorted(self.measurements[eid])[int(len(self.measurements[eid]) / 2)]
        # return self.sum[eid] / self.count[eid]

    def printHeader(self):
        print("#BatchSize", end = "")
        print("     width", end = "")
        print("   filters", end = "")
        print("       mults", end = "")
        print(" |  AVG : ", end = "")
        for eid in range(len(self.measurements)):
            if eid in self.eidToStr:
                print("%10s" % self.eidToStr[eid], end = "")
            else:
                print("Event %4d" % eid, end = "")
        print(" |Median: ", end = "")
        for eid in range(len(self.measurements)):
            if eid in self.eidToStr:
                print("%10s" % self.eidToStr[eid], end = "")
            else:
                print("Event %4d" % eid, end = "")
        print(" | Accuracy", end = "")
        print(" | Count(eid0)")
    
    def printAll(self, batchSize, width, filterCount, accuracy):
        # Avg.
        print("%9d " % batchSize, end = "")
        print("%9d " % width, end = "")
        print("%9d " % filterCount, end = "")
        print("%11d " % (batchSize * width * width * filterCount * 9 * 3), end = "")
        print("%10s"%"", end = "")
        for eid in range(len(self.measurements)):
            if self.count[eid] == 0:
                continue
            print("%10.1f" % (self.sum[eid] / self.count[eid]), end = "")

        print(" %9s"%"", end = "")
        for eid in range(len(self.measurements)):
            if self.count[eid] == 0:
                continue
            median = sorted(self.measurements[eid])[int(len(self.measurements[eid]) / 2)]
            print("%10.1f" % median, end = "")
        print(" %9.2f" % accuracy, end = "")
        print(" %10d" % len(self.measurements[0]))


class GpuProfiler:
    def __init__(self, device):
        self.conv2dBenchCache = {}
        self.benchCacheHit = 0
        self.benchCacheMiss = 0
        self.linearBenchCache = {}
        self.device = device

    def saveProfile(self, path = "gpuProfile.json"):
        with open(path, "w") as outfile:
            data = {"conv2dBenchCache": self.conv2dBenchCache, "linearBenchCache": self.linearBenchCache}
            planInJson = jsonpickle.encode(data, unpicklable=False)
            json.dump(json.loads(planInJson), outfile, indent=2, sort_keys=False)
            print("[GpuProfiler] Saved %d entries." % (len(self.conv2dBenchCache) + len(self.linearBenchCache)))
            print("[GpuProfiler] Cache hit %3.1f %%" % (100 * self.benchCacheHit / (self.benchCacheHit + self.benchCacheMiss)))
    
    def loadProfile(self, path = "gpuProfile.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                if "conv2dBenchCache" in data:
                    self.conv2dBenchCache = data["conv2dBenchCache"]
                if "linearBenchCache" in data:
                    self.linearBenchCache = data["linearBenchCache"]
        except IOError:
            print("[GpuProfiler] No profile file exists at %s." % path)

    def train(self, model, device, train_loader, criterion, optimizer, epoch, perf, profile=False):
        model.train()
        iter_to_capture_start = 50
        iter_to_capture_end = 53
        with torch.autograd.profiler.emit_nvtx():
            iterationCount = 0
            for batch_idx, (data, target) in enumerate(train_loader):        
                start_time = time.time()
            
                ev_zero = torch.cuda.Event(enable_timing=True)
                ev_fp = torch.cuda.Event(enable_timing=True)
                ev_loss = torch.cuda.Event(enable_timing=True)
                ev_bp = torch.cuda.Event(enable_timing=True)
                
                # if profile and iterationCount == iter_to_capture_start:
                #     print("profiler started.")
                #     profiler.start()

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                ev_zero.record()
                output = model(data)
                ev_fp.record()
                output = torch.flatten(output, 1)
                output = F.log_softmax(output, dim=1)
                loss = criterion(output, target)
                
                # bp_start = time.time()
                # torch.cuda.synchronize()
                ev_loss.record()
                # loss.backward()
                output.backward(output)
                ev_bp.record()
                # torch.cuda.synchronize()
                # bpTime = (time.time() - bp_start) * 1E6

                optimizer.step()
                
                # if profile and iterationCount == iter_to_capture_end:
                #     print("profiler ended.")
                #     profiler.stop()

                ev_bp.synchronize()
            
                stop_time = time.time()
                # perf.recordTime(0, 1000 * ev_start.elapsed_time(ev_load))
                # perf.recordTime(1, 1000 * ev_load.elapsed_time(ev_zero))
                perf.recordTime(2, 1000 * ev_zero.elapsed_time(ev_fp))
                # perf.recordTime(3, 1000 * ev_fp.elapsed_time(ev_loss))
                perf.recordTime(4, 1000 * ev_loss.elapsed_time(ev_bp))
                # perf.recordTime(4, bpTime)
                # perf.recordTime(4, 1000 * ev_fp.elapsed_time(ev_bp))
                # perf.recordTime(5, 1000 * ev_bp.elapsed_time(ev_opt))
                # perf.recordTime(6, 1000 * ev_start.elapsed_time(ev_opt))
                perf.recordTime(7, (stop_time - start_time) * 1000 * 1000)
            
                iterationCount += 1

    def benchModel(self, model, inputSize, batchSize, profile=False):
        train_dataset = self.SyntheticDataset(inputSize, batchSize * 200) # 30) # 
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchSize, shuffle=False, pin_memory=True, drop_last=True)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss().cuda(self.device)

        perfStat = Perf({0: 'load', 1: 'zero', 2: 'fp', 3: 'loss', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
        self.train(model.cuda(), self.device, train_loader, criterion, optimizer, 1, perfStat, profile)
        gpuTime = perfStat.getStat(2) + perfStat.getStat(4)
        if profile:
            print("%.f + %.f" % (perfStat.getStat(2), perfStat.getStat(4)))
        return gpuTime, perfStat.getStat(2), perfStat.getStat(4)


    def runConv2dBench(self, config, params, profile=False):
        if str((config, params)) in self.conv2dBenchCache and profile == False:
            self.benchCacheHit += 1
            return self.conv2dBenchCache[str((config, params))]
        self.benchCacheMiss += 1
        batchSize = config[0]
        width = config[1]
        height = config[2]
        inChannels = config[3]
        filterCount = config[4]
        train_dataset = self.SyntheticDataset((inChannels, width, height), batchSize * 200, 100) # 
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchSize, shuffle=False, pin_memory=True, drop_last=True)

        newParams = params.copy()
        newParams["in_channels"] = inChannels
        newParams["out_channels"] = filterCount
        # model = self.Conv2dOp(inChannels, filterCount).to(self.device)
        model = nn.Conv2d(**newParams).to(self.device)

        # optimizer = torch.optim.Adadelta(model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss().cuda(self.device)

        perfStat = Perf({0: 'load', 1: 'zero', 2: 'fp', 3: 'loss', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
        # scheduler = StepLR(optimizer, step_size=1)
        self.train(model, self.device, train_loader, criterion, optimizer, 1, perfStat, profile)
        # scheduler.step()
        gpuTime = perfStat.getStat(2) + perfStat.getStat(4)
        if profile:
            print("%.f + %.f" % (perfStat.getStat(2), perfStat.getStat(4)))
        self.conv2dBenchCache[str((config, params))] = gpuTime
        return gpuTime

    def runLinearBench(self, config, profile=False):
        if str(config) in self.linearBenchCache and profile == False:
            self.benchCacheHit += 1
            return self.linearBenchCache[str(config)]
        self.benchCacheMiss += 1
        batchSize = config[0]
        inFeatures = config[1]
        outFeatures = config[2]
        train_dataset = self.SyntheticDataset((inFeatures), batchSize * 200, num_classes=outFeatures)
        # train_dataset = self.SyntheticDataset((inFeatures), batchSize * 30, num_classes=outFeatures)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchSize, shuffle=False, pin_memory=True, drop_last=True)

        model = self.LinearOp(inFeatures, outFeatures).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss().cuda(self.device)
        
        perfStat = Perf({0: 'load', 1: 'zero', 2: 'fp', 3: 'loss', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
        scheduler = StepLR(optimizer, step_size=1)
        self.train(model, self.device, train_loader, criterion, optimizer, 1, perfStat, profile)
        # scheduler.step()
        if profile:
            print("%.f + %.f" % (perfStat.getStat(2), perfStat.getStat(4)))
        gpuTime = perfStat.getStat(2) + perfStat.getStat(4)
        self.linearBenchCache[str(config)] = gpuTime
        return gpuTime


    class SyntheticDataset(torch.utils.data.dataset.Dataset):
        def __init__(self, input_size, length, num_classes=1000):
            self.tensor = Variable(torch.rand(input_size)).type(torch.FloatTensor)
            self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
            self.length = length
        def __getitem__(self, index):
            return self.tensor, self.target
        def __len__(self):
            return self.length

    # class Conv2dOp(nn.Module):
    #     def __init__(self, inChannels, filterCount, num_classes=1000):
    #         super(GpuProfiler.Conv2dOp, self).__init__()
    #         self.num_classes = num_classes
    #         self.conv1 = nn.Conv2d(inChannels, filterCount, (3, 3), (1, 1), (1, 1))
    #     def forward(self, x):
    #         x = self.conv1(x)
    #         return x
    
    class LinearOp(nn.Module):
        def __init__(self, inFeatures, outFeatures):
            super(GpuProfiler.LinearOp, self).__init__()
            self.linear1 = nn.Linear(inFeatures, outFeatures)
        def forward(self, x):
            x = self.linear1(x)
            return x
