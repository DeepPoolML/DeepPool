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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
import collections
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import math
import jsonpickle
import json
import numpy
import graphviz
from array import array
from typing import Optional, IO, List, Any
from gpuProfiler import GpuProfiler
from jobDescription import TrainingJob
from jobDescription import Layer

import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
import pyprof

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SearchContext:
    def __init__(self, totalGpus: int, globalBatch: int, amplificationLimit: float = 2.0,
            dataParallelBaseline = False, sampleSplit=True, spatialSplit=False, filterSplit=False):
        self.totalGpus = totalGpus
        self.globalBatch = globalBatch
        self.amplificationLimit = amplificationLimit
        self.dataParallelBaseline = dataParallelBaseline
        self.sampleSplit = sampleSplit
        self.spatialSplit = spatialSplit
        self.filterSplit = filterSplit
        print("SearchContext: " + str(self.__dict__))


class CostSim:
    def __init__(self, profiler: GpuProfiler, netBw = 1.25E4, verbose=False):
        self.profiler = profiler
        self.layers: List[Layer] = []
        self.NET_BANDWIDTH = netBw
        self.NET_LATENCY = 140 #40
        self.verbose = verbose

    def generateModuleDescription(self, layerConfigs: list, globalBatch: int):
        # gpuTimeSum = 0
        # profiler.start()
        maxGpuUsed = 0
        for l, config in zip(self.layers, layerConfigs):
            maxGpuUsed = max(maxGpuUsed, self.calcGpusNeeded(l, config, globalBatch))
            # nvtx.range_push("l %d %s" % (l.id, l.name))
            # gpuTime = self.benchGpuTime(l, config, profile=True)
            # gpuTime = self.benchGpuTime(l, config)
            # nvtx.range_pop()
            # gpuTimeSum += gpuTime
        # print("gpuTimeSum: ", gpuTimeSum)
        # profiler.stop()

        return TrainingJob("test", self.layers, layerConfigs, globalBatch, maxGpuUsed, "na")
        
        # job.dumpSingleRunnableModule(15)

    def printAllLayers(self, slient=False):
        #TODO: topological sort of layers. Right now, assume it's sorted.
        for i in range(len(self.layers)):
            self.layers[i].id = i
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # layer.id = i
            prevLayerIds = []
            if layer.prevLayers != None:
                for prevLayer in layer.prevLayers:
                    prevLayerIds.append(prevLayer.id)
            nextLayerIds = []
            for l in layer.nextLayers:
                nextLayerIds.append(l.id)
            if slient == False:
                print("%3d %12s %20s %20s  %s" % (i, layer.name, str(prevLayerIds), str(nextLayerIds), str(layer.params)) )
    
    def computeInputDimensions(self, inputDim):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.inputDim = inputDim
            else:
                prevLayer = layer.prevLayers[0]
                if len(layer.prevLayers) == 1:
                    layer.inputDim = prevLayer.outputDim
                else:
                    if layer.name == "concat":
                        totalChannels = 0
                        for pl in layer.prevLayers:
                            totalChannels += pl.outputDim[0]
                            if prevLayer.outputDim[1] != pl.outputDim[1] or prevLayer.outputDim[2] != pl.outputDim[2]: # width and height must match.
                                print("prevLayer.outputDim: %15s, non-matching other input: %15s" % (prevLayer.outputDim, pl.outputDim))
                        layer.inputDim = (totalChannels, prevLayer.outputDim[1], prevLayer.outputDim[2])
                    if layer.name == "ReLU2d":
                        for pl in layer.prevLayers:
                            if prevLayer.outputDim != pl.outputDim: # this is only correct for additions in Resnet.
                                print("prevLayer.outputDim: %15s, non-matching other input: %15s" % (prevLayer.outputDim, pl.outputDim))
                        layer.inputDim = prevLayer.outputDim

            if layer.name in ["conv2d", "maxPool2d", "avgPool2d"]:
                paddingW = layer.params["padding"][0] if type(layer.params["padding"]) is tuple else layer.params["padding"]
                paddingH = layer.params["padding"][1] if type(layer.params["padding"]) is tuple else layer.params["padding"]
                strideW = layer.params["stride"][0] if type(layer.params["stride"]) is tuple else layer.params["stride"]
                strideH = layer.params["stride"][1] if type(layer.params["stride"]) is tuple else layer.params["stride"]
                kernelSizeW = layer.params["kernel_size"][0] if type(layer.params["kernel_size"]) is tuple else layer.params["kernel_size"]
                kernelSizeH = layer.params["kernel_size"][1] if type(layer.params["kernel_size"]) is tuple else layer.params["kernel_size"]
                
                outWidth = int((layer.inputDim[1] + paddingW * 2 - kernelSizeW) / strideW + 1)
                outHeight = int((layer.inputDim[2] + paddingH * 2 - kernelSizeH) / strideH + 1)
                
                if layer.name == "conv2d":
                    outChannel = layer.params["out_channels"]
                elif layer.name in ["maxPool2d", "avgPool2d"]:
                    outChannel = layer.inputDim[0]
                layer.outputDim = (outChannel, outWidth, outHeight)
            elif layer.name == "adAvgPool2d":
                layer.outputDim = (layer.inputDim[0], layer.params["output_width"], layer.params["output_height"])
            elif layer.name == "linear":
                layer.outputDim = (layer.params["out_features"])
            elif layer.name in ["ReLU2d", "ReLU1d", "ReLU"]:
                layer.outputDim = layer.inputDim
            elif layer.name == "flatten":
                layer.outputDim = int(numpy.prod(layer.inputDim))
            elif layer.name == "concat":
                layer.outputDim = layer.inputDim

            print("%3d %11s %20s %20s %s" % (i, layer.name, str(layer.inputDim), str(layer.outputDim), str(layer.params)) )
    
    def calcInputXfer(self, srcLayer: Layer, destLayer: Layer, srcConfig: tuple, destConfig: tuple, noGpuOverlap = False):
        namesIn2d = ["conv2d", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]
        namesIn1d = ["linear", "ReLU1d"]

        # Don't allow dynamic scaling from real layers to relu.
        reluMultiple = 1
        if destLayer.name in ["ReLU2d", "ReLU1d", "flatten"]:
            reluMultiple = 1000

        if srcLayer.name in namesIn2d and \
                destLayer.name in namesIn2d + ["flatten"]:
            actTime = self.calc2dActivationTime(srcLayer, destLayer, srcConfig, destConfig, noGpuOverlap)
            return (actTime[0] * reluMultiple, actTime[1])
            
        elif srcLayer.name in namesIn1d + ["flatten"] and \
                destLayer.name in namesIn1d:
            actTime = self.calcLinearActivationTime(srcLayer, destLayer, srcConfig, destConfig, noGpuOverlap)
            return (actTime[0] * reluMultiple, actTime[1])
        else:
            print("Can't compute input transfer time from %s to %s." % (srcLayer.name, destLayer.name))

    def calcSyncTime(self, layer, config, ctx):
        if layer.name in ["conv2d"]:
            syncTime = self.calcConv2dSyncTime(layer, config)
        elif layer.name in ["linear"]:
            syncTime = self.calcLinearSyncTime(config, ctx.globalBatch)
        else:
            syncTime = 0
        return syncTime
        
    def calcConv2dSyncTime(self, layer, config, bytesPerParam=4):
        filterCount = config[4]
        kernelSizeW = layer.params["kernel_size"][0] if type(layer.params["kernel_size"]) is tuple else layer.params["kernel_size"]
        kernelSizeH = layer.params["kernel_size"][1] if type(layer.params["kernel_size"]) is tuple else 1
        params = kernelSizeW * kernelSizeH * filterCount + kernelSizeW * kernelSizeH
        size = params * bytesPerParam
        return size / self.NET_BANDWIDTH # Returns microseconds.
        
    def calc2dActivationTime(self, srcLayer: Layer, destLayer: Layer, srcConfig: tuple, destConfig: tuple, noGpuOverlap = False):
        bytesPerParam = 4

        # Compute output dimension of previous and current layer.
        srcS = srcConfig[0]
        srcW = srcConfig[1] * srcLayer.outputDim[1] // srcLayer.inputDim[1] # Adjusts based on input/output ratio. 
        srcH = srcConfig[2] * srcLayer.outputDim[2] // srcLayer.inputDim[2] # It's necessary for pool or conv2d with stride > 1
        srcOutChannel = srcConfig[4] if len(srcConfig) >= 5 else srcConfig[3] # non-convolutional 2d layers don't have filter.
        destS = destConfig[0]
        destW = destConfig[1]
        destH = destConfig[2]
        # destInChannel = destConfig[3]
        destInChannel = srcOutChannel # TODO: temporary hack to handle Inception V3's concat.  
        
        srcNoIndependent = True
        destNoIndependent = True
        if hasattr(srcLayer, 'gpuAssignment') and hasattr(destLayer, 'gpuAssignment'):
            for r in srcLayer.gpuAssignment:
                if r not in destLayer.gpuAssignment:
                    srcNoIndependent = False
                    break
            for r in destLayer.gpuAssignment:
                if r not in srcLayer.gpuAssignment:
                    destNoIndependent = False
                    break
        else:
            # Compute "estimated" number of gpus used for src and dest. This is used only for comparing which side might unshared gpus.
            srcGpus = self.calcGpusNeeded(srcLayer, srcConfig, srcS * destS)
            destGpus = self.calcGpusNeeded(destLayer, destConfig, srcS * destS)
            if srcGpus > destGpus:
                srcNoIndependent = False # no src node is stopped used in dest.
            if srcGpus < destGpus:
                destNoIndependent = False # no dest node is newly used.
        

        # Compute common part that doesn't have to move.
        commonSize = bytesPerParam * min(srcS, destS) * min(srcW, destW) * min(srcH, destH) * min(srcOutChannel, destInChannel)
        if noGpuOverlap:
            commonSize = 0

        # Halo exchange
        if "kernel_size" in destLayer.params: # TODO: hack for adaptive avgPool2D.
            if type(destLayer.params["kernel_size"]) is tuple:
                haloW = int((destLayer.params["kernel_size"][0] - 1) / 2)
                haloH = int((destLayer.params["kernel_size"][1] - 1) / 2)
            else:
                haloW = int((destLayer.params["kernel_size"] - 1) / 2)
                haloH = int((destLayer.params["kernel_size"] - 1) / 2)
        else:
            haloW = 0
            haloH = 0
        haloPixels = 2 * haloW * ((destH + haloH) if destW != destLayer.inputDim[1] else 0)\
                     + 2 * haloH * ((destW + haloW) if destH != destLayer.inputDim[2] else 0)
        haloSize = bytesPerParam * min(srcS, destS) * haloPixels * min(srcOutChannel, destInChannel)

        # compute times
        egressBytes = bytesPerParam * srcS * srcW * srcH * srcOutChannel + (haloSize - commonSize) if srcNoIndependent else bytesPerParam * srcS * srcW * srcH * srcOutChannel + haloSize
        ingressBytes = bytesPerParam * destS * destW * destH * destInChannel + (haloSize - commonSize) if destNoIndependent else bytesPerParam * destS * destW * destH * destInChannel + haloSize
        activationTime = max(egressBytes, ingressBytes) / self.NET_BANDWIDTH
        activationTime += self.NET_LATENCY if activationTime > 0 else 0
        return (2 * activationTime, (egressBytes, ingressBytes, haloSize)) # double to count both forward and backward passes.

    def calcLinearSyncTime(self, config, globalBatch, bytesPerParam=4, alwaysPaySyncTime=False):
        if not alwaysPaySyncTime and config[0] == globalBatch: # No split.
            return 0
        inFeatures = config[1]
        outFeatures = config[2]
        params = inFeatures * outFeatures + outFeatures
        size = params * bytesPerParam
        return size / self.NET_BANDWIDTH # Returns microseconds.
        
    def calcLinearActivationTime(self, srcLayer: Layer, destLayer: Layer, srcConfig: tuple, destConfig: tuple, noGpuOverlap: bool):
        bytesPerParam = 4
        # Prepare variables.
        prevOutFeatures = 0
        if len(srcConfig) >= 4: # prev layer was conv2d.
            # print("%s to %s" % (srcLayer.name, destLayer.name))
            srcS = srcConfig[0]
            srcW = srcConfig[1] * 1 if srcLayer.name == "flatten" else (srcLayer.outputDim[1] // srcLayer.inputDim[1]) # Adjusts based on input/output ratio. 
            srcH = srcConfig[2] * 1 if srcLayer.name == "flatten" else (srcLayer.outputDim[2] // srcLayer.inputDim[2]) # It's necessary for pool or conv2d with stride > 1
            srcOutChannel = srcConfig[4] if len(srcConfig) >= 5 else srcConfig[3] # non-convolutional 2d layers don't have filter.
            srcOutFeatures = srcW * srcH * srcOutChannel
            splitFactor = 1
        elif len(srcConfig) == 3:
            srcS = srcConfig[0]
            srcOutFeatures = srcConfig[2]
            splitFactor = srcLayer.inputDim / srcConfig[1] # This much output must be added up to get the final output.
        else:
            print("[calcLinearActivationTime] error! srcConfig dimensions is not correct.")
        destS = destConfig[0]
        destInFeatures = destConfig[1]

        commonSize = bytesPerParam * min(srcS, destS) * min(srcOutFeatures, destInFeatures)
        if noGpuOverlap:
            commonSize = 0

        # compute times
        egressBytes = bytesPerParam * srcS * srcOutFeatures * splitFactor - commonSize
        ingressBytes = bytesPerParam * destS * destInFeatures * splitFactor - commonSize
        activationTime = max(egressBytes, ingressBytes) / self.NET_BANDWIDTH
        activationTime += self.NET_LATENCY if activationTime > 0 else 0
        # print("activationTime:%.1f, egressBytes:%d, ingressBytes:%d, splitFactor: %d, commonSize: %d" %\
        #     (activationTime, egressBytes, ingressBytes, splitFactor, commonSize))
        return (2 * activationTime, (egressBytes, ingressBytes, splitFactor)) # double to count both forward and backward passes.

    def Conv2d(self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            custom_previous_layers: list = None):
        module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = Layer(module, "conv2d",
                            {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": kernel_size, "stride": stride, "padding": padding},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)
        
        return module

    def MaxPool2d(self,
            kernel_size: _size_2_t,
            stride: _size_2_t,
            padding: _size_2_t = 0,
            # dilation: _size_2_t = 1,
            custom_previous_layers: list = None):
        module = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding) #, dilation=dilation)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = Layer(module, "maxPool2d",
                            {"kernel_size": kernel_size, "stride": stride, "padding": padding},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)

        return module

    def AdaptiveAvgPool2d(self,
            output_size,
            custom_previous_layers: list = None):
        module = nn.AdaptiveAvgPool2d(output_size)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        # stride = (input_size//output_size)  
        # kernel_size = input_size - (output_size-1)*stride  
        # padding = 0
        layer = Layer(module, "adAvgPool2d",
                            {"output_width": output_size[0], "output_height": output_size[1]},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)

        return module

    def AvgPool2d(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
            ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: bool = None,
            custom_previous_layers: list = None):
        module = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                            count_include_pad=count_include_pad, divisor_override=divisor_override)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = Layer(module, "avgPool2d",
                            {"kernel_size": kernel_size, "stride": stride, "padding": padding},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)

        return module

    def Linear(self, in_features: int, out_features: int, bias: bool = True, custom_previous_layers: list = None):
        module = nn.Linear(in_features, out_features, bias)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = Layer(module, "linear",
                            {"in_features": in_features, "out_features": out_features, "bias": bias},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)
        
        return module
    
    def ReLU(self, inplace: bool = False, custom_previous_layers: list = None):
        module = nn.ReLU(inplace=inplace)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        if custom_previous_layers[0].name in ["conv2d", "maxPool2d", "avgPool2d", "adAvgPool2d", "concat"]:
            name = "ReLU2d"
        elif custom_previous_layers[0].name in ["linear"]:
            name = "ReLU1d"
        else:
            name = "ReLU"
        layer = Layer(module, name, {"inplace": inplace, "kernel_size": 1, "stride": 1, "padding": 0}, prevLayers = custom_previous_layers)
        self.layers.append(layer)
        
        return module
    
    def Flatten(self, custom_previous_layers: list = None):
        module = nn.Flatten(start_dim=1)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = Layer(module, "flatten", {"kernel_size": 1}, prevLayers = custom_previous_layers)
        self.layers.append(layer)
        return
    
    class ConcatInputs(nn.Module):
        def __init__(self, dim: int = 1):
            super(CostSim.ConcatInputs, self).__init__()
            self.dim = dim
            
        def forward(self, inputList: List[torch.Tensor]):
            out = torch.cat(inputList, dim=self.dim)
            return out

    def Concat(self, custom_previous_layers: list = None): # concatenates tensors on channel dimension only.
        module = CostSim.ConcatInputs(dim=1)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = Layer(module, "concat", {"kernel_size": 1}, prevLayers = custom_previous_layers)
        self.layers.append(layer)
        return

    def getInitialConfig(self, layer, globalBatch: int):
        if layer.name in ["conv2d"]:
            initCfg = (globalBatch, layer.inputDim[1], layer.inputDim[2], layer.inputDim[0], layer.outputDim[2]) # (batch, width, height, channel, filter)
        elif layer.name in ["linear", "ReLU1d"]:
            initCfg = (globalBatch, layer.inputDim, layer.outputDim)
        elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
            initCfg = (globalBatch, layer.inputDim[1], layer.inputDim[2], layer.inputDim[0]) # (batch, width, height, channel, filter)
        return initCfg

    def listConfigOptions(self, layer, globalBatch: int, totalGpus: int, samplePo2=True, sampleSplit=True, spatialSplit=True, filterSplit=False, pruneHeuristics=False, dataParallelBaseline=False):
        initCfg = self.getInitialConfig(layer, globalBatch)
        totalSplits = int(math.log(totalGpus, 2))

        # generate config candidates.
        sampleSplitOptions = range(totalSplits + 1) if sampleSplit else [0]
        if dataParallelBaseline:
            sampleSplitOptions = [totalSplits]
        if layer.name in ["conv2d"]:
            configCandidates = [(int(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
                                for bs in sampleSplitOptions \
                                    for whs in (range(totalSplits - bs + 1) if spatialSplit else [0]) \
                                        for fs in (range(totalSplits - bs - whs + 1) if filterSplit else [0]) ]
        elif layer.name in ["linear", "ReLU1d"]:
            configCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**ins), int(initCfg[2] / 2**outs) )
                            for bs in sampleSplitOptions \
                                for ins in (range(totalSplits - bs + 1) if filterSplit else [0]) \
                                    for outs in (range(totalSplits - bs - ins + 1) if filterSplit else [0]) ]
        elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
            configCandidates = [(int(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
                                for bs in sampleSplitOptions \
                                    for whs in (range(totalSplits - bs + 1) if spatialSplit else [0]) ]

        # if layer.name in ["conv2d"]:
        #     configCandidates = [(math.ceil(initCfg[0] / replicas), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
        #                         for whs in (range(totalSplits + 1) if spatialSplit else [0]) \
        #                             for fs in (range(totalSplits - whs + 1) if filterSplit else [0]) \
        #                                 for replicas in (([2**ss for ss in range(totalSplits - whs - fs + 1)] if samplePo2 else range(1, 2**(totalSplits - whs - fs) + 1)) if sampleSplit else [1]) ]
        # elif layer.name in ["linear", "ReLU1d"]:
        #     configCandidates = [(math.ceil(initCfg[0] / replicas), math.ceil(initCfg[1] / 2**ins), math.ceil(initCfg[2] / 2**outs) )
        #                         for ins in (range(totalSplits + 1) if filterSplit else [0]) \
        #                             for outs in (range(totalSplits - ins + 1) if filterSplit else [0]) \
        #                                 for replicas in (([2**ss for ss in range(totalSplits - ins - outs + 1)] if samplePo2 else range(1, 2**(totalSplits - ins - outs) + 1)) if sampleSplit else [1]) ]
        #                                 # for replicas in (range(1, 2**(totalSplits - ins - outs) + 1) if sampleSplit else [1]) ]
        # elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
        #     configCandidates = [(math.ceil(initCfg[0] / replicas), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
        #                         for whs in (range(totalSplits + 1) if spatialSplit else [0]) \
        #                             for replicas in (([2**ss for ss in range(totalSplits - whs + 1)] if samplePo2 else range(1, 2**(totalSplits - whs) + 1)) if sampleSplit else [1]) ]
        
        validConfigs = []
        for config in configCandidates:
            invalidConfig = False
            for dim in range(len(config)):
                if config[dim] < 1:
                    invalidConfig = True
                    break
                # add some other rules..
            if not invalidConfig:
                validConfigs.append(config)

        if pruneHeuristics:
            bestGpuTimeByGpusUsed = {}
            prunedConfigs = []
            for config in validConfigs:
                gpuCount = self.calcGpusNeeded(layer, config, globalBatch)
                if gpuCount not in bestGpuTimeByGpusUsed:
                   bestGpuTimeByGpusUsed[gpuCount] = 999999999 
                gpuTime = self.benchGpuTime(layer, config)
                if bestGpuTimeByGpusUsed[gpuCount] > gpuTime:
                    bestGpuTimeByGpusUsed[gpuCount] = gpuTime
            for config in validConfigs:
                gpuCount = self.calcGpusNeeded(layer, config, globalBatch)
                gpuTime = self.benchGpuTime(layer, config)
                if gpuTime <= bestGpuTimeByGpusUsed[gpuCount] * 1.5:
                    prunedConfigs.append(config)
            
            validConfigs = prunedConfigs
        
        return validConfigs
    
    def displayConfigSizeGrowth(self, layer, maxNumGpus):
        options = [(True, True, True), (True, True, False), (True, False, False)]
        totalSplits = int(math.log(maxNumGpus, 2))
        globalBatch = maxNumGpus * 4
        for sampleSplit, spatialSplit, filterSplit in options:
            print("Gpus  Sample Spatial   filter  configListSize  prunedConfigs")
            for allowedSplits in range(totalSplits+1):
                totalGpus = 2 ** allowedSplits
                configs = self.listConfigOptions(layer, globalBatch, totalGpus, sampleSplit=sampleSplit, spatialSplit=spatialSplit, filterSplit=filterSplit)
                prunedConfigs = self.listConfigOptions(layer, globalBatch, totalGpus, sampleSplit=sampleSplit, spatialSplit=spatialSplit, filterSplit=filterSplit, pruneHeuristics=True)
                print( "%3d    %5s   %5s   %5s :   %11d  %11d" % (totalGpus, str(sampleSplit), str(spatialSplit), str(filterSplit), len(configs), len(prunedConfigs) ))

    def calcGpusNeeded(self, layer, config: tuple, globalBatch: int):
        initCfg = self.getInitialConfig(layer, globalBatch)
        gpuCount = 1
        # if len(config) != len(initCfg):
        #     print("[calcGpusNeeded] dimension of configs doesn't match!! %20s layer len(config):%d != len(initCfg):%d" % (layer.name, len(config), len(initCfg)))
        for i in range(len(initCfg)):
            gpuCount *= int(initCfg[i] / config[i])
        return gpuCount
    
    def isConfigDataParallelOnly(self, layer, config: tuple, globalBatch: int):
        initCfg = self.getInitialConfig(layer, globalBatch)
        dpOnly = True
        for i in range(1, len(config)):
            if config[i] != initCfg[i]:
                dpOnly = False
        return dpOnly

    def benchGpuTime(self, layer, config: tuple, profile=False):
        if layer.name in ["conv2d"]:
            gpuTime = self.profiler.runConv2dBench(config, layer.params, profile)
        elif layer.name in ["linear"]:
            gpuTime = self.profiler.runLinearBench(config, profile)
        else:
            gpuTime = 0
        return gpuTime

    def runMultiChainZhihao(self, startLayer, startConfig, globalBatch: int, totalGpus: int):
        k = len(startLayer.nextLayers)
        llist = [[startLayer] for j in range(k)]
        endLayer = None
        for j in range(k):
            l = startLayer.nextLayers[j]
            while len(l.prevLayers) == 1: # Until join happens.
                llist[j].append(l)
                if len(l.nextLayers) > 1:
                    print("[searchMultiChain] ERROR! nested multi-chain. TODO; implement handling of this.")
                l = l.nextLayers[0]
            if endLayer == None:
                endLayer = l
            else:
                assert(endLayer == l)

        print("Found %d chains, branching at %d-th layer, joining at %d-th layer" % (k, startLayer.id, endLayer.id))
        noParallelTimeSum = 0
        t = [[ [] for i in range(len(llist[j])) ] for j in range(k)] # [layer] = list of (config, cumulativeTime, prevConfigIndex)
        for branchIdx in range(k):
            length = len(llist[branchIdx])

            bestConfigList = []
            bestTimeList = []
            bestDataParallelTimeList = []
            for idx in range(length):
                layer = llist[branchIdx][idx]
                initCfg = self.getInitialConfig(layer, globalBatch)

                bestTime = self.benchGpuTime(layer, initCfg)
                bestDataParallelTime = bestTime
                noParallelTimeSum += bestTime
                bestConfig = initCfg                
                for config in (self.listConfigOptions(layer, globalBatch, totalGpus) if idx > 0 else [startConfig]):
                    gpuTime = self.benchGpuTime(layer, config)
                    
                    # Computer all-reduce time
                    # if layer.name in ["conv2d"]:
                    #     syncTime = self.calcConv2dSyncTime(config)
                    # elif layer.name in ["linear"]:
                    #     syncTime = self.calcLinearSyncTime(config, globalBatch)
                    # else:
                    #     syncTime = 0
                    syncTime = 0
                    
                    if idx == 0:
                        t[branchIdx][idx].append((config, syncTime, None, (0, 0, 0, syncTime, (0)) ))
                    else:
                        bestPrevCfgIdx = 0
                        bestCumulativeTime = 99999999999
                        bestTimeComposition = None

                        # WARNING!! Following main branch only!!
                        prevLayer = layer.prevLayers[0]
                        for prevCfgIdx in range(len(t[branchIdx][idx-1])):
                            prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[branchIdx][idx-1][prevCfgIdx]
                            activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevCfg, config)
                            newTime = cumulativeTime + activationTime + gpuTime + syncTime
                            if  newTime < bestCumulativeTime:
                                bestCumulativeTime = newTime
                                bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                                bestPrevCfgIdx = prevCfgIdx
                                
                        t[branchIdx][idx].append((config, bestCumulativeTime, bestPrevCfgIdx, bestTimeComposition ))

                    if gpuTime < bestTime:
                        bestTime = gpuTime
                        bestConfig = config
                
                bestConfigList.append(bestConfig)
                bestTimeList.append(bestTime)
                bestDataParallelTimeList.append(bestDataParallelTime)
        # join at the endLayer. Sum up all times of all branches.
        configToTimeDict = {}
        for endConfig in self.listConfigOptions(endLayer, globalBatch, totalGpus):
            gpuTime = self.benchGpuTime(endLayer, endConfig)
            bestTime = 9999999999999
            optionWithBestTime = []
            sumOfBestTime = 0
            for branchIdx in range(k):
                bestPrevCfgIdx = 0
                bestCumulativeTime = 99999999999
                bestTimeComposition = None
                prevLayer = llist[branchIdx][-1]
                for prevCfgIdx in range(len(t[branchIdx][-1])):
                    prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[branchIdx][-1][prevCfgIdx]
                    activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, endLayer, prevCfg, endConfig)
                    newTime = cumulativeTime + activationTime + gpuTime + syncTime
                    if  newTime < bestCumulativeTime:
                        bestCumulativeTime = newTime
                        bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                        bestPrevCfgIdx = prevCfgIdx
                sumOfBestTime += bestCumulativeTime
                optionWithBestTime.append(bestPrevCfgIdx)

            configToTimeDict[endConfig] = (sumOfBestTime, tuple(optionWithBestTime))

        return (endLayer, configToTimeDict, t)
    
    def searchMultiChain(self, startLayer, startConfig, globalBatch: int, totalGpus: int, verbose=True):
        maximumOptionListSize = 0
        k = len(startLayer.nextLayers)
        llist = [[startLayer] for j in range(k)]
        endLayer = None
        for j in range(k):
            l = startLayer.nextLayers[j]
            while len(l.prevLayers) == 1: # Until join happens.
                llist[j].append(l)
                if len(l.nextLayers) > 1:
                    print("[searchMultiChain] ERROR! nested multi-chain. TODO; implement handling of this.")
                l = l.nextLayers[0]
            if endLayer == None:
                endLayer = l
            else:
                assert(endLayer == l)

        print("Found %d chains, branching at %d-th layer, joining at %d-th layer" % (k, startLayer.id, endLayer.id))

        # Start dynamic programming.
        time_start = time.time()
        def generateAllConfigs(k: int, llist: list):
            if k == 0:
                return [[]]
            configs = []
            for laterPart in generateAllConfigs(k-1, llist[1:]):
                configs.append([(0, startConfig)] + laterPart)

            for nextIndex in range(1, len(llist[0])):
                for config in self.listConfigOptions(llist[0][nextIndex], globalBatch, totalGpus):
                    laterPartList = generateAllConfigs(k-1, llist[1:])
                    # print("[generateAllConfigs] for k=%d, (%d, %s), got %s" % (k, nextIndex, str(config), str(laterPartList)))
                    for laterPart in laterPartList:
                        completePart = [(nextIndex, config)] + laterPart
                        configs.append(completePart)
            return configs
        
        # allCombinedIdx = generateAllConfigs(k, llist)
        print("Total # of cell in table prefore config pruning: %d" % len(generateAllConfigs(k, llist)))
        # TODO: replace this with carefully pruned config matrix. 
        configOptionLists = []
        for j in range(k):
            prevTotalStateCombo = 1
            configOptionLists.append( [ [startConfig] ] )
            print("[Pruning configs] %d-th chain, before prune: [" % j, end="")
            for idx in range(1, len(llist[j])):
                configOptionLists[j].append(self.listConfigOptions(llist[j][idx], globalBatch, totalGpus))
                print(" %2d" % len(configOptionLists[j][idx]), end="")
                prevTotalStateCombo *= len(configOptionLists[j][idx])
            configOptionLists[j].append(self.listConfigOptions(endLayer, globalBatch, totalGpus))
            print(" ] (%d combos)" % prevTotalStateCombo, end="")
            
            for prunIter in range(3):
                print(" ==> after prune (pass %d): [" % (prunIter + 1), end="")
                totalStateCombo = 1
                for idx in range(1, len(llist[j])):
                    prevLayer = llist[j][idx-1]
                    layer = llist[j][idx]
                    nextLayer = llist[j][idx + 1] if (idx + 1 < len(llist[j])) else endLayer
                    selectedConfigs = []
                    for nextConfig in configOptionLists[j][idx + 1]:
                        for prevConfig in configOptionLists[j][idx - 1]:
                            bestConfigByGpuCount = {}
                            for config in configOptionLists[j][idx]:
                                gpusUsed = self.calcGpusNeeded(layer, config, globalBatch)
                                activationTime1, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevConfig, config)
                                activationTime2, activationSizeMatrix = self.calcInputXfer(layer, nextLayer, config, nextConfig)
                                gpuTime = self.benchGpuTime(layer, config)

                                totalTime = activationTime1 + activationTime2 + gpuTime
                                if gpusUsed not in bestConfigByGpuCount or \
                                        totalTime < bestConfigByGpuCount[gpusUsed][0]:
                                    bestConfigByGpuCount[gpusUsed] = (totalTime, config)
                            
                            gpuCountList = bestConfigByGpuCount.keys()
                            previousSelectedTime = 999999999
                            for gpuCount in sorted(gpuCountList):
                                if bestConfigByGpuCount[gpuCount][0] < previousSelectedTime * 0.9: # Don't use more GPUs unless it reduce time by 10% or more.
                                    previousSelectedTime = bestConfigByGpuCount[gpuCount][0]
                                    if bestConfigByGpuCount[gpuCount][1] not in selectedConfigs:
                                        selectedConfigs.append(bestConfigByGpuCount[gpuCount][1])
                    configOptionLists[j][idx] = selectedConfigs
                    print(" %2d" % len(configOptionLists[j][idx]), end="")
                    totalStateCombo *= len(configOptionLists[j][idx])
                print(" ] (%d combos)" % totalStateCombo, end="")

                # Stop prunning if it is converged.
                if totalStateCombo == prevTotalStateCombo:
                    break
                prevTotalStateCombo = totalStateCombo
            print("")
            configOptionLists[j].pop() # remove configs of endLayer

        def generatePrunedCombo(branchIdx: int, configOptionLists: list):
            if branchIdx == len(configOptionLists):
                return [[]]
            
            combos = []
            for idx in range(len(configOptionLists[branchIdx])):
                for config in configOptionLists[branchIdx][idx]:
                    laterPartList = generatePrunedCombo(branchIdx + 1, configOptionLists)
                    # print("[generateAllConfigs] for k=%d, (%d, %s), got %s" % (k, nextIndex, str(config), str(laterPartList)))
                    for laterPart in laterPartList:
                        completePart = [(idx, config)] + laterPart
                        combos.append(completePart)
            return combos
        
        allCombinedIdx = generatePrunedCombo(0, configOptionLists)

        initialIdx = tuple(allCombinedIdx[0])
        t = {}
        # t[initialIdx] = [(numpy.zeros(k, dtype=numpy.int32), numpy.zeros(totalGpus, dtype=numpy.int32), (0, startConfig))]
        t[initialIdx] = [([0 for j in range(k)], [0 for j in range(totalGpus)], (-1, startConfig, 0))]
        # t[initialIdx] = [(array('i', [0 for j in range(k)]), [0 for j in range(totalGpus)], (0, startConfig))]
        optionsConsideredTotal = 0
        optionsAfterMinTotal = 0
        # configOptionLists = [[] for j in range(k)]
        # for j in range(k):
        #     configOptionLists[j].append([startConfig])
        #     for idx in range(1, len(llist[j])):
        #         configOptionLists[j].append(self.listConfigOptions(llist[j][idx], globalBatch, totalGpus))
        if verbose:
            print("Total # of cells in table: %d, configGeneration took: %d sec" % (len(allCombinedIdx), time.time() - time_start))
        for combinedIdxAndConfig in allCombinedIdx[1:]:
            # print(combinedIdx)
            combined = tuple(combinedIdxAndConfig)
            t[combined] = []

            prefilteredOptions = []
            for j in range(k):
                prevIdx = combinedIdxAndConfig[j][0] - 1
                if prevIdx < 0:
                    continue
                currentIdx = combinedIdxAndConfig[j][0]
                layer = llist[j][currentIdx]
                prevLayer = llist[j][prevIdx]
                currentConfig = combinedIdxAndConfig[j][1]
                gpuTime = int(self.benchGpuTime(layer, currentConfig))
                
                for prevConfig in configOptionLists[j][prevIdx]: #prevConfigList:
                    prevCombinedIdxAndConfig = combinedIdxAndConfig.copy()
                    prevCombinedIdxAndConfig[j] = (prevIdx, prevConfig)
                    prevCombined = tuple(prevCombinedIdxAndConfig)
                    
                    for optionIdx in range(len(t[prevCombined])):
                        prevTimeVec, prevGpuReady, prevStep = t[prevCombined][optionIdx]
                    # for (prevTimeVec, prevGpuReady, prevStep) in t[prevCombined]:
                        # compute time.
                        activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevConfig, currentConfig)
                        
                        # First, compute the earliest time it can finish j-th chain.
                        prevBranchReady = prevTimeVec[j]
                        gpusNeeded = self.calcGpusNeeded(layer, currentConfig, globalBatch)
                        newGpuReadyTimeVec = sorted(prevGpuReady, reverse=True)
                        prevGpuReadyTime = newGpuReadyTimeVec[totalGpus - gpusNeeded]
                        newStartTime = max(prevBranchReady, prevGpuReadyTime)
                        # newReadyTime = int(newStartTime + activationTime + gpuTime)
                        newReadyTime = newStartTime + int(activationTime) + gpuTime

                        newTimeVec = prevTimeVec.copy()
                        newTimeVec[j] = newReadyTime

                        gpusAssigned = 0
                        for i in range(totalGpus):
                            if newGpuReadyTimeVec[i] <= newStartTime:
                                newGpuReadyTimeVec[i] = newReadyTime
                                gpusAssigned += 1
                            if gpusAssigned >= gpusNeeded:
                                break
                        # # Experiment... bump always.
                        # bumpToTime = min(newGpuReadyTimeVec)
                        # bumpedTimeVec = [ max(timeElem, bumpToTime) for timeElem in newTimeVec ]
                        # prefilteredOptions.append((bumpedTimeVec, newGpuReadyTimeVec, (j, prevConfig, optionIdx) ))

                        prefilteredOptions.append((newTimeVec, newGpuReadyTimeVec, (j, prevConfig, optionIdx) ))
            
            optionsConsideredTotal += len(prefilteredOptions)

            # Filter by lamportMin.
            skipIdicesForEqual = []
            filteredIndices = []
            for optionIdx in range(len(prefilteredOptions)):
                if optionIdx in skipIdicesForEqual:
                    continue
                (newTimeVec, newGpuReadyTimeVec, step) = prefilteredOptions[optionIdx]
                bumpToTime = min(newGpuReadyTimeVec)
                bumpedTimeVec = [ max(timeElem, bumpToTime) for timeElem in newTimeVec ]

                # check if there's any other result that's clearly better than this.
                foundBetterOption = False
                for compareAgainstIdx in range(len(prefilteredOptions)):
                    if (optionIdx == compareAgainstIdx) or (compareAgainstIdx in filteredIndices):
                        continue
                    (cmpTimeVec, cmpGpuReadyTimeVec, cmpStep) = prefilteredOptions[compareAgainstIdx]
                    better = True
                    equal = True
                    for i in range(k):
                        if bumpedTimeVec[i] < cmpTimeVec[i]:
                            better = False
                        if bumpedTimeVec[i] != cmpTimeVec[i]:
                            equal = False
                    if better and (not equal):
                        foundBetterOption = True
                    if equal:
                        skipIdicesForEqual.append(compareAgainstIdx)
                if foundBetterOption:
                    filteredIndices.append(optionIdx)
                else:
                    t[combined].append(prefilteredOptions[optionIdx])
            # print("[MultiChain] t[%50s] (%3d->%3d options)= %s" %
            #         (str(combined), len(prefilteredOptions), len(t[combined]), str(t[combined])))
            if len(t[combined]) == 0:
                print("  ******  prefilteredOptions: %s" % str(prefilteredOptions))
            maximumOptionListSize = max(maximumOptionListSize, len(t[combined]))
            # optionListSizeList.append(len(t[combined]))
            optionsAfterMinTotal += len(t[combined])


        # TODO: Final join to end layer.
        configToTimeDict = {}
        
        def generatePrunedCombo(branchIdx: int, configOptionLists: list):
            if branchIdx == len(configOptionLists):
                return [[]]
            
            combos = []
            for idx in range(len(configOptionLists[branchIdx])):
                for config in configOptionLists[branchIdx][idx]:
                    laterPartList = generatePrunedCombo(branchIdx + 1, configOptionLists)
                    # print("[generateAllConfigs] for k=%d, (%d, %s), got %s" % (k, nextIndex, str(config), str(laterPartList)))
                    for laterPart in laterPartList:
                        completePart = [(idx, config)] + laterPart
                        combos.append(completePart)
            return combos

        # def generateFinalConfigs(k: int, llist: list):
        #     if k == 0:
        #         return [[]]
        #     configs = []
        #     for config in self.listConfigOptions(llist[0][-1], globalBatch, totalGpus):
        #         laterPartList = generateFinalConfigs(k-1, llist[1:])
        #         for laterPart in laterPartList:
        #             completePart = [(len(llist[0])-1, config)] + laterPart
        #             configs.append(completePart)
        #     return configs
        def generateFinalPrunedCombos(branchIdx: int, configOptionLists: list):
            if branchIdx == len(configOptionLists):
                return [[]]
            combos = []
            for config in configOptionLists[branchIdx][-1]:
                laterPartList = generateFinalPrunedCombos(branchIdx + 1, configOptionLists)
                for laterPart in laterPartList:
                    completePart = [(len(configOptionLists[branchIdx])-1, config)] + laterPart
                    combos.append(completePart)
            return combos

        # finalCombinedList = generateFinalConfigs(k, llist)
        finalCombinedList = generateFinalPrunedCombos(0, configOptionLists)
        for endConfig in self.listConfigOptions(endLayer, globalBatch, totalGpus):
            gpuTime = self.benchGpuTime(endLayer, endConfig)

            bestTime = 9999999999999
            optionWithBestTime = None
            for combinedIdxAndConfig in finalCombinedList:
                # activation time. 
                activationTimeSum = 0
                activationTimeList = []
                for j in range(k):
                    prevConfig = combinedIdxAndConfig[j][1]
                    activationTime, temp = self.calcInputXfer(llist[j][-1], endLayer, prevConfig, endConfig)
                    activationTimeSum += activationTime
                    activationTimeList.append(activationTime)

                # traverse all options..
                for optionIdx in range(len(t[tuple(combinedIdxAndConfig)])):
                    timeVec, gpuReady, prevStep = t[tuple(combinedIdxAndConfig)][optionIdx]
                # for timeVec, gpuReady, prevStep in t[tuple(combinedIdxAndConfig)]:
                    # First, compute the earliest time it can finish j-th chain.
                    endTime = max(timeVec) + activationTimeSum + gpuTime
                    if endTime < bestTime:
                        bestTime = endTime
                        optionWithBestTime = (tuple(combinedIdxAndConfig), optionIdx)
            configToTimeDict[endConfig] = (bestTime, optionWithBestTime)

        if verbose:
            avgOptionListSize = optionsAfterMinTotal / len(allCombinedIdx)
            elapsedTime = time.time() - time_start
            print("[searchMultiChain] took: %d sec (%.3f ms per cell)" % (elapsedTime, 1000 * elapsedTime / len(allCombinedIdx)))
            print("[searchMultiChain] maximumOptionListSize: %d, avg: %.2f, pre-lamportMin: %.2f" % (maximumOptionListSize, avgOptionListSize, optionsConsideredTotal / len(allCombinedIdx)))
        return (endLayer, configToTimeDict, t)

    def displayMultiChainResult(self, endLayer, endConfig: tuple, t, bestJoiningOption):
        bestJoiningCombinedIdxAndConfig, optionIdx = bestJoiningOption
        combined = bestJoiningCombinedIdxAndConfig
        schedule = [] # list of (branch, idx, config, endTime)
        while True:
            newTimeVec, newGpuReadyTimeVec, (j, prevConfig, prevOptionIdx) = t[combined][optionIdx]
            if j == -1: # reached to the branching point.
                break

            schedule.append( (j, combined[j][0], combined[j][1], newTimeVec[j]) )

            nextCombined = list(combined)
            nextCombined[j] = (combined[j][0] - 1, prevConfig)
            combined = tuple(nextCombined)
            optionIdx = prevOptionIdx
        
        for i in range(len(schedule)-1, -1, -1):
            (branch, idx, config, endTime) = schedule[i]
            print("%sLayer(%d, %2d) config: %15s done at %d" % (" "*55*branch, branch, idx, config, endTime))
            
    def searchBestSplits(self, totalGpus: int, globalBatch: int = 16, amplificationLimit: float = 2.0, dataParallelBaseline = False, spatialSplit=False):
        t = [[] for i in range(len(self.layers))] # [layer] = list of (config, cumulativeTime, prevConfigIndex)

        initialConfigs = []
        initialTimes = []
        bestConfigList = []
        bestTimeList = []
        bestDataParallelTimeList = []
        for i in range(len(self.layers)):
            layer = self.layers[i]

            initCfg = self.getInitialConfig(layer, globalBatch)
            initialConfigs.append(initCfg)

            noParallelTime = self.benchGpuTime(layer, initCfg)
            bestTime = noParallelTime
            bestDataParallelTime = bestTime
            initialTimes.append(bestTime)
            bestConfig = initCfg
            
            # generate config candidates.
            totalSplits = int(math.log(totalGpus, 2))
            sampleSplit=True
            # spatialSplit=False # moved to func argument.
            filterSplit=False
            # filterSplit=False
            sampleSplitOptions = range(totalSplits + 1) if sampleSplit else [0]
            if dataParallelBaseline:
                sampleSplitOptions = [totalSplits]
            if layer.name in ["conv2d"]:
                configCandidates = [(int(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
                                    for bs in sampleSplitOptions \
                                        for whs in (range(totalSplits - bs + 1) if spatialSplit else [0]) \
                                            for fs in (range(totalSplits - bs - whs + 1) if filterSplit else [0]) ]
                dpConfigCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**int(whs/2)), int(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], int(initCfg[4] / 2**fs) )
                                    for bs in range(totalSplits + 1) for whs in [0] for fs in [0]]
            elif layer.name in ["linear", "ReLU1d"]:
                configCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**ins), int(initCfg[2] / 2**outs) )
                                for bs in sampleSplitOptions \
                                    for ins in (range(totalSplits - bs + 1) if filterSplit else [0]) \
                                        for outs in (range(totalSplits - bs - ins + 1) if filterSplit else [0]) ]
                dpConfigCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**ins), int(initCfg[2] / 2**outs) )
                                    for bs in range(totalSplits + 1) for ins in [0] for outs in [0] ]
            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                configCandidates = [(int(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
                                    for bs in sampleSplitOptions \
                                        for whs in (range(totalSplits - bs + 1) if spatialSplit else [0]) ]
                dpConfigCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**int(whs/2)), int(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
                                    for bs in range(totalSplits + 1) for whs in [0] ]

            for config in configCandidates:
                # Check validity of config.
                invalidConfig = False
                for dim in range(len(config)):
                    if config[dim] < 1:
                        invalidConfig = True
                        break
                    # add some other rules..
                if invalidConfig:
                    continue
                
                # Benchmark GPU time
                gpuTime = self.benchGpuTime(layer, config)
                
                # Computer all-reduce time
                if layer.name in ["conv2d"]:
                    syncTime = self.calcConv2dSyncTime(layer, config)
                elif layer.name in ["linear"]:
                    syncTime = self.calcLinearSyncTime(config, globalBatch)
                else:
                    syncTime = 0
                
                if i == 0:
                    t[i].append((config, gpuTime + syncTime, None, (0, 0, gpuTime, syncTime, noParallelTime, (0)) ))
                else:
                    bestPrevCfgIdx = 0
                    bestCumulativeTime = 99999999999
                    bestTimeComposition = None
                    bestAmplification = 999999999

                    # WARNING!! Following main branch only!!
                    prevLayer = layer.prevLayers[0]
                    for prevCfgIdx in range(len(t[prevLayer.id])):
                        prevCfg, cumulativeTime, prevConfigIndexOfPrev, prevtimeComposition = t[prevLayer.id][prevCfgIdx]
                        activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevCfg, config)
                        if (layer.name in ["flatten"] or prevLayer.name in ["flatten"]) or (self.verbose and i < 0): #5:
                            print(" %2d " % i, end="")
                            if prevLayer.name in ["conv2d"]:
                                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) => " % (prevLayer.name, *prevCfg), end="")
                            elif prevLayer.name in ["linear", "ReLU1d"]:
                                print("%9s (b=%2d, in=%6d, out=%6d)        => " % (prevLayer.name, *prevCfg), end="")
                            elif prevLayer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d)         => " % (prevLayer.name, *prevCfg), end="")
                            
                            if layer.name in ["conv2d"]:
                                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) " % (layer.name, *config), end="")
                            elif layer.name in ["linear", "ReLU1d"]:
                                print("%9s (b=%2d, in=%6d, out=%6d)        " % (layer.name, *config), end="")
                            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d) " % (layer.name, *config), end="")

                            print("  gpusUsed:%2d->%2d  gpuTime: %6.f   activationTime: %6.f   syncTime: %6.f " % \
                                (self.calcGpusNeeded(prevLayer, prevCfg, globalBatch),
                                self.calcGpusNeeded(layer, config, globalBatch),
                                gpuTime, activationTime, syncTime))
                        
                        gpusUsed = self.calcGpusNeeded(prevLayer, prevCfg, globalBatch)
                        if prevtimeComposition == None:
                            print(t[prevLayer.id][prevCfgIdx])
                        (prevCumulativeTime, prevActivationTime, prevGpuTime, prevSyncTime, prevNoParallelTime, prevActivationSizeMatrix) = prevtimeComposition
                        prevLayerTime = prevGpuTime + prevSyncTime + activationTime
                        amplification = ((prevLayerTime * gpusUsed) / (prevNoParallelTime + prevSyncTime)) if (prevLayerTime > 0 and prevNoParallelTime > 300) else 1
                        layerTime = activationTime + gpuTime + syncTime

                        if cumulativeTime + layerTime < bestCumulativeTime and \
                                (amplification < amplificationLimit or amplification < bestAmplification): # consider if this is within amplification limit or minimum among so far observed.
                            bestCumulativeTime = cumulativeTime + layerTime
                            bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, noParallelTime, activationSizeMatrix)
                            bestAmplification = amplification # although it might not be the best so far.. it should be still within amplification limit.
                            bestPrevCfgIdx = prevCfgIdx
                            
                    t[i].append((config, bestCumulativeTime, bestPrevCfgIdx, bestTimeComposition ))

                if gpuTime < bestTime:
                    bestTime = gpuTime
                    bestConfig = config
                if config in dpConfigCandidates and gpuTime < bestDataParallelTime:
                    # print("bestDataParallelTime updated! config: %s, dpConfigs: %s" % (config, dpConfigCandidates))
                    # print("    allConfigList: %s"%configCandidates)
                    bestDataParallelTime = gpuTime
            
            bestConfigList.append(bestConfig)
            bestTimeList.append(bestTime)
            bestDataParallelTimeList.append(bestDataParallelTime)
            
            # if len(layer.nextLayers) == 1:
            #     print("sequencial transition.")
            # elif len(layer.nextLayers) > 1:
            #     for config in configCandidates:
            #         self.searchMultiChain(layer, config, globalBatch, totalGpus)

        print("Network bandwidth: %5.f Gbps" % (self.NET_BANDWIDTH * 8 / 1000))
        print("Best GPU-only time: %6.1f" % (sum(bestTimeList)))
        # print("Total with maxpool + linear layers: %6.1f" % (sum(bestTimeList) + 425 + 1100))
        
        bestDpTime = 99999999999
        cfgIdx = 0
        for idx in range(len(t[len(t)-1])):
            prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][idx]
            if cumulativeTime < bestDpTime:
                bestDpTime = cumulativeTime
                cfgIdx = prevConfigIndexOfPrev
        bestConfigChain = [None for i in range(len(t))]
        print("Best DP time: %6.f"%bestDpTime)
        # Just print gpu scaling.
        # for i in range(len(t)):
        #     print(" %2d " % i, end="")
        #     layer = self.layers[i]
        #     if layer.name in ["conv2d"]:
        #         print("%9s (b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) => " % (layer.name, *initialConfigs[i]), end="")
        #     elif layer.name in ["linear", "ReLU1d"]:
        #         print("%9s (b=%2d, in=%6d, out=%6d)        => " % (layer.name, *initialConfigs[i]), end="")
        #     elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
        #         print("%9s (b=%2d, w=%3d, h=%3d, c=%4d)         => " % (layer.name, *initialConfigs[i]), end="")
            
        #     gpuTimeScalingDP = (initialTimes[i] / bestDataParallelTimeList[i]) if bestDataParallelTimeList[i] > 0 else 0
        #     gpuTimeScalingMP = (initialTimes[i] / bestTimeList[i]) if bestTimeList[i] > 0 else 0
        #     print("      %6.f    %6.f  %6.f (DP:%4.1fx, MP:%4.1fx)"
        #             % ( bestTimeList[i], bestDataParallelTimeList[i], initialTimes[i], gpuTimeScalingDP, gpuTimeScalingMP))


        for i in range(len(t) - 1, -1, -1):
            bestConfigChain[i] = cfgIdx
            # print("for layer%2d, cfgIdx: %2d" % (i, cfgIdx))
            config, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][cfgIdx]
            cfgIdx = prevConfigIndexOfPrev

        print("Layer    type       initial configuration          => after split configuration            #GPUs time(us) |   prev inptXfer  gpuTime syncTime bestGpuTime dpGpuTime noParallelTime")
        gpuUsecSum = 0
        finalTime = 0
        for i in range(len(bestConfigChain)):
            config, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][bestConfigChain[i]]
            print(" %2d " % i, end="")
            layer = self.layers[i]
            if layer.name in ["conv2d"]:
                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) " % config, end="")
            elif layer.name in ["linear", "ReLU1d"]:
                print("%9s (b=%2d, in=%6d, out=%6d)        => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, in=%6d, out=%6d)        " % config, end="")
            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d)         => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d)         " % config, end="")
            
            gpusUsed = self.calcGpusNeeded(layer, config, globalBatch)
            gpuUsec = gpusUsed * (cumulativeTime - timeComposition[0])
            gpuUsecSum += gpuUsec
            gpuTimeScaling = (initialTimes[i] / timeComposition[2]) if timeComposition[2] > 0 else 0
            gpuTimeScalingDP = (initialTimes[i] / bestDataParallelTimeList[i]) if bestDataParallelTimeList[i] > 0 else 0
            gpuTimeScalingMP = (initialTimes[i] / bestTimeList[i]) if bestTimeList[i] > 0 else 0
            print(" %4d   %6.f   %6.f   %6.f   %6.f   %6.f      %6.f    %6.f  %6.f (%4.1fx) (DP:%4.1fx, MP:%4.1fx)  %6.f  %10s"
                    % (gpusUsed, cumulativeTime, timeComposition[0], timeComposition[1], timeComposition[2], timeComposition[3], bestTimeList[i], bestDataParallelTimeList[i], initialTimes[i], gpuTimeScaling, gpuTimeScalingDP, gpuTimeScalingMP, gpuUsec, str(timeComposition[4])))
            finalTime = max(cumulativeTime, finalTime)
        
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("  Sum                                                                                      ", end="")
        print("        %6.f   %6.f   %6.f   %6.f   %6.f      %6.f    %6.f  %6.f (%4.1fx)                       %6.f"
                % (t[len(bestConfigChain)-1][bestConfigChain[len(bestConfigChain)-1]][1],
                    0,
                    sum(t[i][bestConfigChain[i]][3][1] for i in range(len(bestConfigChain))),
                    sum(t[i][bestConfigChain[i]][3][2] for i in range(len(bestConfigChain))),
                    sum(t[i][bestConfigChain[i]][3][3] for i in range(len(bestConfigChain))),
                    sum(bestTimeList),
                    sum(bestDataParallelTimeList),
                    sum(initialTimes),
                    sum(initialTimes) / sum(t[i][bestConfigChain[i]][3][2] for i in range(len(bestConfigChain))),
                    gpuUsecSum
                    ))
        # print ()
        moduleDesc = self.generateModuleDescription([t[i][bestConfigChain[i]][0] for i in range(len(bestConfigChain))], globalBatch)
        return (moduleDesc, finalTime / 1000., gpuUsecSum / 1000.)

    def searchBestSplitsV2(self, totalGpus: int, globalBatch: int = 16, useZhihaoAlgo = False):
        t = [[] for i in range(len(self.layers))] # [layer] = list of (config, cumulativeTime, prevConfigIndex)

        initialConfigs = []
        initialTimes = []
        bestConfigList = []
        bestTimeList = []
        bestDataParallelTimeList = []
        # for i in range(len(self.layers)):
        #     layer = self.layers[i]
        layer = self.layers[0]
        while True:
            i = layer.id
            
            initCfg = self.getInitialConfig(layer, globalBatch)
            initialConfigs.append(initCfg)

            bestTime = self.benchGpuTime(layer, initCfg)
            bestDataParallelTime = bestTime
            initialTimes.append(bestTime)
            bestConfig = initCfg
            
            for config in self.listConfigOptions(layer, globalBatch, totalGpus):
                # Benchmark GPU time
                gpuTime = self.benchGpuTime(layer, config)
                
                # Computer all-reduce time
                if layer.name in ["conv2d"]:
                    syncTime = self.calcConv2dSyncTime(layer, config)
                elif layer.name in ["linear"]:
                    syncTime = self.calcLinearSyncTime(config, globalBatch)
                else:
                    syncTime = 0
                
                if i == 0:
                    t[i].append((config, gpuTime + syncTime, None, (0, 0, gpuTime, syncTime, (0)) ))
                else:
                    bestPrevCfgIdx = 0
                    bestCumulativeTime = 99999999999
                    bestTimeComposition = None

                    # WARNING!! Following main branch only!!
                    prevLayer = layer.prevLayers[0]
                    for prevCfgIdx in range(len(t[prevLayer.id])):
                        prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[prevLayer.id][prevCfgIdx]
                        activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevCfg, config)
                        # if self.verbose and i < 5:
                        #     print(" %2d " % i, end="")
                        #     if layer.name in ["conv2d"]:
                        #         print("%9s  => " % (layer.name), end="")
                        #         print("(b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) " % config, end="")
                        #     elif layer.name in ["linear", "ReLU1d"]:
                        #         print("%9s (b=%2d, in=%6d, out=%6d)        => " % (layer.name, *initialConfigs[i]), end="")
                        #         print("(b=%2d, in=%6d, out=%6d)        " % config, end="")
                        #     elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                        #         print("%9s (b=%2d, w=%3d, h=%3d, c=%4d)         => " % (layer.name, *initialConfigs[i]), end="")
                        #         print("(b=%2d, w=%3d, h=%3d, c=%4d)         " % config, end="")
                        #     print("  gpusUsed:%2d->%2d  %6.f   %6.f   %6.f " % \
                        #         (self.calcGpusNeeded(prevLayer, prevCfg, globalBatch),
                        #         self.calcGpusNeeded(layer, config, globalBatch),
                        #         gpuTime, activationTime, syncTime))

                        if cumulativeTime + activationTime + gpuTime + syncTime < bestCumulativeTime:
                            bestCumulativeTime = cumulativeTime + activationTime + gpuTime + syncTime
                            bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                            bestPrevCfgIdx = prevCfgIdx
                            
                    t[i].append((config, bestCumulativeTime, bestPrevCfgIdx, bestTimeComposition ))

                if gpuTime < bestTime:
                    bestTime = gpuTime
                    bestConfig = config
                if self.isConfigDataParallelOnly(layer, config, globalBatch) and gpuTime < bestDataParallelTime:
                    bestDataParallelTime = gpuTime
            
            bestConfigList.append(bestConfig)
            bestTimeList.append(bestTime)
            bestDataParallelTimeList.append(bestDataParallelTime)
            
            if len(layer.nextLayers) == 1:
                print("sequencial transition.")
                layer = layer.nextLayers[0]
            elif len(layer.nextLayers) == 0:
                print("Search completed.")
                break
            elif len(layer.nextLayers) > 1:
                while len(layer.nextLayers) > 1:
                    bestTimeByConfig = {}
                    print("Branching out at %d-th layer" % layer.id)
                    for startCfgIdx in range(len(t[layer.id])):
                        startCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[layer.id][startCfgIdx]
                        if useZhihaoAlgo:
                            (endLayer, configToTimeDict, multiChainT) = self.runMultiChainZhihao(layer, startCfg, globalBatch, totalGpus)
                        else:
                            (endLayer, configToTimeDict, multiChainT) = self.searchMultiChain(layer, startCfg, globalBatch, totalGpus)

                        for endConfig in configToTimeDict:
                            newTime = configToTimeDict[endConfig][0] + cumulativeTime
                            if endConfig not in bestTimeByConfig or \
                                    bestTimeByConfig[endConfig][1] > newTime:
                                bestTimeByConfig[endConfig] = (endConfig, newTime, startCfgIdx, (0, 0, 0, 0, (0)) )

                    for endConfig in bestTimeByConfig:
                        t[endLayer.id].append(bestTimeByConfig[endConfig])
                    layer = endLayer
                layer = layer.nextLayers[0]

        print("Network bandwidth: %5.f Gbps" % (self.NET_BANDWIDTH * 8 / 1000))
        print("Best GPU-only time: %6.1f" % (sum(bestTimeList)))
        # print("Total with maxpool + linear layers: %6.1f" % (sum(bestTimeList) + 425 + 1100))
        
        bestDpTime = 99999999999
        cfgIdx = 0
        for idx in range(len(t[-1])):
            prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][idx]
            if cumulativeTime < bestDpTime:
                bestDpTime = cumulativeTime
                cfgIdx = prevConfigIndexOfPrev
        bestConfigChain = [None for i in range(len(t))]
        print("Best DP time: %6.f"%bestDpTime)
        return

        for i in range(len(t) - 1, -1, -1):
            bestConfigChain[i] = cfgIdx
            # print("for layer%2d, cfgIdx: %2d" % (i, cfgIdx))
            config, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][cfgIdx]
            cfgIdx = prevConfigIndexOfPrev

        print("Layer    type       initial configuration          => after split configuration            time (us) |   prev inptXfer  gpuTime syncTime bestGpuTime dpGpuTime noParallelTime")
        for i in range(len(bestConfigChain)):
            config, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][bestConfigChain[i]]
            print(" %2d " % i, end="")
            layer = self.layers[i]
            if layer.name in ["conv2d"]:
                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) " % config, end="")
            elif layer.name in ["linear", "ReLU1d"]:
                print("%9s (b=%2d, in=%6d, out=%6d)        => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, in=%6d, out=%6d)        " % config, end="")
            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d)         => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d)         " % config, end="")
            
            gpuTimeScaling = (initialTimes[i] / timeComposition[2]) if timeComposition[2] > 0 else 0
            print("   %6.f   %6.f   %6.f   %6.f   %6.f      %6.f    %6.f  %6.f (%4.1fx)  %10s"
                    % (cumulativeTime, timeComposition[0], timeComposition[1], timeComposition[2], timeComposition[3], bestTimeList[i], bestDataParallelTimeList[i], initialTimes[i], gpuTimeScaling, str(timeComposition[4])))
        
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("  Sum                                                                                      ", end="")
        print("   %6.f   %6.f   %6.f   %6.f   %6.f      %6.f    %6.f  %6.f (%4.1fx)"
                % (t[len(bestConfigChain)-1][bestConfigChain[len(bestConfigChain)-1]][1],
                    0,
                    sum(t[i][bestConfigChain[i]][3][1] for i in range(len(bestConfigChain))),
                    sum(t[i][bestConfigChain[i]][3][2] for i in range(len(bestConfigChain))),
                    sum(t[i][bestConfigChain[i]][3][3] for i in range(len(bestConfigChain))),
                    sum(bestTimeList),
                    sum(bestDataParallelTimeList),
                    sum(initialTimes),
                    sum(initialTimes) / sum(t[i][bestConfigChain[i]][3][2] for i in range(len(bestConfigChain)))
                    ))


    def searchLinear(self, preStartLayer, preStartConfig, startLayer, ctx: SearchContext):
        layer = startLayer
        while True:
            layer.t = {}

            initCfg = self.getInitialConfig(layer, ctx.globalBatch)
            layer.initCfg = initCfg
            layer.noParallelTime = self.benchGpuTime(layer, initCfg) + self.calcSyncTime(layer, initCfg, ctx)
            
            configCandidates = self.listConfigOptions(layer, ctx.globalBatch, ctx.totalGpus, sampleSplit=ctx.sampleSplit, spatialSplit=ctx.spatialSplit, filterSplit=ctx.filterSplit, dataParallelBaseline=ctx.dataParallelBaseline)
            if self.verbose and layer.id < 3:
                print(" %2d  configCandidates: %s" % (layer.id, str(configCandidates)))
            
            for config in configCandidates:
                # Benchmark GPU time and all-reduce time
                gpuTime = self.benchGpuTime(layer, config)
                syncTime = self.calcSyncTime(layer, config, ctx)
                
                if layer == startLayer:
                    if preStartLayer != None:
                        cumulativeTime, prevLayerTime, prevConfigOfPrev, timeComposition, prevMpIdleTime = preStartLayer.t[preStartConfig]
                        activationTime, activationSizeMatrix = self.calcInputXfer(preStartLayer, layer, preStartConfig, config)
                    else:
                        cumulativeTime = 0
                        activationTime = 0
                        activationSizeMatrix = (0)
                    
                    timeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                    layerTime = activationTime + gpuTime + syncTime
                    newTime = cumulativeTime + layerTime #activationTime + gpuTime + syncTime

                    layer.t[config] = (newTime, layerTime, preStartConfig, timeComposition, 0)
                else:
                    prevLayer = layer.prevLayers[0]
                    bestPrevCfg = None
                    bestCumulativeTime = 99999999999
                    bestLayerTime = None
                    bestTimeComposition = None
                    bestAmplification = 999999

                    for prevCfg in prevLayer.t:
                        cumulativeTime, prevLayerTime, prevConfigOfPrev, timeComposition, prevMpIdleTime = prevLayer.t[prevCfg]
                        activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevCfg, config)
                        
                        # if self.verbose and layer.id < 150:
                        #     print(" %2d " % layer.id, end="")
                        #     if layer.name in ["conv2d"]:
                        #         print("%9s  => " % (layer.name), end="")
                        #         print("(b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) " % config, end="")
                        #     elif layer.name in ["linear", "ReLU1d"]:
                        #         print("%9s  => " % (layer.name), end="")
                        #         print("(b=%2d, in=%6d, out=%6d)        " % config, end="")
                        #     elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                        #         print("%9s  => " % (layer.name), end="")
                        #         print("(b=%2d, w=%3d, h=%3d, c=%4d)         " % config, end="")
                        #     print("  gpusUsed:%2d->%2d  %6.f   %6.f   %6.f " % \
                        #         (self.calcGpusNeeded(prevLayer, prevCfg, ctx.globalBatch),
                        #         self.calcGpusNeeded(layer, config, ctx.globalBatch),
                        #         gpuTime, activationTime, syncTime))
                        
                        gpusUsed = self.calcGpusNeeded(prevLayer, prevCfg, ctx.globalBatch)
                        # if not hasattr(prevLayer, 'noParallelTime'):
                        #     print("no prevLayer.noParallelTime. layerId:%d, name:%s" % (prevLayer.id, prevLayer.name))
                        amplification = ((prevLayerTime * gpusUsed) / (prevLayer.noParallelTime)) if (prevLayerTime > 0 and prevLayer.noParallelTime > 300) else 1
                        layerTime = activationTime + gpuTime + syncTime

                        if cumulativeTime + layerTime < bestCumulativeTime and \
                                (amplification < ctx.amplificationLimit or amplification < bestAmplification): # consider if this is within amplification limit or minimum among so far observed.
                            bestCumulativeTime = cumulativeTime + layerTime
                            bestLayerTime = layerTime
                            bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                            bestAmplification = amplification # although it might not be the best so far.. it should be still within amplification limit.
                            bestPrevCfg = prevCfg

                    layer.t[config] = (bestCumulativeTime, bestLayerTime, bestPrevCfg, bestTimeComposition, 0) # mpIdleTime is 0 for sequencial scan.

                    #     if cumulativeTime + activationTime + gpuTime + syncTime < bestCumulativeTime:
                    #         bestCumulativeTime = cumulativeTime + activationTime + gpuTime + syncTime
                    #         bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                    #         bestPrevCfg = prevCfg

                    # layer.t[config] = (bestCumulativeTime, bestPrevCfg, bestTimeComposition, 0) # mpIdleTime is 0 for sequencial scan.

            while len(layer.nextLayers) > 1:
                layer = self.searchBranch(layer, ctx)
            
            if len(layer.nextLayers) == 0:
                return layer
            elif len(layer.nextLayers) == 1:
                layer = layer.nextLayers[0]
                if len(layer.prevLayers) > 1: # Joining layer.
                    return layer
            else:
                print("Error!")
        return
    
    def searchBranch(self, startLayer: Layer, ctx: SearchContext):
        print("Branching out at %d-th layer" % startLayer.id)

        # First pass. Figure out the best startConfig for every ending config.
        bestTimeByEndCfg = {}
        bestLayerTimeByEndCfg = {}
        bestStartCfgByEndCfg = {}
        bestPrevCfgListByEndCfg = {}
        mpIdleTimeByEndCfg = {}
        joiningLayer = None
        
        if hasattr(startLayer, "bestCfg"):
            startCfgOptions = [startLayer.bestCfg]
        else:
            startCfgOptions = startLayer.t.keys()

        initCfg = self.getInitialConfig(startLayer, ctx.globalBatch)
        startLayer.initCfg = initCfg
        startLayer.noParallelTime = self.benchGpuTime(startLayer, initCfg) + self.calcSyncTime(startLayer, initCfg, ctx)

        for startCfg in startCfgOptions:
            cumulativeTime, prevLayerTime, prevConfigIndexOfPrev, timeComposition, prevMpIdleTime = startLayer.t[startCfg]
            for layer in startLayer.nextLayers:
                endingLayer = self.searchLinear(startLayer, startCfg, layer, ctx)
                if joiningLayer == None:
                    joiningLayer = endingLayer
                else:
                    assert joiningLayer == endingLayer

            for joinConfig in self.listConfigOptions(joiningLayer, ctx.globalBatch, ctx.totalGpus, sampleSplit=ctx.sampleSplit, spatialSplit=ctx.spatialSplit, filterSplit=ctx.filterSplit, dataParallelBaseline=ctx.dataParallelBaseline):
                gpuTime = self.benchGpuTime(joiningLayer, joinConfig)
                syncTime = self.calcSyncTime(joiningLayer, joinConfig, ctx)

                maxBestCumulativeTime = 0
                sumBestActivationTime = 0
                bestCumulativeTimeList = []
                bestPrevCfgList = []
                bestTimeCompositionList = []
                for prevLayer in joiningLayer.prevLayers:
                    bestPrevCfg = None
                    bestCumulativeTime = 99999999999
                    bestActivationTime = None
                    bestTimeComposition = None
                    bestAmplification = 999999

                    for prevCfg in prevLayer.t:
                        cumulativeTime, prevLayerTime, prevConfigIndexOfPrev, timeComposition, prevMpIdleTime = prevLayer.t[prevCfg]
                        activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, joiningLayer, prevCfg, joinConfig)
                        # print("[%d (%s)-> %d (%s)] activationTime: %6.f" % (prevLayer.id, str(prevCfg), joiningLayer.id, str(joinConfig), activationTime))
                        gpusUsed = self.calcGpusNeeded(prevLayer, prevCfg, ctx.globalBatch)
                        amplification = ((prevLayerTime * gpusUsed) / (prevLayer.noParallelTime)) if (prevLayerTime > 0 and prevLayer.noParallelTime > 300) else 1

                        if cumulativeTime + activationTime + gpuTime + syncTime < bestCumulativeTime and \
                                (amplification < ctx.amplificationLimit or amplification < bestAmplification): # consider if this is within amplification limit or minimum among so far observed.
                            bestCumulativeTime = cumulativeTime + activationTime + gpuTime + syncTime
                            bestActivationTime = activationTime
                            bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                            bestAmplification = amplification # although it might not be the best so far.. it should be still within amplification limit.
                            bestPrevCfg = prevCfg
                    
                    # TODO: consider amplifcation here as well. Currently layers just before joiningLayer don't consider amplification.
                    maxBestCumulativeTime = max(maxBestCumulativeTime, bestCumulativeTime)
                    sumBestActivationTime += bestActivationTime
                    bestCumulativeTimeList.append(bestCumulativeTime)
                    bestPrevCfgList.append(bestPrevCfg)
                    bestTimeCompositionList.append(bestTimeComposition)

                mpIdleTime = sum([maxBestCumulativeTime - bestCumulTime for bestCumulTime in bestCumulativeTimeList])
                if joinConfig not in bestTimeByEndCfg or maxBestCumulativeTime < bestTimeByEndCfg[joinConfig]:
                    bestTimeByEndCfg[joinConfig] = maxBestCumulativeTime
                    bestLayerTimeByEndCfg[joinConfig] = sumBestActivationTime + gpuTime + syncTime
                    # print("sumBestActivationTime: %6.f + gpuTime: %6.f + syncTime: %6.f" % (sumBestActivationTime, gpuTime, syncTime))
                    bestStartCfgByEndCfg[joinConfig] = startCfg
                    bestPrevCfgListByEndCfg[joinConfig] = bestPrevCfgList
                    mpIdleTimeByEndCfg[joinConfig] = mpIdleTime

        joiningLayerInitCfg = self.getInitialConfig(joiningLayer, ctx.globalBatch)
        joiningLayer.initCfg = joiningLayerInitCfg
        joiningLayer.noParallelTime = self.benchGpuTime(joiningLayer, joiningLayerInitCfg) + self.calcSyncTime(joiningLayer, joiningLayerInitCfg, ctx)
        joiningLayer.t = {}
        for joinConfig in bestTimeByEndCfg:
            bestTimeByEndCfg[joinConfig]
            bestStartCfgByEndCfg[joinConfig]
            joiningLayer.t[joinConfig] = (bestTimeByEndCfg[joinConfig], bestLayerTimeByEndCfg[joinConfig], bestStartCfgByEndCfg[joinConfig], bestPrevCfgListByEndCfg[joinConfig], mpIdleTimeByEndCfg[joinConfig])
            joiningLayer.branchStartLayer = startLayer

        return joiningLayer

    def to_dot(self, name, globalBatch):
        dot = graphviz.Digraph(name = name)
        for layer in self.layers:
            gpusUsed = self.calcGpusNeeded(layer, layer.bestCfg, globalBatch)
            comment = ""
            # comment = "\n" + str(layer.bestCfg)
            # if hasattr(layer, 'gpuLocalAssignmentDict'):
            #     comment = "\n" + str(layer.gpuLocalAssignmentDict)
            if hasattr(layer, 'startNoOverlapAdjustment'):
                comment = "\n+%.2f ms" % (layer.startNoOverlapAdjustment / 1000)

            node_desc = "%d) %s\nt= %.2f ms (+%.2f)\nGPU=%s%s" % ( #]\nidle=%.1fms" % (
                layer.id, layer.name, layer.t[layer.bestCfg][0] / 1000, layer.t[layer.bestCfg][1] / 1000, layer.gpuAssignment, comment) #, layer.t[layer.bestCfg][4] / 1000) #, str(layer.bestCfg))
            # if node.stage_id is not None:
            #     color = self._colors[node.stage_id % len(self._colors)]
            #     dot.node(node.node_id, node_desc,
            #        color=color, style='filled')
            # else:
            dot.node(str(layer.id), node_desc)
        for layer in self.layers:
            for l in layer.nextLayers:
                dot.edge(str(layer.id), str(l.id))
        dot.render()

    def to_gpuTimeline(self, name, totalGpus, dataParallelBaseline=False):
        gpuTimes = [[] for r in range(totalGpus)]
        maxT = 0
        if dataParallelBaseline:
            dpTime = 0
            for layer in self.layers:
                dpTime += layer.t[layer.bestCfg][1] / 1000
                maxT = max(maxT, layer.t[layer.bestCfg][0] / 1000)
            for r in layer.gpuAssignment:
                gpuTimes[r].append( (0, dpTime) )
        else:
            for layer in self.layers:
                maxT = max(maxT, layer.t[layer.bestCfg][0] / 1000)
                endTime = layer.t[layer.bestCfg][0] / 1000
                startTime = endTime - layer.t[layer.bestCfg][1] / 1000
                for r in layer.gpuAssignment:
                    gpuTimes[r].append( (startTime, endTime) )
        
        fig, axs = plt.subplots(nrows=totalGpus, ncols=1, sharex=True)
        # fig, axs = plt.subplots(nrows=8, ncols=1, sharex=True)
        plt.xlabel("Time (ms)")
        plt.ylabel("GPU Rank")
        fig.set_figheight(2)

        gs1 = matplotlib.gridspec.GridSpec(1, 8)
        gs1.update(wspace=0.0025, hspace=0.005) # set the spacing between axes. 

        for rank, times in enumerate(gpuTimes):
            # if rank >= 8:
            #     break
            print("rank%d" % (rank))
            print(times)
            lastT = 0
            for start, end in sorted(times):
                if start == end:
                    print("skip")
                    continue
                idx = totalGpus - rank - 1
                # idx = 8 - rank - 1
                if lastT > start:
                    print("%5.3f %5.3f %5.3f" % (lastT, start, end))
                # axs[idx].axvspan(lastT, start, facecolor='gray', alpha=0.3)
                axs[idx].axvspan(start, end, facecolor='red', alpha=1)
                lastT = end
            # axs[idx].axvspan(lastT, maxT, facecolor='gray', alpha=0.3)
            axs[rank].set_ylabel("%d"%rank, rotation=0, y=0.2, labelpad=10)
            axs[rank].yaxis.set_ticks([])
            axs[rank].set_xlim(0, 50)

        axs[0].set_title(name)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("gpuTimeline.pdf")

    def searchBestSplitsV3(self, totalGpus: int, globalBatch: int = 16, amplificationLimit: float = 2.0, dataParallelBaseline = False, sampleSplit=True, spatialSplit=False, filterSplit=False):
        """ Parallelization strategy findiing for DeepPool. """
        ctx = SearchContext(totalGpus, globalBatch, amplificationLimit, dataParallelBaseline, sampleSplit=sampleSplit, spatialSplit=spatialSplit, filterSplit=filterSplit)
        finalLayer = self.searchLinear(None, None, self.layers[0], ctx)

        bestTime = 99999999999
        bestLastCfg = None
        bestLastLayerTime = None
        finalLayer.initCfg = self.getInitialConfig(finalLayer, ctx.globalBatch)
        for lastCfg in finalLayer.t:
            cumulativeTime, layerTime, prevCfg, timeComposition, prevMpIdleTime = finalLayer.t[lastCfg]
            if cumulativeTime < bestTime:
                bestTime = cumulativeTime
                bestLastCfg = lastCfg
                bestLastLayerTime = layerTime

        print("Network bandwidth: %5.f Gbps" % (self.NET_BANDWIDTH * 8 / 1000))        
        print("Best Time: %.f" % bestTime)

        gpuUsecSum = 0
        # Initial backward pass
        def backwardLinear(lastLayer, bestLastCfg, packGpus=False, stopAtLayer=None):
            gpuUsecSum = 0
            maxGpusUsed = 0
            startNoOverlapAdjustment = 0
            sideBranchOvertimeSum = 0
            layer = lastLayer
            bestCfg = bestLastCfg
            while True:
                layer.bestCfg = bestCfg
                if layer == stopAtLayer:
                    # gpuUsecSum -= layer.t[bestCfg][1] * self.calcGpusNeeded(layer, bestCfg, ctx.globalBatch) # To avoid double-counting, add startLayer in backwardBranch.
                    return gpuUsecSum, maxGpusUsed, startNoOverlapAdjustment, sideBranchOvertimeSum
                
                gpusUsed = self.calcGpusNeeded(layer, bestCfg, ctx.globalBatch)
                gpuUsecSum += layer.t[bestCfg][1] * gpusUsed
                maxGpusUsed = max(maxGpusUsed, gpusUsed)
                
                if layer.prevLayers == None or len(layer.prevLayers) == 0:
                    return gpuUsecSum, maxGpusUsed, startNoOverlapAdjustment, sideBranchOvertimeSum
                elif len(layer.prevLayers) > 1:
                    cumulativeTime, layerTime, prevConfig, bestPrevCfgListByEndCfg, prevMpIdleTime = layer.t[bestCfg]
                    layer.branchStartLayer.bestCfg = prevConfig
                    self.searchBranch(layer.branchStartLayer, ctx)
                    gpuUsecBlock, gpusUsedBlock, sideBranchOvertime = backwardBranch(layer, bestPrevCfgListByEndCfg, packGpus)
                    gpuUsecSum += gpuUsecBlock
                    maxGpusUsed = max(maxGpusUsed, gpusUsedBlock)
                    sideBranchOvertimeSum += sideBranchOvertime

                    layer = layer.branchStartLayer
                    bestCfg = prevConfig
                elif len(layer.prevLayers) == 1:
                    cumulativeTime, layerTime, prevConfig, timeComposition, prevMpIdleTime = layer.t[bestCfg]
                    if layer.prevLayers[0] == stopAtLayer:
                        startActivationTime, actSizes = self.calcInputXfer(layer.prevLayers[0], layer, prevConfig, layer.bestCfg)
                        startNoOverlapActivationTime, actSizes = self.calcInputXfer(layer.prevLayers[0], layer, prevConfig, layer.bestCfg, True)
                        startNoOverlapAdjustment = startNoOverlapActivationTime - startActivationTime
                        layer.startNoOverlapAdjustment = startNoOverlapAdjustment
                    layer = layer.prevLayers[0]
                    bestCfg = prevConfig
                else:
                    assert False
        
        def backwardBranch(joiningLayer, bestPrevCfgListByEndCfg: list, packGpus):
            gpuUsecSum = 0
            gpusUsedBlock = 0
            branchStartLayer = joiningLayer.branchStartLayer
            blockStartTime = branchStartLayer.t[branchStartLayer.bestCfg][0]
            branchTimeList = []
            for prevLayer, prevCfg in zip(joiningLayer.prevLayers, bestPrevCfgListByEndCfg):
                gpuUsecBranch, gpusUsedBranch, startNoOverlapAdjustment, sideBranchOvertime = backwardLinear(prevLayer, prevCfg, packGpus, stopAtLayer=joiningLayer.branchStartLayer)
                endNoOverlapActivationTime, actSizes = self.calcInputXfer(prevLayer, joiningLayer, prevCfg, joiningLayer.bestCfg, True)
                endActivationTime, actSizes = self.calcInputXfer(prevLayer, joiningLayer, prevCfg, joiningLayer.bestCfg, False)
                endAdjustment = endActivationTime - endNoOverlapActivationTime

                gpuUsecSum += gpuUsecBranch
                gpusUsedBlock += gpusUsedBranch
                if sideBranchOvertime > 0:
                    print("backward branch, joining layer: %d, prevLayerId: %d, sideBranchOvertime: %d" % (joiningLayer.id, prevLayer.id, sideBranchOvertime))
                branchTimeList.append((prevLayer.t[prevCfg][0] - blockStartTime + sideBranchOvertime, gpusUsedBranch, prevLayer.id, startNoOverlapAdjustment + endAdjustment))

            ## GPU packing! Comment out to disable GPU packing.
            branchTimeList.sort(reverse=True)
            maxBranchTime = max([x[0] for x in branchTimeList])
            gpuLocalAssignmentDict = {} # list of lists.
            sideBranchOvertime = 0
            #TODO: This is slightly off.. must consider the last activation time.
            newGpuReadyTimeVec = []
            for i, (branchTime, gpusNeeded, prevLayerId, newGpuActiviationAdjustment) in enumerate(branchTimeList):
                if i >= 1:
                    # Using the same GPU will be cheaper than using new set of GPUs.
                    if maxBranchTime < newGpuActiviationAdjustment or \
                            ((newGpuActiviationAdjustment + branchTime) / branchTime if branchTime > 0 else 1) > amplificationLimit: # TODO; revisit this amplificationLimit.
                        maxBranchTime += branchTime
                        sideBranchOvertime += branchTime
                        print("Running side branch on new GPUs will hurt %.2f ms. So run on main branch. lid: %d" % (newGpuActiviationAdjustment / 1000, prevLayerId))
                    else:
                        branchTime += newGpuActiviationAdjustment
                        print("Added %.3f ms to activation time for lid: %d" % (newGpuActiviationAdjustment, prevLayerId))
                        if branchTime > maxBranchTime:
                            print("side branch time exceeds the main branch time due to activationTime. lid: %d. Added: %.2f" % (prevLayerId, newGpuActiviationAdjustment * gpusNeeded / 1000))
                            sideBranchOvertime += branchTime - maxBranchTime
                            maxBranchTime = branchTime
                            gpuUsecSum += newGpuActiviationAdjustment * gpusNeeded # TODO: this is a slight overestimation for later blocks.

                gpuLocalAssignment = []
                gpusAssigned = 0
                for i in range(len(newGpuReadyTimeVec)):
                    if newGpuReadyTimeVec[i] + branchTime <= maxBranchTime:
                        if packGpus:
                            newGpuReadyTimeVec[i] += branchTime
                        else:
                            newGpuReadyTimeVec[i] = maxBranchTime   #TODO: remove this hack for disabling GPU packing. 
                        gpusAssigned += 1
                        gpuLocalAssignment.append(i)
                    if gpusAssigned >= gpusNeeded:
                        break
                for i in range(len(newGpuReadyTimeVec), len(newGpuReadyTimeVec) + gpusNeeded - gpusAssigned):
                    gpuLocalAssignment.append(i)
                    if packGpus:
                        newGpuReadyTimeVec.append(branchTime)
                    else:
                        newGpuReadyTimeVec.append(maxBranchTime)   #TODO: remove this hack for disabling GPU packing. 
                gpuLocalAssignmentDict[prevLayerId] = gpuLocalAssignment

            joiningLayer.gpuLocalAssignmentDict = gpuLocalAssignmentDict

            print(" GPU packing at layer %d.. reduced gpusUsedBlock %d -> %d" % (branchStartLayer.id, gpusUsedBlock, len(newGpuReadyTimeVec)))
            gpusUsedBlock = len(newGpuReadyTimeVec)
            
            gpusUsed = self.calcGpusNeeded(joiningLayer, joiningLayer.bestCfg, ctx.globalBatch)
            gpuUsecSum += joiningLayer.t[joiningLayer.bestCfg][1] * gpusUsed
            gpusUsedBlock = max(gpusUsedBlock, gpusUsed)
            return gpuUsecSum, gpusUsedBlock, sideBranchOvertime

        gpuUsecSum, maxGpusUsed, startNoOverlapAdjustment, sideBranchOvertimeSum = backwardLinear(finalLayer, bestLastCfg, True)
        
        print("Final layer: ", finalLayer.id, " bestCfg:", finalLayer.bestCfg, " sideBranchOvertimeSum: ", sideBranchOvertimeSum)

        # 2nd backward pass: assign GPUs.
        def gpuAssignLinear(lastLayer, availableRanks, stopAtLayer=None):
            gpuUsecSum = 0
            layer = lastLayer
            while True:
                if layer == stopAtLayer:
                    return
                
                gpusUsed = self.calcGpusNeeded(layer, layer.bestCfg, ctx.globalBatch)
                layer.gpuAssignment = [availableRanks[i] for i in range(gpusUsed)]
                
                if layer.prevLayers == None or len(layer.prevLayers) == 0:
                    return
                elif len(layer.prevLayers) > 1:
                    gpuAssignBranch(layer, availableRanks)
                    layer = layer.branchStartLayer
                elif len(layer.prevLayers) == 1:
                    layer = layer.prevLayers[0]
                else:
                    assert False
        
        def gpuAssignBranch(joiningLayer, availableRanks):
            for prevLayer in joiningLayer.prevLayers:
                branchGpuNeeded = joiningLayer.gpuLocalAssignmentDict[prevLayer.id]
                gpuAssignLinear(prevLayer, [availableRanks[i] for i in branchGpuNeeded], stopAtLayer=joiningLayer.branchStartLayer)
                        
            gpusUsed = self.calcGpusNeeded(joiningLayer, joiningLayer.bestCfg, ctx.globalBatch)
            joiningLayer.gpuAssignment = [availableRanks[i] for i in range(gpusUsed)]

        gpuAssignLinear(finalLayer, list(range(maxGpusUsed)))
        # gpuAssignLinear(finalLayer, list(range(ctx.totalGpus)))



        # Print result
        print("Layer    type       initial configuration          => after split configuration            #GPUs time(us) |   prev inptXfer  gpuTime syncTime bestGpuTime dpGpuTime noParallelTime")
        finalTime = 0
        dpTime = 0
        for layer in self.layers:
            cumulativeTime, layerTime, prevConfigIndexOfPrev, timeComposition, mpIdleTime = layer.t[layer.bestCfg]
            dpTime += layerTime
            nextLayerIds = []
            for l in layer.nextLayers:
                nextLayerIds.append(l.id)

            print(" %3d  %25s  " % (layer.id, str(nextLayerIds)), end="")
            if not hasattr(layer, "bestCfg"):
                print("no bestCfg")
                continue
            if not hasattr(layer, "initCfg"):
                print("no initCfg", end="")
                layer.initCfg = self.getInitialConfig(layer, ctx.globalBatch)
                # continue
            
            if layer.name in ["conv2d"]:
                print("%11s (b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) => " % (layer.name, *layer.initCfg), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) " % layer.bestCfg, end="")
            elif layer.name in ["linear", "ReLU1d"]:
                print("%11s (b=%2d, in=%6d, out=%6d)        => " % (layer.name, *layer.initCfg), end="")
                print("(b=%2d, in=%6d, out=%6d)        " % layer.bestCfg, end="")
            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
                print("%11s (b=%2d, w=%3d, h=%3d, c=%4d)         => " % (layer.name, *layer.initCfg), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d)         " % layer.bestCfg, end="")

            gpusUsed = self.calcGpusNeeded(layer, layer.bestCfg, ctx.globalBatch)
            gpuTime = self.benchGpuTime(layer, layer.bestCfg)
            # gpuUsec = gpusUsed * (cumulativeTime - timeComposition[0])
            # gpuUsecSum += gpuUsec
            print(" %4d   %6.f   %6.f   %6.f   " #%6.f   %6.f      %6.f    %6.f  %6.f (%4.1fx) (DP:%4.1fx, MP:%4.1fx)  %6.f  %10s"
                    % (gpusUsed, cumulativeTime, layer.t[layer.bestCfg][1], gpuTime))#, timeComposition[0], timeComposition[1], timeComposition[2], timeComposition[3])) #, bestTimeList[i], bestDataParallelTimeList[i], initialTimes[i], gpuTimeScaling, gpuTimeScalingDP, gpuTimeScalingMP, gpuUsec, str(timeComposition[4])))
            finalTime = max(cumulativeTime, finalTime)
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("  final                                                                %6.3f ms    %6.3f gpuMsec  DPtime: %6.3f ms, maxGpusUsed: %d"
                 % (finalTime/1000, gpuUsecSum/1000, dpTime/1000, maxGpusUsed))

        # moduleDesc = self.generateModuleDescription([layer.bestCfg for layer in self.layers], ctx.globalBatch)
        moduleDesc = TrainingJob("test", self.layers, [layer.bestCfg for layer in self.layers], ctx.globalBatch, maxGpusUsed, "na")
        # moduleDesc = None
        if ctx.dataParallelBaseline:
            return (moduleDesc, dpTime / 1000., gpuUsecSum / 1000., ctx.totalGpus)
        else:
            return (moduleDesc, (finalTime + sideBranchOvertimeSum) / 1000., gpuUsecSum / 1000., maxGpusUsed)
