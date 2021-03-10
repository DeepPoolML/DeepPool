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

import json
import torch
import torch.nn as nn
from typing import Optional, IO, List, Any

class Layer:
    def __init__(self, module: nn.Module, name: str, params: tuple, prevLayers: list):
        self.id = None      # Assigned later by calling printAllLayers.
        self.name = name
        self.params = params
        self.prevLayers = prevLayers
        if prevLayers is not None:
            for prevLayer in prevLayers:
                prevLayer.nextLayers.append(self)
        self.nextLayers = []
        self.module = module
        self.inputDim = (0, 0, 0)   # (Width, Height, Channel) for 2d convolution
        self.outputDim = (0, 0, 0)  # (Width, Height, Channel)
    
    def dumpForJSON(self):
        prop = {}
        if self.id == None:
            raise Exception("layer is not yet initialized.")

        prop["id"] = self.id
        prop["name"] = self.name
        prop["params"] = self.params
        prop["prevLayers"] = []
        if self.prevLayers != None:
            for prevLayer in self.prevLayers:
                prop["prevLayers"].append(prevLayer.id)
        prop["nextLayers"] = []
        if self.nextLayers != None:
            for nextLayer in self.nextLayers:
                prop["nextLayers"].append(nextLayer.id)
        prop["inputDim"] = self.inputDim
        prop["outputDim"] = self.outputDim
        # prop["inputDim"] = {"width": self.inputDim[0], "height": self.inputDim[1], "channel": self.inputDim[2]}
        # prop["outputDim"] = {"width": self.outputDim[0], "height": self.outputDim[1], "channel": self.outputDim[2]}
        return prop


class TrainingJob:
    def __init__(self, name: str, layers: List[Layer], layerConfigs: List[tuple], globalBatchSize: int, datasetDir: str):
        self.name = name
        self.layers = layers
        self.layerConfigs = layerConfigs
        self.globalBatchSize = globalBatchSize
        self.datasetDir = datasetDir
        self.bytesPerParam = 4
    
    def loadJSON(self, jobInJson: str):
        job = json.loads(jobInJson)
        self.globalBatchSize = job["globalBatchSize"]
        self.layers = []
        self.layerConfigs = []
        for ldsc in job["layers"]:
            print(ldsc)
            prevLayers = [self.layers[prevLayerId] for prevLayerId in ldsc["prevLayers"]]
            l = Layer(None, ldsc["name"], ldsc["params"], prevLayers)
            l.id = ldsc["id"]
            # l.nextLayers = ldsc["nextLayers"]
            l.inputDim = ldsc["inputDim"]
            l.outputDim = ldsc["outputDim"]
            config = ldsc["config"]
            self.layers.append(l)
            self.layerConfigs.append(config)

    def dumpInJSON(self, layers: List[Layer] = None, layerConfigs: list = None):
        if layers is None:
            layers = self.layers
        if layerConfigs is None:
            layerConfigs = self.layerConfigs

        allProps = []
        for l, config in zip(layers, layerConfigs):
            prop = l.dumpForJSON()
            prop["config"] = config
            # prop["tensorTx"] = [{"name": "%d_0" % l.id, "dest": 1, "bytes": 10}]
            # prop["tensorRx"] = [{"name": "%d_0" % l.id, "src": 2, "bytes": 50}]
            allProps.append(prop)
        fullDesc = {"globalBatchSize": self.globalBatchSize, "layers": allProps}
        # return json.dumps(fullDesc, indent=1, sort_keys=False)
        return json.dumps(fullDesc, sort_keys=False)

    def dumpSingleRunnableModule(self, targetRank: int) -> str: # Only supports DP now.
        print("[dumpSingleRunnableModule] generating for rank: %d" % targetRank)
        allProps = []
        srcGpus = None
        maxGpusUsed = 0
        for l, config in zip(self.layers, self.layerConfigs):
            prop = l.dumpForJSON()
            prop["config"] = config
            
            if not self.isConfigDataParallelOnly(l, config, self.globalBatchSize):
                print("[dumpSingleRunnableModule] config was not DP-only.")
                return None
            
            destGpus = self.calcGpusNeeded(l, config, self.globalBatchSize)
            maxGpusUsed = max(maxGpusUsed, destGpus)
            if targetRank >= destGpus: # Not used for this layer.
                configInList = list(config)
                configInList[0] = 0
                prop["config"] = tuple(configInList)

            if srcGpus != None:
                # print("%d srcGpus: %d => destGpus: %d, srcConfig: %s  destConfig: %s" % (l.id, srcGpus, destGpus, str(srcConfig), str(config) ))
                if destGpus > srcGpus: # expanding
                    newGpuCount = destGpus - srcGpus
                    samplesPerSrc = srcConfig[0] - config[0]
                    if samplesPerSrc < 0:
                        print("Error! negative number: %d. destGpus: %d srcGpus: %d" % (samplesPerSrc, destGpus, srcGpus))
                        print("Expanding srcConfig: %s  destConfig: %s" % (str(srcConfig), str(config) ))
                    if targetRank >= srcGpus: # Newly used nodes.
                        newGpuRank = targetRank - srcGpus
                        startSrcNode = int(newGpuRank * srcGpus / newGpuCount)
                        endSrcNode = int((newGpuRank + 1) * srcGpus / newGpuCount)

                        samplesAssigned = 0
                        prop["tensorRx"] = []
                        for src in range(startSrcNode, endSrcNode+1):
                            if src == startSrcNode:
                                samplesAvail = samplesPerSrc * (startSrcNode + 1) - newGpuRank * config[0]
                                if samplesAvail < 0:
                                    print("Error! negative number.")
                                    print("srcConfig: %s  destConfig: %s" % (str(srcConfig), str(config) ))
                                    print("samplesAvail: %d, samplesPerSrc: %d, startSrcNode: %d, newGpuRank: %d" % 
                                        (samplesAvail, samplesPerSrc, startSrcNode, newGpuRank))
                            else:
                                samplesAvail = samplesPerSrc
                            xferSamples = min(samplesAvail, config[0])
                            samplesAssigned += xferSamples
                            xferBytes = xferSamples * self.bytesPerParam
                            prop["tensorRx"].append({"name": "%d_sample_%d" % (l.id, src-startSrcNode),
                                                    "prop": {"xferSamples": xferSamples},
                                                    "src": src,
                                                    "bytes": xferBytes})
                            if samplesAssigned >= config[0]:
                                break
                    elif targetRank < srcGpus: 
                        # send samples after previous layer. Recv nothing for current.
                        startDestNode = int(targetRank * newGpuCount / srcGpus) + srcGpus
                        endDestNode = int((targetRank + 1) * newGpuCount / srcGpus) + srcGpus

                        samplesAssigned = 0
                        allProps[-1]["tensorTx"] = []
                        for dest in range(startDestNode, endDestNode+1):
                            if dest == startDestNode:
                                samplesLeft = config[0] * (startDestNode - srcGpus + 1) - targetRank * samplesPerSrc
                                if samplesLeft < 0:
                                    print("Error! negative number.")
                                    print("srcConfig: %s  destConfig: %s" % (str(srcConfig), str(config) ))
                                    print("samplesLeft: %d, samplesPerSrc: %d, startDestNode: %d, srcGpus: %d, targetRank: %d" &
                                        (samplesLeft, samplesPerSrc, startDestNode, srcGpus, targetRank))
                            else:
                                samplesLeft = config[0]
                            xferSamples = min(samplesLeft, samplesPerSrc)
                            samplesAssigned += xferSamples
                            xferBytes = xferSamples * self.bytesPerParam
                            
                            tensorIdx = targetRank - (dest - srcGpus) * srcGpus / newGpuCount
                            allProps[-1]["tensorTx"].append({"name": "%d_sample_%d" % (l.id, tensorIdx),
                                                        "prop": {"xferSamples": xferSamples},
                                                        "dest": dest,
                                                        "bytes": xferBytes})
                            if samplesAssigned >= samplesPerSrc:
                                break
                elif destGpus < srcGpus: # Shrinking
                    removedGpuCount = srcGpus - destGpus
                    samplesPerDest = config[0] - srcConfig[0]
                    if samplesPerDest < 0:
                        print("Error! negative number: %d. destGpus: %d srcGpus: %d" % (samplesPerDest, destGpus, srcGpus))
                        print("Shrinking srcConfig: %s  destConfig: %s" % (str(srcConfig), str(config) ))
                    if targetRank >= destGpus: # Removed nodes.
                        removedGpuRank = targetRank - destGpus
                        startDestNode = int(removedGpuRank * destGpus / removedGpuCount)
                        endDestNode = int((removedGpuRank + 1) * destGpus / removedGpuCount)

                        samplesAssigned = 0
                        allProps[-1]["tensorTx"] = []
                        for dest in range(startDestNode, endDestNode+1):
                            if dest == startDestNode:
                                samplesLeft = samplesPerDest * (startDestNode + 1) - removedGpuRank * srcConfig[0]
                                if samplesLeft < 0:
                                    print("Error! negative number.")
                                    print("srcConfig: %s  destConfig: %s" % (str(srcConfig), str(config) ))
                            else:
                                samplesLeft = samplesPerDest

                            assert allProps[-1]["config"][0] == srcConfig[0]
                            xferSamples = min(min(samplesLeft, samplesPerDest), srcConfig[0])
                            samplesAssigned += xferSamples
                            xferBytes = xferSamples * self.bytesPerParam
                            startSrcNode = int(dest * removedGpuCount / destGpus) + destGpus
                            tensorIdx = targetRank - startSrcNode
                            allProps[-1]["tensorTx"].append({"name": "%d_sample_%d" % (l.id, tensorIdx),
                                                        "prop": {"xferSamples": xferSamples},
                                                        "dest": dest,
                                                        "bytes": xferBytes})
                            if samplesAssigned >= srcConfig[0]:
                                break
                    elif targetRank < destGpus: 
                        startSrcNode = int(targetRank * removedGpuCount / destGpus) + destGpus
                        endSrcNode = int((targetRank + 1) * removedGpuCount / destGpus) + destGpus

                        samplesAssigned = 0
                        prop["tensorRx"] = []
                        for src in range(startSrcNode, endSrcNode+1): # full transfer
                            if src == startSrcNode:
                                samplesLeft = srcConfig[0] * (startSrcNode - destGpus + 1) - targetRank * samplesPerDest
                                if samplesLeft < 0:
                                    print("Error! negative number.")
                                    print("srcConfig: %s  destConfig: %s" % (str(srcConfig), str(config) ))
                            else:
                                samplesLeft = srcConfig[0]
                            
                            # xferSamples = min(samplesLeft, samplesPerDest)
                            xferSamples = min(min(samplesLeft, samplesPerDest), srcConfig[0])
                            samplesAssigned += xferSamples
                            xferBytes = xferSamples * self.bytesPerParam
                            
                            tensorIdx = src - startSrcNode
                            prop["tensorRx"].append({"name": "%d_sample_%d" % (l.id, tensorIdx),
                                                    "prop": {"xferSamples": xferSamples},
                                                    "src": src,
                                                    "bytes": xferBytes})
                            if samplesAssigned >= samplesPerDest:
                                break

            # TODO: implement tensorTx. & config mod for sampleDim = 0 when not needed.
            allProps.append(prop)
            srcGpus = destGpus
            srcConfig = config

        # Compute dataLoaderOffset & worldSize.
        samplesPerNode = self.layerConfigs[0][0]
        dataLoaderOffset = samplesPerNode * targetRank
        if dataLoaderOffset >= self.globalBatchSize: # This may happen if first layer uses smaller # of GPUs than later ones.
            dataLoaderOffset = 0
            assert allProps[0]["config"][0] == 0

        fullDesc = {"rank": targetRank,
                    "maxGpusUsed": maxGpusUsed,
                    "globalBatchSize": self.globalBatchSize,
                    "dataLoaderOffset": dataLoaderOffset,
                    "layers": allProps}
        # dumpedStr = json.dumps(fullDesc, indent=1, sort_keys=False)
        dumpedStr = json.dumps(fullDesc, sort_keys=False)
        # print(dumpedStr)
        return dumpedStr
    
    # Functions to move:
    # - config to gpu count.
    def getInitialConfig(self, layer: Layer, globalBatch: int):
        if layer.name in ["conv2d"]:
            initCfg = (globalBatch, layer.inputDim[0], layer.inputDim[1], layer.inputDim[2], layer.outputDim[2]) # (batch, width, height, channel, filter)
        elif layer.name in ["linear", "ReLU1d"]:
            initCfg = (globalBatch, layer.inputDim, layer.outputDim)
        elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
            initCfg = (globalBatch, layer.inputDim[0], layer.inputDim[1], layer.inputDim[2]) # (batch, width, height, channel, filter)
        return initCfg

    def calcGpusNeeded(self, layer: Layer, config: tuple, globalBatch: int):
        initCfg = self.getInitialConfig(layer, globalBatch)
        gpuCount = 1
        # if len(config) != len(initCfg):
        #     print("[calcGpusNeeded] dimension of configs doesn't match!! %20s layer len(config):%d != len(initCfg):%d" % (layer.name, len(config), len(initCfg)))
        for i in range(len(initCfg)):
            gpuCount *= int(initCfg[i] / config[i])
        return gpuCount

    def isConfigDataParallelOnly(self, layer: Layer, config: tuple, globalBatch: int):
        initCfg = self.getInitialConfig(layer, globalBatch)
        dpOnly = True
        for i in range(1, len(config)):
            if config[i] != initCfg[i]:
                dpOnly = True
        return dpOnly

def test():
    return

if __name__ == "__main__":
    test()