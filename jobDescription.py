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
import io
from os.path import exists
from typing import Optional, IO, List, Any

class Layer:
    def __init__(self, module, name: str, params: tuple, prevLayers: list):
        self.id = None      # Assigned later by calling printAllLayers.
        self.name = name
        self.params = params
        self.prevLayers = prevLayers
        if prevLayers is not None:
            for prevLayer in prevLayers:
                prevLayer.nextLayers.append(self)
        self.nextLayers = []
        self.module = module
        self.moduleScript = None
        self.inputDim = (0, 0, 0)   # (Channel, Width, Height) for 2d convolution
        self.outputDim = (0, 0, 0)  # (Channel, Width, Height)
        self.must_trace = False

    def getModuleId(self):
        return self.name +\
            json.dumps(self.params, sort_keys=True, separators=('_', '-')) +\
            json.dumps(self.inputDim, sort_keys=False, separators=('_', '-'))
    
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
        if hasattr(self, 'gpuAssignment'):
            prop["gpuAssignment"] = self.gpuAssignment

        if self.module != None:
            moduleId = self.getModuleId()
            saveLocation = "modules/scriptmodule_%s.pt"%moduleId
            if exists(saveLocation): # Skip if module file is already there.
                prop["moduleSavedLocation"] = saveLocation
            else:
                if self.name == "concat":
                    fakeInputs = []
                    for prevLayer in self.prevLayers:
                        inputSize = [1] + list(prevLayer.outputDim)
                        # print("id: ", self.id, " Concat's inputSize: ", inputSize)
                        fakeInputs.append(torch.zeros(inputSize))
                    traced = torch.jit.script(self.module, fakeInputs)
                else:
                    inputSize = [1] + (list(self.inputDim) if type(self.inputDim) == tuple else [self.inputDim])
                    # print("id: ", self.id, " non-concat inputSize: ", inputSize)
                    fakeInput = torch.zeros(tuple(inputSize))
                    if self.must_trace:
                        print("jit tracing...", self.name)
                        traced = torch.jit.trace(self.module, fakeInput)
                    else:
                        print("jit scripting...", self.name)
                        traced = torch.jit.script(self.module, fakeInput)
                # saveLocation = "modules/scriptmodule_%d.pt"%self.id
                torch.jit.save(traced, saveLocation)
                prop["moduleSavedLocation"] = saveLocation

                buffer = io.BytesIO()
                torch.jit.save(traced, buffer)
                self.moduleScript = buffer.getvalue()
                # print("Layer%2d written %5d bytes." % (self.id, len(self.moduleScript)))
                # print(" *** Code ***\n%s" % (traced.code))
        elif hasattr(self, 'moduleSavedLocation'):
            prop["moduleSavedLocation"] = self.moduleSavedLocation

        return prop


class TrainingJob:
    def __init__(self, name: str, layers: List[Layer], layerConfigs: List[tuple], globalBatchSize: int, maxGpusUsed: int, datasetDir: str):
        self.name = name
        self.layers = layers
        self.layerConfigs = layerConfigs
        self.globalBatchSize = globalBatchSize
        self.maxGpusUsed = maxGpusUsed
        self.datasetDir = datasetDir
        self.bytesPerParam = 4
        self.initialBatchSizes = None
        self.sampleIndicesList = None
    
    def loadJSON(self, jobInJson: str):
        job = json.loads(jobInJson)
        self.globalBatchSize = job["globalBatchSize"]
        self.maxGpusUsed = job["maxGpusUsed"]
        self.layers = []
        self.layerConfigs = []
        for ldsc in job["layers"]:
            # print(ldsc)
            prevLayers = [self.layers[prevLayerId] for prevLayerId in ldsc["prevLayers"]]
            l = Layer(None, ldsc["name"], ldsc["params"], prevLayers)
            l.id = ldsc["id"]
            # l.nextLayers = ldsc["nextLayers"]
            l.inputDim = ldsc["inputDim"]
            l.outputDim = ldsc["outputDim"]
            if 'gpuAssignment' in ldsc:
                l.gpuAssignment = ldsc["gpuAssignment"]
            l.bestCfg = ldsc["config"]
            if 'moduleSavedLocation' in ldsc:
                l.moduleSavedLocation = ldsc["moduleSavedLocation"]
            config = ldsc["config"]
            self.layers.append(l)
            self.layerConfigs.append(config)
    
    def getGpusUsed(self):
        return self.maxGpusUsed
        # maxGpusUsed = 0
        # for l, config in zip(self.layers, self.layerConfigs):
        #     destGpus = self.calcGpusNeeded(l, config, self.globalBatchSize)
        #     maxGpusUsed = max(maxGpusUsed, destGpus)
        #     # print("[getGpusUsed] layer: %d, destGpus: %d, maxGpusUsed: %d, config: %s" % (l.id, destGpus, maxGpusUsed, str(config)))
        # return maxGpusUsed

    def dumpInJSON(self, layers: List[Layer] = None, layerConfigs: list = None):
        if layers is None:
            layers = self.layers
        if layerConfigs is None:
            layerConfigs = self.layerConfigs

        allProps = []
        for l, config in zip(layers, layerConfigs):
            prop = l.dumpForJSON()
            prop["config"] = config
            allProps.append(prop)
        fullDesc = {"globalBatchSize": self.globalBatchSize, "maxGpusUsed": self.maxGpusUsed, "layers": allProps}
        # return json.dumps(fullDesc, indent=1, sort_keys=False)
        return json.dumps(fullDesc, sort_keys=False)

    def computeSampleIndicesList(self):
        gpusUsed = self.getGpusUsed()
        specList = [self.dumpSingleRunnableModuleHelper(rank) for rank in range(gpusUsed)]
        sampleIndicesList = []
        for spec in specList:
            sampleOffset = spec["dataLoaderOffset"]
            localBatch = spec["layers"][0]["config"][0] # initialBatchSize or localBatch
            sampleIndices = list(range(sampleOffset, sampleOffset+localBatch))
            sampleIndicesList.append(sampleIndices)
        print("[computeTargetDestList] initial targets: %s" % str(sampleIndicesList))
        
        for lid in range(len(specList[0]["layers"])): # all ranks have the same # of layers.
            for srcRank, spec in enumerate(specList):
                ldsc = spec["layers"][lid]
                if ldsc["config"][0] > 0: # This rank has assigned samples for this layer.
                    if "tensorTx" in ldsc: # send parts of output.
                        ######## Stopped here. replace below code with manipulation on sampleIndicesList.
                        for txItem in ldsc["tensorTx"]:
                            moveCount = txItem["prop"]["xferSamples"]
                            indicesToMove = sampleIndicesList[srcRank][:moveCount]
                            sampleIndicesList[txItem["dest"]].extend(indicesToMove)
                            sampleIndicesList[srcRank] = sampleIndicesList[srcRank][moveCount:]
        print("[computeTargetDestList] final targets: %s" % str(sampleIndicesList))
        initialBatchSizes = [spec["layers"][0]["config"][0] for spec in specList]
        return initialBatchSizes, sampleIndicesList

    def dumpSingleRunnableModule(self, targetRank: int) -> str: # Only supports DP now.
        if self.initialBatchSizes == None:
            self.initialBatchSizes, self.sampleIndicesList = self.computeSampleIndicesList()
        fullDesc = self.dumpSingleRunnableModuleHelper(targetRank)
        fullDesc["initialBatchSizes"] = self.initialBatchSizes
        fullDesc["sampleIndices"] = self.sampleIndicesList[targetRank]
        dumpedStr = json.dumps(fullDesc, sort_keys=False)
        return dumpedStr

    def dumpSingleRunnableModuleHelper(self, targetRank: int) -> str: # Only supports DP now.
        # print("[dumpSingleRunnableModule] generating for rank: %d" % targetRank)
        allProps = []
        for l in self.layers:
            prop = l.dumpForJSON()
            prop["config"] = l.bestCfg
            
            if not self.isConfigDataParallelOnly(l, l.bestCfg, self.globalBatchSize):
                print("[dumpSingleRunnableModule] config was not DP-only.")
                return None
            
            # destGpus = self.calcGpusNeeded(l, config, self.globalBatchSize)
            # maxGpusUsed = max(maxGpusUsed, len(l.gpuAssignment))
            if targetRank not in l.gpuAssignment: # Not used for this layer.
                configInList = list(l.bestCfg)
                configInList[0] = 0
                prop["config"] = tuple(configInList)

            if l.prevLayers == None: # 1st layer.
                allProps.append(prop)
                continue

            for prevLayer in l.prevLayers:
                srcSamples = prevLayer.bestCfg[0]
                dstSamples = l.bestCfg[0]
                srcSamplesAssigned = {}
                dstSamplesAssigned = {}
                for r in prevLayer.gpuAssignment:
                    srcSamplesAssigned[r] = 0
                for r in l.gpuAssignment:
                    dstSamplesAssigned[r] = 0

                commonGpus = set(prevLayer.gpuAssignment).intersection(l.gpuAssignment)
                for r in commonGpus:
                    commonSamples = min(srcSamples, dstSamples)
                    srcSamplesAssigned[r] = commonSamples
                    dstSamplesAssigned[r] = commonSamples

                # Now fill the missing samples by xfer.
                xferNum = 0
                for dstRank in l.gpuAssignment:
                    for srcRank in prevLayer.gpuAssignment:
                        samplesLeftAtSrc = srcSamples - srcSamplesAssigned[srcRank]
                        samplesNeedAtDst = dstSamples - dstSamplesAssigned[dstRank]
                        xferSamples = min(samplesLeftAtSrc, samplesNeedAtDst)

                        if xferSamples == 0:
                            continue

                        xferNum += 1
                        srcSamplesAssigned[srcRank] += xferSamples
                        dstSamplesAssigned[dstRank] += xferSamples
                        
                        xferBytes = xferSamples * self.bytesPerParam
                        
                        assert(dstRank != srcRank)
                        if targetRank == dstRank:
                            # TODO: remove "tensorRx". It's left for python runtime compatibility.
                            if "tensorRx" not in prop:
                                prop["tensorRx"] = []    
                            prop["tensorRx"].append({"name": "%d_from_%d_sample_%d" % (l.id, prevLayer.id, xferNum),
                                                    "prop": {"xferSamples": xferSamples, "prevLayerId": prevLayer.id}, # prevLayerId is necessary for Concat inputs.
                                                    "src": srcRank,
                                                    "bytes": xferBytes})
                            # tensorRxJit is used for CPP runtime.
                            if "tensorRxJit" not in allProps[prevLayer.id]:
                                allProps[prevLayer.id]["tensorRxJit"] = []
                            allProps[prevLayer.id]["tensorRxJit"].append({"name": "%d_from_%d_sample_%d" % (l.id, prevLayer.id, xferNum),
                                                    "prop": {"xferSamples": xferSamples, "nextLayerId": l.id}, # prevLayerId is necessary for Concat inputs.
                                                    "src": srcRank,
                                                    "bytes": xferBytes})
                        if targetRank == srcRank:
                            if "tensorTx" not in allProps[prevLayer.id]:
                                allProps[prevLayer.id]["tensorTx"] = []
                            allProps[prevLayer.id]["tensorTx"].append({"name": "%d_from_%d_sample_%d" % (l.id, prevLayer.id, xferNum),
                                                    "prop": {"xferSamples": xferSamples, "nextLayerId": l.id},
                                                    "dest": dstRank,
                                                    "bytes": xferBytes})
            allProps.append(prop)

        # Compute dataLoaderOffset & worldSize.
        samplesPerNode = self.layers[0].bestCfg[0]
        dataLoaderOffset = samplesPerNode * targetRank
        if dataLoaderOffset >= self.globalBatchSize: # This may happen if first layer uses smaller # of GPUs than later ones.
            dataLoaderOffset = 0
            # assert allProps[0]["config"][0] == 0
            if allProps[0]["config"][0] != 0:
                raise Exception('allProps[0]["config"][0] != 0')
            

        fullDesc = {"rank": targetRank,
                    "maxGpusUsed": self.maxGpusUsed,
                    "globalBatchSize": self.globalBatchSize,
                    "dataLoaderOffset": dataLoaderOffset,
                    "layers": allProps}
        # dumpedStr = json.dumps(fullDesc, indent=1, sort_keys=False)
        return fullDesc
        # dumpedStr = json.dumps(fullDesc, sort_keys=False)
        # # print(dumpedStr)
        # return dumpedStr

    def dumpSingleRunnableModuleHelperOld(self, targetRank: int) -> str: # Only supports DP now.
        # print("[dumpSingleRunnableModule] generating for rank: %d" % targetRank)
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
                    if targetRank >= destGpus and targetRank < srcGpus: # Removed nodes.
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

                            # assert allProps[-1]["config"][0] == srcConfig[0]
                            if allProps[-1]["config"][0] != srcConfig[0]:
                                print("layer:%s, allProps[-1]['config']: %s, srcConfig: %s" % \
                                        (prop["name"], str(allProps[-1]["config"]), str(srcConfig)))
                                raise Exception("allProps[-1]['config'][0] == srcConfig[0] failed. %d %d"% (allProps[-1]['config'][0], srcConfig[0]))

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
            # assert allProps[0]["config"][0] == 0
            if allProps[0]["config"][0] != 0:
                raise Exception('allProps[0]["config"][0] != 0')
            

        fullDesc = {"rank": targetRank,
                    "maxGpusUsed": maxGpusUsed,
                    "globalBatchSize": self.globalBatchSize,
                    "dataLoaderOffset": dataLoaderOffset,
                    "layers": allProps}
        # dumpedStr = json.dumps(fullDesc, indent=1, sort_keys=False)
        return fullDesc
        # dumpedStr = json.dumps(fullDesc, sort_keys=False)
        # # print(dumpedStr)
        # return dumpedStr
    
    # Functions to move:
    # - config to gpu count.
    def getInitialConfig(self, layer: Layer, globalBatch: int):
        if layer.name in ["conv2d"]:
            initCfg = (globalBatch, layer.inputDim[1], layer.inputDim[2], layer.inputDim[0], layer.outputDim[2]) # (batch, width, height, channel, filter)
        elif layer.name in ["linear", "ReLU1d"]:
            initCfg = (globalBatch, layer.inputDim, layer.outputDim)
        elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
            initCfg = (globalBatch, layer.inputDim[1], layer.inputDim[2], layer.inputDim[0]) # (batch, width, height, channel, filter)
        else:
            initCfg = (globalBatch, *layer.inputDim) # (batch, width, height, channel)
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