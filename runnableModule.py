import json
import os
import math
import threading
from typing import TypeVar, Optional, Iterator
import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.datasets
from typing import Optional, List, Any
from logger import Logger

class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = torch.autograd.Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length

T_co = TypeVar('T_co', covariant=True)

class UnevenDistributedSampler(torch.utils.data.distributed.Sampler[T_co]):
    def __init__(self, dataset: torch.utils.data.distributed.Dataset,
                 globalBatchSize: int, localBatch: int,
                 sampleOffset: int, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        self.dataset = dataset
        self.globalBatchSize = globalBatchSize
        self.localBatch = localBatch
        self.sampleOffset = sampleOffset
        if sampleOffset >= globalBatchSize or sampleOffset < 0:
            raise ValueError(
                "Invalid sampleOffset {}, sampleOffset should be in the interval"
                " [0, {}]".format(sampleOffset, globalBatchSize - 1))
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # `type:ignore` is required because Dataset cannot provide a default __len__
        # see NOTE in pytorch/torch/utils/data/sampler.py
        if self.drop_last:
            self.num_iter = math.floor(len(self.dataset) / self.globalBatchSize)  # type: ignore
        else:
            self.num_iter = math.ceil(len(self.dataset) / self.globalBatchSize)  # type: ignore
        self.total_size = self.globalBatchSize * self.num_iter
        self.num_samples = self.num_iter * self.localBatch
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        # remove tail of data to make it evenly divisible.
        indices = indices[:self.total_size]

        # subsample
        subsample = []
        for i in range(self.num_iter):
            start = i * self.globalBatchSize + self.sampleOffset
            end = i * self.globalBatchSize + self.sampleOffset + self.localBatch
            subsample.extend(indices[start:end])
        assert len(subsample) == self.num_samples
        return iter(subsample)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class VisionDataLoaderGenerator:
    @staticmethod
    def genDataLoader(jobInJsonStr, dataDir = None, workers = 4, syntheticDataLength = 1000):
        jobInJson = json.loads(jobInJsonStr)
        globalBatchSize = jobInJson["globalBatchSize"]
        sampleOffset = jobInJson["dataLoaderOffset"]
        localBatch = jobInJson["layers"][0]["config"][0]
        inputDim = jobInJson["layers"][0]["inputDim"]
        # {"globalBatchSize": 16,
        # "rank": 0,
        # "dataLoaderOffset": 2
        # "dataLoaderTargetTx": [{"name": "target_0", "dest": 1, "prop": {"xferSamples": 1}, "bytes": 56}]}, # TODO: implement in dump.
        # "layers": [{"id": 0,
        #             "name": "conv2d",
        #             "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        #             "prevLayers": [], "nextLayers": [1],
        #             "inputDim": [224, 224, 3], "outputDim": [224, 224, 64],
        #             "config": [16, 224, 224, 3, 64],
        #             "tensorTx": [{"name": "0_sample_1_0", "dest": 1, "prop": {"xferSamples": 1}, "bytes": 56}]},
        
        if dataDir is None:
            inputSize = (inputDim[2], inputDim[0], inputDim[1]) # (C, W, H)
            dataset = SyntheticDataset(inputSize, syntheticDataLength)
        else:
            normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            cropSize = (inputDim[1], inputDim[0]) # (H, W)
            dataset = torchvision.datasets.ImageFolder(
                dataDir,
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomResizedCrop(cropSize),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]))

        sampler = None
        if globalBatchSize > localBatch:
            sampler = UnevenDistributedSampler(dataset, globalBatchSize, localBatch,
                                                sampleOffset, shuffle=False, drop_last=True)
        if localBatch > 0:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=localBatch, shuffle=False,
                num_workers=workers, pin_memory=True, sampler=sampler, drop_last=True)
        else:
            loader = None

        # sampleIndices = list(range(sampleOffset, sampleOffset+localBatch))
        return loader #, sampleIndices, inputSize
    
    # @staticmethod
    # def shuffleTargetData(jobInJsonStr, target, commHandler):
    #     ####### Incomplete.

    #     # Send my portions.
    #     jobInJson = json.loads(jobInJsonStr)
    #     sendList = jobInJson["dataLoaderTargetTx"]
    #     sampleSplitSections = [txItem["prop"]["xferSamples"] for txItem in sendList]
    #     remainingSamples = target.shape[0] - sum(sampleSplitSections)
    #     sampleSplitSections.append(remainingSamples)
    #     splittedOutputs = torch.split(target, sampleSplitSections)

    #     for idx, item in enumerate(sendList):
    #         commHandler.sendAsync(splittedOutputs[idx], item["name"], item["dest"])

    #     output = splittedOutputs[-1].clone()

    #     # Receive

    #     return output

class TargetShuffler:
    def __init__(self, commHandler, rank: int, initialBatchSizes: List[int],
            finalSampleIndices: List[int], device: str):
        self.commHandler = commHandler
        self.rank = rank
        self.initialBatchSizes = initialBatchSizes
        self.finalSampleIndices = finalSampleIndices
        self.device = device
    
    def shuffle(self, targetsFromLoader):
        if targetsFromLoader == None:
            targetsFromLoader = torch.zeros(0, device=self.device)
        paddedTarget = []
        for i, initBatchSize in enumerate(self.initialBatchSizes):
            if i == self.rank:
                assert targetsFromLoader.size()[0] == initBatchSize
                paddedTarget.extend(targetsFromLoader.tolist())
            else:
                paddedTarget.extend([0] * initBatchSize)
        tsr = torch.tensor(paddedTarget, dtype=torch.long, device=torch.device(self.device))
        self.commHandler.allReduce(tsr, torch.distributed.ReduceOp, "all")
        allTargetList = torch.chunk(tsr, chunks=tsr.size()[0], dim=0)
        # Logger.log("allTargetList: %s" % str(allTargetList), flush=True)
        myTargetList = [allTargetList[idx] for idx in self.finalSampleIndices]
        # Logger.log("myTargetList: %s" % str(myTargetList), flush=True)
        if len(myTargetList) > 0:
            return torch.cat(myTargetList, dim=0)
        else:
            return None

#### all-gather method. doesn't work with the current NCCL implementation.
        # # tsrSizes = [[initBatchSize] for initBatchSize in self.initialBatchSizes]
        # # tsrList = [torch.zeros(*tsrSize, dtype=torch.LongTensor, device=self.device) for tsrSize in tsrSizes]
        # tsrList = [torch.zeros(initBatchSize, dtype=torch.int, device=torch.device(self.device)) for initBatchSize in self.initialBatchSizes]
        # Logger.log("TargetShuffler.shuffle is about to call allGather. tsrList: %s targetsFromLoader: %s" \
        #         % (str(tsrList), str(targetsFromLoader)), flush=True)
        # self.commHandler.allGather(tsrList, targetsFromLoader, 'all')
        # Logger.log("TargetShuffler.shuffle returned from allGather.", flush=True)
        # allTargets = torch.cat(tsrList, dim=0)
        # allTargetList = torch.chunk(allTargets, chunks=allTargets.size()[0], dim=0)
        # myTargetList = [allTargetList[idx] for idx in self.finalSampleIndices]
        # return torch.cat(myTargetList, dim=0)


class MockCommHandler:
    def __init__(self, conditionVariable = threading.Condition()):
        self.cv = conditionVariable
        self.sentTensors = {}
        
    def send(self, tensor: torch.Tensor, tensorName: str, dest: int):
        with self.cv:
            print("[MockCommHandler] sent %s to %d with %s" % (tensorName, dest, tensor.size()))
            if tensorName not in self.sentTensors:
                self.sentTensors[tensorName] = []
            self.sentTensors[tensorName].append(tensor.clone())
            self.cv.notifyAll()

    def sendAsync(self, tensor: torch.Tensor, tensorName: str, dest: int):
        return self.send(tensor, tensorName, dest)

    def recv(self, tensorName: str, src: int) -> torch.Tensor:
        # # hack for mock unittest..
        # tensorName = tensorName[:-5]
        # if tensorName not in self.sentTensors or len(self.sentTensors[tensorName]) == 0:
        #     tensorToReturn = torch.empty(0)
        # else:
        #     tensorToReturn = self.sentTensors[tensorName].pop()
        with self.cv:
            while (tensorName not in self.sentTensors) or len(self.sentTensors[tensorName]) == 0:
                self.cv.wait()
            # self.cv.wait_for((tensorName in self.sentTensors) and len(self.sentTensors[tensorName]) > 0)
            tensorToReturn = self.sentTensors[tensorName].pop()
            print("[MockCommHandler] recv %s to %d with %s" % (tensorName, src, tensorToReturn.size()))
        return tensorToReturn

class SendSamples(nn.Module):
    def __init__(self, sendList: list, commHandler):
        super(SendSamples, self).__init__()
        if len(sendList) == 0:
            raise Exception("sendList is empty")
        self.sendList = sendList
        self.commHandler = commHandler
    
    def forward(self, x):
        return SendSamplesFunc.apply(x, self.sendList, self.commHandler)

class SendSamplesFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sendList, commHandler):
        ctx.commHandler = commHandler
        ctx.sendList = sendList
        sampleSplitSections = [txItem["prop"]["xferSamples"] for txItem in sendList]
        remainingSamples = x.shape[0] - sum(sampleSplitSections)
        sampleSplitSections.append(remainingSamples)
        splittedOutputs = torch.split(x, sampleSplitSections)

        for idx, item in enumerate(sendList):
            # commHandler.send(splittedOutputs[idx].clone(), item["name"], item["dest"])
            commHandler.sendAsync(splittedOutputs[idx], item["name"], item["dest"])
        # commHandler.waitForAll()
        output = splittedOutputs[-1].clone()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        sendList = ctx.sendList
        # print("SendSamplesFunc backward grad_in: %s" % str(grad_output.size()))
        inputTensorList = []
        for item in sendList:
            additionalInput = ctx.commHandler.recv(item["name"]+"_back", item["dest"])
            inputTensorList.append(additionalInput)
        inputTensorList.append(grad_output)
        inputTensor = torch.cat(inputTensorList, 0)
        # print("                           grad_out: %s" % str(inputTensor.size()))
        return inputTensor, None, None

class ReceiveSamples(nn.Module):
    def __init__(self, recvList: list, commHandler):
        super(ReceiveSamples, self).__init__()
        if len(recvList) == 0:
            raise Exception("recvList is empty")
        self.recvList = recvList
        self.commHandler = commHandler
    
    def forward(self, x):
        return ReceiveSamplesFunc.apply(x, self.recvList, self.commHandler)

class ReceiveSamplesFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, recvList, commHandler):
        ctx.commHandler = commHandler
        ctx.recvList = recvList
        inputTensorList = []
        for rxItem in recvList:
            additionalInput = commHandler.recv(rxItem["name"], rxItem["src"])
            inputTensorList.append(additionalInput)
            print("[ReceiveSamplesFunc] ** additionalInput: %s, requires_grad? %s leaf? %s" % \
                (str(additionalInput.size()), str(additionalInput.requires_grad), str(additionalInput.is_leaf) ))
        
            # additionalInput.retain_grad()
        if x != None:
            inputTensorList.append(x)
        inputTensor = torch.cat(inputTensorList, 0)
        # inputTensor.retain_grad()
        print("[ReceiveSamplesFunc] ** output from ReceiveSamplesFunc.forward: %s, requires_grad? %s leaf? %s  x: %s %s" % \
                (str(inputTensor.size()), str(inputTensor.requires_grad), str(inputTensor.is_leaf), str(x.size() if x != None else None), str(x.requires_grad if x != None else None) ))
        return inputTensor

    @staticmethod
    def backward(ctx, grad_output):
        recvList = ctx.recvList
        sampleSplitSections = [item["prop"]["xferSamples"] for item in recvList]
        remainingSamples = grad_output.shape[0] - sum(sampleSplitSections)
        sampleSplitSections.append(remainingSamples)
        splittedOutputs = torch.split(grad_output, sampleSplitSections)

        for rxIdx, rxItem in enumerate(recvList):
            ctx.commHandler.sendAsync(splittedOutputs[rxIdx], rxItem["name"]+"_back", rxItem["src"])

        output = splittedOutputs[-1]
        return output, None, None

class RunnableModule(nn.Module):
    def __init__(self, specInJson, commHandler, device):
        super(RunnableModule, self).__init__()
        spec = json.loads(specInJson)
        self.rank = spec["rank"]
        self.globalBatchSize = spec["globalBatchSize"]
        self.moduleList = torch.nn.ModuleList()
        self.layersInJson = spec["layers"]
        self.initialBatchSize = self.layersInJson[0]["config"][0]
        self.commHandler = commHandler
        self.device = device
        self.leavesForBackward = []

        for ldsc in self.layersInJson:
            name = ldsc["name"]
            params = ldsc["params"]
            outputDim = ldsc["outputDim"]

            if name == "conv2d":
                module = nn.Conv2d(**params)
            elif name == "maxPool2d":
                module = nn.MaxPool2d(**params)
            elif name in ["avgPool2d", "adAvgPool2d"]:
                module = nn.AdaptiveAvgPool2d((outputDim[0], outputDim[1]))
            elif name == "linear":
                module = nn.Linear(**params)
            elif name in ["ReLU2d", "ReLU1d", "ReLU"]:
                module = nn.ReLU(params["inplace"])
            elif name == "flatten":
                # For VGG and Resnet, we can ignore this for now.
                # Maybe view(-1)?
                module = nn.Flatten(start_dim=1)
                Logger.log("[RunnableModule.__init__] %s layer is not implemented. Safe to ignore for VGG or Resnet" % name)
            elif name == "concat":
                Logger.log("[RunnableModule.__init__] %s layer is not implemented." % name, level=2)
                # Not used in for VGG and Resnet. Only inception needs this.

            # Handle sample transmissions.
            if ldsc["config"][0] > 0: # This rank has assigned samples for this layer.
                if "tensorRx" in ldsc: # receive parts of input.
                    Logger.log("[RunnableModule.__init__] recv tensor found for layer: %d" % ldsc["id"])
                    module = torch.nn.Sequential(ReceiveSamples(ldsc["tensorRx"], self.commHandler), module)
                if "tensorTx" in ldsc: # send parts of output.
                    Logger.log("[RunnableModule.__init__] send tensor found for layer: %d" % ldsc["id"])
                    module = torch.nn.Sequential(module, SendSamples(ldsc["tensorTx"], self.commHandler))

            self.moduleList.append(module)

    def forward(self, x):   
        # Assumes modules and layers are topologically sorted.

        def hook_wrapper(name):
            def hook(grad):
                print("hook_wrapper invoked! %s ; grad: %s" % (name, str(grad.size())) )
            return hook
        # x.requires_grad = True
        # x.register_hook(hook_wrapper("initial input's hook " + str(x.size()) ))
        tensorToReturn = None

        self.outputs = []
        for i, (module, ldsc) in enumerate(zip(self.moduleList, self.layersInJson)):
            if len(ldsc["prevLayers"]) == 0:
                inputTensor = x
            elif len(ldsc["prevLayers"]) == 1:
                inputTensor = self.outputs[ldsc["prevLayers"][0]]
            else:
                Logger.log("[RunnableModule::forward] more than 2 previous layers is not yet supported.", level=2, flush=True)
                raise Exception("[RunnableModule::forward] more than 2 previous layers is not yet supported.")
            
            if ldsc["config"][0] > 0: # This rank has assigned samples for this layer.
                Logger.log("[RunnableModule] forward inputTensor.size(): %s"%str(inputTensor.size() if inputTensor != None else None), level=0)
                if inputTensor == None:    
                    inputTensor = torch.empty(0, device=torch.device(self.device))
                
                output = module(inputTensor)
                Logger.log("[RunnableModule] Layer %d ==> output from running module: %s. requireGrad? %s" % (i, str(output.size()), str(output.requires_grad)), level=0)
                tensorToReturn = output
                runCriterionAndLoss = True
            else: # This rank doesn't participate for this layer.
                # outputDim = [0] + ldsc["outputDim"] if ldsc["outputDim"] is list else [ldsc["outputDim"]]
                # output = torch.empty(*tuple(outputDim), device=torch.device(self.device))
                # Logger.log("[RunnableModule] Layer %d ==> output from running module: %s" % (i, str(output.size())), level=0)
                ######### TODO: stash the current tensorToReturn. So that their backward can be invoked later.
                # - create a method remainingBackward(). which calls backward for all stashed tensors.
                if tensorToReturn != None:
                    self.leavesForBackward.append(tensorToReturn)

                output = None
                runCriterionAndLoss = False
                tensorToReturn = None
            # Logger.log("        ==> final output after sending out samples: %s" % (str(output.size())), level=0)

            if output != None:
                output.register_hook(hook_wrapper(str(ldsc["id"]) + " " + ldsc["name"] + str(output.size()) ))
            self.outputs.append(output)
        return tensorToReturn, runCriterionAndLoss

    def backwardRemainder(self):
        """ Run backward or any obsolete ramainders. """

        Logger.log("backwardRemainder starting. total leaves: %d" % len(self.leavesForBackward), level=0, flush=True)
        while len(self.leavesForBackward) > 0:
            leaf = self.leavesForBackward.pop()
            Logger.log("backwardRemainder: found a leaf (%s). %s" % (str(leaf), str(leaf.requires_grad)), level=0, flush=True)
            # assert leaf.size()[0] == 0
            if leaf.size()[0] != 0:
                Logger.log("leaf.size()[0] == 0 failed. leaf.size(): %s" % str(leaf.size()))
            leaf.backward(leaf) # gradient passed is dummy with 0 sample.
        

def test():
    # testExpandingGpuUsed()
    testUnevenSampler()

def testUnevenSampler():
    testSpecs = [
        r"""{"globalBatchSize": 16,
        "rank": 0,
        "dataLoaderOffset": 0,
        "layers": [{"id": 0,
                    "name": "conv2d",
                    "params": {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1},
                    "prevLayers": [], "nextLayers": [1],
                    "inputDim": [2, 2, 1], "outputDim": [2, 2, 2],
                    "config": [2, 2, 2, 1, 2]} 
                    ]}""",
        r"""{"globalBatchSize": 16,
        "rank": 1,
        "dataLoaderOffset": 2,
        "layers": [{"id": 0,
                    "name": "conv2d",
                    "params": {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1},
                    "prevLayers": [], "nextLayers": [1],
                    "inputDim": [2, 2, 1], "outputDim": [2, 2, 2],
                    "config": [14, 2, 2, 1, 2]} 
                    ]}"""]
    loaders = [VisionDataLoaderGenerator.genDataLoader(jobInJson, dataDir=None, workers=4, syntheticDataLength=1600)
                for jobInJson in testSpecs]
    print("dataset size: %d %d" % (len(loaders[0].dataset), len(loaders[0].sampler.dataset)))
    print("num_iter: %d num_samples: %d %d" % (loaders[0].sampler.num_iter, loaders[0].sampler.num_samples, len(loaders[0].sampler)))
    print("testUnevenSampler: %d, %d" % (len(loaders[0]), len(loaders[1])))
    assert len(loaders[0]) == 100
    assert len(loaders[1]) == 100
    assert len(loaders[0].sampler) == 200
    assert len(loaders[1].sampler) == 1400
    return

def testRunnableModuleBasic():
    testSpec = r"""{"globalBatchSize": 16,
                    "rank": 0,
                    "layers": [{"id": 0,
                                "name": "conv2d",
                                "params": {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1},
                                "prevLayers": [], "nextLayers": [1],
                                "inputDim": [2, 2, 1], "outputDim": [2, 2, 2],
                                "config": [2, 2, 2, 1, 2]},
                                {"id": 1,
                                "name": "ReLU2d",
                                "params": {"inplace": true, "kernel_size": 1, "stride": 1, "padding": 0},
                                "prevLayers": [0], "nextLayers": [],
                                "inputDim": [2, 2, 2], "outputDim": [2, 2, 2],
                                "config": [2, 2, 2, 2]}
                ]}"""
    comm = MockCommHandler()
    module = RunnableModule(testSpec, comm)
    module.initializeAtLocation(0, None)

    module2 = torch.nn.sequencial(nn.Conv2d(**{"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1}),
                                nn.ReLU(inplace=True))

    
    inputSize = (2, 1, 2, 2)
    inputTensor = torch.autograd.Variable(torch.rand(inputSize)).type(torch.FloatTensor)
    print("========== Foward starts ==========")
    output = module.forward(inputTensor)
    print("*** forward pass complete *** output:")
    print(output.size())
    output = torch.flatten(output, 1)
    # output = output.mean() #torch.nn.functional.log_softmax(output, dim=1)
    # criterion = nn.CrossEntropyLoss() #.cuda(self.device)
    # target = torch.autograd.Variable(torch.rand(15)).type(torch.LongTensor)
    # loss = criterion(output, target)
    # module.printAllGrads()
    print("========== Backward starts ==========")
    # loss.backward()
    (1 - output.mean()).backward()
    # module.printAllGrads()

    

def testExpandingGpuUsed():
    testSpec = r"""{"globalBatchSize": 16,
                    "rank": 0,
                    "layers": [{"id": 0,
                                "name": "conv2d",
                                "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                                "prevLayers": [], "nextLayers": [1],
                                "inputDim": [224, 224, 3], "outputDim": [224, 224, 64],
                                "config": [16, 224, 224, 3, 64],
                                "tensorTx": [{"name": "0_sample_1_0", "dest": 1, "prop": {"xferSamples": 1}, "bytes": 56}]},
                                {"id": 1,
                                "name": "ReLU2d",
                                "params": {"inplace": true, "kernel_size": 1, "stride": 1, "padding": 0},
                                "prevLayers": [0], "nextLayers": [2],
                                "inputDim": [224, 224, 64], "outputDim": [224, 224, 64],
                                "config": [15, 224, 224, 64]},
                                {"id": 2,
                                "name": "conv2d",
                                "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                                "prevLayers": [1], "nextLayers": [],
                                "inputDim": [224, 224, 64], "outputDim": [224, 224, 64],
                                "config": [15, 224, 224, 64, 64],
                                "tensorTx": [{"name": "2_sample_1_0", "dest": 1, "prop": {"xferSamples": 1}, "bytes": 56},
                                             {"name": "2_sample_1_1", "dest": 2, "prop": {"xferSamples": 1}, "bytes": 56}
                                            ]}
                        ]}"""
    comm = MockCommHandler()
    module = RunnableModule(testSpec, comm)
    module.initializeAtLocation(0, None)

    inputSize = (16, 3, 224, 224)
    inputTensor = torch.autograd.Variable(torch.rand(inputSize)).type(torch.FloatTensor)
    print("========== Foward starts ==========")
    output = module.forward(inputTensor)
    print("*** forward pass complete *** output:")
    print(output.size())
    output = torch.flatten(output, 1)
    # output = output.mean() #torch.nn.functional.log_softmax(output, dim=1)
    # criterion = nn.CrossEntropyLoss() #.cuda(self.device)
    # target = torch.autograd.Variable(torch.rand(15)).type(torch.LongTensor)
    # loss = criterion(output, target)
    # module.printAllGrads()
    print("========== Backward starts ==========")
    # loss.backward()
    (1 - output.mean()).backward()
    # module.printAllGrads()
 
if __name__ == "__main__":
    test()