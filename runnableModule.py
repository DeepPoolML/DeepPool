import json
import os
import math
import threading
from typing import TypeVar, Optional, Iterator
import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.datasets

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
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=localBatch, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=sampler, drop_last=True)
        return loader

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
            commHandler.sendAsync(splittedOutputs[idx], item["name"], item["dest"])

        output = splittedOutputs[-1].clone()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        sendList = ctx.sendList
        print("SendSamplesFunc backward grad_in: %s" % str(grad_output.size()))
        inputTensorList = []
        for item in sendList:
            additionalInput = ctx.commHandler.recv(item["name"]+"_back", item["dest"])
            inputTensorList.append(additionalInput)
        inputTensorList.append(grad_output)
        inputTensor = torch.cat(inputTensorList, 0)
        print("                           grad_out: %s" % str(inputTensor.size()))
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
        inputTensorList.append(x)
        inputTensor = torch.cat(inputTensorList, 0)
        # print("** output from ReceiveSamplesFunc.forward: %s" % str(inputTensor.size()))
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
    def __init__(self, specInJson, commHandler):
        super(RunnableModule, self).__init__()
        spec = json.loads(specInJson)
        self.rank = spec["rank"]
        self.globalBatchSize = spec["globalBatchSize"]
        self.moduleList = torch.nn.ModuleList()
        self.layersInJson = spec["layers"]
        self.initialBatchSize = self.layersInJson[0]["config"][0]
        self.commHandler = commHandler

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
                print("%s layer is not implemented. Safe to ignore for VGG or Resnet" % name)
            elif name == "concat":
                print("%s layer is not implemented." % name)
                # Not used in for VGG and Resnet. Only inception needs this.

            # Handle sample transmissions.
            if ldsc["config"][0] > 0: # This rank has assigned samples for this layer.
                if "tensorRx" in ldsc: # receive parts of input.
                    print("recv tensor found for layer: %d" % ldsc["id"])
                    module = torch.nn.Sequential(ReceiveSamples(ldsc["tensorRx"], self.commHandler), module)
                if "tensorTx" in ldsc: # send parts of output.
                    print("send tensor found for layer: %d" % ldsc["id"])
                    module = torch.nn.Sequential(module, SendSamples(ldsc["tensorTx"], self.commHandler))

            self.moduleList.append(module)
            
    def initializeAtLocation(self, device, controller):
        self.device = device
        self.controller = controller

    def forward(self, x):   
        # Assumes modules and layers are topologically sorted.

        # def hook_wrapper(name):
        #     def hook(grad):
        #         print("hook_wrapper invoked! %s ; grad: %s" % (name, str(grad.size())) )
        #     return hook
        # x.requires_grad = True
        # x.register_hook(hook_wrapper("initial input's hook " + str(x.size()) ))

        self.outputs = []
        for i, (module, ldsc) in enumerate(zip(self.moduleList, self.layersInJson)):
            if len(ldsc["prevLayers"]) == 0:
                inputTensor = x
            elif len(ldsc["prevLayers"]) == 1:
                inputTensor = self.outputs[ldsc["prevLayers"][0]]
            else:
                print("[RunnableModule::forward] more than 2 previous layers is not yet supported.")
                raise Exception("[RunnableModule::forward] more than 2 previous layers is not yet supported.")
            
            if ldsc["config"][0] > 0: # This rank has assigned samples for this layer.
                # if "tensorRx" in ldsc: # receive parts of input.
                #     inputTensorList = [inputTensor]
                #     for rxItem in ldsc["tensorRx"]:
                #         additionalInput = self.commHandler.recv(rxItem["name"], rxItem["src"])
                        
                #         ## For backprop
                #         def rx_hook_wrapper(tensorRxList):
                #             def hook(grad):
                #                 print("hook_wrapper invoked!")
                #                 sampleSplitSections = [item["prop"]["xferSamples"] for item in tensorRxList]
                #                 remainingSamples = grad.shape[0] - sum(sampleSplitSections)
                #                 sampleSplitSections.append(remainingSamples)
                #                 splittedOutputs = torch.split(grad, sampleSplitSections)
                #                 for idx, item in enumerate(tensorRxList):
                #                     self.commHandler.sendAsync(splittedOutputs[idx], item["name"]+"_back", item["src"])
                #                 output = splittedOutputs[-1].clone()
                #                 return output
                #             return hook
                #         additionalInput.register_hook(rx_hook_wrapper(ldsc["tensorRx"]))
                #         ## End of for backprop

                #         inputTensorList.append(additionalInput)
                #     inputTensor = torch.cat(inputTensorList, 0)

                outputRaw = module(inputTensor)
                print("Layer %d ==> output from running module: %s" % (i, str(outputRaw.size())))

                # if "tensorTx" in ldsc: # send parts of output.
                #     sampleSplitSections = [txItem["prop"]["xferSamples"] for txItem in ldsc["tensorTx"]]
                #     remainingSamples = outputRaw.shape[0] - sum(sampleSplitSections)
                #     sampleSplitSections.append(remainingSamples)
                #     splittedOutputs = torch.split(outputRaw, sampleSplitSections)

                #     # output.register_hook(hook_wrapper_cat_recv)

                #     for txIdx, txItem in enumerate(ldsc["tensorTx"]):
                #         self.commHandler.sendAsync(splittedOutputs[txIdx], txItem["name"], txItem["dest"])
                #     output = splittedOutputs[-1].clone()

                #     ## For backprop
                #     def hook_wrapper_recv(tensorTxList):
                #         def hook(grad):
                #             print("hook_wrapper_recv invoked! grad_in: %s" % str(grad.size()))
                #             inputTensorList = []
                #             for item in tensorTxList:
                #                 additionalInput = self.commHandler.recv(item["name"]+"_back", item["dest"])
                #                 inputTensorList.append(additionalInput)
                #             inputTensorList.append(grad)
                #             inputTensor = torch.cat(inputTensorList, 0)
                #             print("                           grad_out: %s" % str(inputTensor.size()))
                #             return inputTensor

                #         return hook
                #     output.register_hook(hook_wrapper_recv(ldsc["tensorTx"]) )
                #     print("hook registerd")
                #     ## End of for backprop
                # else:
                #     output = outputRaw
                output = outputRaw
                tensorToReturn = output
                runCriterionAndLoss = True

            else: # This rank doesn't participate for this layer.
                outputDim = [0] + ldsc["outputDim"] if ldsc["outputDim"] is list else [ldsc["outputDim"]]
                output = torch.empty(outputDim)
                runCriterionAndLoss = False
            print("        ==> final output after sending out samples: %s" % (str(output.size())))

            # output.register_hook(hook_wrapper(str(ldsc["id"]) + " " + ldsc["name"] + str(output.size()) ))
            self.outputs.append(output)
        return tensorToReturn, runCriterionAndLoss

    # def backward(self, grad):
    #     # Assumes modules and layers are topologically sorted.
    #     self.gradients = [None for i in range(len(self.moduleList))]
    #     for module, ldsc in zip(reversed(self.moduleList), reversed(self.layersInJson)):
    #         if len(ldsc["nextLayers"]) == 0:
    #             gradIn = grad
    #         elif len(ldsc["prevLayers"]) == 1:
    #             gradIn = self.gradients[ldsc["nextLayers"][0]]
    #         else:
    #             print("[RunnableModule::backward] more than 2 previous layers is not yet supported.")
    #             raise Exception("[RunnableModule::backward] more than 2 previous layers is not yet supported.")
            
    #         if ldsc["config"][0] > 0: # This rank has assigned samples for this layer.
    #             if "tensorTx" in ldsc: # receive parts of input.
    #                 sampleSplitSections = [txItem["prop"]["xferSamples"] for txItem in ldsc["tensorTx"]]
    #                 remainingSamples = outputRaw.shape[0] - sum(sampleSplitSections)
    #                 sampleSplitSections.append(remainingSamples)
    #                 splittedOutputs = torch.split(outputRaw, sampleSplitSections)

    #                 # output.register_hook(hook_wrapper_cat_recv)

    #                 for txIdx, txItem in enumerate(ldsc["tensorTx"]):
    #                     self.commHandler.sendAsync(splittedOutputs[txIdx], txItem["name"], txItem["dest"])
    #                 # output = splittedOutputs[-1].clone()
    #                 output = splittedOutputs[-1]

    #             outputRaw = module.backward(gradIn)
    #             print("Layer %d ==> output from backprop module: %s" % (ldsc["id"], str(outputRaw.size())))

    #             if "tensorRx" in ldsc: # previously receive parts of input.
    #                 sampleSplitSections = [txItem["prop"]["xferSamples"] for txItem in ldsc["tensorRx"]]
    #                 remainingSamples = outputRaw.shape[0] - sum(sampleSplitSections)
    #                 sampleSplitSections.append(remainingSamples)
    #                 splittedOutputs = torch.split(outputRaw, sampleSplitSections)

    #                 inputTensorList = [inputTensor]
    #                 for rxIdx, rxItem in enumerate(ldsc["tensorRx"]):
    #                     self.commHandler.sendAsync(splittedOutputs[rxIdx], rxItem["name"]+"_back", rxItem["src"])

    #                 output = splittedOutputs[-1]
    #                 inputTensor = torch.cat(inputTensorList, 1)
    #             else:
    #                 output = outputRaw

    #         else: # This rank doesn't participate for this layer.
    #             outputDim = [0] + ldsc["outputDim"]
    #             output = torch.empty(outputDim)
    #         print("        ==> final output after sending out samples: %s" % (str(output.size())))
    #         output.register_hook(lambda grad: print(grad))
    #         def backwardHook(module, grad_input, grad_output):
    #             print("hoooked!!!!!!!!!!!!!!!!!!!!!")
    #             return
    #         module.register_backward_hook(backwardHook)
    #         print("hook registerd")
    #         self.outputs.append(output)
    #     self.outputs[-1].register_hook(lambda grad: print(grad))
    #     return self.outputs[-1]

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