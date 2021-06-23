import torch
from torch import Tensor
import torch.nn as nn
# from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
# For cost estimator
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from costSimulator import CostSim
from costSimulator import GpuProfiler

class TestModelBasic(nn.Module):
    def __init__(self, n=2, k=2, num_classes=1000, init_weights=True):
        super(TestModelBasic, self).__init__()
        self.start = cs.Conv2d(3, 8, kernel_size=3, padding=1)
        self.middle = nn.ModuleList()
        global startLayer
        startLayer = cs.layers[-1]
        lastLayers = []
        for j in range(k):
            prevLayer = startLayer
            for i in range(n):
                self.middle.append(cs.Conv2d(8, 8, kernel_size=3, padding=1, custom_previous_layers=[prevLayer]))
                prevLayer = cs.layers[-1]
            lastLayers.append(prevLayer)
        
        self.relu = cs.ReLU(inplace=True, custom_previous_layers=lastLayers)
        cs.Flatten()
        self.classifier = nn.Sequential(
            cs.Linear(int(8 * 56 * 56), int(4096)),
            cs.ReLU(True),
            nn.Dropout(),
            cs.Linear(int(4096), int(4096)),
            cs.ReLU(True),
            nn.Dropout(),
            cs.Linear(int(4096), int(num_classes)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TestModelHeavyHead(nn.Module):
    def __init__(self, k=2, num_classes=1000, init_weights=True):
        super(TestModelHeavyHead, self).__init__()
        self.start = cs.Conv2d(3, 32, kernel_size=3, padding=1)
        self.middle = nn.ModuleList()
        global startLayer
        startLayer = cs.layers[-1]
        lastLayers = []
        for j in range(k):
            prevLayer = startLayer
            self.middle.append(cs.Conv2d(32, 32, kernel_size=3, padding=1, custom_previous_layers=[prevLayer]))
            prevLayer = cs.layers[-1]
            self.middle.append(cs.Conv2d(32, 1, kernel_size=3, padding=1, custom_previous_layers=[prevLayer]))
            prevLayer = cs.layers[-1]
            self.middle.append(cs.Conv2d(1, 1, kernel_size=1, padding=0, custom_previous_layers=[prevLayer]))
            prevLayer = cs.layers[-1]
            lastLayers.append(prevLayer)
        
        self.relu = cs.ReLU(inplace=True, custom_previous_layers=lastLayers)
        cs.Flatten()
        self.classifier = nn.Sequential(
            cs.Linear(int(8 * 56 * 56), int(4096)),
            cs.ReLU(True),
            nn.Dropout(),
            cs.Linear(int(4096), int(4096)),
            cs.ReLU(True),
            nn.Dropout(),
            cs.Linear(int(4096), int(num_classes)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



# n=3
# k=3
# globalBatch = 16
# totalGpus = 4

profiler = GpuProfiler("cuda")
profiler.loadProfile()
cs = CostSim(profiler)
# model = TestModelBasic(n=3, k=3)

n=3
k=2
globalBatch = 16
totalGpus = 4

model = TestModelHeavyHead(k=2)
cs.printAllLayers()
cs.computeInputDimensions((3,224,224))

cs.displayConfigSizeGrowth(cs.layers[0],64)

print("\nMulti-chain strategy search test. (totalGpus=%d lengthOfChain=%d chainCount=%d globalBatch=%d)" % (totalGpus, n, k, globalBatch))
# (endLayer, configToTimeDict, t) = cs.searchMultiChain(startLayer, (1, 56, 56, 8, 8), globalBatch, totalGpus)

bestMultiChainTime = 9999999999
bestJiaTime = 9999999999
for startConfig in cs.listConfigOptions(startLayer, globalBatch, totalGpus):
    print(startConfig)
    startGpuTime = cs.benchGpuTime(startLayer, startConfig)

    (endLayer, configToTimeDict, t) = cs.searchMultiChain(startLayer, startConfig, globalBatch, totalGpus)
    (jiaEndLayer, jiaConfigToTimeDict, jiaT) = cs.runMultiChainZhihao(startLayer, startConfig, globalBatch, totalGpus)
    for config in configToTimeDict:
        multiChainTime = configToTimeDict[config][0] + startGpuTime
        jiaTime = jiaConfigToTimeDict[config][0] + startGpuTime
        print(" lastConfig: %20s,   multi-chain algo: %7.1f ms   Zhihao's time: %7.1f ms" %
             (str(config), multiChainTime, jiaTime))
        
        bestMultiChainTime = min(bestMultiChainTime, multiChainTime)
        bestJiaTime = min(bestJiaTime, jiaTime)

    bestConfigToTimeDict = (999999999, None)
    bestEndConfig = None
    for config in configToTimeDict:
        # print(" lastConfig: %20s, time: %.2f" % (str(config), configToTimeDict[config][0]))
        if bestConfigToTimeDict[0] > configToTimeDict[config][0]:
            bestConfigToTimeDict = configToTimeDict[config]
            bestEndConfig = config
    cs.displayMultiChainResult(endLayer, bestEndConfig, t, bestConfigToTimeDict[1])
    # joiningCombined = ((3, (16, 56, 56, 8, 8)), (3, (16, 56, 56, 8, 8)), (3, (16, 28, 28, 8, 8)))
    # cs.displayMultiChainResult(endLayer, bestEndConfig, t, (joiningCombined, 0))

print("Best multi-chain: %.2f  best jia: %.2f" % (bestMultiChainTime, bestJiaTime) )

naiveTotalTime = 0
for i in range(startLayer.id, endLayer.id + 1):
    layer = cs.layers[i]
    config = list(cs.getInitialConfig(layer, globalBatch))
    config[0] = int(config[0] / totalGpus)
    naiveTotalTime += cs.benchGpuTime(layer, tuple(config))
print(" naive DP time w/o all-reduce time: %.2f" % naiveTotalTime)
# cs.searchBestSplits(16, 16)
# cs.searchBestSplits(2, 2)
profiler.saveProfile()

