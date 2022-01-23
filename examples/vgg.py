import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import threading
import os, sys
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from parallelizationPlanner import GpuProfiler
from clusterClient import ClusterClient
from jobDescription import TrainingJob

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, split_count=1, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        split1side = int(split_count**0.5)
        inChannels = int(512 / split1side)
        # print("linear intake features: %d"%int(512 * 7 * 7 / split1side))
        cs.Flatten()
        self.classifier = nn.Sequential(
            cs.Linear(int(inChannels * 7 * 7), int(4096/split1side)),
            cs.ReLU(False),
            nn.Dropout(),
            cs.Linear(int(4096 / split1side), int(4096/split1side)),
            cs.ReLU(False),
            nn.Dropout(),
            cs.Linear(int(4096 / split1side), int(num_classes)),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, split_count=1, batch_norm=False):
    layers = []
    in_channels = 3
    i = 0
    for v in cfg:
        if v == 'M':
            layers += [cs.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # if i == len(cfg) - 2: # last convolutional layer.
                # print("lastConv2d out channel: %d"%v)
            conv2d = cs.Conv2d(in_channels, int(v / split_count), kernel_size=3, padding=1)
            # if batch_norm:
            #     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            # else:
            layers += [conv2d, cs.ReLU(inplace=False)] #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            in_channels = v
        i += 1
    return nn.Sequential(*layers)


        # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        # self.conv1 = nn.Conv2d(  3,  64 / split_count, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d( 64,  64 / split_count, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d( 64, 128 / split_count, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 128 / split_count, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

# layersToSplit = [True, True, 'M', 128, True, 'M', 256, 256, 256, 'M', 512, 512, True, 'M', 512, 512, True, 'M'],
# layersToSplit = [True, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False]
layersToSplit = [True, True, False, True, True, False, True, True, True, False, True, True, True, False, True, True, True, False]


class VGG16(nn.Module):
    def __init__(self, split_count=1, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__()
        self.split_count = split_count
        self.features = nn.ModuleList([])
        in_channels = 3
        # for v in cfg['D']:
        # print(len(cfg['D']))
        # print(len(layersToSplit))
        for i in range(len(cfg['D'])):
            v = cfg['D'][i]
            if v == 'M':
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.features.append(nn.Conv2d(in_channels, int(v / split_count) if layersToSplit[i] else v, kernel_size=3, padding=1))
                in_channels = v

        # split1side = int(split_count**0.5)
        # inChannels = int(512 / split1side)
        # print("linear intake features: %d"%int(512 * 7 * 7 / split1side))
        self.classifier = nn.Sequential(
            nn.Linear(int(512 * 7 * 7 / split_count), int(4096)),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(int(4096), int(4096 / split_count)),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(int(4096 / split_count), int(num_classes)),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
            if cfg['D'][i] != 'M':
                x = torch.nn.functional.relu(x, inplace=False)
                if layersToSplit[i] and i < len(cfg['D']) - 2 and self.split_count > 1:
                    # x = torch.repeat_interleave(x, self.split_count, dim=1)
                    x = x.repeat(1, self.split_count, 1, 1)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(splitCount=1, pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], splitCount), split_count=splitCount, **kwargs)
    # model = VGG16(split_count=splitCount, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def genTestJob(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False):
    profiler = GpuProfiler("cuda")
    profiler.loadProfile()
    global cs
    cs = CostSim(profiler, netBw=1.25E5, verbose=True)
    model = vgg16(pretrained=False)
    cs.printAllLayers()
    cs.computeInputDimensions((3,224,224))
    job = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline)
    return job

"""
def testRunOnCPU():
    # optimizer is not yet implemented.
    def train(loader, model, optimizer = None, criterion = nn.CrossEntropyLoss(), device="cpu"):
        model.to(device)
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            # optimizer.zero_grad()

            print("forward pass is starting.. data: %s" % str(data.size()))
            output, runCriterionAndLoss = model(data)
            # output = torch.flatten(output, 1)
            if runCriterionAndLoss:
                output = F.log_softmax(output, dim=1)
                
                # Hack to match target's sample count with the output at this node.
                if output.size()[0] != target.size()[0]:
                    target = torch.repeat_interleave(target, int(1 + output.size()[0] / target.size()[0]), dim=0)
                    target = target.narrow(0, 0, output.size()[0])

                loss = criterion(output, target)
                print("backward pass is starting")
                loss.backward()
            else:
                output.backward(output) # gradient passed is dummy.
            
            # finish after 1st iteration.
            return
            # optimizer.step()

    comm = MockCommHandler()
    threadList = []
    ## For now just use all gpus.
    for rank, location in enumerate(locations):
        moduleDesc = job.dumpSingleRunnableModule(rank)
        print("%s ==> \n %s" % (location, moduleDesc))
        
        module = RunnableModule(moduleDesc, comm)
        loader = VisionDataLoaderGenerator.genDataLoader(
            moduleDesc, syntheticDataLength=1600)
        train_thread = threading.Thread(name='train_rank%d'%rank, target=train, args=(loader, module,))
        # train_thread = threading.Thread(name='train_rank%d'%rank, target=train, args=(loader, model,))
        threadList.append(train_thread)

    for thread in threadList:
        thread.start()
    for thread in threadList:
        thread.join()
"""

def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, use_be=False):
    profiler = GpuProfiler("cuda")
    profiler.loadProfile()
    global cs
    cs = CostSim(profiler, netBw=netBw, verbose=False, gpuProfileLoc="profile/A100_vgg.prof")
    model = vgg16(pretrained=False)
    
    saveWholeModel = True
    if saveWholeModel:
        fakeInput = torch.zeros(cs.layers[0].inputDim)
        traced = torch.jit.script(model, fakeInput)
        saveLocation = "modules/vgg16.pt"
        torch.jit.save(traced, saveLocation)
        print("Saved whole model to %s" % saveLocation)

    # model = vgg11()
    # model = resnet34()
    cs.printAllLayers(slient=True)
    cs.computeInputDimensions((3,224,224))
    # job = cs.searchBestSplits(4, 16, dataParallelBaseline=True)
    # job = cs.searchBestSplits(4, 16)
    # job = cs.searchBestSplits(gpuCount, globalBatch, dataParallelBaseline=True)
    job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    print("Searching for parallelization strategy is completed.\n")

    jobInJson = job.dumpInJSON()
    # print("\n*** General description ***\n")
    # print(jobInJson)
    # print("\n\n")

    # print("\n*** GPU specific description for rank:0 ***\n")
    # print(job.dumpSingleRunnableModule(0))
    # print("\n\n")

    # for rank in range(4):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    job2 = TrainingJob("test", None, None, 0, 0, "")
    job2.loadJSON(jobInJson)
    assert(jobInJson == job2.dumpInJSON())
    print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    # print(jobInJson)

    # testRunOnCPU()

    profiler.saveProfile()

    if not spatialSplit:
        cc = ClusterClient()
        jobName = "vgg16_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        jobName += "_BE" if use_be else ""
        cc.submitTrainingJob(jobName, jobInJson, use_be)

    if simResultFilename != None:
        f = open(simResultFilename, "a")
        f.write("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, gpuCount, iterMs, gpuMs))
        f.close()

        if gpuCount == 8:
            f = open(simResultFilename, "r")
            print(f.read())
            f.close()



# def runAllConfigs(clusterType: str):
#     if clusterType == "V100":
#         netBw = 22937
#     elif clusterType == "A100":
#         netBw = 2.66E5
#     else:
#         print("Wrong cluster type. Put either V100 or A100")

#     gpuCounts = [1, 2, 4, 8]
#     # gpuCounts = [1, 2, 4]
#     # globalBatchSize = 16
#     globalBatchSize = 8
#     # limitAndBaseline = [(1.5, False, False), (2.0, False, False), (2.5, False, False), (99, False, False), (2.0, True, False)]
#     limitAndBaseline = [(99, False, True)]
#     for lim, baseline, spatialSplit in limitAndBaseline:
#         f = open("vgg16_%s_lim%2.1f_sim.data" % ("DP" if baseline else "MP", lim), "w")
#         f.write("#batch GPUs IterMs  GpuMs\n")
#         f.close()

#         for gpuCount in gpuCounts:
#             preSize = os.stat('runtimeResult.data').st_size
#             main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit)
#             # check exp finished.
#             print("runtimeResult.data's original size: ", preSize)
#             while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
#                 time.sleep(10)
#             print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
        
#         if not spatialSplit:
#             fw = open("vgg16_%s_lim%2.1f_run.data" % ("DP" if baseline else "MP", lim), "w")
#             fr = open('runtimeResult.data', "r")
#             fw.write("#batch GPUs IterMs  GpuMs\n")
#             fw.write(fr.read())
#             fw.close()
#             fr.close()

#         fr = open('runtimeResult.data', "w")
#         fr.close()
        
def runAllConfigs(modelName: str, clusterType: str):
    if clusterType == "V100":
        netBw = 22937
    elif clusterType == "A100":
        netBw = 2.66E5
    else:
        print("Wrong cluster type. Put either V100 or A100")

    gpuCounts = [1, 2, 4, 8]
    # gpuCounts = [1, 2, 4]
    globalBatchSize = 16
    # globalBatchSize = 8
    # limitAndBaseline = [(2.0, True, False), (99, False, False), (4.0, False, False), (2.5, False, False)]
    # limitAndBaseline = [(99, False, True)]
    limitAndBaseline = [(99, True, False)]
    # limitAndBaseline = []
    for lim, baseline, spatialSplit in limitAndBaseline:
        simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim)
        f = open(simResultFilename, "w")
        f.write("#batch GPUs IterMs  GpuMs\n")
        f.close()

        for gpuCount in gpuCounts:
            preSize = os.stat('runtimeResult.data').st_size
            main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit, simResultFilename=simResultFilename)
            # check exp finished.
            print("runtimeResult.data's original size: ", preSize)
            while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
                time.sleep(10)
            print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
        
        if not spatialSplit:
            fw = open("%s_%s_b%d_lim%2.1f_run.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim), "w")
            fr = open('runtimeResult.data', "r")
            fw.write("#batch GPUs IterMs  GpuMs\n")
            fw.write(fr.read())
            fw.close()
            fr.close()

        fr = open('runtimeResult.data', "w")
        fr.close()

    # #################################
    # ## Profiling by batch size.
    # #################################
    # globalBatchSizes = [1,2,4,8,16,32,64,128]
    # lim, baseline, spatialSplit = (2.0, True, False)
    # simResultFilename = "%s_%s_varyBatch_sim.data" % (modelName, "DP" if baseline else "MP")
    # f = open(simResultFilename, "w")
    # f.write("#batch GPUs IterMs  GpuMs\n")
    # f.close()

    # # for gpuCount in gpuCounts:
    # gpuCount = 1
    # for globalBatchSize in globalBatchSizes:
    #     preSize = os.stat('runtimeResult.data').st_size
    #     main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit, simResultFilename=simResultFilename)
    #     # check exp finished.
    #     print("runtimeResult.data's original size: ", preSize)
    #     while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
    #         time.sleep(10)
    #     print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
    
    # if not spatialSplit:
    #     fw = open("%s_%s_varyBatch_run.data" % (modelName, "DP" if baseline else "MP"), "w")
    #     fr = open('runtimeResult.data', "r")
    #     fw.write("#batch GPUs IterMs  GpuMs\n")
    #     fw.write(fr.read())
    #     fw.close()
    #     fr.close()

    # fr = open('runtimeResult.data', "w")
    # fr.close()

def runStrongScalingBench(modelName='vgg16'):
    profiler = GpuProfiler("cuda")
    global cs
    netBw = 2.66E5
    cs = CostSim(profiler, netBw=netBw, verbose=False)
    inputSize = (3,224,224)
    if modelName == 'vgg11':
        model = vgg11(pretrained=False)
    elif modelName == 'vgg16':
        model = vgg16(pretrained=False)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", pytorch_total_params)
    
    print("Model: ", modelName)
    print("BatchSize  iterMs    fpMs    bpMs")
    for batchSize in [2 ** exp for exp in range(1, 9)]:
        iterTime, fpTime, bpTime = profiler.benchModel(model, inputSize, batchSize)
        print(" %8d  %6.1f  %6.1f  %6.1f" %
            (batchSize, iterTime / 1000, fpTime / 10000, bpTime / 1000))


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True)
    elif len(sys.argv) >= 4:
        use_be = len(sys.argv) > 4 and int(sys.argv[4]) == 1
        if sys.argv[3] == "DP":
            main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True, use_be=use_be)
        else:
            main(int(sys.argv[1]), int(sys.argv[2]), amplificationLimit=float(sys.argv[3]), use_be=use_be)
    elif len(sys.argv) == 2:
        print("Run all configs")
        runAllConfigs("vgg16", sys.argv[1])
    elif len(sys.argv) == 1:
        for modelName in ['vgg11', 'vgg16']:
            runStrongScalingBench(modelName)
    else:
        print("Wrong number of arguments.\nUsage: ")
