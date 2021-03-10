import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import threading
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from parallelizationPlanner import GpuProfiler
from cluster import ClusterClient
from jobDescription import TrainingJob
from runnableModule import RunnableModule
from runnableModule import MockCommHandler
from runnableModule import VisionDataLoaderGenerator

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
            cs.ReLU(True),
            nn.Dropout(),
            cs.Linear(int(4096 / split1side), int(4096/split1side)),
            cs.ReLU(True),
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
            layers += [conv2d, cs.ReLU(inplace=True)] #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
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
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(4096), int(4096 / split_count)),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(4096 / split_count), int(num_classes)),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
            if cfg['D'][i] != 'M':
                x = torch.nn.functional.relu(x, inplace=True)
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

profiler = GpuProfiler("cuda")
profiler.loadProfile()
cs = CostSim(profiler, verbose=True)
# model = vgg16(pretrained=False)
model = vgg11()
# model = resnet34()
cs.printAllLayers()
cs.computeInputDimensions((224,224,3))
job = cs.searchBestSplits(4, 16)


jobInJson = job.dumpInJSON()
job2 = TrainingJob("test", None, None, 0, "")
job2.loadJSON(jobInJson)
assert(jobInJson == job2.dumpInJSON())
print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
print(jobInJson)

locations = ["a", "b", "c", "d"]


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


# cc = ClusterClient("172.31.70.173", 12345)
# cc.submitTrainingJob(jobInJson)

profiler.saveProfile()

