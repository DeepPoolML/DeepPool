import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


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



def vgg16(splitCount=1, pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    # model = VGG(make_layers(cfg['D'], splitCount), split_count=splitCount, **kwargs)
    model = VGG16(split_count=splitCount, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model