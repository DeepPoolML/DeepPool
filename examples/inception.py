from collections import namedtuple
import warnings
import torch
import time
from torch import nn, Tensor
import torch.nn.functional as F
# from .utils import load_state_dict_from_url
from typing import Callable, Any, Optional, Tuple, List
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from gpuProfiler import GpuProfiler
from clusterClient import ClusterClient
from jobDescription import TrainingJob

__all__ = ['Inception3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


def inception_v3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> "Inception3":
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    # if pretrained:
    #     if 'transform_input' not in kwargs:
    #         kwargs['transform_input'] = True
    #     if 'aux_logits' in kwargs:
    #         original_aux_logits = kwargs['aux_logits']
    #         kwargs['aux_logits'] = True
    #     else:
    #         original_aux_logits = True
    #     kwargs['init_weights'] = False  # we are loading weights from a pretrained model
    #     model = Inception3(**kwargs)
    #     state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    #     if not original_aux_logits:
    #         model.aux_logits = False
    #         model.AuxLogits = None
    #     return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None
    ) -> None:
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = cs.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = cs.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        layerBeforeAux = cs.layers[-1]
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768, custom_previous_layer=layerBeforeAux)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = cs.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        cs.Flatten()
        self.fc = cs.Linear(2048, num_classes)
        # if init_weights:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #             import scipy.stats as stats
        #             stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #             X = stats.truncnorm(-2, 2, scale=stddev)
        #             values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
        #             values = values.view(m.weight.size())
        #             with torch.no_grad():
        #                 m.weight.copy_(values)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        prevLayer = cs.layers[-1]
        outputLayers = []
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1, custom_previous_layers=[prevLayer])
        outputLayers.append(cs.layers[-1])

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)
        outputLayers.append(cs.layers[-1])

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
        outputLayers.append(cs.layers[-1])

        self.branch_pool_1 = cs.AvgPool2d(kernel_size=3, stride=1, padding=1, custom_previous_layers=[prevLayer])
        self.branch_pool_2 = conv_block(in_channels, pool_features, kernel_size=1)
        outputLayers.append(cs.layers[-1])
        cs.Concat(custom_previous_layers=outputLayers)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        prevLayer = cs.layers[-1]
        outputLayers = []
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2, custom_previous_layers=[prevLayer])
        outputLayers.append(cs.layers[-1])

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)
        outputLayers.append(cs.layers[-1])

        self.branch_pool_1 = cs.MaxPool2d(kernel_size=3, stride=2, custom_previous_layers=[prevLayer])
        outputLayers.append(cs.layers[-1])
        cs.Concat(custom_previous_layers=outputLayers)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool_1(x)
        # F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        prevLayer = cs.layers[-1]
        outputLayers = []
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1, custom_previous_layers=[prevLayer])
        outputLayers.append(cs.layers[-1])

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        outputLayers.append(cs.layers[-1])

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        outputLayers.append(cs.layers[-1])

        self.branch_pool_1 = cs.AvgPool2d(kernel_size=3, stride=1, padding=1, custom_previous_layers=[prevLayer])
        self.branch_pool_2 = conv_block(in_channels, 192, kernel_size=1)
        outputLayers.append(cs.layers[-1])
        cs.Concat(custom_previous_layers=outputLayers)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # branch_pool = self.branch_pool(branch_pool)
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        custom_previous_layer = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        if custom_previous_layer == None:
            prevLayer = cs.layers[-1]
        else:
            prevLayer = custom_previous_layer
        outputLayers = []

        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        outputLayers.append(cs.layers[-1])

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)
        outputLayers.append(cs.layers[-1])

        self.branch_pool_1 = cs.MaxPool2d(kernel_size=3, stride=2, custom_previous_layers=[prevLayer])
        outputLayers.append(cs.layers[-1])
        cs.Concat(custom_previous_layers=outputLayers)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        # branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool_1(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        prevLayer = cs.layers[-1]
        outputLayers = []
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1, custom_previous_layers=[prevLayer])
        outputLayers.append(cs.layers[-1])

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1, custom_previous_layers=[prevLayer])
        prevLayer3x3_2 = cs.layers[-1]
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), custom_previous_layers=[prevLayer3x3_2])
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), custom_previous_layers=[prevLayer3x3_2])
        cs.Concat(custom_previous_layers=[cs.layers[-2], cs.layers[-1]])
        # self.branch3x3_2a = conv_block(384, 384*2, kernel_size=(1, 3), padding=(0, 1))
        outputLayers.append(cs.layers[-1])

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1, custom_previous_layers=[prevLayer])
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        prevLayer3x3dbl_3 = cs.layers[-1]
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), custom_previous_layers=[prevLayer3x3dbl_3])
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), custom_previous_layers=[prevLayer3x3dbl_3])
        cs.Concat(custom_previous_layers=[cs.layers[-2], cs.layers[-1]])
        # self.branch3x3dbl_3a = conv_block(384, 384*2, kernel_size=(1, 3), padding=(0, 1))

        outputLayers.append(cs.layers[-1])

        self.branch_pool_1 = cs.AvgPool2d(kernel_size=3, stride=1, padding=1, custom_previous_layers=[prevLayer])
        self.branch_pool_2 = conv_block(in_channels, 192, kernel_size=1)
        outputLayers.append(cs.layers[-1])
        cs.Concat(custom_previous_layers=outputLayers)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.prepool = cs.AvgPool2d(kernel_size=5, stride=3)
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.postpool = cs.AdaptiveAvgPool2d((1, 1))
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        # x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.prepool(x)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.postpool(x)
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = cs.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



# profiler = GpuProfiler("cuda")
# profiler.loadProfile()
# cs = CostSim(profiler)
# model = Inception3()

# cs.printAllLayers()
# cs.computeInputDimensions((3, 299,299))

# globalBatch = 16
# totalGpus = 4

# # cs.searchBestSplitsV2(totalGpus, globalBatch)
# # cs.searchBestSplits(8, globalBatch)


# for startLayerId in [6, 15, 24, 33]:
#     startLayer = cs.layers[startLayerId]

#     bestMultiChainTime = 9999999999
#     bestJiaTime = 9999999999
#     startConfig = cs.listConfigOptions(startLayer, globalBatch, totalGpus)[0]
#     startAndEndConfigsToTime = {}
#     startAndEndConfigsToTimeJia = {}
#     for startConfig in cs.listConfigOptions(startLayer, globalBatch, totalGpus):
#         print(startConfig)
#         startGpuTime = cs.benchGpuTime(startLayer, startConfig)
#         # (endLayer, configToTimeDict, t) = cs.searchMultiChain(startLayer, startConfig, globalBatch, totalGpus)
#         (jiaEndLayer, jiaConfigToTimeDict, jiaT) = cs.runMultiChainZhihao(startLayer, startConfig, globalBatch, totalGpus)
#         # startAndEndConfigsToTime[startConfig] = configToTimeDict
#         startAndEndConfigsToTimeJia[startConfig] = jiaConfigToTimeDict
#         for config in jiaConfigToTimeDict:
#             multiChainTime = jiaConfigToTimeDict[config][0] + startGpuTime
#             jiaTime = jiaConfigToTimeDict[config][0] + startGpuTime
#             print(" lastConfig: %20s,   multi-chain algo: %7.1f ms   Zhihao's time: %7.1f ms" % (str(config), multiChainTime, jiaTime))    
#             bestMultiChainTime = min(bestMultiChainTime, multiChainTime)
#             bestJiaTime = min(bestJiaTime, jiaTime)

#         bestConfigToTimeDict = (999999999, None)
#         bestEndConfig = None
#         for config in jiaConfigToTimeDict:
#             if bestConfigToTimeDict[0] > jiaConfigToTimeDict[config][0]:
#                 bestConfigToTimeDict = jiaConfigToTimeDict[config]
#                 bestEndConfig = config
#         cs.displayMultiChainResult(jiaEndLayer, bestEndConfig, jiaT, bestConfigToTimeDict[1])

#     print("Best multi-chain: %.2f  best jia: %.2f" % (bestMultiChainTime, bestJiaTime) )


# profiler.saveProfile()




def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False):
    profiler = GpuProfiler("cuda")
    profiler.loadProfile()
    global cs
    cs = CostSim(profiler, netBw=netBw, verbose=True, gpuProfileLoc="layerGpuProfileA100.txt")
    model = Inception3(aux_logits=False)
    cs.printAllLayers(slient=True)
    cs.computeInputDimensions((3,299,299))
    # job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)

    # if dataParallelBaseline:
    #     dpIterUsec, dpFpUsec, dpBpUsec = profiler.benchModel(model, (3, 299, 299), int(globalBatch / gpuCount))
    #     print("(DP baseline) whole model bench: %.1f ms (fp: %.1f, bp: %.1f)" % (dpIterUsec / 1000, dpFpUsec / 1000, dpBpUsec / 1000))

    job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))
    profiler.saveProfile()
    cs.to_dot(simResultFilename, globalBatch)
    # cs.to_gpuTimeline("Inception v3, Burst Parallel", maxGpusUsed, dataParallelBaseline)
    jobInJson = job.dumpInJSON()

    # for rank in range(gpuCount):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    job2 = TrainingJob("test", None, None, 0, 0, "")
    job2.loadJSON(jobInJson)
    assert(jobInJson == job2.dumpInJSON())
    print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    # print(jobInJson)
    
    if not spatialSplit and not simOnly:
        cc = ClusterClient()
        jobName = "InceptionV3_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        cc.submitTrainingJob(jobName, jobInJson)

    if simResultFilename != None:
        f = open(simResultFilename, "a")
        f.write("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))
        f.close()

        if gpuCount == 8:
            f = open(simResultFilename, "r")
            print(f.read())
            f.close()


def runAllConfigs(modelName: str, clusterType: str, simOnly=True):
    if clusterType == "V100":
        netBw = 22937
    elif clusterType == "A100":
        netBw = 2.66E5
    elif clusterType == "10Gbps":
        netBw = 1.25E3
    elif clusterType == "100Gbps":
        netBw = 1.25E4
    elif clusterType == "10Tbps":
        netBw = 1.25E6
    else:
        print("Wrong cluster type. Put either V100 or A100")

    gpuCounts = [1, 2, 4, 8]
    # gpuCounts = [1, 2, 4]
    globalBatchSize = 64
    # globalBatchSize = 16
    # globalBatchSize = 8
    limitAndBaseline = [(2.0, True, False), (1.5, False, False), (2.0, False, False), (5.0, False, False), (10.0, False, False)]
    # limitAndBaseline = [(99, False, True)]
    # limitAndBaseline = []
    for lim, baseline, spatialSplit in limitAndBaseline:
        simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim)
        f = open(simResultFilename, "w")
        f.write("#batch GPUs IterMs  GpuMs\n")
        f.close()

        for gpuCount in gpuCounts:
            if not simOnly:
                preSize = os.stat('runtimeResult.data').st_size
            main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit, simResultFilename=simResultFilename, simOnly=simOnly)
            # check exp finished.
            if not simOnly:
                print("runtimeResult.data's original size: ", preSize)
                while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
                    time.sleep(10)
                print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
        
        if not spatialSplit and not simOnly:
            fw = open("%s_%s_b%d_lim%2.1f_run.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim), "w")
            fr = open('runtimeResult.data', "r")
            fw.write("#batch GPUs IterMs  GpuMs\n")
            fw.write(fr.read())
            fw.close()
            fr.close()

        fr = open('runtimeResult.data', "w")
        fr.close()

def runStrongScalingBench():
    profiler = GpuProfiler("cuda")
    global cs
    netBw = 2.66E5
    cs = CostSim(profiler, netBw=netBw, verbose=False)
    inputSize = (3,224,224)
    model = Inception3(aux_logits=False)
    
    print("Model: ", "Inception3")
    print("BatchSize  iterMs    fpMs    bpMs")
    for batchSize in [2 ** exp for exp in range(1, 9)]:
        iterTime, fpTime, bpTime = profiler.benchModel(model, inputSize, batchSize)
        print(" %8d  %6.1f  %6.1f  %6.1f" %
            (batchSize, iterTime / 1000, fpTime / 10000, bpTime / 1000))

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 3:
        gpuCount = int(sys.argv[1])
        globalBatchSize = int(sys.argv[2])
        simResultFilename = "%s_%s_b%d_sim.data" % ("inception", "DP", globalBatchSize)
        main(gpuCount, globalBatchSize, dataParallelBaseline=True)
    elif len(sys.argv) == 4:
        gpuCount = int(sys.argv[1])
        globalBatchSize = int(sys.argv[2])
        amplificationLimit = float(sys.argv[3])
        simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % ("inception", "MP", globalBatchSize, amplificationLimit)
        main(gpuCount, globalBatchSize, amplificationLimit, simResultFilename = simResultFilename)#, netBw = 1.25E4)
    elif len(sys.argv) == 2:
        print("Run all configs")
        runAllConfigs("inceptionV3", sys.argv[1])
    elif len(sys.argv) == 1:
        runStrongScalingBench()
    else:
        print("Wrong number of arguments.\nUsage: ")
