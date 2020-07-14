from torchvision.models.resnet import Bottleneck
import torch.nn as nn
import torchvision
import math

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Res5Block(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3]):
        super(Res5Block, self).__init__()

        self.inplanes = 1024

        self.layer4 = self._make_layer(block, 512, layers[0], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self._load_pretrained_layers()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        res5_feats = self.layer4(x)

        return res5_feats

    def _load_pretrained_layers(self):

        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.resnet50(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())[258:-2]

        assert len(param_names) == len(pretrained_param_names)

        for i, param in enumerate(param_names):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict, strict=True)

        print("Pretrained parameters with ImageNet have been loaded in Res5 block.")


if __name__ == '__main__':
    import torch

    res5block = Res5Block(Bottleneck, [3])
    print(res5block)

    input = torch.randn((3,1024,19,19))
    output = res5block(input)
    print(output.shape)