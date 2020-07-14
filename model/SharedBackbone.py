import torch.nn as nn
import torchvision
import math
from torchvision.models.resnet import BasicBlock, Bottleneck, resnet50

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBackbone(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3,4,6], num_classes=1000):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.load_pretrained_layers()

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        res3_feats = self.layer2(x) # 1/8   512 channel
        res4_feats = self.layer3(res3_feats)    # 1/16  1024 channel

        return res3_feats, res4_feats


    def load_pretrained_layers(self):

        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.resnet50(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())[:-2] + list(pretrained_state_dict.keys())[258:-2]

        # print(param_names)
        # print(pretrained_param_names)

        for i, param in enumerate(param_names):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict, strict=True)

        print("Pretraied parameters with ImageNet have been loaded in before res4 layer")


if __name__ == '__main__':
    import torch

    model = ResNetBackbone()
    # print(model)

    tensor = torch.randn((3,3,300,300))
    out1, out2 = model(tensor)
    print(out1.shape, out2.shape)