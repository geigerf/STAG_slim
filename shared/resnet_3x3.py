import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.init as weight_init

__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, addConv=False):
    """3x3 convolution with padding"""
    # If an additional convolution should be used, remove the padding in the
    # ResNet blocks
    if addConv:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                         stride=1, padding=0, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                         stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 addConv=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, addConv=addConv)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, addConv=addConv)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, inplanes=64,
                 dropout=0.2, stride=1, dilation=1, kernel=3,
                 addConv=False, dropoutMax=0):
        self.inplanes = inplanes
        self.stride = stride
        self.dilation = dilation
        self.kernel = kernel
        self.addConv = addConv
        self.dropoutMax = dropoutMax
        super(ResNet, self).__init__()
        # Stride, dilation and kernel size are applied to 1/4 of the input
        # convolution filters
        if stride+dilation > 2 or kernel > 3:
            self.conv1 = nn.Conv2d(1, 3*inplanes//4, kernel_size=3,
                                   stride=stride, dilation=1, padding=1,
                                   bias=False)
            self.conv2 = nn.Conv2d(1, inplanes//4, kernel_size=kernel,
                                   stride=stride, dilation=dilation,
                                   padding=kernel//2 * dilation, bias=False)
            self.bn1 = nn.BatchNorm2d(3*inplanes//4)
            self.bn2 = nn.BatchNorm2d(inplanes//4)
        # Stride is also applied to all convolutions, and the maxpooling is
        # removed if there is a stride greater than 1
        else:
            self.conv1 = nn.Conv2d(1, inplanes, kernel_size=3, stride=stride,
                                   dilation=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(inplanes)
        # Dropout before the maxpooling layer
        if dropoutMax != 0:
            self.dropoutM = nn.Dropout2d(p=dropoutMax, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(10, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        weight_init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #    elif isinstance(m, nn.BatchNorm2d):
        #        weight_init.constant_(m.weight, 1)
        #        weight_init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, self.addConv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        #import pdb; pdb.set_trace()
        # Apply stride, dilation or kernel size to 1/4 of the input convolution
        # filters and concatenate the results afterwards
        if self.stride+self.dilation > 2 or self.kernel > 3:
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x2 = self.conv2(x)
            x2 = self.bn2(x2)
            x = torch.cat([x1, x2], dim=1)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
        x = self.relu(x)
        # Optional dropout before maxpooling
        if self.dropoutMax != 0:
            x = self.dropoutM(x)
        # If a stride larger than 1 was used before, maxpooling is not required
        if self.stride == 1:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


def resnet18(pretrained=False, inplanes=64, dropout=0.2, stride=1,
             dilation=1, addConv=False, dropoutMax=0, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], inplanes=inplanes,
                   dropout=dropout, stride=stride, dilation=dilation,
                   addConv=addConv, dropoutMax=dropoutMax, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
